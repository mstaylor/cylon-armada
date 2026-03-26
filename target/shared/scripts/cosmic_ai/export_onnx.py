"""Export AstroMAE model to ONNX format with optional model parallelism partitioning.

Converts the pre-trained PyTorch AstroMAE model to ONNX so it can run
via onnxruntime-node in the Path B (Node.js + WASM) pipeline. Optionally
partitions the ONNX graph into subgraphs for model parallelism across
Lambda functions using FMI.

Usage:
    # Full model export
    python export_onnx.py \
        --model-path /path/to/model.pt \
        --output-path /path/to/astromae.onnx

    # Export with model parallelism partitioning
    python export_onnx.py \
        --model-path /path/to/model.pt \
        --output-dir /path/to/partitions/ \
        --partition
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def _count_parameters(model):
    """Count parameters and estimate memory per named child module."""
    stage_info = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        # float32 = 4 bytes per param; activations roughly 2x params for inference
        param_mb = (params * 4) / (1024 * 1024)
        estimated_peak_mb = param_mb * 3  # params + gradients buffer + activations
        stage_info[name] = {
            "parameters": params,
            "param_mb": round(param_mb, 2),
            "estimated_peak_mb": round(estimated_peak_mb, 2),
        }
    return stage_info


def estimate_memory(model):
    """Estimate per-stage memory for model parallelism partitioning.

    Returns a partition report with recommended Lambda memory configs.
    """
    stage_info = _count_parameters(model)
    total_params = sum(s["parameters"] for s in stage_info.values())

    # Group into logical stages for AstroMAE
    # Stage 0 (ViT): patch_embed, cls_token, pos_embed, blocks, fc_norm, head, vit_block
    # Stage 1 (Inception): inception_model
    # Stage 2 (Fusion): concat_block
    vit_names = {"patch_embed", "cls_token", "pos_embed", "blocks", "fc_norm", "head", "vit_block"}
    inception_names = {"inception_model"}
    fusion_names = {"concat_block"}

    def _aggregate(names):
        params = sum(stage_info.get(n, {}).get("parameters", 0) for n in names)
        param_mb = (params * 4) / (1024 * 1024)
        peak_mb = param_mb * 3
        return {
            "parameters": params,
            "param_pct": round(100 * params / total_params, 1) if total_params > 0 else 0,
            "param_mb": round(param_mb, 2),
            "estimated_peak_mb": round(peak_mb, 2),
            "components": [n for n in names if n in stage_info],
        }

    report = {
        "total_parameters": total_params,
        "total_param_mb": round((total_params * 4) / (1024 * 1024), 2),
        "stages": {
            "stage_0_vit": _aggregate(vit_names),
            "stage_1_inception": _aggregate(inception_names),
            "stage_2_fusion": _aggregate(fusion_names),
        },
        "recommended_lambda_memory": {},
    }

    # Recommend Lambda memory: round up to nearest 256MB, minimum 256MB
    for stage_name, info in report["stages"].items():
        peak = info["estimated_peak_mb"]
        # Add 128MB headroom for runtime + ONNX Runtime overhead
        recommended = max(256, int(((peak + 128) + 255) // 256 * 256))
        report["recommended_lambda_memory"][stage_name] = recommended

    return report


# ---------------------------------------------------------------------------
# Full model export
# ---------------------------------------------------------------------------

def export(model_path, output_path, image_size=224, batch_size=1):
    """Export AstroMAE model to ONNX.

    Args:
        model_path: Path to pre-trained PyTorch model checkpoint.
        output_path: Path for the output ONNX file.
        image_size: Input image spatial dimension (default: 224).
        batch_size: Batch dimension for the exported model.

    Returns:
        (output_path, memory_report) tuple.
    """
    device = "cpu"

    logger.info("Loading model from %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(checkpoint, "module"):
        model = checkpoint.module
    else:
        model = checkpoint
    model = model.eval().to(device)

    # Memory estimation
    mem_report = estimate_memory(model)
    logger.info("Total parameters: %d (%.1f MB)",
                mem_report["total_parameters"], mem_report["total_param_mb"])
    for stage_name, info in mem_report["stages"].items():
        logger.info("  %s: %d params (%.1f%%), est. peak %.1f MB → Lambda %d MB",
                     stage_name, info["parameters"], info["param_pct"],
                     info["estimated_peak_mb"],
                     mem_report["recommended_lambda_memory"][stage_name])

    # AstroMAE expects [image, magnitude] as input
    dummy_image = torch.randn(batch_size, 5, image_size, image_size, device=device)
    dummy_magnitude = torch.randn(batch_size, 5, device=device)

    logger.info("Exporting to ONNX: %s", output_path)
    torch.onnx.export(
        model,
        ([dummy_image, dummy_magnitude],),
        output_path,
        input_names=["image", "magnitude"],
        output_names=["redshift"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "magnitude": {0: "batch_size"},
            "redshift": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("ONNX export verified: %.1f MB → %s", size_mb, output_path)

    return output_path, mem_report


# ---------------------------------------------------------------------------
# Model parallelism: stage-level export
# ---------------------------------------------------------------------------

class _ViTStage(nn.Module):
    """Stage 0: ViT encoder branch — extracts visual features."""
    def __init__(self, model):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.blocks = model.blocks
        self.fc_norm = model.fc_norm
        self.head = model.head
        self.vit_block = model.vit_block

    def forward(self, image):
        B = image.shape[0]
        x = self.patch_embed(image)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        x = self.vit_block(x)
        return x


class _InceptionStage(nn.Module):
    """Stage 1: Inception + magnitude branch — extracts photometric features."""
    def __init__(self, model):
        super().__init__()
        self.inception_model = model.inception_model

    def forward(self, image, magnitude):
        return self.inception_model([image, magnitude])


class _FusionStage(nn.Module):
    """Stage 2: Fusion — concatenates ViT and Inception outputs, predicts redshift."""
    def __init__(self, model):
        super().__init__()
        self.concat_block = model.concat_block

    def forward(self, vit_out, inception_out):
        x = torch.cat((vit_out, inception_out), dim=1)
        return self.concat_block(x)


def export_partitioned(model_path, output_dir, image_size=224, batch_size=1):
    """Export AstroMAE as partitioned ONNX subgraphs for model parallelism.

    Creates 3 ONNX files:
        stage_0_vit.onnx      — ViT encoder (image → vit_features)
        stage_1_inception.onnx — Inception branch ([image, magnitude] → incep_features)
        stage_2_fusion.onnx   — Fusion (vit_features + incep_features → redshift)

    Also writes partition_manifest.json with per-stage metadata.

    Args:
        model_path: Path to pre-trained PyTorch model checkpoint.
        output_dir: Directory for partition output files.
        image_size: Input image spatial dimension.
        batch_size: Export batch size.

    Returns:
        (manifest_path, memory_report) tuple.
    """
    device = "cpu"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model from %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(checkpoint, "module"):
        model = checkpoint.module
    else:
        model = checkpoint
    model = model.eval().to(device)

    mem_report = estimate_memory(model)

    dummy_image = torch.randn(batch_size, 5, image_size, image_size, device=device)
    dummy_magnitude = torch.randn(batch_size, 5, device=device)

    stages = [
        {
            "name": "stage_0_vit",
            "module": _ViTStage(model),
            "inputs": (dummy_image,),
            "input_names": ["image"],
            "output_names": ["vit_features"],
            "dynamic_axes": {"image": {0: "batch_size"}, "vit_features": {0: "batch_size"}},
        },
        {
            "name": "stage_1_inception",
            "module": _InceptionStage(model),
            "inputs": (dummy_image, dummy_magnitude),
            "input_names": ["image", "magnitude"],
            "output_names": ["inception_features"],
            "dynamic_axes": {
                "image": {0: "batch_size"},
                "magnitude": {0: "batch_size"},
                "inception_features": {0: "batch_size"},
            },
        },
        {
            "name": "stage_2_fusion",
            "module": _FusionStage(model),
            "inputs": (
                torch.randn(batch_size, 1096, device=device),
                torch.randn(batch_size, 2120, device=device),
            ),
            "input_names": ["vit_features", "inception_features"],
            "output_names": ["redshift"],
            "dynamic_axes": {
                "vit_features": {0: "batch_size"},
                "inception_features": {0: "batch_size"},
                "redshift": {0: "batch_size"},
            },
        },
    ]

    manifest = {
        "model_path": model_path,
        "image_size": image_size,
        "total_parameters": mem_report["total_parameters"],
        "total_param_mb": mem_report["total_param_mb"],
        "stages": [],
    }

    import onnx

    for stage in stages:
        onnx_path = os.path.join(output_dir, f"{stage['name']}.onnx")
        logger.info("Exporting %s → %s", stage["name"], onnx_path)

        torch.onnx.export(
            stage["module"],
            stage["inputs"],
            onnx_path,
            input_names=stage["input_names"],
            output_names=stage["output_names"],
            dynamic_axes=stage["dynamic_axes"],
            opset_version=17,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        stage_mem = mem_report["stages"].get(stage["name"], {})

        stage_entry = {
            "name": stage["name"],
            "onnx_file": f"{stage['name']}.onnx",
            "onnx_size_mb": round(size_mb, 2),
            "input_names": stage["input_names"],
            "output_names": stage["output_names"],
            "parameters": stage_mem.get("parameters", 0),
            "param_pct": stage_mem.get("param_pct", 0),
            "estimated_peak_mb": stage_mem.get("estimated_peak_mb", 0),
            "recommended_lambda_memory_mb": mem_report["recommended_lambda_memory"].get(
                stage["name"], 512
            ),
        }
        manifest["stages"].append(stage_entry)

        logger.info("  %s: %.1f MB ONNX, %d params (%.1f%%), Lambda %d MB",
                     stage["name"], size_mb,
                     stage_entry["parameters"], stage_entry["param_pct"],
                     stage_entry["recommended_lambda_memory_mb"])

    # Write manifest
    manifest_path = os.path.join(output_dir, "partition_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Partition manifest: %s", manifest_path)
    return manifest_path, mem_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export AstroMAE to ONNX")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to pre-trained PyTorch model")
    parser.add_argument("--output-path", type=str, default="astromae.onnx",
                        help="Output ONNX file path (full model export)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for partitioned export")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image spatial dimension")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Export batch size (dynamic axes enabled regardless)")
    parser.add_argument("--partition", action="store_true",
                        help="Export partitioned subgraphs for model parallelism")
    parser.add_argument("--memory-report", action="store_true",
                        help="Print memory estimation report only (no export)")

    args = parser.parse_args()

    if args.memory_report:
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model = checkpoint.module if hasattr(checkpoint, "module") else checkpoint
        report = estimate_memory(model.eval())
        print(json.dumps(report, indent=2))
    elif args.partition:
        output_dir = args.output_dir or os.path.splitext(args.output_path)[0] + "_partitions"
        export_partitioned(args.model_path, output_dir, args.image_size, args.batch_size)
    else:
        export(args.model_path, args.output_path, args.image_size, args.batch_size)