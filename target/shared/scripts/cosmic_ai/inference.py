"""AstroMAE inference module — adapted for agentic workflows.

Refactored from AI-for-Astronomy/code/Anomaly Detection/Inference/inference.py.
Instead of a standalone CLI script, this is a callable module that returns
structured predictions and metadata for downstream LLM task generation.

Original: arXiv:2501.06249 — Scalable Cosmic AI Inference using Cloud
Serverless Computing with FMI.
"""

import gc
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def load_data(data_path, device="cpu"):
    """Load an SDSS data partition (.pt file).

    Returns:
        TensorDataset with tensors: (images, magnitudes, redshifts)
    """
    return torch.load(data_path, map_location=device, weights_only=False)


def load_model(model_path, device="cpu"):
    """Load a pre-trained AstroMAE model checkpoint.

    Returns:
        Model in eval mode on the specified device.
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(model, "module"):
        model = model.module
    return model.eval()


def run_inference(
    model,
    dataset,
    batch_size=512,
    device="cpu",
    num_workers=0,
):
    """Run inference on a dataset and return per-sample results.

    Args:
        model: Pre-trained AstroMAE model in eval mode.
        dataset: TensorDataset with (images, magnitudes, redshifts).
        batch_size: Inference batch size.
        device: Device for computation ('cpu' or 'cuda').
        num_workers: DataLoader workers.

    Returns:
        dict with:
            predictions: numpy array of predicted redshifts
            true_redshifts: numpy array of true redshift labels
            magnitudes: numpy array of magnitude values (N, 5)
            metrics: dict with timing and throughput info
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False,
        num_workers=num_workers,
    )

    predictions = []
    true_redshifts = []
    magnitudes_all = []
    total_data_bits = 0

    start = time.perf_counter()

    with torch.no_grad():
        for data in dataloader:
            images = data[0].to(device)
            mags = data[1].to(device)
            labels = data[2]

            output = model([images, mags])

            predictions.append(output.cpu().numpy().flatten())
            true_redshifts.append(labels.numpy().flatten())
            magnitudes_all.append(data[1].numpy())

            total_data_bits += (
                images.element_size() * images.nelement() * 8
                + mags.element_size() * mags.nelement() * 8
            )

            gc.collect()

    elapsed = time.perf_counter() - start
    num_samples = len(dataset)

    return {
        "predictions": np.concatenate(predictions),
        "true_redshifts": np.concatenate(true_redshifts),
        "magnitudes": np.concatenate(magnitudes_all),
        "metrics": {
            "total_time_s": round(elapsed, 4),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "num_batches": len(dataloader),
            "throughput_bps": total_data_bits / elapsed if elapsed > 0 else 0,
            "samples_per_sec": num_samples / elapsed if elapsed > 0 else 0,
            "device": device,
        },
    }


def compute_metrics(predictions, true_redshifts):
    """Compute standard photometric redshift evaluation metrics.

    Adapted from Plot_Redshift.py::err_calculate().

    Returns:
        dict with MAE, MSE, bias, precision (NMAD), R2 score
    """
    predictions = np.asarray(predictions)
    true_z = np.asarray(true_redshifts)

    mae = np.mean(np.abs(predictions - true_z))
    mse = np.mean((predictions - true_z) ** 2)

    delta_z = (predictions - true_z) / (1 + true_z)
    bias = np.mean(delta_z)
    nmad = 1.48 * np.median(np.abs(delta_z - np.median(delta_z)))

    ss_res = np.sum((true_z - predictions) ** 2)
    ss_tot = np.sum((true_z - np.mean(true_z)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": round(float(mae), 6),
        "mse": round(float(mse), 6),
        "bias": round(float(bias), 6),
        "precision_nmad": round(float(nmad), 6),
        "r2": round(float(r2), 6),
    }