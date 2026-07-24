# Cosmic AI Experiments — cylon-armada

## Overview

Cosmic AI integrates the **AstroMAE** deep learning model (arXiv:2501.06249 —
*Scalable Cosmic AI Inference using Cloud Serverless Computing with FMI*) into
the cylon-armada context-reuse framework. It demonstrates the framework on a
real scientific workload — photometric redshift prediction from SDSS galaxy
observations — rather than purely synthetic tasks.

Original source: `/home/parallels/AI-for-Astronomy/`

---

## Scripts

### Python (shared)

| File | Purpose |
|------|---------|
| `target/shared/scripts/cosmic_ai/inference.py` | Runs AstroMAE on SDSS `.pt` data partitions, returns predictions + metrics |
| `target/shared/scripts/cosmic_ai/task_generator.py` | Converts inference results into LLM analysis tasks using 5 templates |
| `target/shared/scripts/cosmic_ai/export_onnx.py` | Exports PyTorch AstroMAE to ONNX; optionally partitions into 3 stages for model parallelism |
| `target/shared/scripts/cosmic_ai/blocks/` | AstroMAE model architecture (ViT + Inception + Fusion) |

### Node.js (Lambda Path B)

| File | Purpose |
|------|---------|
| `target/aws/scripts/lambda/node/inference.mjs` | ONNX Runtime inference on Lambda; supports full-model and partitioned (model parallel) modes |
| `target/aws/scripts/lambda/node/task_generator.mjs` | Port of Python task_generator for Node.js path |
| `target/aws/scripts/lambda/node/armada_worker.mjs` | Lambda handler for model parallelism stages (ViT, Inception, Fusion) |

### Step Functions

| File | Purpose |
|------|---------|
| `target/aws/scripts/step_functions/workflow_model_parallel.asl.json` | Splits AstroMAE across 2 Lambda workers via FMI |

---

## What the Scripts Do

### AstroMAE Inference

Runs the pre-trained AstroMAE model on SDSS galaxy observations:

```
Input:
  images:     Float32 [B, 5, 224, 224]  — B galaxies × 5 SDSS bands × 224×224 pixels
  magnitudes: Float32 [B, 5]            — u, g, r, i, z photometric magnitudes

Output:
  predictions: Float32 [B, 1]           — predicted photometric redshift z_pred
  metrics: MAE, MSE, bias, NMAD, R², throughput (Gbps), samples/sec
```

### LLM Task Generation

`task_generator.py/mjs` converts inference results into semantically clustered
LLM analysis tasks using 5 templates:

| Template | Triggered when | Example |
|----------|---------------|---------|
| `redshift_analysis` | Normal prediction (even index) | "Analyze z_pred=0.451 (true z=0.443) for galaxy with magnitudes u=22.1, g=20.9..." |
| `color_classification` | Normal prediction (odd index) | "Given color indices u-g=1.22, g-r=0.57... classify morphological type" |
| `outlier_analysis` | Residual in top 10% | "AstroMAE predicted z=0.621, true z=0.312 — analyze this error" |
| `batch_summary` | Per batch (2 per run) | "Summarize batch MAE=0.021, bias=-0.003 for large-scale structure surveys" |
| `cost_analysis` | Per inference run | "Compare serverless cost for 500K galaxies at 0.38 Gbps vs HPC GPU" |

Tasks cluster naturally — two galaxies at similar redshifts with similar
magnitudes produce embeddings with cosine similarity ≥ 0.85, triggering
context reuse without any additional configuration.

### Model Parallelism Partitioning

`export_onnx.py` partitions AstroMAE into 3 stages for FMI-based model
parallelism across Lambda functions:

```
Stage 0 (ViT):       patch_embed + Transformer → vit_features [B, 1096]
Stage 1 (Inception): Inception + magnitude branch → inception_features [B, 2120]
Stage 2 (Fusion):    MLP concatenation → redshift [B, 1]
```

Stages 0 and 1 run in parallel on separate Lambda workers; FMI exchanges the
intermediate tensors; Stage 2 runs after both complete.

---

## Experiments

### Experiment 1 — Context Reuse on Cosmic AI Tasks

**Status: Partially run** (via `mixed_scientific` scenario)

The `mixed_scientific.json` scenario includes 6 astronomy tasks sampled from
the Cosmic AI task templates, mixed with epidemiology, hydrology, and seismology
tasks. These participate in the standard Phase 1/Phase 2 cylon-armada sweep.

**What is measured:**
- Cache hit rate for astronomy tasks as world size scales
- Cross-domain reuse: do astronomy tasks ever match non-astronomy contexts?
- Cost savings from context reuse on real scientific inference prompts

**To run a dedicated Cosmic AI sweep:**
```bash
conda run -n cylon_dev python3 \
  target/aws/scripts/experiment/cloud_sweep.py \
  --arch lambda-nodejs \
  --scenario mixed_scientific \
  --world-sizes 1 2 4 8 16 32 64 \
  --scaling weak \
  --task-count 1 \
  --runs 4 \
  --context-backend cylon \
  --fmi-channel direct
```

---

### Experiment 2 — Model Parallelism via FMI

**Status: Built, not yet run**

**Workflow:** `cylon-armada-model-parallel-workflow`
(`workflow_model_parallel.asl.json`)

Splits AstroMAE inference across 2 Lambda workers using Step Functions Parallel
state + FMI tensor exchange:

```
ParallelStages
├── Branch A: Stage0_ViT  (rank 0) — runs ViT encoder
└── Branch B: Stage1_Inception (rank 1) — runs Inception encoder
         ↓
    Stage2_Fusion — combines vit_features + inception_features → redshift
```

**What is measured:**
- `model_parallelism_overhead_ms` — extra latency vs single-Lambda full inference
- FMI tensor exchange bandwidth for intermediate activations
- Cost comparison: 1 Lambda (full model) vs 2 Lambdas (partitioned) × 2 invocations

**Prerequisites:**
1. AstroMAE model checkpoint exported to ONNX partitions and uploaded to S3
2. FMI sweep stable (currently in progress)

**To run:**
```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-model-parallel-workflow \
  --name "model-parallel-test-$(date +%s)" \
  --input file://target/aws/scripts/step_functions/test_events/model_parallel_test_event.json
```

---

### Experiment 3 — Full Cosmic AI Sweep with Real SDSS Data

**Status: Planned**

End-to-end pipeline:
```
SDSS .pt partitions
        ↓
AstroMAE inference (run_inference())
        ↓
task_generator (generate_tasks_from_results())
        ↓
armada_init → FMI sweep (ws1-ws64)
        ↓
context reuse metrics + cost savings
```

**Prerequisites:**
- AstroMAE model checkpoint available at S3 URI (`ONNX_MODEL_S3` env var)
- SDSS data partitions available (`.pt` files from AI-for-Astronomy repo)
- FMI sweep validated end-to-end

**What is measured (in addition to standard sweep metrics):**
- AstroMAE inference accuracy: MAE, NMAD, R² on real galaxy predictions
- Inference throughput (Gbps, samples/sec) on Lambda
- Task generation latency (inference → LLM task pipeline)
- Context reuse rate specifically for astronomy tasks

---

## Relationship to Prior Work

| Paper | What cylon-armada adds |
|-------|----------------------|
| arXiv:2501.06249 (Cosmic AI, FMI) | Context reuse layer on top of AstroMAE inference; FMI used for context broadcast not just data shuffle |
| Cylon distributed DataFrame (FMI scaling) | Same FMI collective infrastructure applied to LLM context sharing instead of DataFrame operations |

The key research contribution: Cosmic AI demonstrated that serverless FMI works
for HPC-scale astronomy inference. cylon-armada demonstrates that the same FMI
infrastructure can drive **cost reduction through context reuse**, a different axis
of optimization on the same workload.

---

*Last updated: May 2026*
*Experiments 1 (partial) complete · Experiment 2 built · Experiment 3 planned*