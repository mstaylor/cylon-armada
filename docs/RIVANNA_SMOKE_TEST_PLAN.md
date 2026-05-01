# Rivanna Smoke Test Plan
## cylon-armada on UVA HPC — CPU and GPU Nodes

**Status**: In progress  
**Goal**: Validate `armada_ecs_runner.py` runs correctly under SLURM via Apptainer on both CPU (standard partition) and GPU (bii-gpu partition) before running the full Phase 1 experiment matrix.

---

## Architecture

Rivanna uses Apptainer (formerly Singularity) — Docker images are converted to `.sif` files pulled from Docker Hub. The pattern mirrors `cylon/target/rivanna/scripts/ucc-ucx-redis/cylon-experiment-setup-apptainer.py`.

```
Docker Hub (mstaylor/...)
        │  apptainer pull
        ▼
  Rivanna scratch (.sif)
        │  apptainer exec / run
        ▼
  runArmada.sh  ← conda activate cylon_dev + PYTHONPATH
        │
        ▼
  armada_ecs_runner.py  ← reads env vars, calls Bedrock, writes to S3
        │
        ▼
  s3://staylor.dev2/results/rivanna/
```

---

## Files

| File | Purpose |
|------|---------|
| `target/rivanna/scripts/runArmada.sh` | Container entrypoint — activates conda, sets PYTHONPATH, exec runner |
| `target/rivanna/scripts/armada-experiment-setup.py` | Generates SLURM scripts (full parameter sweeps) |
| `target/rivanna/scripts/Makefile` | `smoke-cpu`, `smoke-gpu`, `image-pull`, `image-pull-gpu` targets |

---

## Prerequisites

### 1. Push images to Docker Hub (local machine)

```bash
ECR=448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada

docker tag $ECR:cylon-armada-python qad5gv/cylon-armada-python:latest
docker tag $ECR:cylon-armada-gpu    qad5gv/cylon-armada-gpu:latest

docker push qad5gv/cylon-armada-python:latest
docker push qad5gv/cylon-armada-gpu:latest
```

### 2. Pull SIF images on Rivanna (login node)

> **Space warning**: CPU SIF ~6 GB, GPU SIF ~20 GB. Apptainer's layer cache adds
> 2–3× overhead during the pull (~50–70 GB total). This exceeds the default `$HOME`
> quota. The Makefile redirects `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to
> `$SCRATCH` automatically — do not override this.

```bash
cd /scratch/$USER/cylon-armada/target/rivanna/scripts

# CPU image (~6 GB SIF, ~15 GB scratch during pull)
make image-pull DOCKER_USER=qad5gv

# GPU image (~20 GB SIF, ~50 GB scratch during pull) — allow 30–60 min
make image-pull-gpu DOCKER_USER=qad5gv

# Optional: clear cache after both pulls to reclaim scratch space (~30 GB)
make image-cache-clean
```

Pull from a **login node** only — compute nodes have no internet access.

### 3. AWS credentials on Rivanna

Run once on the Rivanna login node:

```bash
module load awscli

aws configure
# AWS Access Key ID:     <s3User access key from AWS console>
# AWS Secret Access Key: <s3User secret key>
# Default region name:   us-east-1
# Default output format: json
```

Credentials are stored in `~/.aws/credentials` — home directory, NFS-mounted on all Rivanna compute nodes automatically. Nothing goes to scratch.

Validate the credentials are working before submitting any jobs:

```bash
module load awscli

# Confirm identity and account
aws sts get-caller-identity
# Expected output:
# {
#     "UserId": "...",
#     "Account": "448324707516",
#     "Arn": "arn:aws:iam::448324707516:user/s3User"
# }

# Confirm Bedrock access (the most critical permission)
aws bedrock list-foundation-models --region us-east-1 \
    --query 'modelSummaries[?modelId==`amazon.titan-embed-text-v2:0`].modelId' \
    --output text
# Expected: amazon.titan-embed-text-v2:0

# Confirm S3 results bucket access
aws s3 ls s3://staylor.dev2/results/ --region us-east-1
# Expected: list of results directories (no AccessDenied error)
```

### 4. Verify Redis reachability

```bash
module load redis

redis-cli -h dev-cylon-redis1.aws-cylondata.com -p 6379 ping
# Expected: PONG
```

If this fails, Rivanna compute nodes cannot reach the Redis instance. Options:
- Open port 6379 on the Redis security group for Rivanna egress IPs
- Use `CONTEXT_BACKEND=redis` in a mode that tolerates Redis failure (context stored in DynamoDB only)

---

## Running Smoke Tests

From the Rivanna login node:

```bash
# Load modules once per session (Python prereqs required for python/3.9.16)
# All complex dependencies live inside the Apptainer SIF — no conda needed on host
module load intel-compilers/2023.1.0 impi/2021.9.0 python/3.9.16
module load apptainer awscli

cd /scratch/$USER/cylon-armada/target/rivanna/scripts

# CPU smoke (standard partition, SIMD=numpy)
make smoke-cpu

# GPU smoke (bii-gpu partition, SIMD=gcylon)
make smoke-gpu

# Check queue
make q

# Watch queue
make qq
```

Each smoke test submits one SLURM job: 3 tasks, world_size=1, weak scaling.  
The third task is a duplicate of the first — expected cache hit rate ~33%.

---

## Validation Checklist

| Check | CPU | GPU |
|-------|-----|-----|
| Job reaches RUNNING state | ☐ | ☐ |
| No `ModuleNotFoundError` in log | ☐ | ☐ |
| `conda activate cylon_dev` succeeds | ☐ | ☐ |
| Bedrock embedding call succeeds | ☐ | ☐ |
| `backend: numpy` in stopwatch | ☐ | — |
| `backend: gcylon` in stopwatch | — | ☐ |
| `savings_pct > 0` (cache hit on task 3) | ☐ | ☐ |
| Results in `s3://staylor.dev2/results/rivanna/weak/` | ☐ | ☐ |
| No `CYLON_SESSION_ID` warning in logs | ☐ | ☐ |
| No Redis connection error in logs | ☐ | ☐ |

---

## Full Experiment Runs (after smoke tests pass)

Use `armada-experiment-setup.py` for the full parameter sweep:

```bash
# CPU — hydrology scenario, threshold/dimension sweep
python armada-experiment-setup.py \
    -d /scratch/$USER/cylon-armada/cylon-armada.sif \
    -r1 dev-cylon-redis1.aws-cylondata.com \
    -n 1 -t 1 -c 10 \
    --tasks 4 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON \
    --runs 3 --scenario hydrology \
    -l1 /scratch/$USER/armada-results -l2 /output \
    -p2 standard

# GPU — hydrology scenario, gcylon backend
python armada-experiment-setup.py \
    -d /scratch/$USER/cylon-armada/cylon-armada-gpu.sif \
    -r1 dev-cylon-redis1.aws-cylondata.com \
    -n 1 -t 1 -c 10 \
    --tasks 4 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends GCYLON \
    --runs 3 --scenario hydrology --gpu \
    -l1 /scratch/$USER/armada-results-gpu -l2 /output \
    -p2 bii-gpu
```

---

## Open Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| Apptainer cache fills `$HOME` quota | High | Fixed: Makefile sets `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to scratch; run `make image-cache-clean` after pull |
| Redis reachability from Rivanna compute nodes | High | Test with `redis-cli ping` before submitting; open port 6379 to Rivanna egress IPs if needed |
| GPU SIF pull time | Medium | ~20 GB SIF, allow 30–60 min from login node |
| AWS credentials propagation to compute nodes | ✅ | `aws configure` on login node; `~/.aws/credentials` NFS-mounted on all nodes |
| `pycylon.simd` not available in CPU image | Low | Falls back to numpy automatically |