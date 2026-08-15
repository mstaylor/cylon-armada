# Rivanna Smoke Test Plan
## cylon-armada on UVA HPC — CPU and GPU Nodes

**Status**: In progress  
**Goal**: Validate `armada_ecs_runner.py` runs correctly under SLURM via Apptainer on both CPU (standard partition) and GPU (bii-gpu partition) before running the full Phase 1 experiment matrix.

---

## Architecture

Rivanna uses Apptainer (formerly Singularity) — Docker images are converted to `.sif` files pulled from Docker Hub. There are two independent pipelines on Rivanna, but they share **one image** (`cylon-armada-uccucx-python`). It's a strict superset of the FMI-only Lambda image — same shared scripts, same `armada_ecs_runner.py`, same FMI support, plus UCX/UCC built from source — and Rivanna has none of Lambda's constraints against running UCC/UCX (see below), so there's no reason to pull a second, FMI-only image here too. (`cylon-armada-fmi-python` is Lambda-specific.)

The agentic pipeline does not use any distributed communication library at all on Rivanna/ECS — `armada_ecs_runner.py` simulates `world_size` as in-process Python threads (`ThreadPoolExecutor`) sharing memory directly, not real separate processes, so there is nothing to broadcast across a network. (FMI is only exercised by the genuinely-distributed Lambda deployment, `armada_executor.py`, where Step Functions Maps each rank to a separate Lambda invocation.) Experiment B's collectives, by contrast, are real separate processes and use UCC/UCX directly — Rivanna's compute nodes have real, routable network connectivity (no NAT), so there's no need for FMI's rendezvous/NAT-traversal machinery either, which exists specifically to work around Lambda's inability to accept inbound connections.

**Agentic pipeline** — pattern mirrors `cylon/target/rivanna/scripts/ucc-ucx-redis/cylon-experiment-setup-apptainer.py`:

```
Docker Hub (mstaylor/cylon-armada-uccucx-python)
        │  apptainer pull
        ▼
  Rivanna scratch (cylon-armada-uccucx.sif)
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

**Experiment B collectives (UCC/UCX)** — real multi-rank fan-out via `srun -n $WORLD_SIZE`; each rank is a separate process, self-assigning its UCC rank via redis-OOB (`INCR` at `REDIS_HOST:REDIS_PORT` — not MPI, not rendezvous). Mirrors `cylon/target/rivanna/scripts/scaling_job.slurm`'s `srun -n N python script.py` pattern:

```
Docker Hub (mstaylor/cylon-armada-uccucx-python)
        │  apptainer pull
        ▼
  Rivanna scratch (cylon-armada-uccucx.sif)
        │  srun -n $WORLD_SIZE apptainer exec  (N real processes)
        ▼
  runExpBCollectives.sh  ← conda activate cylon_dev + PYTHONPATH
        │
        ▼
  python -m experiment.exp_b_collectives --channel ucc ...
        │  (UCC redis-OOB self-assigns each process its rank)
        ▼
  /output (bind-mounted local scratch — no S3 needed for this path yet)
```

---

## Files

| File | Purpose |
|------|---------|
| `target/rivanna/scripts/runArmada.sh` | Agentic pipeline container entrypoint — activates conda, sets PYTHONPATH, exec runner |
| `target/rivanna/scripts/armada-experiment-setup.py` | Generates SLURM scripts for the agentic pipeline (full parameter sweeps) |
| `target/rivanna/scripts/runExpBCollectives.sh` | Experiment B collectives container entrypoint — activates conda, sets PYTHONPATH, execs `exp_b_collectives.py` |
| `target/rivanna/scripts/run_exp_b_collectives.slurm` | Experiment B collectives SLURM script — real multi-rank `srun -n $WORLD_SIZE` fan-out |
| `target/rivanna/scripts/Makefile` | `smoke-cpu`, `smoke-gpu`, `expb-smoke`, `image-pull-gpu`, `image-pull-uccucx` targets |

---

## Prerequisites

### 1. Push images to Docker Hub (local machine)

```bash
ECR=448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada

docker tag $ECR:cylon-armada-gpu           qad5gv/cylon-armada-gpu:latest
docker tag $ECR:cylon-armada-uccucx-python qad5gv/cylon-armada-uccucx-python:latest

docker push qad5gv/cylon-armada-gpu:latest
docker push qad5gv/cylon-armada-uccucx-python:latest
```

`cylon-armada-uccucx-python` (built from `Dockerfile.uccucx.python`, adds UCX v1.19.1 + UCC v1.6.0 built from source) covers both pipelines on Rivanna (see Architecture above). `cylon-armada-fmi-python` (built from `Dockerfile.fmi.python`) is Lambda-specific — push it for Lambda deployment separately, but it isn't needed on Rivanna.

### 2. Pull SIF images on Rivanna (login node)

> **Space warning**: UCC/UCX SIF ~6-7 GB, GPU SIF ~20 GB. Apptainer's layer cache
> adds 2–3× overhead during the pull. This exceeds the default `$HOME`
> quota. The Makefile redirects `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to
> `$SCRATCH` automatically — do not override this.

```bash
cd /scratch/$USER/cylon-armada/target/rivanna/scripts

# UCC/UCX image (~6-7 GB SIF) — both the agentic pipeline and Experiment B collectives
make image-pull-uccucx DOCKER_USER=qad5gv

# GPU image (~20 GB SIF, ~50 GB scratch during pull) — allow 30–60 min
make image-pull-gpu DOCKER_USER=qad5gv

# Optional: clear cache after pulls to reclaim scratch space
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

**Note — this only confirms basic reachability, not UCC's OOB protocol.** A plain
`PING` exercises simple Redis GET/SET (what the agentic pipeline's context cache
uses). Experiment B's UCC channel additionally needs Redis to carry the `INCR`-based
rank-assignment and worker-address exchange multiple real processes do concurrently
during `srun -n $WORLD_SIZE` — this has only been validated locally and has **not**
yet been confirmed from Rivanna compute nodes. `make expb-smoke` (world_size=4) is
the first real test of that path; see the Open Issues table.

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

# Experiment B collectives smoke (world_size=4, real multi-rank UCC)
make expb-smoke

# Check queue
make q

# Watch queue
make qq
```

The agentic-pipeline smoke tests (`smoke-cpu`/`smoke-gpu`) each submit one SLURM job: 3 tasks, world_size=1, weak scaling. The third task is a duplicate of the first — expected cache hit rate ~33%.

`expb-smoke` submits a different kind of job: `--ntasks=4` real separate processes (not threads), each running `exp_b_collectives.py --channel ucc` and self-assigning its rank via redis-OOB. It measures 4 collectives (`scatter reduce broadcast barrier`) at 2 message sizes — a fast connectivity/correctness check, not a full sweep. Override world size with `make expb-smoke EXPB_WORLD_SIZE=8`, or invoke `run_exp_b_collectives.slurm` directly for full control over `COLLECTIVES`/`MSG_SIZES`/`WARMUP`/`REPS`/`RUNS` (see the usage comment at the top of that file).

---

## Validation Checklist

### Agentic pipeline (CPU / GPU)

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

### Experiment B collectives (`expb-smoke`)

| Check | Status |
|-------|--------|
| Job reaches RUNNING state with 4 tasks | ☐ |
| No `ModuleNotFoundError` in log | ☐ |
| `conda activate cylon_dev` succeeds | ☐ |
| All 4 ranks connect via UCC redis-OOB (no `INCR`/rank-assignment errors) | ☐ |
| Every collective produces a latency row (scatter/reduce/broadcast/barrier) | ☐ |
| `exp_b_collectives_results.csv` written to `$EXPB_RESULTS` | ☐ |
| `env.finalize()` exits cleanly (no MPI_Barrier segfault/abort) | ☐ |

---

## Full Experiment Runs (after smoke tests pass)

Use `armada-experiment-setup.py` for the full parameter sweep:

```bash
# CPU — hydrology scenario, threshold/dimension sweep
python armada-experiment-setup.py \
    -d /scratch/$USER/cylon-armada/cylon-armada-uccucx.sif \
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
| Basic Redis reachability from Rivanna compute nodes | ✅ | Confirmed via the `ws1` agentic-pipeline smoke test (`s3://staylor.dev2/results/rivanna/weak/rivanna_smoke_ws1_*`, 2026-05-03) — plain GET/SET works |
| UCC OOB rank-exchange over Redis from Rivanna (multi-rank) | High — unvalidated | Not yet confirmed: the `ws1` result above only exercises simple Redis GET/SET, not UCC's `INCR`-based concurrent rank assignment + worker-address exchange across real separate processes. `make expb-smoke` (world_size=4) is the first test of this specific path — no results yet |
| `cylon-armada-uccucx-python` image built and pushed to ECR/Docker Hub | ✅ | Built from `Dockerfile.uccucx.python` (adds UCX v1.19.1 + UCC v1.6.0 from source); run `make image-pull-uccucx` on Rivanna to pull it before `make expb-smoke`/`make cpu`/`make smoke-cpu` |
| GPU SIF pull time | Medium | ~20 GB SIF, allow 30–60 min from login node |
| AWS credentials propagation to compute nodes | ✅ | `aws configure` on login node; `~/.aws/credentials` NFS-mounted on all nodes |
| `pycylon.simd` not available in CPU image | Low | Falls back to numpy automatically |