I # Experiment Playbook

Step-by-step guide for executing Phase 1 experiments.

---

## Pipeline Overview

```
1. Local Validation     → verify pipeline works before spending AWS money
2. AWS Deployment       → push images, apply Terraform, upload scripts
3. Experiment Execution → run scenarios across execution paths
4. Results Analysis     → collect data, generate visualizations, validate metrics
```

---

## Stage 1: Local Validation

### 1.1 Environment Setup

```bash
conda activate cylon_dev

# Verify dependencies
python -c "import boto3, redis, numpy, langchain_aws, llama_index; print('Python deps OK')"

# Build Cython SIMD extension (Path A2)
cd python/simd
CYLON_PREFIX=$CYLON_PREFIX python setup.py build_ext --inplace
cd -
```

### Host Services (Redis)

Redis runs on the **host OS** (Mac), not inside the Parallels VM.
The VM connects via the Parallels network IP. See [Cylon ENVIRONMENT_SETUP.md](../../cylon/ENVIRONMENT_SETUP.md) for details.

```bash
# On the host OS (Mac):
brew install redis
redis-server
```

```bash
# In the Parallels VM — find the host IP:
ip neighbor show
# Look for: 10.211.55.2 dev enp0s5 ... REACHABLE

# Set environment variables for the VM:
export REDIS_HOST=10.211.55.2
export REDIS_PORT=6379
```

Redis is the **primary persistence layer** (default). Context metadata JSON and
workflow membership sets are stored in Redis alongside the search embeddings,
enabling cross-invocation reuse within the TTL window.

### Persistence Configuration

The context manager has two independent persistence layers:

| Layer | Default | Env var to enable | Purpose |
|-------|---------|-------------------|---------|
| Redis metadata | **ON** | `REDIS_HOST` / `REDIS_PORT` | Primary — cross-invocation reuse within TTL |
| DynamoDB | **OFF** | `DYNAMO_TABLE_NAME` | Optional — durable long-term history + analytics |

**Redis only (default)** — recommended for experiments:
```bash
export REDIS_HOST=10.211.55.2
# DYNAMO_TABLE_NAME not set → DynamoDB disabled
```

**Redis + DynamoDB** — for durable audit trail:
```bash
export REDIS_HOST=10.211.55.2
export DYNAMO_TABLE_NAME=cylon-armada-context-store
export DYNAMO_ENDPOINT_URL=http://10.211.55.2:8100  # local dev only

# Start DynamoDB Local (host OS):
docker run -p 8100:8000 amazon/dynamodb-local:latest -jar DynamoDBLocal.jar -sharedDb

# Create table (from VM):
aws dynamodb create-table \
    --table-name cylon-armada-context-store \
    --attribute-definitions \
        AttributeName=context_id,AttributeType=S \
        AttributeName=workflow_id,AttributeType=S \
        AttributeName=created_at,AttributeType=S \
    --key-schema \
        AttributeName=context_id,KeyType=HASH \
        AttributeName=workflow_id,KeyType=RANGE \
    --global-secondary-indexes \
        'IndexName=workflow_id-created_at-index,KeySchema=[{AttributeName=workflow_id,KeyType=HASH},{AttributeName=created_at,KeyType=RANGE}],Projection={ProjectionType=ALL}' \
    --billing-mode PAY_PER_REQUEST \
    --endpoint-url $DYNAMO_ENDPOINT_URL
```

**In-memory only** — no external services required (contexts lost between runs):
```bash
export CONTEXT_BACKEND=redis   # or cylon
# REDIS_HOST not needed — set persist_to_redis=False in code or use --no-persist flag
```

### 1.2 Run Tests

```bash
# Python (69 tests)
python -m pytest tests/ -v

# Node.js (33 tests)
cd target/aws/scripts/lambda/node && npm test && cd -
```

### 1.3 Smoke Test

Ensure `REDIS_HOST` is set (see above). `DYNAMO_TABLE_NAME` is optional — omit to use Redis-only persistence.

```bash
# Redis backend (default)
target/shared/scripts/experiment/examples/smoke_test.sh

# Cylon ContextTable backend
target/shared/scripts/experiment/examples/smoke_test.sh --backend cylon
```

Or run the runner directly:

```bash
# Redis backend
python target/shared/scripts/experiment/runner.py \
    --context-backend redis \
    --tasks-file target/shared/scripts/experiment/scenarios/hydrology.json \
    --tasks 4 \
    --thresholds 0.8 \
    --dimensions 256 \
    --output target/shared/scripts/experiment/results/smoke_test

# Cylon ContextTable backend
python target/shared/scripts/experiment/runner.py \
    --context-backend cylon \
    --tasks-file target/shared/scripts/experiment/scenarios/hydrology.json \
    --tasks 4 \
    --thresholds 0.8 \
    --dimensions 256 \
    --output target/shared/scripts/experiment/results/smoke_test
```

Checklist:
- [ ] Result files created (`*_summary.csv`, `*_stopwatch.csv`)
- [ ] `savings_pct` > 0 for reuse runs
- [ ] `savings_pct` = 0 for baseline runs
- [ ] Stratified sampling selected tasks from different categories
- [ ] No errors in logs

### 1.4 Node.js Smoke Test (Path B)

The Node.js path supports two context backends:
- **wasm** (default): WASM SIMD128 via cylon-wasm — requires cylon-wasm built
- **redis**: Pure JS `Float32Array` dot product — no WASM dependency

#### WASM backend setup (one-time)

Requires cylon-wasm built (`wasm-pack build --target nodejs --release` in `cylon/rust/cylon-wasm`).

```bash
# Create cylon_host stub and verify WASM setup
target/shared/scripts/experiment/examples/setup_wasm.sh

# Custom pkg path (if built elsewhere):
target/shared/scripts/experiment/examples/setup_wasm.sh --wasm-pkg /path/to/pkg
```

#### Similarity benchmark

```bash
# WASM SIMD benchmark (default)
target/shared/scripts/experiment/examples/run_nodejs_simd.sh

# JS dot product benchmark (no WASM required)
target/shared/scripts/experiment/examples/run_nodejs_simd.sh --context-backend redis

# With custom params
target/shared/scripts/experiment/examples/run_nodejs_simd.sh --dim 1024 --n 5000
```

Expected output: `comparisons_per_sec` > 100,000 for 256-dim with WASM, `avg_search_ms` < 10ms for 1000 embeddings.

#### Route task experiment

```bash
# WASM backend (requires Redis + Bedrock + WASM)
target/shared/scripts/experiment/examples/run_nodejs_route.sh

# Redis/JS backend (requires Redis + Bedrock, no WASM)
target/shared/scripts/experiment/examples/run_nodejs_route.sh --context-backend redis

# With custom params
target/shared/scripts/experiment/examples/run_nodejs_route.sh --tasks 8 --threshold 0.9
```

Or run the runner directly:

```bash
# WASM backend
node target/aws/scripts/lambda/node/run_experiment.mjs \
    --action simd_benchmark --dim 256 --n 1000

# Redis/JS backend
node target/aws/scripts/lambda/node/run_experiment.mjs \
    --context-backend redis \
    --action simd_benchmark --dim 256 --n 1000
```

### 1.5 S3 Upload (Optional)

Both Python and Node.js runners support `--s3-bucket` for uploading results to S3.
Omit to keep results local only.

```bash
# Python with S3 upload
python target/shared/scripts/experiment/runner.py \
    --context-backend cylon \
    --tasks-file target/shared/scripts/experiment/scenarios/hydrology.json \
    --tasks 4 --thresholds 0.8 --dimensions 256 \
    --s3-bucket cylon-armada-results \
    --s3-prefix experiments/hydrology \
    --output target/shared/scripts/experiment/results/smoke_test

# Node.js with S3 upload
node run_experiment.mjs \
    --action simd_benchmark --dim 256 --n 1000 \
    --s3-bucket cylon-armada-results \
    --s3-prefix experiments/simd
```

### Output Files

Each experiment produces two files (matching Cylon's scaling.py pattern):

| File | Content | Used By |
|------|---------|---------|
| `*_stopwatch.csv` | cloudmesh benchmark (system info + timings) | Full audit trail |
| `*_summary.csv` | Data-only CSV (timings + metrics, no machine info) | Results pipeline aggregator |

Both Python and Node.js produce identical CSV column names for cross-path comparison.

### Multiple Runs (Standard Deviation)

Use `--runs N` to repeat each configuration N times. Each run writes to a
separate subdirectory (`run_1/`, `run_2/`, ...). The results pipeline
aggregator computes mean and std dev across runs automatically.

```bash
# 3 runs per config for std dev
python target/shared/scripts/experiment/runner.py \
    --context-backend redis \
    --tasks 4 8 --thresholds 0.8 --dimensions 256 \
    --runs 3 \
    --output target/shared/scripts/experiment/results/multi_run
```

---

## Stage 2: AWS Deployment

### 2.1 Build and Push Docker Images

One image per runtime — each Lambda function uses a different CMD override.

```bash
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Python (Path A1/A2)
docker build -t cylon-armada-python -f docker/Dockerfile.python .
docker tag cylon-armada-python $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-python:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-python:latest

# Node.js (Path B)
docker build -t cylon-armada-nodejs -f docker/Dockerfile.nodejs .
docker tag cylon-armada-nodejs $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-nodejs:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-nodejs:latest

# Docker Hub (for Rivanna / Singularity)
docker tag cylon-armada-python docker.io/$DOCKER_USER/cylon-armada-python:latest
docker push docker.io/$DOCKER_USER/cylon-armada-python:latest
```

### 2.2 Verify Deployment

```bash
# Test Python init Lambda
aws lambda invoke \
    --function-name cylon-armada-init \
    --payload '{"workflow_id":"test","tasks":["Hello world"],"config":{}}' \
    /tmp/init_test.json && cat /tmp/init_test.json

# Test Node.js init Lambda
aws lambda invoke \
    --function-name cylon-armada-init-node \
    --payload '{"workflow_id":"test","tasks":["Hello world"],"config":{}}' \
    /tmp/init_node_test.json && cat /tmp/init_node_test.json
```

---

## Stage 3: Experiment Execution

### 3.1 Task Sampling

Tasks are selected using **stratified sampling** by default — evenly spaced across the task list with seeded jitter. This ensures coverage of all task categories at any subset size.

```bash
# Stratified (default) — 4 tasks, one from each category
python runner.py --tasks-file scenarios/hydrology.json --tasks 4

# Sequential — first N tasks (for debugging)
python runner.py --tasks-file scenarios/hydrology.json --tasks 4 --sampling sequential

# Random — uniform sample with seed
python runner.py --tasks-file scenarios/hydrology.json --tasks 4 --sampling random --seed 99
```

Same seed + same strategy = identical task selection across runs. Baseline and reuse runs use the same sampled tasks for paired comparison.

### 3.2 Experiment Matrix

Each scenario runs a parameter sweep across Python and Node.js paths, plus an optional LlamaIndex baseline.

**Python (Path A — cylon-armada)**

| Variable | Values |
|----------|--------|
| Context backend | cylon, redis |
| Similarity threshold | 0.70, 0.80, 0.90 |
| Embedding dimensions | 256, 512, 1024 |
| SIMD backend | NUMPY, PYCYLON, CYTHON_BATCH |
| Baseline | yes, no |

Per scenario: 2 × 3 × 3 × 3 × 2 = **108 configurations**

**Node.js (Path B — cylon-armada)**

| Variable | Values |
|----------|--------|
| Context backend | wasm, redis |
| Similarity threshold | 0.70, 0.80, 0.90 |
| Embedding dimensions | 256, 512, 1024 |
| Baseline | yes, no |

Per scenario: 2 × 3 × 3 × 2 = **36 configurations**

**LlamaIndex baseline (comparator — `--include-llamaindex`)**

| Variable | Values |
|----------|--------|
| System | llamaindex |
| Task counts | same as cylon runs |
| Embedding dimensions | 256, 1024 |
| Reuse rate | always 0% (LLM called for every task) |

One config per (task_count, dimension) pair — no threshold or SIMD backend dimensions since LlamaIndex always calls the LLM. The `system` column in the aggregated CSV distinguishes `cylon` from `llamaindex` rows for direct cost and latency comparison.

> **Why LlamaIndex is Python-only (no Node.js/WASM equivalent)**
>
> Although a LlamaIndex JS package exists (`llamaindex` on npm), it cannot be
> used as a comparator for Path B. LlamaIndex JS implements its own pure-JS
> `VectorStoreIndex` with no hook to substitute an external WASM search engine.
> Plugging in `cylon-wasm` would require replacing LlamaIndex's entire vector
> store layer — effectively rebuilding `context_handler.mjs`.
>
> For Path B the equivalent baseline is the existing `baseline` mode
> (`system=cylon`, `threshold=1.0`): all tasks call the LLM, reuse rate = 0%,
> identical cost profile to LlamaIndex. The Node.js path's primary performance
> claim is the WASM SIMD128 search itself — no existing framework including
> LlamaIndex JS uses it, making the WASM vs pure-JS comparison (cylon WASM
> vs cylon NUMPY fallback) the result of interest on that path.

### 3.3 Run All Scenarios (Local)

```bash
# Run all scenarios with default parameter sweep (redis backend)
target/shared/scripts/experiment/examples/run_all_scenarios.sh

# Run all scenarios with Cylon ContextTable backend
target/shared/scripts/experiment/examples/run_all_scenarios.sh --backend cylon

# Or run individual scenarios
target/shared/scripts/experiment/examples/run_scenario.sh hydrology
target/shared/scripts/experiment/examples/run_scenario.sh hydrology --backend cylon
target/shared/scripts/experiment/examples/run_scenario.sh epidemiology
target/shared/scripts/experiment/examples/run_scenario.sh seismology
target/shared/scripts/experiment/examples/run_scenario.sh mixed_scientific

# With LlamaIndex baseline for comparison
python target/shared/scripts/experiment/runner.py \
    --context-backend redis \
    --tasks-file target/shared/scripts/experiment/scenarios/hydrology.json \
    --tasks 4 8 16 32 \
    --thresholds 0.80 \
    --dimensions 256 \
    --include-llamaindex \
    --runs 3 \
    --output target/shared/scripts/experiment/results/hydrology_comparison

# With custom parameters
target/shared/scripts/experiment/examples/run_scenario.sh hydrology \
    --backend cylon \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH

# Scenario 1: Astronomical inference (cosmic-ai — dynamic tasks)
python target/shared/scripts/experiment/runner.py \
    --context-backend cylon \
    --cosmic-ai \
    --data-path /path/to/sdss/data.pt \
    --model-path /path/to/astromae/model.pt \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/shared/scripts/experiment/results/scenario1_cosmicai
```

### 3.4 Run on AWS (Step Functions)

The Step Functions workflow follows the cylon paper architecture:
`ArmadaInit` (embed tasks) → `ProcessTasks` (Map: route each task) → `AggregateResults`.

Infrastructure config (Redis, DynamoDB, Bedrock models) is set via Lambda
environment variables — only the workflow payload (tasks + thresholds) is
passed through Step Functions.

```bash
PYTHON_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw python_workflow_arn)
NODEJS_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw nodejs_workflow_arn)

# Run a scenario via Step Functions (Python path)
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-th08-d256',
    'tasks': tasks,
    'config': {
        'similarity_threshold': '0.80',
        'embedding_dimensions': '256'
    }
}))
")" --query 'output' --output text > target/shared/scripts/experiment/results/aws_hydrology.json

# Node.js path
aws stepfunctions start-sync-execution \
    --state-machine-arn $NODEJS_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-node-th08',
    'tasks': tasks,
    'config': {
        'similarity_threshold': '0.80',
        'embedding_dimensions': '256'
    }
}))
")" --query 'output' --output text > target/shared/scripts/experiment/results/aws_hydrology_node.json
```

### 3.5 Per-Scenario Checklist

For each scenario:

- [ ] Clear context store before each run
- [ ] Run baseline (threshold=1.0) first
- [ ] Run reuse experiment with same tasks (same seed)
- [ ] Run LlamaIndex baseline with `--include-llamaindex` (same tasks, same seed)
- [ ] Verify result JSON has all expected fields
- [ ] Check `reuse_rate` is within expected range for the scenario
- [ ] Check LlamaIndex `reuse_rate` = 0% and `total_cost` > cylon reuse `total_cost`
- [ ] Save raw results to `target/shared/scripts/experiment/results/`

---

## Stage 4: Results Analysis

### 4.1 Results Pipeline

The results pipeline follows the cylon paper pattern: download → aggregate → charts → notebook.
It parses `_summary.csv` files, groups by experiment name, computes cross-run mean and
standard deviation (N-1), and generates publication-quality charts.

```bash
cd target/shared/scripts

# Full pipeline from local results
python -m results.pipeline \
    --local-dir experiment/results/ \
    --output-dir experiment/output/ \
    --chart-format svg

# Individual steps
python -m results.pipeline --local-dir experiment/results/ --step aggregate
python -m results.pipeline --local-dir experiment/results/ --step charts
python -m results.pipeline --local-dir experiment/results/ --step notebook

# Download from S3 first (AWS experiments)
python -m results.pipeline \
    --config configs/experiment_config.yaml \
    --step download --step aggregate --step charts
```

#### Pipeline outputs

| File | Content |
|------|---------|
| `aggregated_results.csv` | One row per config: `{metric}_mean`, `{metric}_std` |
| `cost_savings.svg` | Reuse vs. baseline cost bar chart |
| `reuse_rate.svg` | Reuse rate by threshold and context backend |
| `latency_breakdown.svg` | Stacked bar: search vs. LLM latency |
| `cost_scaling.svg` | Cost vs. task count per platform |
| `infrastructure_comparison.svg` | Cost and latency across platforms |
| `threshold_sensitivity.svg` | Reuse rate and savings vs. threshold |
| `simd_comparison.svg` | Search latency by SIMD backend |
| `dimension_impact.svg` | Search latency and reuse vs. embedding dimensions |
| `context_reuse_results.ipynb` | Interactive Jupyter notebook with all charts |

### 4.2 Key Metrics

| Metric | CSV Column | Unit |
|--------|-----------|------|
| Total cost | `total_cost_mean` / `_std` | USD |
| Baseline cost | `baseline_cost_mean` / `_std` | USD |
| Savings % | `savings_pct_mean` / `_std` | % |
| Reuse rate | `reuse_rate_mean` / `_std` | % |
| Cache hits | `cache_hits_mean` / `_std` | count |
| LLM calls | `llm_calls_mean` / `_std` | count |
| Total latency | `total_ms_mean` / `_std` | seconds |
| Search latency | `search_latency_ms_mean` / `_std` | seconds |
| LLM latency | `llm_latency_ms_mean` / `_std` | seconds |

### 4.3 Success Criteria Validation

| Metric | Target | Check |
|--------|--------|-------|
| Cost reduction | 60-80% | `savings_pct_mean` across all scenarios |
| Reuse quality | >0.80 ROUGE-L | Compare reused vs baseline responses |
| Search latency | <20ms / 1000 contexts | `search_latency_ms_mean` |
| SIMD speedup | >2x vs numpy | Path A1/A2 vs NUMPY comparison |
| All 5 scenarios | Complete | All result files present |
| Statistical rigor | 3+ runs per config | `num_runs` ≥ 3 in aggregated CSV |
| LlamaIndex comparison | cylon cost < llamaindex cost | Filter `system` column: cylon `total_cost_mean` < llamaindex `total_cost_mean` at all task counts |
| LlamaIndex reuse | llamaindex `reuse_rate` = 0% | Confirms baseline has no zero-cost path |

---

## Cost Estimation

With 3 runs per config for standard deviation:

| Component | Per Run | × 270 Configs × 3 Runs | Total |
|-----------|---------|------------------------|-------|
| Titan V2 embeddings | ~$0.001 | $0.81 | $0.81 |
| Nova Lite LLM | ~$0.01 | $8.10 | $8.10 |
| Lambda compute | ~$0.002 | $1.62 | $1.62 |
| DynamoDB (optional) | ~$0.001 | $0.81 | $0.81 |
| ElastiCache | $0.017/hr | ~12 hrs | $0.20 |
| **Total (Lambda only)** | | | **~$11.54** |

Additional infrastructure costs (per experiment batch):

| Infrastructure | Instance | Est. Cost |
|----------------|----------|-----------|
| ECS Fargate | 0.25 vCPU / 512MB | ~$0.01/run |
| ECS EC2 (GPU) | g4dn.xlarge (T4) | ~$0.53/hr |
| Rivanna HPC | GPU partition (A100) | Allocation-based |

---

## Stage 5: Multi-Infrastructure Experiments

The same container image runs across all infrastructure targets — only the
orchestration layer changes.

### Infrastructure Overview

| Platform | Orchestration | Container Source | GPU |
|----------|--------------|------------------|-----|
| AWS Lambda | Step Functions Map | ECR | No |
| AWS Fargate | ECS Task / Step Functions `ecs:RunTask` | ECR | No |
| AWS ECS EC2 | ECS Task on GPU instance (g4dn.xlarge) | ECR | NVIDIA T4 |
| Rivanna HPC | SLURM job array | Docker Hub → Singularity | NVIDIA A100/V100 |

### 5.1 ECS Fargate Deployment

Fargate uses the same Python/Node.js image from ECR — no GPU, serverless compute.

```bash
# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://target/aws/scripts/ecs/task_definition_fargate.json

# Run a task
aws ecs run-task \
    --cluster cylon-armada \
    --task-definition cylon-armada-fargate \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --overrides '{"containerOverrides":[{"name":"armada","command":["armada_executor.handler"],"environment":[{"name":"WORKFLOW_ID","value":"exp-fargate-001"}]}]}'
```

### 5.2 ECS EC2 GPU Deployment

Uses `Dockerfile.gpu` with CUDA runtime for GPU-accelerated Cylon SIMD.
Deploy on `g4dn.xlarge` (NVIDIA T4, 16GB VRAM) or `g5.xlarge` (NVIDIA A10G, 24GB).

```bash
# Build GPU image
docker build -t cylon-armada-gpu -f docker/Dockerfile.gpu .
docker tag cylon-armada-gpu $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-gpu:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cylon-armada-gpu:latest

# Register GPU task definition
aws ecs register-task-definition \
    --cli-input-json file://target/aws/scripts/ecs/task_definition_gpu.json

# Run on GPU instance
aws ecs run-task \
    --cluster cylon-armada-gpu \
    --task-definition cylon-armada-gpu \
    --launch-type EC2 \
    --overrides '{"containerOverrides":[{"name":"armada","command":["armada_executor.handler"]}]}'
```

### 5.3 Rivanna HPC Deployment

Rivanna uses Singularity/Apptainer containers built from the Docker Hub image.

```bash
# On Rivanna — pull and convert Docker image to Singularity
module load apptainer
apptainer pull cylon-armada.sif docker://docker.io/$DOCKER_USER/cylon-armada-python:latest

# Submit experiment job array
sbatch target/rivanna/scripts/run_experiment.slurm

# Submit GPU experiment
sbatch target/rivanna/scripts/run_experiment_gpu.slurm
```

#### SLURM Job Structure

```
sbatch run_experiment.slurm
  └─ SLURM array index → rank
     └─ apptainer exec cylon-armada.sif python -m armada_executor ...
```

Each SLURM array task runs one `armada_executor` instance inside the
Singularity container. A coordinator job runs `armada_init` first,
writes task payloads to shared storage, then the array tasks process
them in parallel.

### 5.4 Experiment Matrix by Infrastructure

| Variable | Lambda | Fargate | ECS GPU | Rivanna |
|----------|--------|---------|---------|---------|
| Node counts | 1-1000 (concurrent) | 1-10 | 1-4 | 1-32 |
| Task counts | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 |
| SIMD backends | NUMPY, CYTHON | NUMPY, CYTHON | NUMPY, CYTHON, GPU | NUMPY, CYTHON, GPU |
| Context backends | redis, cylon | redis, cylon | redis, cylon | redis, cylon |
| Runs per config | 3-5 | 3-5 | 3-5 | 3-5 |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ThrottlingException` from Bedrock | Too many concurrent LLM calls | Reduce `MaxConcurrency` in ASL, add delays |
| Lambda timeout (300s) | Large task count or slow Bedrock | Increase timeout or reduce tasks per run |
| Redis connection refused | Lambda not in VPC with ElastiCache | Add VPC config to Lambda via Terraform |
| `ModuleNotFoundError: pycylon` | Docker image missing pycylon | Verify Dockerfile build includes Cylon |
| Empty search results | Context store empty | Run baseline first, check workflow_id matches across prepare/route calls |
| Contexts not reused across Lambda invocations | Redis persistence off | Ensure `REDIS_HOST` is set and `persist_to_redis=True` (default) |
| Stratified sampling returns fewer tasks than requested | Deduplication on small task lists | Use `--sampling sequential` or increase task pool |