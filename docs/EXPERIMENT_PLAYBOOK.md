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

All Lambda functions and the ECS runner share a **single ECR repository** — each image is
differentiated by tag. Lambda functions use a CMD override per function; ECS tasks run
`armada_ecs_runner.py` directly. Task definitions are created by `terraform apply` (not manually).

```bash
ECR_REPO=$(terraform -chdir=target/aws/scripts/terraform output -raw ecr_repository_url)

aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin $ECR_REPO

# Python image — Lambda (Path A1/A2) + ECS runner
docker build -t cylon-armada-python -f docker/Dockerfile.python .
docker tag cylon-armada-python $ECR_REPO:python-latest
docker push $ECR_REPO:python-latest

# Node.js image — Lambda (Path B)
docker build -t cylon-armada-nodejs -f docker/Dockerfile.nodejs .
docker tag cylon-armada-nodejs $ECR_REPO:nodejs-latest
docker push $ECR_REPO:nodejs-latest

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

# Verify ECS task definition is registered (created by terraform apply)
aws ecs describe-task-definition \
    --task-definition cylon-armada-python \
    --query 'taskDefinition.{family:family,revision:revision,status:status}'
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

All four execution paths are triggered via Step Functions. Fetch ARNs from Terraform outputs:

```bash
PYTHON_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw python_workflow_arn)
NODEJS_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw nodejs_workflow_arn)
FARGATE_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw ecs_fargate_workflow_arn)
EC2_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw ecs_ec2_workflow_arn)
```

#### Lambda — Python path (Map: one invocation per task)

The Step Functions workflow follows the cylon paper architecture:
`ArmadaInit` (embed tasks) → `ProcessTasks` (Map: route each task) → `AggregateResults`.

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-lambda-th08-d256',
    'tasks': tasks,
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'}
}))
")" --query 'output' --output text > target/shared/scripts/experiment/results/aws_hydrology_lambda.json
```

#### Lambda — Node.js path

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $NODEJS_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-nodejs-th08',
    'tasks': tasks,
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'}
}))
")" --query 'output' --output text
```

#### ECS Fargate — weak scaling (world_size controls thread pool)

The ECS workflow triggers a single Fargate task that runs the full experiment and writes
results directly to S3. `scaling=weak` means each of the `world_size` threads processes
an equal share of the task list.

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $FARGATE_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-fargate-weak-4',
    'tasks': tasks,
    'scaling': 'weak',
    'world_size': 4,
    's3_scripts_bucket': '',
    's3_scripts_prefix': 'scripts/'
}))
")"
```

#### ECS EC2 — strong scaling

`scaling=strong` keeps the total task list fixed while increasing `world_size` — measures
speedup as more parallel threads process the same workload.

```bash
for world_size in 1 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $EC2_SFN \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:32]
print(json.dumps({
    'workflow_id': 'exp-hydrology-ec2-strong-${world_size}',
    'tasks': tasks,
    'scaling': 'strong',
    'world_size': ${world_size},
    's3_scripts_bucket': '',
    's3_scripts_prefix': 'scripts/'
}))
")"
done
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

The ECS task definition (`cylon-armada-python`) is registered by `terraform apply` against
the `CylonFargateExperiments` cluster. Experiments are triggered via Step Functions, which
injects per-run config as environment variable overrides and waits for task completion.

```bash
FARGATE_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw ecs_fargate_workflow_arn)

# Weak scaling — world_size scales with task count
for world_size in 1 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $FARGATE_SFN \
        --name "fargate-weak-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks']
print(json.dumps({
    'workflow_id': 'exp-fargate-weak-w${world_size}',
    'tasks': tasks[:world_size * 4],
    'scaling': 'weak',
    'world_size': ${world_size},
    's3_scripts_bucket': '',
    's3_scripts_prefix': 'scripts/'
}))
")"
done

# Strong scaling — fixed 32 tasks, increasing parallelism
for world_size in 1 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $FARGATE_SFN \
        --name "fargate-strong-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:32]
print(json.dumps({
    'workflow_id': 'exp-fargate-strong-w${world_size}',
    'tasks': tasks,
    'scaling': 'strong',
    'world_size': ${world_size},
    's3_scripts_bucket': '',
    's3_scripts_prefix': 'scripts/'
}))
")"
done
```

Results land in S3 under `results/ecs-fargate/<workflow_id>/`.

### 5.2 ECS EC2 Deployment

The same `cylon-armada-python` task definition runs on `CylonEC2Experiments` with
`LaunchType: EC2`. For GPU experiments, build and push the GPU image first.

```bash
ECR_REPO=$(terraform -chdir=target/aws/scripts/terraform output -raw ecr_repository_url)
EC2_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw ecs_ec2_workflow_arn)

# (Optional) GPU image for CUDA-accelerated SIMD
docker build -t cylon-armada-gpu -f docker/Dockerfile.gpu .
docker tag cylon-armada-gpu $ECR_REPO:gpu-latest
docker push $ECR_REPO:gpu-latest

# Strong scaling — sweep world_size, fixed task list
for world_size in 1 2 4 8 16; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $EC2_SFN \
        --name "ec2-strong-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:32]
print(json.dumps({
    'workflow_id': 'exp-ec2-strong-w${world_size}',
    'tasks': tasks,
    'scaling': 'strong',
    'world_size': ${world_size},
    's3_scripts_bucket': '',
    's3_scripts_prefix': 'scripts/'
}))
")"
done
```

Results land in S3 under `results/ecs-ec2/<workflow_id>/`.

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

| Variable | Lambda | ECS Fargate | ECS EC2 | Rivanna |
|----------|--------|-------------|---------|---------|
| Scaling modes | N/A (Map-parallel) | weak, strong | weak, strong | weak, strong |
| World sizes | 1-1000 (Map concurrency) | 1, 2, 4, 8 | 1, 2, 4, 8, 16 | 1, 2, 4, 8, 16, 32 |
| Task counts | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 |
| SIMD backends | NUMPY, CYTHON | NUMPY, CYTHON | NUMPY, CYTHON, GPU | NUMPY, CYTHON, GPU |
| Context backends | redis, cylon | redis, cylon | redis, cylon | redis, cylon |
| Orchestration | Step Functions Map | Step Functions `ecs:runTask.sync` | Step Functions `ecs:runTask.sync` | SLURM job array |
| Results destination | Step Functions output | S3 `results/ecs-fargate/` | S3 `results/ecs-ec2/` | S3 `results/rivanna/` |
| Runs per config | 3-5 | 3-5 | 3-5 | 3-5 |

#### ECS scaling checklist

For each `(platform, scaling_mode, world_size)` combination:

- [ ] Run with `scaling=weak` — verify total tasks = world_size × tasks_per_worker
- [ ] Run with `scaling=strong` — verify all world_size threads processed same task list
- [ ] Check `throughput_tasks_per_sec` increases with world_size (weak) or stays flat (strong ideal)
- [ ] Check `speedup = time(world_size=1) / time(world_size=N)` approaches N for strong scaling
- [ ] Verify `reuse_rate` is consistent with Lambda path at same threshold
- [ ] Confirm results in S3 under correct prefix before next run

### Phase 4: RuVector Backend (Node.js Path B, Large-Scale)

RuVector (`@ruvector/core`, NAPI-RS) is planned as a 4th context backend for Phase 4
large-scale experiments. It provides HNSW approximate nearest neighbor search with
SimSIMD acceleration (AVX-512/NEON) — a fundamentally different search architecture
from cylon-wasm's exact linear scan.

**Why Phase 4 and not phase 1 or 2:**
- At current experiment scales (4-48 tasks), linear scan and HNSW show identical
  latency — there is no data to differentiate them
- The benefit appears at N > 10K contexts, where HNSW's O(log n) complexity
  significantly outperforms linear O(n) scan
- RuVector has no Python SDK (CLI/subprocess only), so it applies to Path B only

**Research question it answers:**
At large context stores, does HNSW approximate search (RuVector) maintain the
cylon-armada reuse rate while reducing search latency vs cylon-wasm exact scan?
The reuse decision logic, Step Functions orchestration, and cost tracking remain
unchanged — only the search backend is swapped.

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
| ECS task fails immediately | Missing IAM permissions | Check `cylon-armada-ecs-task-role` has Bedrock + S3 + DynamoDB permissions |
| ECS task stuck in PENDING | No EC2 capacity in cluster | Check cluster has running instances; for Fargate verify subnet has capacity |
| Step Functions ECS timeout | Task ran longer than SFN Express limit (5 min) | Switch to STANDARD workflow type or reduce world_size/task count |
| ECS results not in S3 | `RESULTS_BUCKET` or `RESULTS_PREFIX` env var wrong | Check task definition env vars and ContainerOverrides in CloudWatch logs |
| Strong scaling shows no speedup | Redis contention or single-threaded GIL | Use separate `ContextManager` per thread (already done in runner); check Redis connection pool |
---

## Stage 6: FMI Integration Experiments

These experiments validate the three Phase 2 FMI capabilities. All require a Cylon Lambda container with `fmilib` installed and a reachable rendezvous server (for `direct` channel experiments).

**Prerequisites:**
```bash
# Verify rendezvous server is reachable
RENDEZVOUS_HOST=$(terraform -chdir=target/aws/scripts/terraform output -raw rendezvous_host)
RENDEZVOUS_PORT=$(terraform -chdir=target/aws/scripts/terraform output -raw rendezvous_port)
curl -s http://$RENDEZVOUS_HOST:$RENDEZVOUS_PORT/health

# Verify FMI is available in the container
docker run --rm cylon-armada-python python -c \
    "from communicator.fmi_bridge import FMIBridge; b = FMIBridge(1, 0); print('FMI available:', b.available)"
```

---

### 6.1 Context Broadcast Experiments

**Research question:** Does FMI Arrow IPC broadcast eliminate the need for Redis in the Cylon backend? Does it improve latency vs. N independent Redis reads at scale?

**Experimental matrix:**

| Variable | Values |
|----------|--------|
| Context backend | `cylon` (IPC broadcast), `redis` (per-worker load) |
| FMI channel type | `redis`, `direct` |
| World size | 1, 2, 4, 8, 16 |
| Context store size | 0 (cold), 100, 500, 1000 contexts pre-loaded |
| Tasks per worker | 4 (weak scaling) |

**Run — FMI context broadcast (Cylon backend, redis channel):**

```bash
for world_size in 1 2 4 8 16; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "fmi-broadcast-redis-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks']
# weak scaling: world_size * 4 tasks
print(json.dumps({
    'workflow_id': 'fmi-broadcast-redis-w${world_size}',
    'tasks': tasks[:${world_size}*4],
    'scaling': 'weak',
    'world_size': ${world_size},
    'fmi_channel_type': 'redis',
    'fmi_hint': 'fast',
    'results_s3_dir': 'results/fmi/broadcast_redis/',
    'experiment_name': 'fmi_broadcast_redis_ws${world_size}',
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'},
}))
")"
done
```

**Run — FMI context broadcast (Cylon backend, direct TCPunch channel):**

```bash
for world_size in 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "fmi-broadcast-direct-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks']
print(json.dumps({
    'workflow_id': 'fmi-broadcast-direct-w${world_size}',
    'tasks': tasks[:${world_size}*4],
    'scaling': 'weak',
    'world_size': ${world_size},
    'fmi_channel_type': 'direct',
    'fmi_hint': 'low_latency',
    'results_s3_dir': 'results/fmi/broadcast_direct/',
    'experiment_name': 'fmi_broadcast_direct_ws${world_size}',
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'},
}))
")"
done
```

**Baseline — no FMI (Redis backend, per-worker load):**

```bash
for world_size in 1 2 4 8 16; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "no-fmi-redis-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks']
print(json.dumps({
    'workflow_id': 'no-fmi-redis-w${world_size}',
    'tasks': tasks[:${world_size}*4],
    'scaling': 'weak',
    'world_size': ${world_size},
    'results_s3_dir': 'results/fmi/no_fmi_redis/',
    'experiment_name': 'no_fmi_redis_ws${world_size}',
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'},
}))
")"
done
```

**Success criteria:**

- [ ] `fmi_broadcast_*` results show `search_latency_ms` equivalent to or better than `no_fmi_redis_*` at world_size ≥ 4
- [ ] `fmi_broadcast_direct_*` shows lower broadcast latency than `fmi_broadcast_redis_*`
- [ ] `reuse_rate` is consistent between FMI and non-FMI runs (same threshold, same tasks)
- [ ] No FMI-related errors in CloudWatch logs at any world_size

---

### 6.2 Progressive Context Sync Experiments

**Research question:** Does broadcasting new embeddings immediately after LLM calls improve within-run reuse rates for semantically related task batches?

**Experimental matrix:**

| Variable | Values |
|----------|--------|
| FMI progressive sync | enabled (Cylon FMI container), disabled (baseline) |
| Task semantic overlap | low (~10%), medium (~40%), high (~70%) |
| World size | 2, 4, 8 |
| Similarity threshold | 0.75, 0.80, 0.85 |

Task batches with controlled semantic overlap are created from `hydrology.json` (high overlap — watershed/streamflow tasks are semantically close) vs. `mixed_scientific.json` (low overlap — cross-domain tasks).

**Run — progressive sync enabled (FMI available):**

```bash
for world_size in 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "prog-sync-high-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
# High-overlap scenario: all tasks from same domain
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:${world_size}*4]
print(json.dumps({
    'workflow_id': 'prog-sync-high-w${world_size}',
    'tasks': tasks,
    'scaling': 'weak',
    'world_size': ${world_size},
    'fmi_channel_type': 'redis',
    'results_s3_dir': 'results/fmi/prog_sync_high/',
    'experiment_name': 'prog_sync_high_ws${world_size}',
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'},
}))
")"
done
```

**Baseline — no progressive sync (non-FMI container or world_size=1):**

```bash
for world_size in 2 4 8; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "no-sync-high-w${world_size}-$(date +%s)" \
        --input "$(python -c "
import json
tasks = json.load(open('target/shared/scripts/experiment/scenarios/hydrology.json'))['tasks'][:${world_size}*4]
print(json.dumps({
    'workflow_id': 'no-sync-high-w${world_size}',
    'tasks': tasks,
    'scaling': 'weak',
    'world_size': ${world_size},
    'results_s3_dir': 'results/fmi/no_sync_high/',
    'experiment_name': 'no_sync_high_ws${world_size}',
    'config': {'similarity_threshold': '0.80', 'embedding_dimensions': '256'},
}))
")"
done
```

**Metric to compare:** `reuse_rate_mean` between `prog_sync_*` and `no_sync_*` at the same world_size and task overlap level.

**Success criteria:**

- [ ] `prog_sync_high_*` shows `reuse_rate` > `no_sync_high_*` at world_size ≥ 4 with high-overlap tasks
- [ ] `prog_sync_*` vs `no_sync_*` difference narrows for low-overlap (`mixed_scientific`) tasks
- [ ] `total_ms` for `prog_sync_*` is within 15% of `no_sync_*` (sync overhead is acceptable)
- [ ] No duplicate context entries caused by concurrent broadcasts

---

### 6.3 Model Parallelism Experiments (AstroMAE)

**Research question:** Does FMI direct channel tensor exchange remove the Step Functions 256KB limit, enabling larger batch inference? What is the FMI latency overhead vs. single-Lambda inference?

**Prerequisites:**
```bash
# Export AstroMAE ONNX partitions
python target/shared/scripts/cosmic_ai/export_onnx.py \
    --model-path /path/to/astromae/model.pt \
    --output-dir /tmp/onnx_partitions \
    --partition 3  # stage 0 (ViT), stage 1 (Inception), stage 2 (Fusion)

# Upload ONNX partitions to S3
ONNX_BUCKET=$(terraform -chdir=target/aws/scripts/terraform output -raw results_bucket)
aws s3 cp /tmp/onnx_partitions/stage_0.onnx s3://$ONNX_BUCKET/models/astromae/stage_0.onnx
aws s3 cp /tmp/onnx_partitions/stage_1.onnx s3://$ONNX_BUCKET/models/astromae/stage_1.onnx
aws s3 cp /tmp/onnx_partitions/stage_2.onnx s3://$ONNX_BUCKET/models/astromae/stage_2.onnx
```

**Experimental matrix:**

| Variable | Values |
|----------|--------|
| Mode | single-Lambda (full model), 2-Lambda FMI split |
| FMI channel | `redis`, `direct` |
| Batch size | 1, 4, 8, 16, 32 |
| Runs per config | 5 |

**Run — 2-Lambda FMI parallel (direct channel):**

The `workflow_model_parallel.asl.json` triggers a Parallel state with rank=0 (ViT) and rank=1 (Inception). Each Lambda runs `model_parallel_stage` via `run_action.py`.

```bash
MODEL_PARALLEL_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw model_parallel_workflow_arn)
ONNX_BUCKET=$(terraform -chdir=target/aws/scripts/terraform output -raw results_bucket)

for batch_size in 1 4 8 16 32; do
    python -c "
import json, base64, numpy as np
# Synthetic SDSS-like input: (B, 224, 224, 1) image tensor
x = np.random.randn(${batch_size}, 224, 224, 1).astype(np.float32)
print(json.dumps({
    'workflow_id': 'astromae-fmi-direct-b${batch_size}',
    'fmi_config': {
        'world_size': 2,
        'fmi_channel_type': 'direct',
        'fmi_hint': 'low_latency',
    },
    'onnx_config': {
        's3_bucket': '$ONNX_BUCKET',
        'stage_0_key': 'models/astromae/stage_0.onnx',
        'stage_1_key': 'models/astromae/stage_1.onnx',
        'fusion_key':  'models/astromae/stage_2.onnx',
    },
    'input_b64': base64.b64encode(x.tobytes()).decode(),
    'input_shape': list(x.shape),
    'batch_size': ${batch_size},
}))
" | xargs -d'\n' aws stepfunctions start-sync-execution \
        --state-machine-arn $MODEL_PARALLEL_SFN \
        --name "astromae-direct-b${batch_size}-$(date +%s)" \
        --input
done
```

**Metric to collect:**

| Metric | Column in results | Description |
|--------|------------------|-------------|
| Stage 0 latency | `stage_latency_ms.stage_0` | ViT encoder time |
| Stage 1 latency | `stage_latency_ms.stage_1` | Inception branch time |
| FMI allgather latency | `fmi_latency_ms` | Tensor exchange time |
| Fusion latency | `stage_latency_ms.fusion` | Fusion stage time |
| Total end-to-end | sum of above | Compare to single-Lambda baseline |

**Success criteria:**

- [ ] `batch_size=32` succeeds with FMI (would exceed 256KB Step Functions limit without FMI)
- [ ] `fmi_latency_ms` < 20ms for `direct` channel at all batch sizes
- [ ] `fmi_latency_ms` < 100ms for `redis` channel at all batch sizes
- [ ] Total FMI end-to-end latency ≤ 1.5× single-Lambda latency at batch_size=1 (parallelism overhead bounded)
- [ ] ViT + Inception stages run concurrently (verify from CloudWatch timestamps)
- [ ] Prediction output shape is `(B, 1)` for all batch sizes

---

### 6.4 FMI Results Analysis

FMI experiment results follow the same aggregation pipeline as Phase 1:

```bash
cd target/shared/scripts

# Aggregate FMI broadcast results
python -m results.pipeline \
    --local-dir experiment/results/fmi/ \
    --output-dir experiment/output/fmi/ \
    --chart-format svg

# Compare FMI vs. no-FMI reuse rates
python -c "
import pandas as pd
df = pd.read_csv('experiment/output/fmi/aggregated_results.csv')
fmi   = df[df['experiment_name'].str.startswith('fmi_broadcast')]
nofmi = df[df['experiment_name'].str.startswith('no_fmi')]
print('Reuse rate — FMI vs. no-FMI:')
print(fmi[['world_size', 'reuse_rate_mean']].merge(
    nofmi[['world_size', 'reuse_rate_mean']],
    on='world_size', suffixes=('_fmi', '_nofmi')
))
"
```

**Key charts to generate:**

| Chart | X axis | Y axis | Grouping |
|-------|--------|--------|----------|
| Broadcast latency vs. world_size | world_size | `search_latency_ms_mean` | FMI channel (redis vs. direct) |
| Progressive sync benefit | semantic overlap | `reuse_rate_mean` | FMI sync on/off |
| FMI tensor exchange latency | batch_size | `fmi_latency_ms` | channel type |
| Total AstroMAE latency | batch_size | total latency (ms) | single-Lambda vs. 2-Lambda FMI |
