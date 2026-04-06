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


---

## Stage 6: FMI Integration Experiments

These experiments validate the three Phase 2 FMI capabilities. Each AWS experiment has a dedicated JSON input file under `target/shared/scripts/experiment/scenarios/fmi/` — pass it directly to the CLI with `--input file://`.

**Prerequisites:**
```bash
# Fetch ARNs
PYTHON_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw python_workflow_arn)
MODEL_PARALLEL_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw model_parallel_workflow_arn)
ONNX_BUCKET=$(terraform -chdir=target/aws/scripts/terraform output -raw results_bucket)

# Verify rendezvous server (required for fmi_channel_type=direct experiments)
RENDEZVOUS_HOST=$(terraform -chdir=target/aws/scripts/terraform output -raw rendezvous_host)
curl -s http://$RENDEZVOUS_HOST:10000/health

# Verify FMI is present in the deployed container
aws lambda invoke \
    --function-name cylon-armada-executor \
    --payload '{"task_description":"test","embedding_b64":"","workflow_id":"fmi-check","rank":0,"world_size":1}' \
    /tmp/fmi_check.json && grep -i "fmi" /tmp/fmi_check.json
```

Input files are in `target/shared/scripts/experiment/scenarios/fmi/`. Set `SCENARIOS` to that path for brevity:

```bash
SCENARIOS=target/shared/scripts/experiment/scenarios/fmi
```

---

### 6.1 Context Broadcast Experiments

**Research question:** Does FMI Arrow IPC broadcast eliminate Redis round-trips for the Cylon backend? Does the direct (TCPunch) channel reduce broadcast latency vs Redis at scale?

#### 6.1.1 FMI broadcast — Redis channel

One execution per world_size. Each file has the correct task count for weak scaling (ws1=4, ws2=8, ws4=16, ws8=32 tasks):

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-redis-ws1-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_redis_ws1.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-redis-ws2-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_redis_ws2.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-redis-ws4-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_redis_ws4.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-redis-ws8-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_redis_ws8.json
```

#### 6.1.2 FMI broadcast — Direct channel (TCPunch)

Requires rendezvous server reachable from Lambda. Uses `fmi_channel_type: direct`, `fmi_hint: low_latency`:

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-direct-ws2-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_direct_ws2.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-direct-ws4-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_direct_ws4.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "fmi-broadcast-direct-ws8-$(date +%s)" \
    --input file://$SCENARIOS/broadcast_direct_ws8.json
```

#### 6.1.3 Baseline — no FMI (world_size=1)

The `broadcast_redis_ws1.json` file with `world_size: 1` is the natural baseline — no FMI broadcast is triggered at world_size=1. Run it 3 times for statistical consistency:

```bash
for run in 1 2 3; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $PYTHON_SFN \
        --name "baseline-no-fmi-run${run}-$(date +%s)" \
        --input file://$SCENARIOS/broadcast_redis_ws1.json
done
```

**Success criteria:**

- [ ] All executions succeed with no FMI errors in CloudWatch logs
- [ ] `search_latency_ms` at ws4 and ws8 is ≤ ws1 baseline (broadcast amortized across workers)
- [ ] `fmi_broadcast_direct_*` shows lower broadcast overhead than `fmi_broadcast_redis_*` at same world_size
- [ ] `reuse_rate` is consistent across all world_sizes (same threshold, same tasks)

---

### 6.2 Progressive Context Sync Experiments

**Research question:** Does broadcasting new embeddings immediately after LLM calls improve within-run reuse rates for related task batches?

Two task sets:
- **High overlap** (`prog_sync_high_*`): all hydrology tasks — semantically similar, expected high within-run reuse gain
- **Low overlap** (`prog_sync_low_*`): cross-domain mixed_scientific tasks — low within-run reuse expected

#### 6.2.1 Progressive sync — high semantic overlap

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-high-ws2-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_high_ws2.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-high-ws4-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_high_ws4.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-high-ws8-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_high_ws8.json
```

#### 6.2.2 Progressive sync — low semantic overlap

```bash
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-low-ws2-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_low_ws2.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-low-ws4-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_low_ws4.json

aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --name "prog-sync-low-ws8-$(date +%s)" \
    --input file://$SCENARIOS/prog_sync_low_ws8.json
```

#### 6.2.3 Baselines (ws1, no FMI sync)

Use the corresponding `prog_sync_high_ws2.json` files but compare against ws1 results (which have no sync by definition). The `broadcast_redis_ws1.json` file serves as the ws1 hydrology baseline; for low-overlap use `prog_sync_low_ws2.json` with `world_size` edited to 1, or run the low-overlap tasks sequentially via the local runner:

```bash
# Local low-overlap baseline (no FMI, no world_size parallelism)
python target/shared/scripts/experiment/runner.py \
    --context-backend redis \
    --tasks-file target/shared/scripts/experiment/scenarios/mixed_scientific.json \
    --tasks 8 16 32 \
    --thresholds 0.80 \
    --dimensions 256 \
    --output target/shared/scripts/experiment/results/fmi/no_sync_low_baseline
```

**Success criteria:**

- [ ] `prog_sync_high_ws4` and `ws8` show higher `reuse_rate` than `prog_sync_high_ws2` (more workers = more cross-worker reuse opportunities)
- [ ] `prog_sync_high_*` `reuse_rate` > `prog_sync_low_*` `reuse_rate` at same world_size (confirms semantic overlap drives the benefit)
- [ ] `total_ms` overhead for sync is < 15% of routing time (sync doesn't dominate latency)
- [ ] No duplicate context_ids in aggregated stopwatch CSV

---

### 6.3 Model Parallelism Experiments (AstroMAE via FMI)

**Research question:** Does FMI `allgather_tensors` via direct channel remove the 256KB Step Functions payload limit, enabling batch_size ≥ 32 inference?

#### 6.3.1 Prerequisites — export and upload ONNX partitions

```bash
# Export AstroMAE ONNX partitions (run once)
python target/shared/scripts/cosmic_ai/export_onnx.py \
    --model-path /path/to/astromae_weights.pt \
    --output-dir /tmp/onnx_partitions \
    --partition 3

# Upload to S3
aws s3 cp /tmp/onnx_partitions/stage_0.onnx s3://$ONNX_BUCKET/models/astromae/stage_0.onnx
aws s3 cp /tmp/onnx_partitions/stage_1.onnx s3://$ONNX_BUCKET/models/astromae/stage_1.onnx
aws s3 cp /tmp/onnx_partitions/stage_2.onnx s3://$ONNX_BUCKET/models/astromae/stage_2.onnx
```

#### 6.3.2 Create input files per batch size

Create one input file per `(batch_size, channel_type)` configuration. The input tensor is a synthetic SDSS-like array; replace `input_b64` and `input_shape` with real data for production runs.

```bash
# Helper: generate a base64-encoded random float32 tensor of given shape
gen_tensor_b64() {
    python3 -c "
import base64, numpy as np, sys
shape = [int(x) for x in sys.argv[1].split(',')]
arr = np.random.randn(*shape).astype(np.float32)
print(base64.b64encode(arr.tobytes()).decode())
" "$1"
}

# Create input files for batch sizes 1, 4, 8, 16, 32
mkdir -p $SCENARIOS/model_parallel

for batch_size in 1 4 8 16 32; do
    tensor_b64=$(gen_tensor_b64 "${batch_size},224,224,1")
    cat > $SCENARIOS/model_parallel/astromae_direct_b${batch_size}.json << JSON
{
  "workflow_id": "astromae-fmi-direct-b${batch_size}",
  "fmi_config": {
    "world_size": 2,
    "fmi_channel_type": "direct",
    "fmi_hint": "low_latency"
  },
  "onnx_config": {
    "s3_bucket": "$ONNX_BUCKET",
    "stage_0_key": "models/astromae/stage_0.onnx",
    "stage_1_key": "models/astromae/stage_1.onnx",
    "fusion_key":  "models/astromae/stage_2.onnx"
  },
  "input_b64": "${tensor_b64}",
  "input_shape": [${batch_size}, 224, 224, 1],
  "batch_size": ${batch_size}
}
JSON
done
```

#### 6.3.3 Run model parallelism experiments

```bash
# FMI direct channel — batch sweep
for batch_size in 1 4 8 16 32; do
    aws stepfunctions start-sync-execution \
        --state-machine-arn $MODEL_PARALLEL_SFN \
        --name "astromae-direct-b${batch_size}-$(date +%s)" \
        --input file://$SCENARIOS/model_parallel/astromae_direct_b${batch_size}.json
done
```

**Success criteria:**

- [ ] `batch_size=32` succeeds (would exceed 256KB Step Functions limit without FMI)
- [ ] `fmi_latency_ms` < 20ms for direct channel at all batch sizes
- [ ] Prediction output field is present in result JSON with shape `[batch_size, 1]`
- [ ] ViT and Inception stages complete in parallel (check CloudWatch timestamps — overlap expected)
- [ ] Total end-to-end latency ≤ 1.5× single-Lambda baseline at batch_size=1

---

### 6.4 FMI Results Analysis

```bash
cd target/shared/scripts

# Aggregate all FMI results
python -m results.pipeline \
    --local-dir experiment/results/fmi/ \
    --output-dir experiment/output/fmi/ \
    --chart-format svg

# Quick comparison: FMI broadcast vs baseline reuse rate
python3 -c "
import pandas as pd
df = pd.read_csv('experiment/output/fmi/aggregated_results.csv')
broadcast = df[df['experiment_name'].str.startswith('fmi_broadcast')]
baseline  = df[df['experiment_name'] == 'fmi_broadcast_redis_ws1']
print(broadcast[['experiment_name','world_size','reuse_rate_mean','search_latency_ms_mean']]
      .sort_values('world_size'))
"

# Progressive sync benefit: high vs low overlap
python3 -c "
import pandas as pd
df = pd.read_csv('experiment/output/fmi/aggregated_results.csv')
high = df[df['experiment_name'].str.startswith('prog_sync_high')]
low  = df[df['experiment_name'].str.startswith('prog_sync_low')]
print('High overlap reuse rate:')
print(high[['world_size','reuse_rate_mean']].sort_values('world_size'))
print('Low overlap reuse rate:')
print(low[['world_size','reuse_rate_mean']].sort_values('world_size'))
"
```
