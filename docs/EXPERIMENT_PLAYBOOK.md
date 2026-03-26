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
python -c "import boto3, redis, numpy, langchain_aws; print('Python deps OK')"

# Build Cython SIMD extension (Path A2)
cd python/simd
CYLON_PREFIX=$CYLON_PREFIX python setup.py build_ext --inplace
cd -
```

### Host Services (Redis + DynamoDB Local)

Redis and DynamoDB Local run on the **host OS** (Mac), not inside the Parallels VM.
The VM connects via the Parallels network IP. See [Cylon ENVIRONMENT_SETUP.md](../../cylon/ENVIRONMENT_SETUP.md) for details.

```bash
# On the host OS (Mac):
# Redis — install and run via Homebrew
brew install redis
redis-server

# DynamoDB Local — run via Docker on the host
docker run -p 8100:8000 amazon/dynamodb-local:latest -jar DynamoDBLocal.jar -sharedDb
```

```bash
# In the Parallels VM — find the host IP:
ip neighbor show
# Look for: 10.211.55.2 dev enp0s5 ... REACHABLE

# Set environment variables for the VM:
export REDIS_HOST=10.211.55.2
export REDIS_PORT=6379
export DYNAMO_ENDPOINT_URL=http://10.211.55.2:8100
```

```bash
# Create DynamoDB table (from VM, pointing to host)
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

### 1.2 Run Tests

```bash
# Python (69 tests)
python -m pytest tests/ -v

# Node.js (33 tests)
cd target/aws/scripts/lambda/node && npm test && cd -
```

### 1.3 Smoke Test

Ensure `REDIS_HOST` and `DYNAMO_ENDPOINT_URL` are set (see above).

```bash
# Minimal run: 4 tasks, stratified sampling, single config
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/hydrology.json \
    --tasks 4 \
    --thresholds 0.8 \
    --dimensions 256 \
    --output target/experiments/results/smoke_test

# Verify output
cat target/experiments/results/smoke_test/*_summary.json | python -m json.tool
```

Checklist:
- [ ] Result JSON files created (`*_summary.csv`, `*_stopwatch.csv`, `*_metrics.json`)
- [ ] `savings_pct` > 0 for reuse runs
- [ ] `savings_pct` = 0 for baseline runs
- [ ] Stratified sampling selected tasks from different categories
- [ ] No errors in logs

### 1.4 Node.js Smoke Test (Path B — WASM SIMD)

Requires cylon-wasm built (`wasm-pack build --target nodejs --release` in `cylon/rust/cylon-wasm`).

```bash
cd target/aws/scripts/lambda/node

# Ensure cylon_host stub exists for local testing
mkdir -p /path/to/cylon/rust/cylon-wasm/pkg/node_modules/cylon_host
cat > /path/to/cylon/rust/cylon-wasm/pkg/node_modules/cylon_host/index.js << 'STUB'
module.exports = {
    host_get_rank: () => 0,
    host_get_world_size: () => 1,
    host_barrier: () => {},
    host_broadcast: (p, l, r) => l,
    host_all_to_all: (p, l, r) => l,
    host_gather: (p, l, r, o) => l,
    host_scatter: (p, l, r, o) => l,
    host_all_gather: (p, l, o) => l,
};
STUB

# SIMD benchmark (no AWS dependencies — pure WASM SIMD throughput)
CYLON_WASM_BINDINGS=/path/to/cylon/rust/cylon-wasm/pkg/cylon_wasm.js \
CYLON_WASM_PATH=/path/to/cylon/rust/cylon-wasm/pkg/cylon_wasm_bg.wasm \
node run_experiment.mjs \
    --action simd_benchmark \
    --dim 256 --n 1000 --iterations 100 \
    --name nodejs_simd_d256_n1000 \
    --output ../../../../experiments/results/smoke_nodejs

# Verify output
cat ../../../../experiments/results/smoke_nodejs/*_metrics.json | python -m json.tool
```

Expected output: `comparisons_per_sec` > 100,000 for 256-dim, `avg_search_ms` < 10ms for 1000 embeddings.

```bash
# Full route_task experiment (requires Redis + Bedrock)
CYLON_WASM_BINDINGS=/path/to/cylon/rust/cylon-wasm/pkg/cylon_wasm.js \
CYLON_WASM_PATH=/path/to/cylon/rust/cylon-wasm/pkg/cylon_wasm_bg.wasm \
REDIS_HOST=10.211.55.2 \
node run_experiment.mjs \
    --action route_task \
    --tasks-file ../../../../experiments/scenarios/hydrology.json \
    --tasks 4 --threshold 0.8 --dimensions 256 \
    --name nodejs_route_hydrology_t4 \
    --output ../../../../experiments/results/smoke_nodejs

cd -
```

### 1.5 S3 Upload (Optional)

Both Python and Node.js runners support `--s3-bucket` for uploading results to S3.
Omit to keep results local only.

```bash
# Python with S3 upload
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/hydrology.json \
    --tasks 4 --thresholds 0.8 --dimensions 256 \
    --s3-bucket cylon-armada-results \
    --s3-prefix experiments/hydrology \
    --output target/experiments/results/smoke_test

# Node.js with S3 upload
node run_experiment.mjs \
    --action simd_benchmark --dim 256 --n 1000 \
    --s3-bucket cylon-armada-results \
    --s3-prefix experiments/simd
```

### Output Files

Each experiment produces three files (matching Cylon's scaling.py pattern):

| File | Content | Used By |
|------|---------|---------|
| `*_stopwatch.csv` | cloudmesh benchmark (system info + timings) | Full audit trail |
| `*_summary.csv` | Data-only CSV (timings + metrics, no machine info) | Results pipeline aggregator |
| `*_metrics.json` | All timings + metrics as JSON | Quick inspection, Jupyter notebooks |

Both Python and Node.js produce identical CSV formats for cross-path comparison.

---

## Stage 2: AWS Deployment

### 2.1 Build and Push Docker Images

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
```

### 2.2 Deploy Infrastructure

```bash
cd target/aws/scripts/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform plan
terraform apply

terraform output -json > ../../deployment_outputs.json
```

### 2.3 Upload Scripts to S3

```bash
aws s3 sync target/shared/scripts/ s3://cylon-armada-scripts/target/shared/scripts/ \
    --exclude "*.pyc" --exclude "__pycache__/*" --exclude "*.so"
```

### 2.4 Verify Deployment

```bash
# Test Python Lambda
aws lambda invoke \
    --function-name cylon-armada-worker \
    --payload '{"S3_BUCKET":"cylon-armada-scripts","S3_OBJECT_NAME":"target/shared/scripts","S3_OBJECT_TYPE":"folder","SCRIPT":"/tmp/target/shared/scripts/run_action.py","ACTION":"prepare_tasks","action_payload":{"workflow_id":"test","tasks":["Hello world"],"config":{}}}' \
    /tmp/lambda_test.json && cat /tmp/lambda_test.json

# Test Node.js Lambda
aws lambda invoke \
    --function-name cylon-armada-worker-node \
    --payload '{"action":"simd_benchmark","action_payload":{"dim":256,"n":100,"iterations":10}}' \
    /tmp/node_test.json && cat /tmp/node_test.json
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

Each scenario runs a parameter sweep:

| Variable | Values |
|----------|--------|
| Similarity threshold | 0.70, 0.80, 0.90 |
| Embedding dimensions | 256, 512, 1024 |
| Execution path | NUMPY, PYCYLON, CYTHON_BATCH |
| Baseline | yes, no |

Per scenario: 3 × 3 × 3 × 2 = **54 configurations**

### 3.3 Run All Scenarios (Local)

```bash
# Scenario 1: Astronomical inference (cosmic-ai — dynamic tasks)
python target/experiments/runner.py \
    --cosmic-ai \
    --data-path /path/to/sdss/data.pt \
    --model-path /path/to/astromae/model.pt \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/experiments/results/scenario1_cosmicai

# Scenario 2: Hydrology
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/hydrology.json \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/experiments/results/scenario2_hydrology

# Scenario 3: Epidemiology
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/epidemiology.json \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/experiments/results/scenario3_epidemiology

# Scenario 4: Seismology
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/seismology.json \
    --tasks 8 16 32 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/experiments/results/scenario4_seismology

# Scenario 5: Mixed scientific + benchmarks
python target/experiments/runner.py \
    --tasks-file target/experiments/scenarios/mixed_scientific.json \
    --tasks 16 32 48 \
    --thresholds 0.70 0.80 0.90 \
    --dimensions 256 512 1024 \
    --backends NUMPY PYCYLON CYTHON_BATCH \
    --output target/experiments/results/scenario5_mixed
```

### 3.4 Run on AWS (Step Functions)

```bash
PYTHON_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw python_workflow_arn)
NODEJS_SFN=$(terraform -chdir=target/aws/scripts/terraform output -raw nodejs_workflow_arn)
REDIS_HOST=$(terraform -chdir=target/aws/scripts/terraform output -raw redis_endpoint)

# Run a scenario via Step Functions (Python path)
aws stepfunctions start-sync-execution \
    --state-machine-arn $PYTHON_SFN \
    --input "$(python -c "
import json
tasks = json.load(open('target/experiments/scenarios/hydrology.json'))['tasks'][:16]
print(json.dumps({
    'workflow_id': 'exp-hydrology-th08-d256',
    'tasks': tasks,
    'config': {
        'llm_model_id': 'anthropic.claude-3-haiku-20240307-v1:0',
        'embedding_model_id': 'amazon.titan-embed-text-v2:0',
        'embedding_dimensions': '256',
        'similarity_threshold': '0.80',
        'redis_host': '$REDIS_HOST',
        'redis_port': '6379'
    }
}))
")" --query 'output' --output text > target/experiments/results/aws_hydrology.json
```

### 3.5 Per-Scenario Checklist

For each scenario:

- [ ] Clear context store before each run
- [ ] Run baseline (threshold=1.0) first
- [ ] Run reuse experiment with same tasks (same seed)
- [ ] Verify result JSON has all expected fields
- [ ] Check `reuse_rate` is within expected range for the scenario
- [ ] Save raw results to `target/experiments/results/`

---

## Stage 4: Results Analysis

### 4.1 Collect Results

```bash
python -c "
import json, glob
results = []
for f in sorted(glob.glob('target/experiments/results/*_summary.json')):
    results.append(json.load(open(f)))
json.dump(results, open('target/experiments/results/all_summaries.json', 'w'), indent=2)
print(f'Collected {len(results)} experiment summaries')
"
```

### 4.2 Key Metrics

| Metric | JSON Path | Unit |
|--------|-----------|------|
| Total cost | `cost_summary.total_cost` | USD |
| Baseline cost | `cost_summary.baseline_cost` | USD |
| Savings % | `cost_summary.savings_pct` | % |
| Reuse rate | `reuse_stats.reuse_rate` | % |
| Cache hits | `reuse_stats.cache_hits` | count |
| LLM calls | `reuse_stats.llm_calls` | count |
| Avg latency | `latency.avg_per_task_ms` | ms |
| Wall clock | `wall_clock_ms` | ms |

### 4.3 Visualizations (Jupyter Notebooks)

Create in `target/experiments/analysis/`:

1. **Cost reduction curves** — savings % vs similarity threshold, one line per dimension
2. **Reuse rate by domain** — bar chart: astronomy, hydrology, epidemiology, seismology
3. **Latency distributions** — box plots for Path A1 vs A2 vs B
4. **Path comparison** — scatter: cost vs latency for all 3 paths
5. **Dimension sweep** — reuse quality vs embedding dimension at fixed threshold
6. **Scaling** — workflow time vs task count (4, 8, 16, 32, 48)
7. **Cross-domain isolation** — heatmap of similarity scores between domains
8. **Cost breakdown** — stacked bar: embedding + LLM + avoided cost per scenario

### 4.4 Success Criteria Validation

| Metric | Target | Check |
|--------|--------|-------|
| Cost reduction | 60-80% | `savings_pct` across all scenarios |
| Reuse quality | >0.80 ROUGE-L | Compare reused vs baseline responses |
| Search latency | <20ms / 1000 contexts | Latency from results |
| SIMD speedup | >2x vs numpy | Path A1/A2 vs NUMPY comparison |
| All 5 scenarios | Complete | All result files present |

---

## Cost Estimation

| Component | Per Run | × 270 Runs | Total |
|-----------|---------|-----------|-------|
| Titan V2 embeddings | ~$0.001 | $0.27 | $0.27 |
| Claude Haiku LLM | ~$0.01 | $2.70 | $2.70 |
| Lambda compute | ~$0.002 | $0.54 | $0.54 |
| DynamoDB | ~$0.001 | $0.27 | $0.27 |
| ElastiCache | $0.017/hr | ~4 hrs | $0.07 |
| **Total** | | | **~$3.85** |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ThrottlingException` from Bedrock | Too many concurrent LLM calls | Reduce `MaxConcurrency` in ASL, add delays |
| Lambda timeout (300s) | Large task count or slow Bedrock | Increase timeout or reduce tasks per run |
| Redis connection refused | Lambda not in VPC with ElastiCache | Add VPC config to Lambda via Terraform |
| `ModuleNotFoundError: pycylon` | Docker image missing pycylon | Verify Dockerfile build includes Cylon |
| Empty search results | Context store empty | Run baseline first, check workflow_id |
| Stratified sampling returns fewer tasks than requested | Deduplication on small task lists | Use `--sampling sequential` or increase task pool |