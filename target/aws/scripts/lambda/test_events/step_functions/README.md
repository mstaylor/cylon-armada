# Step Functions Test Payloads

Pre-generated input payloads for manual Step Functions execution — one per
(scenario × world_size × runtime). Use these to run and validate individual
experiments without the sweep script.

## Directory Structure

```
step_functions/
├── README.md
├── index.json
├── epidemiology/
│   ├── nodejs_epidemiology_ws1.json   — Node.js FMI, ws=1
│   ├── nodejs_epidemiology_ws2.json   — Node.js FMI, ws=2
│   ├── ...
│   ├── nodejs_epidemiology_ws64.json
│   ├── python_epidemiology_ws1.json   — Python Redis, ws=1
│   └── ...
├── hydrology/
├── seismology/
└── mixed_scientific/
```

## State Machine ARNs

| Runtime | State Machine |
|---------|--------------|
| Node.js (FMI) | `arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-nodejs-workflow` |
| Python (Redis) | `arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-python-workflow` |

---

## How to Run a Single Experiment

### Option A — AWS Console

1. Open [Step Functions Console](https://us-east-1.console.aws.amazon.com/states/home?region=us-east-1)
2. Select `cylon-armada-nodejs-workflow`
3. Click **Start execution**
4. Paste the contents of e.g. `epidemiology/nodejs_epidemiology_ws2.json` as the input
5. Click **Start execution**
6. Monitor the execution graph in real time

### Option B — AWS CLI

```bash
# Run Node.js FMI experiment — epidemiology ws2
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-nodejs-workflow \
  --name "manual-epidemiology-ws2-$(date +%s)" \
  --input file://epidemiology/nodejs_epidemiology_ws2.json

# Run Python Redis experiment — hydrology ws4
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-python-workflow \
  --name "manual-hydrology-ws4-$(date +%s)" \
  --input file://hydrology/python_hydrology_ws4.json
```

### Option C — cloud_sweep.py dry-run then fire

```bash
# Preview what would fire
conda run -n cylon_dev python3 \
  /home/parallels/cylon-armada/target/aws/scripts/experiment/cloud_sweep.py \
  --arch lambda-nodejs \
  --scenario epidemiology \
  --world-sizes 2 \
  --scaling weak \
  --task-count 1 \
  --runs 1 \
  --context-backend cylon \
  --fmi-channel direct \
  --dry-run

# Then remove --dry-run to actually fire
```

---

## How to Check Results

### 1 — Poll S3 for the metrics file

```bash
# Watch for result (replace experiment_name with value from payload)
aws s3 ls s3://staylor.dev2/results/lambda-nodejs/epidemiology/weak/ | grep manual
```

### 2 — Download and inspect metrics

```bash
aws s3 cp \
  s3://staylor.dev2/results/lambda-nodejs/epidemiology/weak/lambda_nodejs_epidemiology_weak_ws2_manual_metrics.json \
  - | python3 -m json.tool
```

Key fields to check:
| Field | What it means |
|-------|--------------|
| `cache_hits` | Number of tasks served from cache (should be 0 on run 1) |
| `reuse_rate` | Fraction cached (0.0 → 1.0) |
| `avg_latency_ms` | Average per-task latency |
| `wall_clock_ms` | Total workflow time |
| `savings_pct` | Cost savings vs full LLM baseline |

### 3 — Check Lambda logs for FMI pairing

```bash
# Follow executor logs live
aws logs tail /aws/lambda/cylon-armada-executor-node --follow --format short | \
  grep -v "NodeDeprecation\|no longer support\|LTS"
```

Look for:
- `FMI communicator ready: rank=X worldSize=Y` — FMI init succeeded
- `Paired partnerId: X to pair_name: fmi_pairXXNONBLOCKING` — TCPunch paired
- `FMI broadcast: rank 0 sending N contexts` — context broadcast running
- `Task timed out after 300s` — FMI pairing failed (increase timeout or retry)

### 4 — Check Redis directly

```bash
redis-cli -h dev-cylon-redis1.aws-cylondata.com \
  KEYS "result:*manual*"
```

---

## Payload Details

Each payload contains:

```json
{
  "workflow_id": "manual-{scenario}-ws{N}",
  "tasks": ["task 0 text", "task 1 text", ...],  // N tasks, one per rank
  "world_size": N,
  "scaling": "weak",
  "context_backend": "cylon",      // FMI path  (nodejs)
                  // or "redis"    // Redis path (python)
  "fmi_channel_type": "direct",    // TCPunch P2P (nodejs only)
  "results_s3_dir": "results/lambda-nodejs/{scenario}/weak/",
  "experiment_name": "lambda_nodejs_{scenario}_weak_ws{N}_manual",
  "config": {
    "llm_model_id": "amazon.nova-lite-v1:0",
    "embedding_model_id": "amazon.titan-embed-text-v2:0",
    "embedding_dimensions": 1024,
    "similarity_threshold": 0.85,
    "region": "us-east-1"
  }
}
```

**Important**: `world_size = len(tasks)`. For FMI to work, exactly `world_size`
Map items fire — one executor per rank. Each executor handles one task and pairs
with all other ranks via TCPunch for context broadcast.

---

## Validating FMI Step by Step

To isolate and validate FMI at each world size:

```bash
# 1. ws1 — no FMI, just Redis path baseline
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-nodejs-workflow \
  --name "validate-ws1-$(date +%s)" \
  --input file://epidemiology/nodejs_epidemiology_ws1.json

# 2. ws2 — simplest FMI pair (rank 0 + rank 1)
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-nodejs-workflow \
  --name "validate-ws2-$(date +%s)" \
  --input file://epidemiology/nodejs_epidemiology_ws2.json

# 3. ws4 — binomial tree broadcast across 4 workers
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:448324707516:stateMachine:cylon-armada-nodejs-workflow \
  --name "validate-ws4-$(date +%s)" \
  --input file://epidemiology/nodejs_epidemiology_ws4.json
```

If ws2 works but ws4 fails — TCPunch pairing for 3+ simultaneous peers is the issue.
If ws1 works but ws2 fails — FMI init / rendezvous connectivity issue.