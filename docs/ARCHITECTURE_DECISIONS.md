# Architecture Decisions — cylon-armada
## Context-Based Cost Optimization for Multi-Agent LLM Workflows

---

## ADR-001: Embedding Payload Offload via Redis (Non-FMI Path)

**Status**: Implemented (May 2026)  
**Context**: Phase 1 scaling experiments  
**Decision maker**: mstaylor

### Problem

AWS Lambda Step Functions payloads are hard-capped at **6 MB** (output) and **256 KB** (input per Map item). When `armada_init` embeds all tasks via Bedrock and returns the embeddings inline in the Step Functions response, the payload size is:

```
tasks × embedding_dimensions × 4 bytes (float32) + task text + metadata
```

At 64 tasks × 1024 dimensions: ~262 KB embeddings alone. With task text and metadata this easily exceeds 6 MB, causing HTTP 413 errors that prevent large-scale (world_size > 1) Lambda experiments.

This is a fundamental architectural constraint of AWS Step Functions — not a bug.

### Decision

**Non-FMI path**: `armada_init` stores each task embedding in Redis keyed by `embedding:{workflow_id}:{rank}` (TTL 1 hour) and replaces `embedding_b64` in the Step Functions payload with a lightweight `embedding_key` string. `armada_executor` retrieves the embedding from Redis before routing.

```
armada_init                armada_executor
    │                           │
    ├─ embed task_i ──────────► Redis: embedding:{wf_id}:{i}
    │                           │
    └─ SFN payload: {           └─ GET embedding:{wf_id}:{rank}
         embedding_key: "..."       │
         task_description: "..."    └─ route(task, embedding)
       }
```

**FMI path** (Phase 2): embeddings are broadcast peer-to-peer via TCPunch direct channels, eliminating the Redis round-trip entirely. This is the key research contribution.

### Consequences

- **Enables**: Lambda experiments at world_size = 1, 2, 4, 8, 16, 32, 64 without payload limit errors
- **Adds**: one Redis GET per executor invocation (~1-5ms latency overhead)
- **Creates**: the architectural baseline against which FMI is compared:

| Path | Embedding transfer | Latency overhead | Redis dependency |
|------|-------------------|-----------------|-----------------|
| Non-FMI (Redis offload) | init → Redis → executor | Redis RTT per task | Required |
| FMI direct (TCPunch) | init → TCPunch → executor | ~0ms | Not required |

This overhead difference is quantifiable and becomes a primary thesis data point.

### Implementation

- `armada_init.py`: `embedding_key` offload when `REDIS_HOST` is set
- `armada_executor.py`: Redis GET when `embedding_key` present; `embedding_b64` inline fallback for backward compatibility
- Backward compatible: if Redis unavailable, `embedding_b64` remains inline

---

## ADR-002: Scaling Mode Definitions (Weak vs Strong)

**Status**: Implemented (May 2026)  
**Reference**: `cylon/target/shared/scripts/scaling/scaling.py` lines 281-286

### Definition (aligned with cylon scaling.py)

```python
# cylon scaling.py pattern:
if scaling == 'w':   # weak
    num_rows = data['rows']               # work per worker = constant
    max_val  = num_rows * world_size      # total grows with world_size
else:                # strong
    max_val  = data['rows']               # total work = constant
    num_rows = int(data['rows'] / world_size)  # work per worker shrinks
```

Applied to cylon-armada (`task_count` = `rows`):

| Mode | Tasks in TASKS_JSON | Per-worker tasks | Total work |
|------|-------------------|-----------------|-----------|
| **Weak** | `task_count × world_size` | `task_count` (constant) | Grows |
| **Strong** | `task_count` | `task_count / world_size` | Fixed |

### Task count choice

`task_count = 64` chosen so strong scaling at `world_size = 64` gives 1 task/worker (minimum meaningful unit). Task tiling is used for weak scaling when `task_count × world_size` exceeds scenario size — valid because production LLM workloads naturally see recurring task types.

### World sizes

```
1, 2, 4, 8, 16, 32, 64
```

Lambda experiments are limited by `start_sync_execution` 5-minute timeout at large world sizes (see ADR-001 — embedding offload removes the payload limit, but execution time limits remain for very large weak scaling configurations).

---

## ADR-003: Context Backend Selection (CONTEXT_BACKEND)

**Status**: Implemented

| Backend | Storage | Cross-invocation reuse | Use case |
|---------|---------|----------------------|---------|
| `cylon` | In-memory Arrow ContextTable + Redis Arrow IPC snapshot | Requires `load_from_redis()` | ECS/Rivanna (long-lived process) |
| `redis` | Redis key-value (embedding + metadata) | Native — reads Redis on every call | Lambda (stateless invocations) |

**Lambda experiments use `redis` backend** because each Lambda invocation starts a fresh Python process. The cylon backend's in-memory table is empty at startup; the Redis IPC snapshot restore (`load_from_redis()`) adds latency and complexity that the pure redis backend avoids.

**ECS/Rivanna experiments use `cylon` backend** because the process is long-lived — the in-memory ContextTable accumulates contexts across all tasks in the run, with Redis as the persistence layer for cross-run reuse.

---

## ADR-004: Results S3 Path Convention

```
s3://staylor.dev2/results/{architecture}/{scenario}/{scaling}/
    {experiment_name}_metrics.json
    {experiment_name}_stopwatch.csv
    {experiment_name}_summary.csv
```

Where:
- `architecture`: `lambda-python`, `lambda-nodejs`, `ecs-fargate`, `ecs-ec2-cpu`, `ecs-ec2-gpu`, `rivanna`, `rivanna-gpu`
- `scenario`: `hydrology`, `epidemiology`, `seismology`, `mixed_scientific`
- `scaling`: `weak`, `strong`
- `experiment_name`: `{arch}_{scenario}_{scaling}_ws{world_size}_run{n}_{tag}`