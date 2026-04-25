# Experiment Status — cylon-armada
## Context-Based Cost Optimization for Multi-Agent LLM Workflows

**Last updated**: 2026-04-25
**Repository**: cylon-armada
**Phase**: 1 infrastructure complete — ready for full experiment runs

---

## Infrastructure Status

All six execution paths are deployed and smoke-tested end-to-end on AWS.

| Workflow | State Machine | Status | S3 Results Path |
|----------|--------------|--------|-----------------|
| Lambda — Python | `cylon-armada-python-workflow` | ✅ PASSING | `results/lambda/` |
| Lambda — Node.js | `cylon-armada-nodejs-workflow` | ✅ PASSING | `results/lambda/` |
| Model Parallel (AstroMAE) | `cylon-armada-model-parallel-workflow` | ✅ PASSING (smoke/null models) | — |
| ECS Fargate | `cylon-armada-ecs-fargate-workflow` | ✅ PASSING | `results/ecs-fargate/` |
| ECS EC2 CPU | `cylon-armada-ecs-ec2-cpu-workflow` | ✅ PASSING | `results/ecs-ec2/` |
| ECS EC2 GPU | `cylon-armada-ecs-ec2-gpu-workflow` | ✅ PASSING | `results/ecs-ec2-gpu/` |

### Smoke Test Results (S3: `staylor.dev2`)

All runs: 3 tasks, `world_size=1`, `scaling=weak`, `context_backend=cylon`, `embedding_dimensions=1024`, `similarity_threshold=0.85`.
Third task is a duplicate of the first — expected cache hit rate ~33%.

| Platform | Date | Tasks | Cache Hits | Reuse Rate | Total Cost | Avg Latency |
|----------|------|-------|-----------|------------|------------|-------------|
| Lambda Python | 2026-04-17 | 1 | — | — | — | — |
| Lambda Python (ws3) | 2026-04-18 | — | — | — | — | — |
| ECS Fargate | 2026-04-21 | 3 | 1 | 33.3% | $0.067 | ~1560ms |
| ECS EC2 CPU | 2026-04-23 | 3 | 1 | 33.3% | $0.067 | ~1560ms |
| ECS EC2 GPU (gcylon) | 2026-04-25 | 3 | 1 | 33.3% | $0.056 | ~1187ms |

---

## Terraform Infrastructure

Three independent Terraform modules — apply/destroy independently to control cost.

| Module | Path | Manages | State |
|--------|------|---------|-------|
| `terraform/` | `target/aws/scripts/terraform/` | Lambda functions, Fargate cluster, DynamoDB, Redis ECS service, Step Functions (all workflows) | Applied |
| `terraform-ec2/` | `target/aws/scripts/terraform-ec2/` | EC2 launch template, ASG, capacity provider, CPU ECS task def, EC2 SFN | Applied |
| `terraform-gpu/` | `target/aws/scripts/terraform-gpu/` | GPU launch template, ASG (g4dn.xlarge), capacity provider, GPU ECS task def, GPU SFN | Applied |

### Key Architecture Decisions (from debugging)

- **`host` network mode** for EC2 tasks: `awsvpc` secondary ENIs in the default VPC have no public IP and can't reach Bedrock/S3 without a NAT gateway. `host` mode shares the instance's primary ENI.
- **Capacity provider strategy** in ASLs: ECS scales the ASG on demand when a task is submitted — no pre-provisioned instances needed. GPU ASG defaults to `min=0, desired=0`.
- **GPU image root volume**: 100 GB required (default 30 GB fills up with cuDF/conda layers).
- **`runCyloninLambda.sh` execute bit**: GPU Dockerfile was missing `chmod +x` — fixed in `docker/Dockerfile.gpu`.
- **S3 hot-reload**: shared scripts (`context/`, `chain/`, `cost/`) are loaded from S3 at task startup — fixes take effect without image rebuilds.

### ECR Image Tags

| Tag | Content | Used by |
|-----|---------|---------|
| `cylon-armada-python` | Python Lambda + ECS runner (CPU SIMD) | Lambda Python, ECS Fargate, ECS EC2 CPU |
| `cylon-armada-nodejs` | Node.js Lambda (WASM SIMD128) | Lambda Node.js |
| `cylon-armada-gpu` | GPU runner with gcylon/cuDF (CUDA SIMD) | ECS EC2 GPU |
| `rendezvous` | TCPunch rendezvous server | FMI direct channel experiments |

---

## What Remains for the PhD

### Phase 1 — Full Experiment Runs (NEXT)

The infrastructure is proven. Now run the full parameter sweeps defined in `EXPERIMENT_PLAYBOOK.md §3.2`.

**Experiment matrix per scenario:**

| Variable | Values |
|----------|--------|
| Context backend | `cylon`, `redis` |
| Similarity threshold | `0.70`, `0.80`, `0.90` |
| Embedding dimensions | `256`, `512`, `1024` |
| SIMD backend (Python) | `NUMPY`, `PYCYLON`, `CYTHON_BATCH` |
| Baseline vs reuse | both |
| Runs per config | 3 (for std dev) |

**Scenarios** (4 scientific domains):
1. Hydrology
2. Epidemiology
3. Seismology
4. Mixed Scientific
5. Astronomical inference (cosmic-ai — dynamic tasks from AstroMAE)

**Per-platform runs:**

| Platform | Scaling | World sizes | Priority |
|----------|---------|------------|---------|
| Lambda Python | Map-parallel | 1–1000 (SFN concurrency) | High |
| Lambda Node.js | Map-parallel | 1–1000 | High |
| ECS Fargate | weak + strong | 1, 2, 4, 8 | High |
| ECS EC2 CPU | weak + strong | 1, 2, 4, 8, 16 | High |
| ECS EC2 GPU (gcylon) | weak + strong | 1, 2, 4, 8 | Medium |
| Rivanna HPC | weak + strong | 1, 2, 4, 8, 16, 32 | Medium |

**Success criteria** (from `EXPERIMENT_PLAYBOOK.md §4.3`):
- Cost reduction: 60–80% (`savings_pct_mean`)
- Search latency: <20ms per 1000 contexts
- SIMD speedup: >2× vs numpy
- Statistical rigor: ≥3 runs per config
- LlamaIndex comparison: cylon cost < llamaindex cost at all task counts

### Phase 2 — FMI Integration Experiments

Already defined in `EXPERIMENT_PLAYBOOK.md §6`. Infrastructure deployed.

| Experiment | Research Question | Status |
|------------|-----------------|--------|
| Context Broadcast (Redis channel) | Does FMI broadcast eliminate Redis round-trips? | Ready to run |
| Context Broadcast (TCPunch direct) | Does direct channel reduce broadcast latency? | Ready to run |
| Progressive Context Sync | Does immediate sync improve within-run reuse? | Ready to run |
| Model Parallelism (AstroMAE) | Does FMI allgather remove 256KB SFN payload limit? | Smoke tested ✅ |

### Phase 3 — Rivanna HPC

- Pull `cylon-armada-python` from Docker Hub → Singularity on Rivanna
- SLURM job arrays for strong/weak scaling at larger world sizes (16, 32)
- A100/V100 GPU experiments with gcylon at HPC scale
- Status: infrastructure not yet set up on Rivanna side

### Phase 4 — RuVector Backend (Large-Scale Node.js)

- RuVector (`@ruvector/core`, NAPI-RS): HNSW approximate search with SimSIMD
- Only differentiates from WASM SIMD at N > 10K contexts (O(log n) vs O(n))
- Applies to Path B (Node.js) only — no Python SDK
- Status: deferred until Phase 1/2 data collected

### Thesis Writing Milestones

| Milestone | Depends on |
|-----------|-----------|
| Phase 1 results chapter draft | Full experiment runs + results pipeline |
| Phase 2 results chapter draft | FMI experiments |
| Phase 3 results chapter draft | Rivanna runs |
| Comparative analysis (all platforms) | All phases complete |
| Final submission | Committee approval |

---

## Immediate Next Steps

1. **Commit all infrastructure changes** — large set of terraform, ASL, Dockerfile, and shared script changes accumulated
2. **Run Phase 1 full experiment matrix** — use `EXPERIMENT_PLAYBOOK.md §3.3` scripts; start with Lambda Python + hydrology scenario (cheapest, fastest)
3. **Run results pipeline** — `python -m results.pipeline` to generate charts and aggregated CSV
4. **Fix `pricing:GetProducts` IAM permission** — added to all three terraform modules; apply to silence non-fatal warning in ECS logs *(done in code, needs `terraform apply`)*

---

## Open Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| `pricing:GetProducts` AccessDeniedException | Low | Non-fatal warning; fix in terraform, needs apply |
| Node.js Lambda Node workers not implemented for ECS | N/A | Intentional — WASM SIMD is CPU-only, no GPU benefit |
| Rivanna Singularity setup | Medium | Needed for Phase 3; not blocking Phase 1/2 |
| AstroMAE ONNX models not yet uploaded to S3 | Medium | Needed for real (non-smoke) model parallel runs |
| `savings_pct` always 0.0 in ECS metrics | Medium | `baseline_cost` not computed in `armada_ecs_runner.py` — needs paired baseline run logic |