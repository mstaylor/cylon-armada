# Experiment Status — cylon-armada
## Zero-Copy Agentic Data Orchestration: Modeling Distributed AI Workflows as Communication Operators over Apache Arrow and Cylon

**Last updated**: 2026-06-13
**Repository**: cylon-armada
**Proposal**: `docs/uva_phd_proposal.pdf`

---

## Phase Model (revised June 2026)

The proposal reorganizes the work so the **core scientific contribution is delivered on
high-performance infrastructure first**, with serverless as a studied follow-on. This
supersedes the earlier serverless-first / "context-cost-optimization" phase numbering.

| Phase | Scope | Platform | Status |
|-------|-------|----------|--------|
| **Phase 0** | Serverless BSP viability — *can FaaS sustain barrier-synchronized collectives?* | AWS Lambda vs EC2 | ✅ **COMPLETE** (prior work, arXiv:2501.06249) |
| **Phase 1** | **Agentic Arrow data plane + collective mapping (the core thesis)** | **Rivanna HPC + EC2/EFA** | 🔜 **PRIMARY — not yet run** |
| **Phase 2** | Serverless extension — *when is serverless preferable?* (duty-cycle δ*, stragglers, fault tolerance) | AWS Lambda / Fargate | ⏳ Follow-on |

**Phase 0 result (in hand):** Cylon on Lambda with direct P2P (NAT-traversal TCP) scales
**within 6.5% of EC2 at 64 nodes** (15.85× vs 16.96× speedup); allreduce ≈13 ms at N=32;
barrier sync grows as log₂N. This settles the serverless-BSP question and is the foundation
on which Phase 2 builds — it is *not* a risk to the Phase 1 contribution.

> **Old → new mapping.** The earlier docs called the Lambda context-reuse sweep "Phase 1
> (Proof-of-Concept)." That work is now **Phase 0** (serverless viability, complete) plus the
> serverless arm of **Phase 2**. The new **Phase 1** is the HPC/Rivanna data plane, which the
> existing Lambda infrastructure does *not* yet cover.

---

## Experiment Map (Proposal §VI)

Experiments are organized into three tiers and tied to hypotheses **H1–H5**. The 464 completed
Lambda runs are **preliminary serverless evidence** feeding Experiments B and E; the HPC arm
(Phase 1) is the primary deliverable and is pending.

| Exp | Tier | Question | Hyp. | Phase | Status |
|-----|------|----------|------|-------|--------|
| **A** | 1 Microbench | Arrow IPC throughput vs JSON/HTTP, Pickle/gRPC, Protobuf/gRPC | H2 | 1 | ⏳ pending |
| **A2** | 1 Microbench | Arrow schema-compatibility validation | H2 | 1 | ⏳ pending |
| **B** | 1 Microbench | Cylon collective latency/throughput, N=4–64 | H1 | 0/1 | ◑ partial (Lambda data exists) |
| **C** | 2 Integration | Resource-deterministic scheduling reduces exec-time variance (CV ≥50%↓) | H1 | 1 | ⏳ pending |
| **D** | 2 Integration | Data-communication reduction; validates Formula 1: D(P\*) ≤ D(P_stateless)/N | H2 | 1 | ⏳ pending — **vehicle chosen 2026-09-24: population-scale anomaly detection over SDSS, not Cosmic AI.** Spec: `docs/superpowers/specs/2026-09-24-expD-population-anomaly-design.md` |
| **E** | 2 Integration | **End-to-end pipeline speedup S(N); validates Formula 2. Primary baseline = Ray/Plasma; HTTP = diagnostic** | — | 1 | ◑ HTTP arm only (S(64)=4.2×); Ray arm pending |
| **F** | 3 Infra | Arrow Flight cross-region transport | — | 2 | ⏳ pending |
| **G** | 2 Integration | Asynchronous batching / pipeline overlap | — | 1 | ⏳ pending |
| **H** | 2 Integration | LLM generation throughput preserved (no degradation) | H3 | 1 | ⏳ pending |
| **I** | 3 Infra | Straggler & cold-start jitter at p99, N≤64 | — | 2 | ⏳ pending |
| **J** | 3 Infra | ContextTable search latency, semantic hit-rate h, recall vs FAISS, L3 degradation | H4 | 1/2 | ◑ search-latency measured: **11.1 µs per stored embedding, linear to 200K; one worker fails between 200K and 1M** (2026-09-22) |
| **K** | 3 Infra | Fault tolerance / MTTR; 2PC + WAL roll-forward recovery (MTTR ≥80%↓) | H5 | 2 | ⏳ pending |
| **L** | 3 Infra | Strong scaling, Amdahl serial fraction, serverless-vs-HPC gap, duty-cycle δ* | — | 2 | ⏳ pending |

**Hypotheses (Proposal §I-A):** H1 communication-overhead reduction (CV ≥50%↓, collective
latency ≥90%↓) · H2 zero-copy throughput gain + D(P\*) bound · H3 LLM-throughput isolation ·
H4 semantic-reuse cost reduction (≥70% at h≥0.70, recall ≥0.95) · H5 serverless-native fault
tolerance (MTTR ≥80%↓).

---

## Infrastructure Status

All six cloud execution paths are deployed and smoke-tested end-to-end on AWS. These cover
**Phase 0 / Phase 2 (serverless + container)** and Contribution #8 (deployment validation).
**Phase 1's primary platform — Rivanna HPC + EC2/EFA — is not yet stood up.**

| Workflow | State Machine | Phase | Status | S3 Results Path |
|----------|--------------|-------|--------|-----------------|
| Lambda — Python | `cylon-armada-python-workflow` | 0/2 | ✅ PASSING | `results/lambda/` |
| Lambda — Node.js | `cylon-armada-nodejs-workflow` | 0/2 | ✅ PASSING | `results/lambda/` |
| Model Parallel (AstroMAE) | `cylon-armada-model-parallel-workflow` | 2 | ✅ PASSING (smoke/null models) | — |
| ECS Fargate | `cylon-armada-ecs-fargate-workflow` | 2 | ✅ PASSING | `results/ecs-fargate/` |
| ECS EC2 CPU | `cylon-armada-ecs-ec2-cpu-workflow` | 2 | ✅ PASSING | `results/ecs-ec2/` |
| ECS EC2 GPU | `cylon-armada-ecs-ec2-gpu-workflow` | 2 | ✅ PASSING | `results/ecs-ec2-gpu/` |
| **Rivanna HPC + EC2/EFA** | — | **1** | ❌ **not yet set up (Phase 1 primary)** | `results/rivanna/` (planned) |

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

### Preliminary serverless evidence (feeds Experiments B, E)

464 runs completed on Lambda (Python + Node.js, ws1–ws64, 4 domains, 4 runs/config). Headline:
**S(64) = 4.2× over a concurrent stateless-HTTP baseline**, Lambda within 6.5% of EC2 at N=64.
These are the **HTTP-baseline arm of Experiment E only** — the **Ray/Plasma baseline (the
load-bearing comparison) and the HPC/Rivanna arm are still to run**, and S(N) is expected to
compress against Ray.

---

## Terraform Infrastructure

Three independent Terraform modules — apply/destroy independently to control cost. (Cover the
serverless + container paths; a Rivanna/EFA path is not Terraform-managed.)

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

## What Remains for the PhD (by revised phase)

### Phase 1 — Agentic data plane on HPC (CORE THESIS, delivered first)

The primary deliverable. Runs on Rivanna + EC2/EFA, where BSP suitability is unquestioned;
no Lambda dependency, no 15-min ceiling.

- **Stand up the Rivanna / EC2-EFA path** — Singularity image from `cylon-armada-python`; libfabric/EFA communicator; SLURM job arrays for weak + strong scaling (N up to 64).
- **Run the Tier 1–2 experiments on HPC:** A, A2, B (collectives), C (scheduling variance), D (data-movement bound), **E (S(N) vs Ray/Plasma — the headline)**, G (batching overlap), H (LLM throughput isolation).
- **ContextTable evaluation (J):** semantic hit-rate h, recall vs FAISS, L3 residency/degradation.
- **Success criteria (replace the old "60–80% cost reduction"):** H1 CV ≥50%↓ & collective latency ≥90%↓ · H2 D(P\*) ≤ D(P_stateless)/N & embedding/upsert throughput gain · H3 no LLM-throughput degradation · S(N) speedup over **Ray** (not just HTTP) · H4 recall ≥0.95 at h≥0.70.

### Phase 2 — Serverless extension (follow-on)

Port the proven data plane onto Lambda/Fargate and characterize *when serverless wins*.

- **Experiment I** — straggler & cold-start jitter at p99, N≤64 (compute-skew characterization — the variable-length Reason operator is the dominant source of barrier variance).
- **Experiment K** — serverless-native fault tolerance: 2PC checkpointing + WAL roll-forward recovery; MTTR ≥80%↓ vs full-epoch replay.
- **Experiment L** — strong scaling, Amdahl serial fraction, serverless-vs-HPC gap, and the duty-cycle threshold **δ\*** at which reserved EC2 becomes cheaper per TB than Lambda.
- **Experiment F** — Arrow Flight cross-region transport.
- Existing FMI work that feeds Phase 2: Context Broadcast (Redis vs TCPunch channels), Progressive Context Sync, Model Parallelism (AstroMAE allgather — already smoke-tested ✅).

### Cross-cutting — Deployment validation (Contribution #8)

Six AWS paths + Rivanna at world sizes 1–64, confirming the same operator abstraction runs
unmodified across all backends.

### Thesis Writing Milestones

| Milestone | Depends on |
|-----------|-----------|
| Phase 1 (HPC data plane) results chapter | Rivanna/EFA experiments A–H, J |
| Phase 0 serverless-viability chapter | Already in hand (arXiv:2501.06249) |
| Phase 2 (serverless extension) results chapter | Experiments I, K, L, F |
| Comparative analysis (HPC vs serverless, vs Ray) | Phases 1 + 2 complete |
| Final submission | Committee approval |

---

## Immediate Next Steps

1. **Stand up the Rivanna / EC2-EFA path** — this is the gating item for the Phase 1 core thesis; everything else is secondary.
2. **Run the Ray/Plasma baseline N-sweep for Experiment E** — the load-bearing comparison; current 4.2× is vs HTTP only and will compress against Ray. Local wiring is done; the cloud sweep is next.
2a. **Cosmic AI Lambda arms** — one executor rebuild carrying pycylon FMI and the ContextTable, then Arm A reproduction, Arm B (pycylon FMI) ws=1, 2, then sweep, then Arm C at tolerance 0.0005 with `redshift_analysis` reuse disabled.
3. **Commit accumulated infrastructure changes** — terraform, ASL, Dockerfile, shared-script changes.
4. **Re-run the serverless sweep as Phase 2 evidence** (not as the headline) once the data plane is validated on HPC.

---

## Open Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| Rivanna / EC2-EFA path not set up | **High** | Gates the Phase 1 core thesis (new primary platform) |
| Exp D vehicle chosen: population-scale anomaly detection | **Resolved 2026-09-24** | Cosmic AI divides its input, not its state, so work that is independent per object cannot demonstrate communication. The replacement runs over the same SDSS data and the same runtime: scatter the catalogue, infer per galaxy, **redistribute by subpopulation and REDUCE partial statistics**, then characterise the summaries. The reduce is intrinsic, since no worker can compute a population statistic from its own block, which is why this exercises Table III's mapping without forcing. **Critical design decision:** the redistribution must carry feature vectors, not summary scalars. Scalars move 166 MB total and only 9× what Cosmic AI moves today; features at 4 KB per galaxy move 10.6 GB, four orders of magnitude more, and that is the difference between an experiment with a signal and one without. Spec: `docs/superpowers/specs/2026-09-24-expD-population-anomaly-design.md` |
| Cosmic AI's per-object framing does not survive scale | **Medium** | Reported as a finding rather than a defect. Measured: inference is 0.134 s/galaxy against a 3.31 s model call, so the science is under 8% of runtime and commentary is 92%. Ungated, 99.6% of galaxies accept one of five stored answers; gated, 88% accept one of 146. A task where eight of nine objects take an interchangeable answer has an output nearly constant across its input, so the reuse rate measures redundancy in the task design rather than an optimisation. The population framing cuts model calls from 304,200 to ~100 across SDSS, wall clock from 5.9 h to ~1.6 h at N=64, and cost from ~$58 to ~$0.02. Cosmic AI is not being rewritten; it remains the Exp E vehicle. |
| Cosmic AI cannot carry the communication claim (H2 / Exp D) | **Superseded by the row above** | Measured 2026-09-22. Cosmic AI partitions *queries*, not *state*: each rank owns its galaxies end to end, so the entire cross-rank data plane is one AllGather of ~281 KB per rank per run. The one route by which it might have grown communication-heavy — partitioned Retrieve over a large stored-embedding set — is closed: stored embeddings grow **sublinearly in galaxies (exponent 0.498, better fit than linear)**, reaching only ~6,800 at full SDSS scale, where search is 2.3% of a model call. Exp D needs a **state-partitioned** workload. Proposed vehicle: retrieval over a knowledge base too large for one worker, which forces Retrieve→REDUCE by construction and makes the stateless S3-mediated baseline the one the serverless literature criticises. See `docs/Retrieve_Scaling_Findings_2026-09-22.md`. |
| Reuse validity gate: per-template fix shipped to production (3/3 templates), photometry_classification hardened through two adversarial review rounds | **High, all three templates shipped 2026-09-29** | Ungated, reuse is 0.996 and the store saturates at 5 entries — embedding cosine similarity alone is not a valid correctness signal. The deployed z_pred-only gate was 0.884 reuse / **0.500 answer-level agreement** (95% CI [0.386, 0.564], n=120) — barely better than chance, decomposed and fixed per template. `outlier_analysis`: unchanged, already correct (0.917). `redshift_analysis`: `reuse_key` is the **signed** residual (`z_pred − z_true`), measured at 0.89-0.925 across two independent samples. `photometry_classification`: since the task is deterministic (`|standardized value| > 2` per band), `reuse_key` is now assigned up front from ground truth alone (`photometry_reuse_key_for`), a hit returns a response synthesized from the query's own magnitudes (`synthesize_photometry_response`) rather than a matched candidate's stored text, and a fresh response is re-verified against ground truth at write time (`verified_photometry_reuse_key`) before its key is trusted. `is_photometry_key` reserves `[1000, 1031]` so a categorical photometry key can never collide with a continuous redshift/outlier key under any configured tolerance. Two adversarial review rounds found and closed: reuse never actually firing (a query-side key was missing), borrowed-text responses, a tolerance-based key collision, a config-override breaking ground truth, a reuse-metrics double-count, three vacuous tests, and unvalidated malformed inputs. 566 tests pass (18 pre-existing skips). See `docs/Reuse_Gate_Validity_Findings_2026-09-28.md` (Parts 10-13) for full detail, including the two items deliberately left unfixed (no discriminator between `redshift_analysis`/`outlier_analysis` keys; a malformed magnitude vector can crash prompt generation before gating runs) and the corrected storage-persistence gap description. **The closing claim that all three templates can contribute measured reuse is superseded by the row below.** |
| Reuse gate tolerance operating characteristic (Arm C Phase 1, spec §4.4) | **Resolved 2026-10-02; changes Arm C** | Full specified sweep over all 1253 galaxies at 0.0005 to 0.05 plus ungated (3,242 Bedrock calls, $0.38), replacing the sampled 18 to 40 pair method. The pre-declared fit/report rule selects **no** tolerance. `redshift_analysis`: raw agreement (0.79 to 0.93) is a base-rate artefact, since about 85% of fresh verdicts are "accurate"; at the deployed 0.0091 kappa is 0.04 against a 0.71 self-agreement ceiling and the false-normal rate is 0.85 (self 0.16). Keyed on z_pred alone, kappa never exceeds 0.12, which is the outcome §4.4 named in advance: this reuse cannot be made safe from inference-time information, and the shipped signed-residual key is an oracle. `outlier_analysis`: false-normal at most 0.06 at 0.0091 or tighter, near self (0.03 to 0.05). `photometry_classification`: tolerance-independent (0.877 reuse); synthesized answers are 100% correct against ground truth while fresh Nova Lite answers are 66.5% correct. **Arm C decisions:** `REUSE_KEY_TOLERANCE=0.0005`; `redshift_analysis` reuse disabled and reported as a negative result; `outlier_analysis` gated; photometry savings reported as rule-synthesized answers, not reused model output. See `docs/superpowers/findings/2026-10-02-armc-phase1-tolerance-sweep.md`. |
| Ray/Plasma baseline in Experiment E | **High, in progress 2026-10-02** | Required to answer the "strawman baseline" critique; 4.2× is vs HTTP only. `ray-native` and `ray-cylon` are wired end to end in `run_cosmic_local.py` with a final cross-rank barrier, a shared Ray namespace and per-run cleanup; 342 tests pass. The comparable `ray-native` total is `run_s + teardown_barrier_s` against Armada's `run_s`, and `ray_cluster_s` is reported separately. Next is the cloud N-sweep. See `docs/superpowers/findings/2026-10-02-ray-arms-wiring-implementation.md`. |
| Cosmic AI Lambda data plane, Arms A/B/C | **In progress 2026-10-02** | Arm A runs end to end on Lambda (smoke at ws=1 and ws=2, all stage fields present, ranks_aggregated equals world_size). One image, `cosmic-ai-executor`, for all arms (spec §7.1), being extended with pycylon FMI and the ContextTable so Arm B moves to pycylon FMI (v2 rendezvous) and Arm C can run zero-copy. See `docs/superpowers/findings/2026-10-02-cosmic-ai-arms-status.md`. |
| `pricing:GetProducts` AccessDeniedException | Low | Non-fatal warning; fix in terraform, needs apply |
| Node.js Lambda workers not implemented for ECS | N/A | Intentional — WASM SIMD is CPU-only, no GPU benefit |
| AstroMAE ONNX models not yet uploaded to S3 | Medium | Needed for real (non-smoke) model-parallel runs |
| `savings_pct` always 0.0 in ECS metrics | Low | Legacy cost metric; de-emphasized under data-orchestration framing |
| L3 noisy-neighbor (multi-tenant) condition not in Exp J | Medium | Reviewer-raised; Exp J currently single-tenant only |