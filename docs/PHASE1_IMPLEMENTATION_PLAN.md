# Phase 1: Proof-of-Concept Implementation Plan
## Context-Based Cost Optimization for Multi-Agent LLM Workflows

**Status**: Draft
**Repository**: cylon-armada

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Breakdown](#component-breakdown)
4. [Benchmark Strategy](#benchmark-strategy)
5. [Local Development](#local-development)
6. [Lambda Deployment](#lambda-deployment)
7. [Test Scenarios](#test-scenarios)
8. [Success Criteria](#success-criteria)
9. [Risks & Open Questions](#risks-open-questions)

---

## Overview

Phase 1 delivers a working proof-of-concept that validates the core thesis: **semantic similarity-based context reuse reduces LLM costs by 60-80% in multi-agent workflows without sacrificing output quality**. This is validated with real Bedrock API calls and real dollar costs.

### Scope

Build an end-to-end context reuse system with three execution paths:

- **Path A1 — Python + pycylon (per-call SIMD)**: Lambda functions using the existing Cylon C++ framework with native SIMD similarity search. Python loop calls `cosine_similarity_f32` per embedding pair. Extends the Docker images in `cylon/docker/aws/lambda/`.
- **Path A2 — Python + Cython (batch SIMD)**: Same Lambda environment as A1, but the entire similarity search loop is pushed into C/C++ via a Cython extension module. Single Python→C++ boundary crossing for the full search.
- **Path B — Node.js + WASM**: Lambda functions using the Cylon WASM module (`cylon-wasm`) with WASM SIMD128 similarity search. Extends `wasm_handler.mjs`.

All three paths are benchmarked against each other to produce comparative performance data for the thesis.

### Deliverables

- Working context store (DynamoDB + Redis) with embedding storage and retrieval
- Similarity engine using Cylon SIMD (`cosine_similarity_f32`) via both native and WASM paths
- Context router that decides reuse vs. new LLM call based on similarity threshold
- 4+ test scenarios executing 32+ tasks against AWS Bedrock with real cost capture
- Experimental results demonstrating 60-80% cost reduction
- Jupyter notebooks with visualizations (cost curves, reuse rates, latency distributions)
- Proposal paper draft with experimental methodology and results

### Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM Provider | AWS Bedrock only | Existing AWS infrastructure, no API key management |
| Embedding Model | Amazon Titan Text Embeddings v2 | See [Embedding Model Selection](#embedding-model-selection) below |
| Persistent Storage | DynamoDB | Serverless, free tier for PoC, matches production design |
| Hot Cache | Redis (ElastiCache) | Already used by FMI/Cylon Lambda infrastructure |
| Cost Tracking | Cylon CostTracker | Existing framework in `cylon/target/shared/scripts/scaling/costlib/` — extended for Bedrock costs |
| LLM Chain Execution | LangChain | Chain/LCEL patterns for Bedrock invocation; matches architecture diagram; same code local and Lambda |
| SIMD Operations | Verified | `cosine_similarity_f32`, `dot_product_f32`, `euclidean_distance_f32` in `cylon-wasm/src/simd.rs` with full test coverage |

### Embedding Model Selection

Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) is chosen for both practical and experimental reasons.

**Bedrock Embedding Model Comparison:**

| Model | Dimensions | Cost (per 1K tokens) | Notes |
|-------|-----------|----------------------|-------|
| Amazon Titan Text Embeddings V2 | 256 / 512 / 1024 (configurable) | ~$0.00002 | Cheapest, configurable dims, native Bedrock |
| Amazon Titan Text Embeddings V1 | 1536 (fixed) | ~$0.0001 | Higher cost, fixed dimensions |
| Cohere Embed English V3 | 1024 (fixed) | ~$0.0001 | Good quality, 5x cost of Titan V2 |

**Why Titan V2:**

1. **Configurable dimensions as experiment variable.** Titan V2 uniquely supports 256, 512, and 1024 dimensions via the `dimensions` parameter. This creates a 2D parameter space (dimension × similarity threshold) for experiments — the same embedding model produces vectors of different sizes, isolating the effect of dimensionality on reuse accuracy and cost.

2. **SIMD performance implications.** Different embedding dimensions directly affect SIMD computation costs:
   - 1024-dim → 256 SIMD ops (4×f32 lanes per v128 op)
   - 512-dim → 128 SIMD ops
   - 256-dim → 64 SIMD ops

   This provides a measurable performance axis: does reducing dimensions improve similarity search latency enough to offset any quality loss? The Cylon SIMD engine (`cosine_similarity_f32`) processes 4 floats per WASM SIMD128 instruction, making this directly quantifiable.

3. **Cost advantage.** At ~$0.00002 per 1K tokens, Titan V2 is 5× cheaper than alternatives. For a PoC generating thousands of embeddings, this keeps experimental costs under $1 total.

4. **Native Bedrock integration.** No external API keys or network configuration — invoked directly via `bedrock-runtime` with the same IAM credentials used for LLM calls.

**Experiment Design Implication:** The dimension sweep (256/512/1024) combined with threshold sweep (0.70–0.95) produces a grid of experimental configurations, each measuring: reuse rate, cost savings, output quality (BLEU/ROUGE vs. baseline), and SIMD search latency.

### cosmic-ai Integration (Phase 1)

The context reuse architecture is validated on real scientific workloads using the **cosmic-ai** astronomical inference pipeline (arXiv:2501.06249). This demonstrates applicability beyond general-purpose NLP tasks.

**Integration approach:**
1. **AstroMAE inference** → run pre-trained model on SDSS photometric data partitions
2. **Task generation** → dynamically generate LLM analysis tasks from real inference results (redshift predictions, magnitude profiles, error metrics)
3. **Context reuse** → similar galaxy observations produce semantically overlapping analysis prompts, creating natural context reuse opportunities

**Components (under `target/experiments/cosmic_ai/`):**
- `blocks/` — AstroMAE model architecture (ViT + Inception, photoz, NormalCell) — from AI-for-Astronomy
- `inference.py` — refactored inference module (callable, returns structured results)
- `task_generator.py` — generates LLM tasks from inference output with configurable templates
- `export_onnx.py` — export AstroMAE to ONNX for Node.js inference and model parallelism

**Node.js / ONNX support:** The AstroMAE model is exported to ONNX format and runs via `onnxruntime-node` on the Path B Node.js Lambda runtime. This gives Path B a complete end-to-end pipeline: ONNX inference → task generation → WASM SIMD context reuse.

**cosmic-ai AWS infrastructure** (`AI-for-Astronomy/aws/`) uses the same 3-stage serverless pattern as cylon-armada: Initialize → Distributed Map → Summarize. Same S3 script runner, same Lambda containers, same Step Functions orchestration.

### Serverless Model Parallelism via FMI (Phase 1)

For models that exceed a single Lambda's memory or compute capacity, the system supports **model parallelism across Lambda functions** using Cylon's FMI communicator for inter-Lambda tensor exchange.

**Architecture:**

AstroMAE has a natural parallel split — the ViT encoder and Inception branch are **independent** and can execute concurrently on separate Lambda containers:

```
                     ┌─────────────────────────┐
                     │  Input: [image, magnitude]│
                     └──────────┬──────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Lambda 0 (rank=0)│      │ Lambda 1 (rank=1)│
        │                  │      │                  │
        │ ONNX Stage 0:   │      │ ONNX Stage 1:   │
        │ ViT Encoder      │      │ Inception Branch │
        │  patch_embed     │      │  conv2d_init     │
        │  transformer     │      │  inception blocks│
        │  blocks          │      │  magnitude MLP   │
        │  fc_norm + head  │      │                  │
        │  vit_block       │      │                  │
        │                  │      │                  │
        │ Output: (B,1096) │      │ Output: (B,2120) │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └──────────┬──────────────┘
                            │ FMI all-gather
                            ▼
                 ┌──────────────────────┐
                 │ Lambda 0 or 1        │
                 │ ONNX Stage 2:        │
                 │ Fusion (concat_block)│
                 │ Output: (B, 1)       │
                 │ → redshift prediction│
                 └──────────────────────┘
```

**Key design decisions:**

1. **ONNX graph partitioning:** The full ONNX model is split into 2-3 subgraphs at export time. Each Lambda loads only its assigned subgraph — reducing per-Lambda memory and enabling models that don't fit in a single container.

2. **FMI tensor exchange:** Intermediate activations are exchanged via Cylon's FMI communicator:
   - **Direct channel (TCPunch):** <10ms latency via Rendezvous Server — primary for pipeline stages
   - **Redis channel:** Fallback for tensor exchange, also used for coordination
   - **S3 channel:** For very large intermediate tensors (>100MB)

3. **Parallel branches:** AstroMAE's ViT and Inception branches run concurrently (data parallelism within model parallelism). The FMI `all-gather` collects outputs from both branches before the fusion stage.

4. **Generalization:** While AstroMAE is the Phase 1 demo model, the ONNX partition + FMI exchange pattern works for any model — including large language models split across Lambda containers. This is a key thesis contribution: **serverless model parallelism using FMI-based communication.**

**Components:**
- `export_onnx.py` — export and optionally partition the ONNX model
- Node.js: `onnxruntime-node` for ONNX inference in Lambda containers
- Python: `onnxruntime` for ONNX inference (alternative to PyTorch for lighter containers)
- FMI communicator (Redis/Direct channel) for inter-Lambda tensor exchange

**Memory-Aware Partitioning:**

Each ONNX subgraph has different memory requirements. The export/partition tool estimates per-stage memory so the orchestrator can assign appropriate Lambda memory configurations:

```
┌────────────────────────────────────────────────────────────────┐
│ ONNX Partition Report                                          │
├──────────┬──────────┬──────────────┬──────────────────────────┤
│ Stage    │ Params   │ Est. Memory  │ Lambda Memory Config     │
├──────────┼──────────┼──────────────┼──────────────────────────┤
│ 0 (ViT)  │ ~60-70%  │ ~350MB peak  │ 1024 MB                  │
│ 1 (Incep)│ ~30-40%  │ ~200MB peak  │ 512 MB                   │
│ 2 (Fuse) │ <1%      │ ~50MB peak   │ 256 MB (runs on rank 0)  │
└──────────┴──────────┴──────────────┴──────────────────────────┘
```

Peak memory per Lambda = (subgraph size) + (local activations) + (FMI exchange buffer)

**Memory management strategy:**

1. **Per-stage Lambda memory config** — Step Functions payload specifies memory per rank. The ViT stage (heavier transformer blocks) gets more memory than the Inception stage (lighter CNNs).

2. **S3 lazy loading** — each Lambda downloads only its assigned ONNX subgraph from S3 at cold start. The full model is never loaded on any single Lambda.

3. **Deep partitioning for very large models** — if a single stage exceeds Lambda's 10GB container limit, transformer blocks can be further split across Lambdas (e.g., blocks 0-5 on rank 0, blocks 6-11 on rank 1) with FMI pipelining activations between stages.

4. **Memory estimation at export time** — `export_onnx.py` reports per-stage parameter counts and estimated peak memory, enabling the orchestrator to auto-configure Lambda memory settings.

**Benchmark dimensions:**
- Single Lambda (full model) vs. 2-Lambda parallel (ViT + Inception split)
- FMI Direct channel vs. Redis channel for tensor exchange latency
- Memory utilization per stage (actual vs. estimated)
- Model size threshold at which parallelism becomes beneficial

### Swarm Orchestration (Future Phase — Custom Implementation)

Multi-agent swarm patterns will be implemented as a cylon-armada native capability in a future phase, built on the Cylon Communicator and Step Functions infrastructure. This is not an integration of an external project — it will be a custom implementation drawing from cognitive diversity and topology-based coordination concepts.

### Out of Scope (Phase 1)

- Multi-agent cognitive diversity patterns (future phase)
- Advanced Step Functions patterns (choice states, error handling workflows) (Phase 2)
- Hierarchical indexing / FAISS (future optimization)
- Cross-provider experiments (OpenAI comparison)
- Production multi-region deployment

---

## Architecture

The full system architecture is defined in the [Master Architecture Reference](../../../cylon/docs/design/MASTER_ARCHITECTURE_REFERENCE.md). This section identifies which layers and components are implemented in Phase 1, and what is deferred.

### Phase 1 Scope within the Architecture

The Master Architecture defines 6 layers and 4 core components. Phase 1 implements a subset:

| Layer / Component | Phase 1 Status | Notes |
|-------------------|---------------|-------|
| Layer 1: Research Environment | **Partial** | Jupyter notebooks + experiment runner. No analysis tools yet. |
| Layer 2: Orchestration (Step Functions) | **Included** | Step Functions state machine orchestrates task distribution, parallel Lambda execution, and result aggregation |
| Layer 3: Agent Processing | **Core focus** | Context Manager, Context Router, LangChain Executor implemented. Agent Coordinator simplified (no cognitive diversity). |
| Layer 4: Data Processing (Cylon) | **Complete** | SIMD operations, Cylon Communicator (Redis + Direct channels), Arrow data format. See [Cylon Integration](#cylon-integration). |
| Layer 5: Storage | **Partial** | DynamoDB + Redis. No S3 checkpointing in Phase 1. |
| Layer 6: LLM Provider | **Partial** | AWS Bedrock only (no Azure OpenAI, no OpenAI) |

### Phase 1 Data Flow

```
Task Input
    │
    ▼
┌──────────────────┐
│  Embed Task       │ ← Amazon Titan Text Embeddings V2
│  (Bedrock)        │   (configurable: 256/512/1024 dims)
└────────┬─────────┘
         │ embedding vector
         ▼
┌──────────────────┐
│  Similarity       │ ← Cylon SIMD: cosine_similarity_f32
│  Search           │   A1: per-call C++ | A2: Cython batch | B: WASM SIMD128
└────────┬─────────┘
         │ similarity score
         ▼
┌──────────────────┐
│  Context Router   │ ← Threshold comparison (configurable: 0.70–0.95)
│  Decision         │
└───┬──────────┬───┘
    │          │
    ▼          ▼
  REUSE       NEW CALL
    │          │
    │          ▼
    │   ┌──────────────┐
    │   │  Bedrock LLM  │ ← Claude (via Bedrock)
    │   │  Invocation    │
    │   └──────┬───────┘
    │          │ response + cost
    │          ▼
    │   ┌──────────────┐
    │   │  Store Context │ ← DynamoDB (persistent) + Redis (cache)
    │   │  + Embedding   │
    │   └──────┬───────┘
    │          │
    ▼          ▼
┌──────────────────┐
│  Return Response  │ ← CostTracker records: reuse vs. new call cost
│  + Cost Metrics   │
└──────────────────┘
```

### Execution Path Architecture

All three paths implement the same data flow but use different SIMD backends:

```
                         ┌─────────────────────────────────┐
                         │        Shared Components         │
                         │  DynamoDB │ Redis │ Bedrock │     │
                         │  CostTracker │ Experiment Runner  │
                         │  LangChain (Bedrock integration)  │
                         └──┬──────────┬──────────┬────────┘
                            │          │          │
              ┌─────────────┘          │          └─────────────┐
              ▼                        ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ Path A1: Python +    │ │ Path A2: Python +    │ │ Path B: Node.js +   │
│ pycylon (per-call)   │ │ Cython (batch)       │ │ WASM               │
│                      │ │                      │ │                      │
│ Runtime: Python 3.10 │ │ Runtime: Python 3.10 │ │ Runtime: Node.js 18 │
│ Docker: cylon/docker/│ │ Docker: cylon/docker/│ │ Handler:             │
│   aws/lambda/        │ │   aws/lambda/        │ │   wasm_handler.mjs  │
│ SIMD: Native C++     │ │ SIMD: Native C++     │ │ SIMD: WASM SIMD128  │
│   AVX2/SSE           │ │   AVX2/SSE           │ │ Bindings:            │
│ Search: Python loop, │ │ Search: Single C++   │ │   wasm-bindgen      │
│   1 call per pair    │ │   call, batch search │ │ Size: ~50MB          │
│ Size: ~800MB         │ │ Size: ~800MB         │ │   (WASM only)       │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

**Architecture Diagram Cross-Reference** (see `docs/Serverless HPC Framework for Distributed AI.png`):

| Diagram Component | Phase 1 Implementation | Path |
|-------------------|----------------------|------|
| Step Function Orchestrator | `target/aws/scripts/step_functions/workflow.asl.json` | All |
| Workflow State Manager | Integrated into Step Functions state machine (Map state + result aggregation) | All |
| Agent Coordinator Lambda | `target/shared/scripts/coordinator/agent_coordinator.py` — triggered by Step Functions | A1/A2 |
| Context Manager Lambda | `target/shared/scripts/context/manager.py`, `target/aws/scripts/lambda/node/context_handler.mjs` | All |
| langChain Executor Lambda | `target/shared/scripts/chain/executor.py`, `target/aws/scripts/lambda/node/context_handler.mjs` | All |
| Context Aware Request Router Lambda | `target/shared/scripts/context/router.py`, `target/aws/scripts/lambda/node/context_handler.mjs` | All |
| Cache Manager Lambda | Integrated into Context Manager (Redis ops) | All |
| Context Similarity Lambda | Integrated into Context Router (SIMD search) | All |
| Amazon Titan | Embedding Service (`amazon.titan-embed-text-v2:0`) | All |
| Claude | LLM invocation via langChain Executor | All |
| Llama | **Deferred** — single LLM model for Phase 1 | — |
| AWS EventBridge | **Deferred** — no event-driven triggers | — |
| Performance Processor Lambda | **Deferred** — metrics captured inline | — |
| Cost Processor Lambda | `target/shared/scripts/cost/bedrock_pricing.py` (inline, not separate Lambda) | A1/A2 |
| Redis ElastiCache | Shared storage layer | All |
| Amazon S3 | **Deferred** — no checkpointing in Phase 1 | — |
| Rendezvous Server | Existing Cylon Communicator infrastructure | All |

### Cylon Integration

Cylon is not just a SIMD library in this system — it is the **foundation layer** providing distributed computing primitives, data representation, and inter-Lambda communication. The context reuse system is built on top of Cylon's existing infrastructure.

**Cylon provides 5 capabilities to this project:**

#### 1. SIMD-Accelerated Similarity Search

The Context Similarity Engine uses Cylon's SIMD operations for cosine similarity:

- **Path A1/A2 (Python):** pycylon C++ bindings → native AVX2/SSE SIMD
- **Path B (Node.js):** cylon-wasm Rust module → WASM SIMD128
- **Source:** `cylon/rust/cylon-wasm/src/simd.rs` (Rust/WASM), native C++ via pycylon

All three execution paths use Cylon's `cosine_similarity_f32` — the difference is the call pattern (per-pair, batch, or WASM).

#### 2. Cylon Communicator (Inter-Lambda Communication)

Cylon implements a custom communicator abstraction (`cylon/cpp/src/cylon/net/communicator.hpp`) that supports multiple backends: MPI, UCX, UCC, and a serverless-oriented backend inspired by FMI (Fault-tolerant Message Interface). The FMI-inspired backend provides three channel types for Lambda-to-Lambda communication:

| Channel | Implementation | Use Case |
|---------|---------------|----------|
| **Direct** (TCPunch) | NAT hole-punching via Rendezvous Server → direct TCP peer connections | Primary for Lambda — lowest latency (<10ms) |
| **Redis** | Redis key-value store as message bus | Fallback when Direct is unavailable; also used for OOB coordination |
| **S3** | S3 object storage for message passing | Large data transfers, checkpointing |

**Phase 1 usage:** The Agent Coordinator uses the Cylon Communicator with the Redis channel to distribute tasks and collect results across parallel Lambda workers. The Direct channel (TCPunch) is available but Redis is simpler for the PoC. The channel type is configurable via `FMIConfig`:

```python
from pycylon.net import FMIConfig, CommType

config = FMIConfig(
    rank=rank, world_size=world_size,
    host=rendezvous_host, port=rendezvous_port,
    maxtimeout=30,
    channel_type="redis",       # or "direct", "s3"
    redis_host=redis_host,
    redis_port=redis_port
)
```

#### 3. Rendezvous Server

The Rendezvous Server (`cylon/target/aws/scripts/rendezvous/src/server.ts`) coordinates Lambda peer discovery:

1. Each Lambda worker registers with the Rendezvous Server (local IP:port)
2. Server responds with the worker's NAT-mapped IP:port
3. Workers use exchanged addresses for Direct channel (TCPunch) or fall back to Redis channel

The Rendezvous Server is already deployed at `cylon-rendezvous.aws-cylondata.com:10000` (see `cylon/docker/aws/lambda/fmi/fmi.json`).

#### 4. Apache Arrow Data Format

Cylon is built on Apache Arrow for columnar data representation. In this project, Arrow format is used for:

- **Embedding storage in Redis:** Float32 Arrow arrays for zero-copy transfer between Context Manager and Context Router
- **Batch similarity search (Path A2):** Contiguous Arrow-compatible float32 buffers passed to Cython without serialization overhead
- **Inter-Lambda data transfer:** Arrow IPC format via Cylon Communicator channels

#### 5. Cost Tracking Framework

The existing `CostTracker` in `cylon/target/shared/scripts/scaling/costlib/` provides Lambda duration, S3, and infrastructure cost tracking. Phase 1 extends this with `BedrockCostTracker` for LLM and embedding costs, inheriting the pricing resolution pattern (config → AWS API → defaults).

### DynamoDB Schema (Phase 1)

**Table: `context-store`**

| Attribute | Type | Role |
|-----------|------|------|
| `context_id` | String | Partition Key (UUID) |
| `workflow_id` | String | Sort Key |
| `task_description` | String | Original task text |
| `embedding` | Binary | Float32 array (256/512/1024 dims) |
| `embedding_dim` | Number | Dimension of the embedding (256, 512, or 1024) |
| `response` | String | LLM response text |
| `model_id` | String | Bedrock model used |
| `cost_input_tokens` | Number | Input token count |
| `cost_output_tokens` | Number | Output token count |
| `cost_usd` | Number | Total cost in USD |
| `created_at` | String | ISO timestamp |
| `reuse_count` | Number | Times this context was reused |

**GSI**: `workflow_id-created_at-index` for querying all contexts in a workflow.

### Redis Cache Layout (Phase 1)

```
# Hot embedding cache (for fast similarity search)
embedding:{context_id} → binary float32 array     TTL: 1h

# Recent context responses (for fast reuse)
context:{context_id}   → JSON {response, metadata}  TTL: 1h

# Workflow-level index
workflow:{workflow_id}  → SET of context_ids          TTL: 2h
```

### Cost Tracking Integration

The existing `CostTracker` in `cylon/target/shared/scripts/scaling/costlib/` is extended with Bedrock-specific metrics:

| Metric | Source | Description |
|--------|--------|-------------|
| `bedrock_llm_cost` | Bedrock response metadata | Per-call LLM cost (input + output tokens) |
| `bedrock_embedding_cost` | Bedrock response metadata | Per-call embedding cost |
| `lambda_compute_cost` | Lambda duration | SIMD search + overhead |
| `cache_hit_savings` | Context Router | Cost avoided by reuse |
| `total_workflow_cost` | Aggregated | Sum of all costs for a workflow run |
| `baseline_cost` | Calculated | What the workflow would cost without reuse |
| `savings_pct` | Calculated | `1 - (total_workflow_cost / baseline_cost)` |

---

## Component Breakdown

### Project Structure

Follows the Cylon `target/` directory pattern — shared libraries in `target/shared/scripts/`, Lambda-specific handlers in `target/aws/scripts/lambda/`, experiments in `target/experiments/`.

```
cylon-armada/
├── docs/
│   └── PHASE1_IMPLEMENTATION_PLAN.md           ← this document
├── target/
│   ├── shared/scripts/                          ← shared libraries (reusable across targets)
│   │   ├── context/
│   │   │   ├── manager.py                      ← Context Manager (DynamoDB + Redis)
│   │   │   ├── router.py                       ← Context Router (similarity threshold logic)
│   │   │   └── embedding.py                    ← Embedding Service (Bedrock Titan V2)
│   │   ├── chain/
│   │   │   └── executor.py                     ← LangChain Executor (Bedrock LLM chains)
│   │   ├── simd/                                ← compiled .so placed here after build
│   │   ├── run_action.py                        ← Action dispatcher (downloaded from S3, executed by Lambda)
│   │   ├── coordinator/
│   │   │   └── agent_coordinator.py            ← Agent Coordinator (triggered by Step Functions)
│   │   └── cost/
│   │       ├── bedrock_pricing.py              ← Bedrock cost extension for CostTracker
│   │       └── experiment_tracker.py           ← Per-experiment cost aggregation
│   ├── aws/scripts/
│   │   ├── lambda/
│   │   │   ├── python/
│   │   │   │   └── handler.py                  ← S3 script runner (follows Cylon lambda_entry1.py pattern)
│   │   │   └── node/
│   │   │       ├── context_handler.mjs          ← Path B: WASM SIMD128 context reuse handler
│   │   │       ├── inference.mjs                ← ONNX inference (full model + partitioned stages)
│   │   │       ├── task_generator.mjs           ← Generate LLM tasks from inference results
│   │   │       └── package.json                 ← Node.js dependencies (onnxruntime-node, AWS SDK)
│   │   ├── step_functions/
│   │   │   ├── workflow.asl.json               ← Step Functions ASL — Python (S3 script runner)
│   │   │   ├── workflow_nodejs.asl.json        ← Step Functions ASL — Node.js (direct dispatch)
│   │   │   └── workflow_model_parallel.asl.json ← Step Functions ASL — model parallelism
│   │   └── terraform/                           ← Terraform modules (Phase 2 — manual deploy for Phase 1)
│   └── experiments/
│       ├── runner.py                           ← Experiment Runner (general + cosmic-ai)
│       ├── cosmic_ai/                          ← Astronomical inference experiments
│       │   ├── __init__.py
│       │   ├── inference.py                    ← AstroMAE inference module (PyTorch, adapted)
│       │   ├── task_generator.py               ← Generate LLM tasks from SDSS data
│       │   ├── export_onnx.py                  ← ONNX export + model parallelism partitioning
│       │   └── blocks/                         ← Model architecture (from AI-for-Astronomy)
│       │       ├── model_vit_inception.py      ← ViT + Inception model
│       │       ├── photoz.py                   ← Inception block + magnitude model
│       │       └── normal_cell.py              ← Transformer cell with PCM
│       ├── scenarios/                          ← Scenario configuration files (JSON/YAML)
│       ├── analysis/                           ← Jupyter notebooks for visualization
│       └── results/                            ← Experiment output data
├── config/
│   ├── dynamo_tables.json                      ← DynamoDB table definitions
│   └── experiment_defaults.yaml                ← Default experiment parameters
├── python/
│   └── simd/
│       ├── batch_search.pyx                    ← Cython source: batch SIMD search (Path A2)
│       └── setup.py                            ← Cython build config (requires CYLON_PREFIX)
├── docker/
│   ├── Dockerfile.python                       ← Path A1/A2 Docker image
│   └── Dockerfile.nodejs                       ← Path B Docker image
├── tests/
│   ├── test_context_manager.py
│   ├── test_context_router.py
│   ├── test_embedding.py
│   ├── test_chain_executor.py
│   ├── test_batch_search.py
│   └── test_cost_tracker.py
```

### Component 1: Context Manager

**Responsibility:** Store and retrieve contexts with embeddings from DynamoDB and Redis.

**Implementations:** `target/shared/scripts/context/manager.py` and `target/aws/scripts/lambda/node/context_handler.mjs`

**Interface:**

```python
class ContextManager:
    def __init__(self, dynamo_table: str, redis_host: str, redis_port: int):
        """Initialize DynamoDB and Redis clients."""

    def store_context(self, context_id: str, workflow_id: str,
                      task_description: str, embedding: np.ndarray,
                      response: str, cost_metadata: dict) -> None:
        """Store context in DynamoDB and cache in Redis."""

    def get_context(self, context_id: str) -> dict:
        """Retrieve context — Redis first, DynamoDB fallback."""

    def get_all_embeddings(self, workflow_id: str = None) -> list[tuple[str, np.ndarray]]:
        """Return all (context_id, embedding) pairs for similarity search.
        Optionally filter by workflow_id."""

    def increment_reuse_count(self, context_id: str) -> None:
        """Increment reuse_count in DynamoDB."""

    def get_workflow_contexts(self, workflow_id: str) -> list[dict]:
        """Query GSI for all contexts in a workflow."""
```

**Dependencies:**
- Python: `boto3` (DynamoDB), `redis-py` (Redis), `numpy` (embedding serialization)
- Node.js: `@aws-sdk/client-dynamodb`, `ioredis`

**Notes:**
- Embeddings stored as Binary in DynamoDB (float32 → bytes via `np.ndarray.tobytes()` / `Float32Array.buffer`)
- Redis cache uses TTL-based eviction; DynamoDB is the source of truth
- Both paths read/write the same DynamoDB table and Redis keys

---

### Component 2: Context Router

**Responsibility:** Find similar contexts using SIMD-accelerated cosine similarity and make reuse decisions.

**Implementations:** `target/shared/scripts/context/router.py` and `target/aws/scripts/lambda/node/context_handler.mjs`

**Interface:**

```python
class ContextRouter:
    def __init__(self, context_manager: ContextManager,
                 threshold: float = 0.85, top_k: int = 5):
        """Initialize with similarity threshold and top-k results."""

    def find_similar(self, query_embedding: np.ndarray,
                     workflow_id: str = None) -> list[dict]:
        """Return top-k contexts above threshold.
        Each result: {context_id, similarity, response, metadata}."""

    def should_reuse(self, query_embedding: np.ndarray,
                     workflow_id: str = None) -> tuple[bool, dict]:
        """Decision function. Returns (reuse: bool, best_match_or_none).
        Records decision metrics for cost tracking."""

    def route(self, task_description: str, query_embedding: np.ndarray,
              workflow_id: str, chain_executor, cost_tracker) -> dict:
        """Full routing pipeline:
        1. Check similarity → reuse or new call
        2. If new call: delegate to langChain Executor, store context
        3. Record costs either way
        Returns {response, source: 'cache'|'llm', cost, similarity}."""
```

**SIMD Integration:**

| Path | Import | Function | Boundary Crossings |
|------|--------|----------|--------------------|
| A1 (Python per-call) | `import pycylon` | `pycylon.simd.cosine_similarity_f32(a, b)` | N (one per embedding) |
| A2 (Cython batch) | `from simd.batch_search import batch_cosine_search` | `batch_cosine_search(query, all_embeddings, threshold)` | 1 (entire search) |
| B (Node.js WASM) | `import { simd_cosine_similarity_f32 } from 'cylon-wasm'` | `simd_cosine_similarity_f32(new Float32Array(a), new Float32Array(b))` | N (one per embedding) |
| Fallback (Python) | `import numpy` | `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` | N/A (pure Python) |

---

### Component 3: Embedding Service

**Responsibility:** Generate embeddings via Amazon Titan Text Embeddings V2 with configurable dimensions.

**Implementations:** `target/shared/scripts/context/embedding.py` and `target/aws/scripts/lambda/node/context_handler.mjs`

**Interface:**

```python
class EmbeddingService:
    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0",
                 dimensions: int = 1024, region: str = "us-east-1"):
        """Initialize Bedrock runtime client."""

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        Returns float32 numpy array of configured dimension."""

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.
        Uses sequential Bedrock calls (no native batch API)."""

    def get_cost(self, token_count: int) -> float:
        """Calculate embedding cost: token_count * $0.00002 / 1000."""
```

**Bedrock Invocation:**

```python
response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({
        "inputText": text,
        "dimensions": self.dimensions,  # 256, 512, or 1024
        "normalize": True
    })
)
embedding = json.loads(response["body"].read())["embedding"]
```

---

### Component 4: Agent Coordinator + Step Functions Orchestrator

**Responsibility:** Orchestrate multi-task workflows using AWS Step Functions for task distribution, parallel Lambda execution, and result aggregation. The Step Functions state machine handles the workflow lifecycle; individual Lambda workers handle context routing and LLM execution.

**Implementations:**
- `target/aws/scripts/step_functions/workflow.asl.json` — Step Functions state machine (ASL)
- `target/shared/scripts/coordinator/agent_coordinator.py` — Coordinator Lambda (prepares tasks, aggregates results)

**Step Functions Workflow:**

```
                    ┌─────────────────────┐
                    │  Start Workflow      │
                    │  (input: tasks,      │
                    │   config, workflow_id)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Prepare Tasks       │ ← Coordinator Lambda
                    │  (embed all tasks,   │   generates payloads
                    │   partition by rank)  │   with embeddings
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Map State           │ ← Step Functions parallel
                    │  (parallel workers)  │   execution (max concurrency
                    │                      │   configurable)
                    │  ┌────────────────┐  │
                    │  │ Worker Lambda  │  │ ← Context Router:
                    │  │ (per task:     │  │   embed → search → reuse
                    │  │  route →       │  │   or LLM call via
                    │  │  reuse/LLM)    │  │   ChainExecutor
                    │  └────────────────┘  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Aggregate Results   │ ← Coordinator Lambda
                    │  (collect costs,     │   merges per-task results,
                    │   compute savings)   │   computes cost summary
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  End Workflow        │
                    │  (output: results,   │
                    │   cost_summary,      │
                    │   reuse_stats)       │
                    └─────────────────────┘
```

**Coordinator Lambda Interface:**

```python
class AgentCoordinator:
    def __init__(self, config: BedrockConfig = None):
        """Initialize with Bedrock config (resolved from env/payload/file)."""

    def prepare_tasks(self, event: dict) -> dict:
        """Step 1 — called by Step Functions 'Prepare Tasks' state.
        - Receives workflow_id, tasks list, and config from Step Functions input
        - Embeds all tasks via EmbeddingService
        - Returns list of task payloads with embeddings for the Map state."""

    def aggregate_results(self, event: dict) -> dict:
        """Step 3 — called by Step Functions 'Aggregate Results' state.
        - Receives array of per-task results from the Map state
        - Aggregates cost metrics via BedrockCostTracker
        - Returns {results, cost_summary, reuse_stats}."""
```

**Worker Lambda** (invoked by Step Functions Map state):

Each worker receives a single task payload and runs the Context Router pipeline:
1. Deserialize the pre-computed embedding
2. Run similarity search (SIMD backend depends on path: A1/A2/B)
3. Reuse cached response or invoke ChainExecutor for new LLM call
4. Return `{response, source, cost_usd, similarity, latency}` to Step Functions

**Cylon Communicator Integration:**

Step Functions handles the orchestration (task distribution, parallel execution, result collection), replacing the need for direct Cylon Communicator coordination in the workflow layer. However, the Cylon Communicator (Redis channel) remains available for:

1. **Context sharing between parallel workers:** Workers can share discovered contexts in real-time via Cylon all-gather, improving reuse rates for later tasks in the same Map execution
2. **Benchmark comparison:** Direct Lambda invocation via Cylon Communicator can be benchmarked against Step Functions orchestration overhead
3. **Channel configuration:** Workers receive `rank`/`world_size` in their Step Functions payload for Cylon Communicator initialization

```python
# Worker Lambda initializes Cylon Communicator from Step Functions payload
config = FMIConfig(
    rank=event["rank"],
    world_size=event["world_size"],
    channel_type="redis",
    redis_host=event["redis_host"],
    redis_port=event["redis_port"]
)
```

---

### Component 5: Cost Tracker Extension

**Responsibility:** Extend the existing `CostTracker` framework with Bedrock LLM and embedding cost tracking.

**Implementation:** `target/shared/scripts/cost/bedrock_pricing.py`

**Pricing Resolution:**

Follows the same precedence pattern as `AWSPricing` in `costlib/aws_pricing.py`:

```
Config file → AWS Pricing API → Static defaults
```

```python
@dataclass
class BedrockPricing:
    """Bedrock model pricing (per 1K tokens)."""
    claude_input_per_1k: float = 0.003     # Claude 3 Haiku input
    claude_output_per_1k: float = 0.015    # Claude 3 Haiku output
    titan_embed_per_1k: float = 0.00002    # Titan Text Embeddings V2

    @classmethod
    def from_aws_api(cls, region: str = "us-east-1") -> "BedrockPricing":
        """Pull current Bedrock prices from AWS Pricing API (boto3 pricing client).
        Falls back to static defaults on API failure or missing data."""

    @classmethod
    def from_config(cls, config_path: str) -> "BedrockPricing":
        """Load pricing overrides from YAML/JSON config file."""

    @classmethod
    def resolve(cls, region: str = "us-east-1",
                config_path: str = None) -> "BedrockPricing":
        """Resolve pricing with precedence:
        1. Config file (if provided and exists)
        2. AWS Pricing API (live prices)
        3. Static defaults (hardcoded fallback)
        Logs which source was used for auditability."""
```

**Cost Tracker Interface:**

```python
class BedrockCostTracker:
    def __init__(self, pricing: BedrockPricing = None, region: str = "us-east-1"):
        """Initialize with resolved pricing. If pricing is None,
        calls BedrockPricing.resolve() to auto-detect."""

    def record_llm_call(self, input_tokens: int, output_tokens: int,
                        model_id: str) -> float:
        """Record an LLM call and return its cost in USD."""

    def record_embedding_call(self, token_count: int) -> float:
        """Record an embedding call and return its cost in USD."""

    def record_cache_hit(self, avoided_input_tokens: int,
                         avoided_output_tokens: int) -> float:
        """Record a cache hit and return the avoided cost in USD."""

    def get_summary(self) -> dict:
        """Return {total_cost, baseline_cost, savings_pct,
                   llm_calls, embedding_calls, cache_hits,
                   pricing_source: 'aws_api'|'config'|'defaults',
                   cost_breakdown: {llm, embedding, lambda}}."""
```

**Integration:** Wraps the existing `CostTracker` from `cylon/target/shared/scripts/scaling/costlib/aws_pricing.py`, adding Bedrock-specific methods alongside the existing Lambda/S3/StepFn tracking.

---

### Component 6: Experiment Runner

**Responsibility:** Execute experiment scenarios with parameter sweeps and collect structured results.

**Implementation:** `target/experiments/runner.py`

**Interface:**

```python
class ExperimentRunner:
    def __init__(self, coordinator: AgentCoordinator,
                 embedding_service: EmbeddingService):
        """Initialize with coordinator and embedding service."""

    def run_scenario(self, scenario_config: dict) -> dict:
        """Execute a single scenario:
        - Run baseline (no reuse)
        - Run with context reuse at configured threshold/dimensions
        - Compare costs, quality, latency
        Returns structured results."""

    def run_sweep(self, scenario_config: dict,
                  thresholds: list[float],
                  dimensions: list[int]) -> list[dict]:
        """Run a parameter sweep: threshold × dimension grid.
        Returns results for each configuration."""

    def export_results(self, results: list[dict], output_dir: str) -> None:
        """Export results as JSON + CSV for Jupyter analysis."""
```

**Scenario Configuration (YAML):**

```yaml
scenario:
  name: "code_review_32_tasks"
  description: "32 code review tasks with varying similarity"
  tasks_file: "scenarios/code_review_tasks.json"

sweep:
  thresholds: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
  dimensions: [256, 512, 1024]

bedrock:
  llm_model: "anthropic.claude-3-haiku-20240307-v1:0"
  embedding_model: "amazon.titan-embed-text-v2:0"
  region: "us-east-1"

execution:
  path: "python"  # or "nodejs"
  use_simd: true
  redis_host: "localhost"
  dynamo_table: "context-store"
```

---

### Component 7: WASM Context Handler

**Responsibility:** Node.js Lambda handler for Path B, extending the existing `wasm_handler.mjs` with context reuse operations.

**Implementation:** `target/aws/scripts/lambda/node/context_handler.mjs`

**Operations:**

```javascript
const OPERATIONS = {
    'embed_and_search': async (params) => {
        // 1. Embed task via Bedrock (BedrockRuntimeClient)
        // 2. SIMD similarity search via cylon-wasm cosine_similarity_f32
        // 3. Return best match or null
    },
    'store_context': async (params) => {
        // Store context + embedding in DynamoDB
    },
    'route_task': async (params) => {
        // Full routing pipeline (embed → search → reuse or LLM call)
    }
};
```

**Based on:** `cylon/target/aws/scripts/lambda/wasm_handler.mjs` — same lazy-load WASM pattern, same Lambda handler signature, new operations for context reuse.

---

### Component 8: LangChain Executor

**Responsibility:** Execute LLM chains via AWS Bedrock using LangChain. This is the component that constructs prompts, invokes the LLM, and parses responses. The Context Router delegates to this when a new LLM call is needed (no cache hit).

**Implementations:** `target/shared/scripts/chain/executor.py` and `target/aws/scripts/lambda/node/context_handler.mjs`

**Interface:**

```python
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChainExecutor:
    def __init__(self, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
                 region: str = "us-east-1"):
        """Initialize LangChain ChatBedrock client."""
        self.llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"temperature": 0.0}
        )

    def execute(self, task_description: str,
                system_prompt: str = None) -> dict:
        """Execute a single LLM chain for the given task.
        Returns {response, input_tokens, output_tokens, latency_ms}."""

    def execute_with_context(self, task_description: str,
                             similar_context: dict) -> dict:
        """Execute an LLM chain augmented with a similar (but below-threshold)
        context for reference. Used when similarity is close but not reusable.
        Returns {response, input_tokens, output_tokens, latency_ms}."""

    def build_chain(self, prompt_template: ChatPromptTemplate) -> RunnableSequence:
        """Build a reusable LCEL chain: prompt | llm | parser.
        Allows custom prompt templates per scenario."""
```

**Node.js (LangChain.js):**

```javascript
import { ChatBedrock } from "@langchain/aws";
import { ChatPromptTemplate } from "@langchain/core/prompts";

class ChainExecutor {
    constructor(modelId = "anthropic.claude-3-haiku-20240307-v1:0", region = "us-east-1") {
        this.llm = new ChatBedrock({ model: modelId, region });
    }

    async execute(taskDescription, systemPrompt = null) { /* ... */ }
    async executeWithContext(taskDescription, similarContext) { /* ... */ }
}
```

**Dependencies:**
- Python: `langchain-aws`, `langchain-core`
- Node.js: `@langchain/aws`, `@langchain/core`

**Architecture Diagram Mapping:** This is the **langChain Executor Lambda** in the architecture diagram. It sits between the Context Aware Request Router and the Bedrock models (Titan, Claude, Llama).

---

### Component 9: Cython Batch Similarity Search

**Responsibility:** Push the entire similarity search loop into C/C++ via a Cython extension, eliminating per-embedding Python→C++ boundary crossing overhead. This is the Path A2 optimization.

**Implementation:** `python/simd/batch_search.pyx` (source), compiled `.so` installed to `target/shared/scripts/simd/`

**Interface:**

```cython
# batch_search.pyx
import numpy as np
cimport numpy as np

def batch_cosine_search(
    np.ndarray[np.float32_t, ndim=1] query,
    np.ndarray[np.float32_t, ndim=2] embeddings,
    float threshold,
    int top_k = 5
) -> list:
    """Search all embeddings against query in a single C-level loop.

    Args:
        query: float32 array of shape (dim,)
        embeddings: float32 array of shape (N, dim) — contiguous buffer
        threshold: minimum cosine similarity to include
        top_k: max results to return

    Returns:
        List of (index, similarity) tuples, sorted descending.
        Single Python→C++ boundary crossing for the entire search.
    """
```

**Build:**

```python
# python/simd/setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("batch_search.pyx"),
    include_dirs=[np.get_include()]
)
```

**Integration with Context Router:**

```python
# In router.py — Path A2 mode
from simd.batch_search import batch_cosine_search

# Collect all embeddings into a contiguous numpy array
all_embeddings = np.vstack([emb for _, emb in context_manager.get_all_embeddings()])
results = batch_cosine_search(query_embedding, all_embeddings, self.threshold, self.top_k)
```

**Performance Advantage:** For N stored embeddings, Path A1 makes N Python→C++ calls. Path A2 makes 1 call and iterates entirely in C, avoiding N-1 boundary crossings plus Python loop overhead. The difference scales with N and is a key benchmark variable.

---

## Benchmark Strategy

### Experimental Design

Every experiment runs two modes to produce a direct cost comparison:

1. **Baseline** — All tasks execute fresh LLM calls via the langChain Executor. No similarity search, no context reuse. This establishes the "without optimization" cost.
2. **Context Reuse** — Tasks are routed through the Context Router. Similar contexts are reused; only novel tasks trigger LLM calls. This measures the optimization.

The difference between baseline cost and reuse cost is the measured savings.

### Parameter Sweep Grid

Each scenario is executed across a 3D parameter space:

| Dimension | Values | Count |
|-----------|--------|-------|
| Similarity threshold | 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 | 6 |
| Embedding dimension | 256, 512, 1024 | 3 |
| Execution path | A1 (Python per-call), A2 (Cython batch), B (Node.js WASM) | 3 |

**Total configurations per scenario:** 6 × 3 × 3 = **54 configurations**

Each configuration runs the full task set (baseline + reuse), producing paired cost/quality/latency data.

### Metrics Collected

**Per-task metrics:**

| Metric | Unit | Source |
|--------|------|--------|
| `embedding_latency` | ms | Time to generate embedding via Bedrock Titan V2 |
| `search_latency` | ms | Time for SIMD similarity search across all stored contexts |
| `llm_latency` | ms | Time for Bedrock LLM call (only if new call) |
| `total_latency` | ms | End-to-end task processing time |
| `similarity_score` | float | Best match similarity (0.0–1.0) |
| `decision` | enum | `reuse` or `new_call` |
| `input_tokens` | int | Tokens sent to LLM (0 if reused) |
| `output_tokens` | int | Tokens received from LLM (0 if reused) |
| `task_cost_usd` | float | Total cost for this task |

**Per-workflow metrics:**

| Metric | Unit | Description |
|--------|------|-------------|
| `total_cost` | USD | Sum of all task costs |
| `baseline_cost` | USD | Cost without context reuse |
| `savings_pct` | % | `1 - (total_cost / baseline_cost)` |
| `reuse_rate` | % | Tasks reused / total tasks |
| `avg_similarity` | float | Mean similarity score across all tasks |
| `quality_score` | float | BLEU/ROUGE of reused responses vs. fresh LLM responses |

**Per-path metrics (SIMD comparison):**

| Metric | Unit | Description |
|--------|------|-------------|
| `search_latency_p50` | ms | Median similarity search time |
| `search_latency_p99` | ms | 99th percentile search time |
| `search_throughput` | ops/sec | Similarity comparisons per second |
| `cold_start_latency` | ms | First invocation overhead (Lambda cold start) |
| `warm_latency` | ms | Subsequent invocations (warm Lambda) |

### Path Comparison: A1 vs. A2 vs. B

The three execution paths are benchmarked on the same task sets to isolate SIMD implementation performance:

| Comparison | What it measures |
|------------|-----------------|
| A1 vs. A2 | Python loop overhead — how much does eliminating per-embedding boundary crossings improve search latency? Scales with context store size. |
| A1 vs. B | Native C++ SIMD vs. WASM SIMD128 — does native compilation outperform WASM, and by how much? |
| A2 vs. B | Optimized native batch vs. WASM per-call — best-case Python path vs. Node.js path. |
| All paths | Cold start comparison — ~800MB Python Docker image vs. ~50MB WASM module. |

**Scaling test:** Run similarity search over increasing context store sizes (100, 500, 1000, 5000, 10000 embeddings) at each dimension (256, 512, 1024) to produce latency curves per path.

### Quality Measurement

Context reuse is only useful if reused responses are actually good enough. Quality is measured by comparing reused responses against what a fresh LLM call would have produced:

1. **Baseline run**: Execute all tasks with fresh LLM calls. Store all responses.
2. **Reuse run**: Execute same tasks with context reuse enabled.
3. **For each reused task**: Compare the reused response against the baseline (fresh) response using:
   - **ROUGE-L**: Longest common subsequence overlap
   - **BLEU**: N-gram precision
   - **Semantic similarity**: Embed both responses, compute cosine similarity

**Quality threshold**: A reuse decision is "correct" if the quality score exceeds a configurable minimum (default: 0.80 ROUGE-L). This produces a precision metric: what percentage of reuse decisions maintained acceptable quality?

**Tradeoff curve**: Plotting threshold vs. quality produces the key insight — at what similarity threshold does reuse quality degrade?

### Statistical Rigor

- **Repetitions**: Each configuration runs 3× to capture variance
- **Confidence intervals**: Report 95% CI for all aggregate metrics
- **Paired tests**: Baseline vs. reuse uses paired comparisons (same task set) to control for task difficulty variance
- **Cold start isolation**: First Lambda invocation is recorded separately and excluded from warm latency statistics

### Output Artifacts

Each experiment run produces:

| Artifact | Format | Location |
|----------|--------|----------|
| Raw task-level results | JSON | `python/experiments/results/{scenario}_{timestamp}.json` |
| Aggregated metrics | CSV | `python/experiments/results/{scenario}_{timestamp}_summary.csv` |
| Parameter sweep results | CSV | `python/experiments/results/{scenario}_{timestamp}_sweep.csv` |
| Cost curves | Jupyter notebook | `python/experiments/analysis/cost_analysis.ipynb` |
| Latency distributions | Jupyter notebook | `python/experiments/analysis/latency_analysis.ipynb` |
| Path comparison | Jupyter notebook | `python/experiments/analysis/path_comparison.ipynb` |
| Quality analysis | Jupyter notebook | `python/experiments/analysis/quality_analysis.ipynb` |

---

## Local Development

### Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Path A1/A2, experiments, coordination |
| Node.js | 18+ | Path B |
| Docker | 24+ | DynamoDB Local, Redis, Lambda testing |
| AWS CLI | 2.x | Bedrock access, credential configuration |
| Cython | 3.x | Path A2 batch search compilation |
| Rust + wasm-pack | stable | Building cylon-wasm (if modifying SIMD ops) |

### AWS Credentials

Bedrock calls require valid AWS credentials with `bedrock:InvokeModel` permission. Local development uses the same credentials as Lambda deployment:

```bash
# Option 1: AWS CLI profile (recommended for local dev)
aws configure --profile cylon-armada
export AWS_PROFILE=cylon-armada

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

Required IAM permissions:
- `bedrock:InvokeModel` (Titan Embeddings + Claude)
- `dynamodb:*` (context-store table — or use DynamoDB Local)
- `lambda:InvokeFunction` (only when testing Lambda invocation)

### Local Services (Docker Compose)

DynamoDB Local and Redis run locally to avoid AWS costs during development:

```yaml
# docker-compose.yml
services:
  dynamodb-local:
    image: amazon/dynamodb-local:latest
    ports:
      - "8000:8000"
    command: "-jar DynamoDBLocal.jar -sharedDb"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
docker compose up -d
```

**DynamoDB Local table creation:**

```bash
aws dynamodb create-table \
  --table-name context-store \
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
  --endpoint-url http://localhost:8000
```

### Python Setup (Path A1 / A2)

```bash
cd cylon-armada/python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langchain-aws langchain-core boto3 redis numpy pyyaml
pip install rouge-score nltk  # quality metrics
pip install jupyter matplotlib plotly pandas  # analysis
pip install cython  # Path A2
pip install pytest  # testing

# Build Cython batch search (Path A2)
cd simd
python setup.py build_ext --inplace
cd ..
```

**Verify pycylon is available** (requires Cylon conda environment from `~/cylon`):

```bash
conda activate cylon_dev
python -c "import pycylon; print('pycylon available')"
```

### Node.js Setup (Path B)

```bash
cd cylon-armada/node

npm install
# Dependencies: @langchain/aws @langchain/core
#               @aws-sdk/client-dynamodb @aws-sdk/client-bedrock-runtime
#               ioredis
```

**Verify cylon-wasm is available:**

```bash
node -e "const cylon = require('cylon-wasm'); console.log('cylon-wasm available');"
```

### Running Locally

**Quick smoke test** — verify Bedrock connectivity and end-to-end pipeline:

```bash
cd cylon-armada/python
python -m pytest tests/ -v -k "test_embedding"  # verifies Bedrock Titan V2
python -m pytest tests/ -v -k "test_chain"       # verifies LangChain + Claude
python -m pytest tests/ -v -k "test_context"     # verifies DynamoDB Local + Redis
python -m pytest tests/ -v -k "test_batch"       # verifies Cython build
```

**Run a single scenario locally:**

```bash
python -m experiments.runner \
  --scenario config/scenarios/smoke_test.yaml \
  --local \
  --dynamo-endpoint http://localhost:8000 \
  --redis-host localhost
```

The `--local` flag tells the runner to execute tasks in-process rather than invoking Lambda functions. The same Context Manager, Context Router, and LangChain Executor code runs locally and on Lambda.

### Local vs. Lambda Differences

| Concern | Local | Lambda |
|---------|-------|--------|
| DynamoDB | DynamoDB Local (localhost:8000) | AWS DynamoDB (us-east-1) |
| Redis | Local Docker (localhost:6379) | ElastiCache (VPC) |
| Bedrock | Same (remote AWS API) | Same (remote AWS API) |
| SIMD (A1) | pycylon from conda env | pycylon in Docker image |
| SIMD (A2) | Cython built locally | Cython built in Docker image |
| SIMD (B) | cylon-wasm npm module | cylon-wasm in Lambda layer |
| Execution | In-process (sequential) | Lambda invocations (parallel) |
| Cost tracking | Same BedrockCostTracker | Same + Lambda compute costs |

The code paths are identical — only the endpoint configuration and execution model differ. This is enforced by the `ContextManager`, `ChainExecutor`, and `ContextRouter` accepting endpoint URLs as constructor parameters.

---

## Lambda Deployment

### Docker Image Strategy

Phase 1 requires two Docker images — one for Python (Path A1/A2) and one for Node.js (Path B). Both extend existing Cylon Lambda Docker images.

#### Path A1/A2: Python + pycylon + Cython

Extends `cylon/docker/aws/lambda/Dockerfile` which already includes:
- Ubuntu 22.04, Miniconda, Python 3.10
- Cylon C++ built with Redis+FMI flags (`-DCYLON_USE_REDIS=1 -DCYLON_FMI=1`)
- pycylon bindings, boto3, redis-py, awslambdaric
- hiredis, redis-plus-plus (C++ Redis clients)

**S3 Script Runner Pattern:**

The Python Lambda follows Cylon's `lambda_entry1.py` pattern: instead of baking application code into the Docker image, the handler downloads scripts from S3 at runtime. This decouples code updates from container builds.

```
Step Functions → Lambda handler (handler.py)
                      │
                      ├── 1. Set env vars from event payload
                      ├── 2. Download scripts from S3 bucket
                      ├── 3. Execute run_action.py via subprocess
                      └── 4. Return result JSON to Step Functions
```

The Docker image contains only the Lambda handler, dependencies, and Cylon — the context reuse scripts (`context/`, `chain/`, `cost/`, `simd/`, `coordinator/`) are uploaded to S3 and downloaded at invocation time.

**Dockerfiles:** `docker/Dockerfile.python` and `docker/Dockerfile.nodejs`

**Estimated image sizes:**
- Python (Path A1/A2): ~850MB (base Cylon image ~800MB + LangChain ~50MB)
- Node.js (Path B): ~50-80MB (Node.js runtime + WASM module + LangChain.js)

### Lambda Function Configuration

#### Function Definitions

| Function | Runtime | Memory | Timeout | Description |
|----------|---------|--------|---------|-------------|
| `cylon-armada-worker` | Python 3.10 (Docker) | 1024 MB | 300s | Path A1/A2 — S3 script runner, all actions (prepare/route/aggregate) |
| `cylon-armada-worker-node` | Node.js 18 (Docker) | 512 MB | 300s | Path B — WASM SIMD context routing |

**Memory rationale:**
- Python worker at 1024 MB: pycylon + embedding arrays + LangChain overhead
- Node.js worker at 512 MB: WASM module is lightweight, smaller embedding overhead

**Note:** With the S3 script runner pattern, a single Python Lambda function handles all three actions (prepare_tasks, route_task, aggregate_results). The `ACTION` field in the Step Functions payload determines which function to execute.

#### Step Functions State Machine

| Resource | Type | Description |
|----------|------|-------------|
| `context-reuse-workflow` | Express | Workflow state machine — uses Map state for parallel task execution |

**Express vs. Standard:** Express workflow is chosen because:
- Workflows complete in <5 minutes (within Express limit)
- Lower cost ($0.000001 per state transition vs. $0.025 per transition for Standard)
- Supports synchronous invocation from experiment runner

#### IAM Role

**Lambda execution role** — shared by all Lambda functions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockAccess",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
            ]
        },
        {
            "Sid": "DynamoDBAccess",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query",
                "dynamodb:UpdateItem", "dynamodb:Scan", "dynamodb:DeleteItem"
            ],
            "Resource": [
                "arn:aws:dynamodb:us-east-1:*:table/context-store",
                "arn:aws:dynamodb:us-east-1:*:table/context-store/index/*"
            ]
        }
    ]
}
```

**Step Functions execution role** — allows the state machine to invoke Lambda functions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "InvokeLambda",
            "Effect": "Allow",
            "Action": ["lambda:InvokeFunction"],
            "Resource": [
                "arn:aws:lambda:us-east-1:*:function:context-reuse-*",
                "arn:aws:lambda:us-east-1:*:function:context-coordinator"
            ]
        }
    ]
}
```

**Experiment runner IAM** — for local/ECS experiment execution (starts Step Functions workflows):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "StepFunctionsAccess",
            "Effect": "Allow",
            "Action": [
                "states:StartExecution",
                "states:StartSyncExecution",
                "states:DescribeExecution",
                "states:ListExecutions"
            ],
            "Resource": ["arn:aws:states:us-east-1:*:stateMachine:context-reuse-workflow"]
        }
    ]
}
```

#### VPC Configuration

The Python and Node.js worker functions require VPC access for ElastiCache Redis:

- **Subnets:** Same private subnets as existing Cylon Lambda functions
- **Security Group:** Allow outbound to Redis port 6379 + Bedrock endpoints (HTTPS 443)
- **NAT Gateway:** Required for Bedrock API access from private subnets

The coordinator function does **not** need VPC access (no Redis communication — it only invokes other Lambdas).

### ECR Repository Setup

```bash
# Create repositories
aws ecr create-repository --repository-name cylon-armada/context-reuse-python
aws ecr create-repository --repository-name cylon-armada/context-reuse-nodejs

# Build and push Path A
cd cylon-armada
docker build -f docker/Dockerfile.python -t context-reuse-python .
docker tag context-reuse-python:latest <account>.dkr.ecr.us-east-1.amazonaws.com/cylon-armada/context-reuse-python:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/cylon-armada/context-reuse-python:latest

# Build and push Path B
docker build -f docker/Dockerfile.nodejs -t context-reuse-nodejs .
docker tag context-reuse-nodejs:latest <account>.dkr.ecr.us-east-1.amazonaws.com/cylon-armada/context-reuse-nodejs:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/cylon-armada/context-reuse-nodejs:latest
```

### DynamoDB Table Provisioning

```bash
aws dynamodb create-table \
    --table-name context-store \
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
    --region us-east-1
```

PAY_PER_REQUEST (on-demand) keeps costs near zero for PoC workloads and avoids capacity planning.

### Deployment Checklist

| Step | Command / Action | Depends On |
|------|-----------------|------------|
| 1. ECR repos | `aws ecr create-repository` | AWS account |
| 2. DynamoDB table | `aws dynamodb create-table` | AWS account |
| 3. ElastiCache Redis | Existing Cylon cluster or new `cache.t3.micro` | VPC |
| 4. Build Python Docker image | `docker build -f docker/Dockerfile.python` | Cylon base image built |
| 5. Build Node.js Docker image | `docker build -f docker/Dockerfile.nodejs` | cylon-wasm built |
| 6. Push images to ECR | `docker push` | Steps 4-5 |
| 7. Create IAM roles | Lambda execution role + Step Functions execution role | — |
| 8. Create Lambda functions | `aws lambda create-function --package-type Image` | Steps 6-7 |
| 9. Configure VPC for workers | Attach subnets + security group | Step 8, VPC exists |
| 10. Create Step Functions state machine | `aws stepfunctions create-state-machine` from `workflow.asl.json` | Steps 7-8 |
| 11. Verify connectivity | Smoke test: start workflow → embed → search → LLM call → aggregate | All above |

---

## Test Scenarios

### Scenario 1: Astronomical Inference Analysis (cosmic-ai)

**Purpose:** Demonstrate context reuse on real astronomical inference workloads. Tasks are dynamically generated from AstroMAE predictions on SDSS photometric data.

**Source:** AI-for-Astronomy / cosmic-ai (arXiv:2501.06249)

**Tasks:** Generated dynamically from real inference results via configurable templates:
- Per-galaxy redshift analysis (similar photometric profiles → high semantic overlap)
- Color-based morphological classification (similar color indices → clustered prompts)
- Outlier analysis for prediction errors (top 10% residuals)
- Batch-level performance summaries and cost analysis

**Task generation pipeline:**
1. Load SDSS data partition (`.pt` file from `cosmicai-data` S3 bucket)
2. Run AstroMAE model inference → predictions, magnitudes, redshifts
3. Generate LLM analysis tasks from real observations via configurable templates
4. Feed tasks through the context reuse pipeline

**Expected reuse rate:** 50-65% — galaxies with similar magnitude profiles produce highly overlapping analysis prompts.

**Key question:** Does context reuse provide the same cost savings on scientific domain tasks as on general-purpose NLP tasks?

**Configuration:** Templates configurable via JSON config file, direct parameter override, or `COSMIC_AI_CONFIG` environment variable.

---

### Scenario 2: Hydrological Analysis (High Similarity)

**Purpose:** Validate context reuse with watershed assessment and flood risk tasks. Similar catchment parameters and hydrological regimes produce semantically overlapping prompts.

**Source:** Biocomplexity Institute — hydrology research domain

**Tasks:** 32 hydrological analysis tasks across 4 categories (8 per category):
- Watershed flood risk assessment (similar drainage area, precipitation, soil type)
- Climate change impact modeling (similar temperature/precipitation scenarios)
- Water quality and nutrient loading (similar agricultural parameters)
- Groundwater and sediment dynamics (moderate similarity)

**Expected reuse rate:** 50-60% — catchments with similar parameters produce highly reusable flood risk and water balance analyses.

**Key question:** How precisely can embeddings distinguish between watersheds with different parameters but similar analytical frameworks?

---

### Scenario 3: Epidemiological Modeling (Moderate Similarity)

**Purpose:** Test context reuse with disease spread modeling where similar transmission parameters and population structures produce partially reusable analyses.

**Source:** Biocomplexity Institute — computational epidemiology research domain

**Tasks:** 32 epidemiological analysis tasks across 4 categories:
- Disease spread modeling (similar R0, serial interval, population size)
- Intervention evaluation (vaccination, school closures, contact tracing)
- Surveillance and forecasting (detection systems, resistance emergence)
- Population health and severity (age structure, behavioral dynamics)

**Expected reuse rate:** 45-55% — epidemic models with similar parameters cluster well, but intervention-specific details limit reuse.

**Key question:** Does the system correctly reuse base epidemic models while generating fresh analyses for different intervention strategies?

---

### Scenario 4: Seismological Hazard Assessment (Moderate Similarity)

**Purpose:** Validate context reuse with earthquake hazard and risk analysis tasks where similar magnitude, depth, and tectonic settings produce reusable hazard evaluations.

**Source:** Biocomplexity Institute — earthquake prediction research domain

**Tasks:** 32 seismological analysis tasks across 4 categories:
- Probabilistic seismic hazard assessment (similar fault parameters, distance, site conditions)
- Ground motion prediction and aftershock modeling (similar magnitude/distance scenarios)
- Structural vulnerability and lifeline analysis (similar building types, infrastructure)
- Induced seismicity and catalog analysis (moderate similarity)

**Expected reuse rate:** 45-55% — sites with similar distance-to-fault, VS30, and design earthquake produce highly overlapping hazard assessments.

**Key question:** How does the similarity engine handle the trade-off between reusing a hazard assessment for a "similar enough" site versus generating a fresh site-specific analysis?

---

### Scenario 5: Mixed Scientific Workload + Benchmarks

**Purpose:** Simulate a realistic multi-domain research workflow with tasks spanning all scientific domains plus system performance benchmarks. Tests cross-domain context isolation and overall cost reduction.

**Source:** Biocomplexity Institute — cross-domain research workload

**Tasks:** 48 tasks across all domains:
- 6 astronomy (cosmic-ai redshift, galaxy classification, model performance)
- 6 hydrology (watershed analysis, climate impact, water quality)
- 6 epidemiology (disease spread, interventions, surveillance)
- 6 seismology (hazard assessment, ground motion, vulnerability)
- 24 system benchmarks (SIMD throughput, path comparison, cost analysis, scalability, FMI overhead, Arrow IPC performance)

**Expected reuse rate:** 40-55% — within-domain tasks reuse well; cross-domain reuse should be near zero (validating domain isolation). Benchmark tasks test system performance rather than LLM content.

**Key question:** What is the overall cost reduction in a realistic multi-domain scientific workflow? Does cross-domain contamination occur (e.g., an epidemiology response reused for seismology)?

---

### Scenario Design Principles

1. **Controlled similarity distribution:** Each scenario has a known distribution of similar/dissimilar tasks so reuse rates can be predicted and validated.
2. **Reproducible task sets:** Task descriptions are stored as JSON files in `target/experiments/scenarios/` or generated dynamically from data (cosmic-ai) and version-controlled.
3. **Baseline pairing:** Every reuse run has a corresponding baseline run with the same tasks, enabling paired statistical comparison.
4. **Cross-path consistency:** All three execution paths (A1, A2, B) run the same task sets to ensure fair comparison.

### Task Sampling Strategy

When `--tasks N` is less than the total tasks in a scenario file, the experiment runner selects a subset using a configurable sampling strategy:

| Strategy | Flag | Behavior | Use Case |
|----------|------|----------|----------|
| **Stratified** (default) | `--sampling stratified` | Picks evenly across the task list with jitter within each stratum. For 4 tasks from a 32-task file with 4 categories: selects 1 from each category. | Production experiments — ensures coverage of all task categories regardless of subset size. |
| **Sequential** | `--sampling sequential` | Takes the first N tasks in order. | Debugging and quick validation — predictable, fast. |
| **Random** | `--sampling random` | Uniform random sample. | Robustness testing — verify results aren't sensitive to task selection. |

All strategies are seeded (`--seed 42` default) for reproducibility. Same seed + same strategy = same tasks every run.

**Why stratified is the default:** Scenario files are structured with intentional category grouping (e.g., hydrology.json has 8 watershed + 8 climate + 8 water quality + 8 groundwater tasks). Sequential sampling with `--tasks 4` would only test watershed tasks, missing the cross-category reuse patterns that the experiment is designed to measure. Stratified sampling guarantees coverage of the full similarity distribution at any subset size.

**Example:**
```bash
# 4 tasks from hydrology (one from each of 4 categories)
python runner.py --tasks-file scenarios/hydrology.json --tasks 4

# Same but sequential (first 4 = all watershed)
python runner.py --tasks-file scenarios/hydrology.json --tasks 4 --sampling sequential

# Random with different seed
python runner.py --tasks-file scenarios/hydrology.json --tasks 4 --sampling random --seed 99

# Full scenario (no sampling needed)
python runner.py --tasks-file scenarios/hydrology.json --tasks 32
```

---

## Success Criteria

### Primary Metrics (Must Achieve)

| Metric | Target | Measured By |
|--------|--------|-------------|
| Cost reduction | **60-80%** vs. baseline | `savings_pct` from BedrockCostTracker across all scenarios |
| Reuse quality | **>0.80 ROUGE-L** for reused responses vs. fresh | Quality analysis comparing reused vs. baseline responses |
| Similarity search latency | **<20ms** for 1000 contexts (512-dim) | `search_latency_p50` per-path metrics |
| SIMD speedup | **>2x** vs. numpy scalar fallback | Path A1 vs. Fallback latency comparison |
| End-to-end pipeline | **All 5 scenarios** complete successfully on Lambda | Experiment runner completes without errors |

### Secondary Metrics (Should Achieve)

| Metric | Target | Measured By |
|--------|--------|-------------|
| Cython batch speedup (A2 vs. A1) | **>1.5x** at 1000+ contexts | `search_latency` comparison A1 vs. A2 |
| Cold start overhead | **<5s** for Python, **<2s** for Node.js | `cold_start_latency` per path |
| Reuse rate (mixed workload) | **40-55%** | Scenario 4 reuse rate |
| Experimental cost | **<$5 total** across all experiments | Sum of all BedrockCostTracker summaries |
| Dimension sweep insight | Measurable quality/latency tradeoff across 256/512/1024 | Parameter sweep results show non-trivial differences |

### Deliverable Checklist

- [x] Working context store (Cylon ContextTable + DynamoDB + Redis, configuration-driven backend)
- [x] Context Router with configurable threshold making correct reuse/new-call decisions
- [x] LangChain Executor invoking Claude via Bedrock and returning structured responses
- [x] Cython batch similarity search (Path A2) compiled and functional
- [x] All 3 execution paths (A1, A2, B) — Python S3 script runner + Node.js direct dispatch
- [x] Cylon FMI Communicator coordinating parallel Lambda workers (context broadcast, cost reduction)
- [x] BedrockCostTracker capturing real USD costs with pricing resolution (Python + Node.js)
- [x] Terraform infrastructure (Lambda, Step Functions, DynamoDB, ECR, S3, IAM)
- [x] cosmic-ai integration (AstroMAE inference, task generation, ONNX export, model parallelism)
- [x] 5 test scenarios with 32-48 tasks each (4 general-purpose + 1 cosmic-ai from real data)
- [ ] Deploy to AWS, execute experiments, capture results
- [ ] Jupyter notebooks with cost curves, latency distributions, path comparisons, quality analysis
- [ ] Proposal paper draft with experimental methodology, results, and visualizations

---

## Risks & Open Questions

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **pycylon SIMD bindings not exposing cosine_similarity** | Path A1/A2 blocked | Medium | Verify binding exists; fallback to calling Cylon C++ directly via ctypes or adding the binding |
| **Cython batch search doesn't outperform per-call** | Weakens A2 benchmark narrative | Low | Python→C++ overhead is well-documented; if gap is small, the finding itself is interesting |
| **Bedrock throttling during experiments** | Experiments stall or produce incomplete data | Medium | Use exponential backoff; stagger experiment runs; request limit increase if needed |
| **WASM SIMD128 not enabled on Lambda Node.js runtime** | Path B has no SIMD acceleration | Low | WASM SIMD128 is supported on Node.js 16+; verify with feature detection at startup |
| **DynamoDB Scan for get_all_embeddings is slow at scale** | Search latency dominated by data retrieval, not SIMD | Medium | Cache all embeddings in Redis; for >10K contexts, consider pagination or pre-loading |
| **LangChain Bedrock integration changes** | API breakage during development | Low | Pin langchain-aws version; wrap calls in abstraction layer |

### Open Questions

1. **Arrow format for embeddings — worth it?** Using Arrow IPC for embedding storage in Redis adds complexity but enables zero-copy transfer. For the PoC, raw float32 bytes may be sufficient. Decision: start with raw bytes, benchmark, and add Arrow if serialization shows up in profiling.

2. **Cylon Communicator channel selection for benchmarks.** Redis channel is simpler for Phase 1, but Direct (TCPunch) has lower latency. Should we benchmark both channel types as an additional variable, or fix to Redis for the PoC?

3. **Quality metric selection.** ROUGE-L and BLEU measure surface-level text similarity. For code-related tasks, these may not capture semantic correctness. Consider adding a task-specific evaluator (e.g., LLM-as-judge) for quality assessment.

4. **Embedding model warmth.** Titan V2 may have cold start latency on first invocation. Should we pre-warm the model before experiment runs, or include cold start in the measurements?

5. **Context store growth during experiments.** As experiments run, the context store accumulates entries from prior runs. Should each experiment start with a clean store, or should we test with a pre-populated store to simulate production conditions?

6. **Lambda concurrency limits.** Running 32+ parallel Lambda workers may hit account concurrency limits. Verify reserved concurrency is available or request an increase.