<img src="img.png" width="100%">

# Cylon Armada

**Context-Based Cost Optimization for Multi-Agent LLM Workflows**

Cylon Armada reduces LLM costs by 60-80% in multi-agent workflows through intelligent context reuse based on semantic similarity. Built on the [Cylon](https://github.com/mstaylor/cylon) distributed computing platform, it provides SIMD-accelerated similarity search and fault-tolerant execution in a serverless architecture.

## Key Metrics

| Metric | Target |
|--------|--------|
| Cost Reduction | 60-80% |
| Similarity Search | <20ms for 1,000 contexts |
| SIMD Speedup | 2-4x vs scalar |
| Scalability | 10,000+ contexts per node |
| Context Reuse Rate | 60-80% (task-dependent) |

## Core Innovation

The **Context Similarity Engine** identifies semantically similar LLM contexts using embeddings and SIMD-accelerated cosine similarity (via Cylon), enabling intelligent reuse of LLM outputs across agents and workflows. Instead of making redundant LLM calls, agents check for existing similar contexts first - paying only the cost of an embedding lookup rather than a full LLM invocation.

### SIMD-Accelerated Similarity Search

The similarity engine uses **SIMD (Single Instruction, Multiple Data)** to accelerate embedding comparisons. Rather than comparing embedding dimensions one at a time (scalar), SIMD processes 4 float32 values in a single CPU instruction using 128-bit vector registers:

```
Scalar:  a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]  →  4 instructions
SIMD:    [a[0],a[1],a[2],a[3]] * [b[0],b[1],b[2],b[3]]  →  1 instruction
```

For a 1536-dimension embedding (e.g., `text-embedding-3-small`), the dot product at the core of cosine similarity drops from 1536 multiply-accumulate operations to ~384. Cylon implements this in Rust with WASM SIMD128 intrinsics (`f32x4_mul`, `f32x4_add`) and exposes the operations through PyCylon:

- **`cosine_similarity_f32(a, b)`** - SIMD dot product + normalization
- **`dot_product_f32(a, b)`** - SIMD 4-wide multiply-accumulate
- **`euclidean_distance_f32(a, b)`** - SIMD difference-squared-sum

Real-world speedup is **2-4x** vs scalar (memory bandwidth and cache effects prevent a full theoretical 4x). At 1,000 stored contexts, a full similarity scan completes in under 20ms.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Orchestration Layer (AWS Step Functions)         │
│    Workflow Definition → Task Distribution → Aggregation │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│             Agent Processing Layer (Lambda)               │
│  Context Manager │ Context Router │ Agent Coordinator    │
│  Chain Executor  ← core research contribution            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            Data Processing Layer (Cylon)                  │
│    SIMD Operations │ Distributed Processing │ Checkpoints│
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   Storage Layer                           │
│       DynamoDB │ ElastiCache Redis │ S3                   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 LLM Provider Layer                        │
│        AWS Bedrock │ Azure OpenAI │ OpenAI               │
└─────────────────────────────────────────────────────────┘

Analysis & Visualization: Jupyter Notebooks (graphs and charts only)
```

## Components

### Context Manager
Stores and retrieves LLM contexts with their embeddings. Uses DynamoDB for persistence and Redis for hot caching with <5ms read latency on cache hits.

### Context Router
Finds similar contexts using Cylon's SIMD-accelerated cosine similarity. Searches 1,000 contexts in under 20ms with a configurable similarity threshold (default 0.85).

### Agent Coordinator
Orchestrates multi-agent teams with cognitive diversity patterns: role specialization, perspective diversity, hierarchical reasoning, and ensemble voting. Agents share a common context pool for cross-agent reuse.

### Chain Executor
Runs multi-step LLM workflows with fault tolerance. Checkpoints state to S3 after each step (Arrow IPC + zstd compression) and resumes from the last checkpoint on failure.

## Building and Deployment

The entire environment (conda, PyCylon, communication backends, dependencies) is baked into Docker images. Which image you build depends on your deployment target. All Dockerfiles live in the [Cylon repo](https://github.com/mstaylor/cylon) under `docker/`.

### Deployment Targets

| Target | Communication | GPU | Use Case |
|--------|--------------|-----|----------|
| **Lambda** | Redis OOB | No | Serverless agent execution |
| **ECS / Fargate** | UCX/UCC + Redis | Optional (EC2 + GPU) | Long-running container workloads |
| **Rivanna (HPC)** | UCX/UCC + Libfabric + Redis | Yes | High-performance cluster experiments |

### Lambda

Serverless execution with Redis for out-of-band communication. Includes boto3, `awslambdaric`, and cloudmesh.

```bash
docker build -t cylon-lambda -f docker/aws/lambda/Dockerfile docker/aws/lambda/
```

| | |
|---|---|
| **Cylon build flags** | `-DCYLON_USE_REDIS=1 -DCYLON_FMI=1` |
| **Conda env** | `cylon_NoUCX.yml` |
| **Entrypoint** | `awslambdaric` → Lambda handler |

### ECS / Fargate

Container-based execution with UCX/UCC for high-throughput inter-node communication. For GPU workloads, attach an EC2 GPU instance (T4, A10G, V100) and use the gcylon image.

```bash
# ECS (CPU)
docker build -t cylon-ecs -f docker/aws/ecs/Dockerfile docker/aws/ecs/

# Fargate (CPU)
docker build -t cylon-fargate -f docker/aws/ecs-fargate/Dockerfile docker/aws/ecs-fargate/

# ECS with GPU (gcylon - requires NVIDIA GPU instance)
docker build -t cylon-gcylon -f docker/gcylon/Dockerfile docker/gcylon/
```

| | |
|---|---|
| **Cylon build flags (CPU)** | `-DCYLON_UCX=1 -DCYLON_UCC=1 -DCYLON_USE_REDIS=1` |
| **Cylon build flags (GPU)** | Same as CPU + gcylon/pygcylon with cuDF/RMM (CUDA 12.8) |
| **Conda env** | `cylon_NoUCX.yml` |
| **Deps** | UCX v1.19, UCC v1.6, hiredis, redis-plus-plus |
| **GPU deps** | RAPIDS 24.10 (cudf, libcudf, rmm), Rust `--features gpu` |

### Rivanna (HPC)

HPC cluster deployment with UCX/UCC, libfabric, and a custom UCX branch for remote address overriding. Supports both CPU and GPU experiments.

```bash
# CPU
docker build -t cylon-rivanna -f docker/rivanna/Dockerfile docker/rivanna/

# GPU (gcylon)
docker build -t cylon-gcylon -f docker/gcylon/Dockerfile docker/gcylon/
```

| | |
|---|---|
| **Cylon build flags (CPU)** | `-DCYLON_UCX=1 -DCYLON_UCC=1 -DCYLON_USE_REDIS=1 -DCYLON_LIBFABRIC=1` |
| **Cylon build flags (GPU)** | Same as CPU + gcylon/pygcylon with cuDF/RMM |
| **Conda env** | `cylon_rivanna_NoUCX.yml` |
| **Deps** | UCX (override-remote-address branch), UCC, libfabric, hiredis, redis-plus-plus |

### EC2 with Libfabric

For EC2-based experiments that need libfabric (e.g., AWS EFA), add the libfabric build flag to any ECS image:

| | |
|---|---|
| **Additional build flag** | `-DCYLON_LIBFABRIC=1 -DLIBFABRIC_INSTALL_PREFIX=$LIBFABRIC_HOME/install` |

### Configuration

```bash
# Environment variables (all targets)
REDIS_HOST=localhost
REDIS_PORT=6379
OPENAI_API_KEY=sk-...
SIMILARITY_THRESHOLD=0.85
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
AWS_REGION=us-east-1
```

## Project Structure

```
cylon-armada/
├── armada/
│   ├── context/           # Context Manager and Router
│   ├── agents/            # Agent Coordinator and team patterns
│   ├── chains/            # Chain Executor and workflow definitions
│   ├── providers/         # LLM provider abstraction layer
│   └── experiments/       # Experiment runner and analysis tools
├── deployment/
│   └── terraform/         # AWS infrastructure as code
├── notebooks/             # Jupyter notebooks (graphs and charts)
├── tests/                 # Unit and integration tests
└── scripts/               # Build, deploy, and smoke test scripts
```

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Languages | Python 3.9+, C/C++ (via PyCylon), Rust (via Cylon) |
| Data Processing | PyCylon (Cython bindings), SIMD ops, Arrow IPC |
| GPU Processing | gcylon, pygcylon, cuDF, RMM (CUDA 12.8, RAPIDS 24.10) |
| Compute | AWS Lambda, ECS/Fargate, Rivanna HPC |
| Communication | Redis OOB (Lambda), UCX/UCC (ECS, Rivanna), Libfabric (Rivanna, EC2) |
| Storage | DynamoDB, ElastiCache Redis, S3 |
| LLM Providers | AWS Bedrock, Azure OpenAI, OpenAI |
| Embeddings | text-embedding-3-small (1536-dim) |
| Visualization | Jupyter Notebooks (Matplotlib, Plotly) |
| Infrastructure | Terraform, Docker, GitHub Actions |

## How Context Reuse Works

```
1. Agent receives task
2. Generate embedding for the task
3. Search for similar contexts (PyCylon SIMD cosine similarity)
4. If similarity >= threshold (0.85):
     → Return cached context (cost: ~$0.00002)
5. Else:
     → Make LLM call (cost: ~$0.002)
     → Store result + embedding for future reuse
```

At a 75% reuse rate, this reduces per-task cost from $0.00202 to $0.00052 - a **74% reduction**.

## Roadmap

- [x] **Phase 0** - Cylon foundation (PyCylon, SIMD, distributed ops, libfabric + Cython bindings)
- [ ] **Phase 1** - Proof-of-concept (context store, similarity engine, demo workflow)
- [ ] **Phase 2** - Core components deployed to AWS (Context Manager, Router, Coordinator, Executor)
- [ ] **Phase 3** - Multi-agent orchestration (cognitive diversity patterns, 1000+ agents)
- [ ] **Phase 4** - Large-scale experiments (100K+ tasks, publication-quality results, GPU benchmarks)
- [ ] **Phase 5** - Thesis and publications

## Related Projects

- [Cylon](https://github.com/mstaylor/cylon) - Distributed data processing framework (foundation)
- [ruv-FANN](https://github.com/mstaylor/ruv-FANN) - Multi-agent orchestration patterns

## License

TBD

## References

- Cylon: A Fast, Scalable, Universal Distributed Data Processing Framework (Frontiers paper)
- Polychroniou et al., "Rethinking SIMD Vectorization for In-Memory Databases" (SIGMOD 2015)
