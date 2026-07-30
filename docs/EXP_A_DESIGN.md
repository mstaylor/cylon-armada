# Experiment A / A2 Zero-Copy Data Plane Benchmark Design

**Status:** proposed (pre-implementation review)
**Validates:** Contribution C1 (zero-copy Arrow data plane), Arrow schema compatibility.
**Proposal refs:** Table VIII row A/A2; Validation §Experiment A; IV1 (dim), IV2 (batch size), payload-schema-type axis.

## 1. What the proposal asks for

**Experiment A, Zero-Copy Data Plane Throughput Benchmark.** Compare the Apache Arrow/Cylon
zero-copy data plane against serialization-based baselines (JSON, Pickle, Protobuf). Metrics:
**throughput** and **memory copies**. Expected: Arrow outperforms JSON, Pickle, and Protobuf, with
the largest gain vs JSON and a smaller but measurable gain vs Protobuf. The comparison isolates the
benefit of *reduced memory copies*.

**Experiment A2, Arrow Schema Compatibility.** Validate the zero-copy edge condition
`schema_out(Oi) == schema_in(Oj)` across the operator DAG. Plus the proposal's payload-schema-type
axis: dense fixed (`FixedSizeList<Float32>` embeddings) vs variable-length nested (`LargeUtf8`
document chunks). Arrow's zero-copy advantage is strongest for dense, weakest for variable.

## 2. Representative workload payload

The dominant operator payload (>90% of edges) is the **embedding batch**: `N × D` float32 stored as
Arrow `FixedSizeList<Float32>[D]`, the schema held by `ContextTable`
(`cpp/src/context/context_table.cpp:46-61`). The benchmark serializes this structure through the
same code paths the framework uses for inter-operator transfer, so measured throughput reflects the
framework's data plane rather than an arbitrary payload.

**Grid** (proposal IV1/IV2):
- Batch size `N ∈ {100, 500, 1000, 5000}` rows
- Embedding dim `D ∈ {256, 512, 1024}` (Titan v2 dims used across all existing data; 768 optional)
- Payload type ∈ {`embeddings` (dense FixedSizeList), `doc_chunks` (variable LargeUtf8)}

**Data-scale axis (separate sweep).** The node/agent axis (Experiment B/E, `N ≤ 64`) and the
*data* axis are distinct dimensions and are argued separately. To exercise the data plane at the
row counts characteristic of the prior Cylon data-engineering work, and to substantiate the
petabyte-class motivation rather than only the 1 TB memory-bounded subset, Experiment A adds a
single-node data-scale sweep on the same code paths:

- Corpus size `R ∈ {10^4, 10^5, 10^6, 10^7}` embedding rows (10K to 10M), with `R` extended toward
  10^8 to 10^9 on the HPC/EC2-EFA arm where memory permits.
- Reported as throughput vs `R` and as bytes transferred, holding `D` fixed at 1024.

This lets the throughput result hold the *platform constant* while varying data volume by four to
five orders of magnitude, so the "modest node count" observation does not carry over to the data
dimension. The scalability of the underlying Cylon collective layer at billions of rows is
**inherited from prior published work** and cited as foundation, not re-derived here (see
`docs/EXPERIMENT_PLATFORM_STRATEGY.md`).

## 3. Formats compared (each = serialize → wire-bytes → deserialize → usable array)

| Format | What it is | Why it's here |
|--------|-----------|---------------|
| **`arrow_ipc`** | Arrow RecordBatch IPC stream (`pa.ipc.new_stream`) | The proposed zero-copy data-plane edge; deserialization reads directly into the FixedSizeList buffer. This is what crosses an operator edge (the proposal's Arrow-IPC hypothesis), not the ContextTable store snapshot |
| **`json`** | `json.dumps(list)` → `json.loads` → `np.array` | JSON-over-HTTP baseline (proposal); corresponds to the metadata path (`manager.py:245`) |
| **`pickle`** | `pickle.dumps(arr, HIGHEST_PROTOCOL)` → `pickle.loads` | Pickle-over-gRPC baseline (proposal) |
| **`protobuf`** | `EmbeddingBatch { int32 n; int32 dim; repeated float values [packed]; }` | Protobuf-over-gRPC baseline (proposal). Python protobuf marshals repeated floats element-wise (slow); the proposal notes this is a GIL artifact and proposes a native Go comparison. `_pb2.py` committed so the harness needs no `protoc` at run time |
| **`flatbuffers`** | zero-copy `[float]` vector (`fbs/embedding_batch.fbs`) | FlatBuffers-over-gRPC baseline (proposal). The zero-copy binary format; isolates Arrow's columnar-layout advantage from generic zero-copy binary. Generated code committed so no `flatc` at run time |

Supplementary formats (not proposal Exp A baselines, kept for transparency):

| Format | What it is | Why it's here |
|--------|-----------|---------------|
| `base64_tobytes` | `base64(np.tobytes)` → `np.frombuffer` | The framework's current FMI transfer encoding (`run_action.py:90`) |
| `contexttable_ipc` | `ContextTable.to_ipc()` / `from_ipc()` (full 10-column store) | The store snapshot, not the data-plane edge. Transferred zero-copy (`pyarrow_wrap_buffer` + `MakeFromIpcBuffer`) with the O(N) context_id index build deferred to first keyed access. Above pickle at D=1024, near the bare RecordBatch. The ContextTable is the subject of Experiment J |

## 4. Metrics (per format × N × D × payload-type)

| Metric | Definition |
|--------|-----------|
| `serialize_ms`, `deserialize_ms`, `roundtrip_ms` | median over measured reps (warmup discarded) |
| `usable_access_ms` | time to obtain an `(N,D)` float32 you can run SIMD on (Arrow = O(1) view; others materialize) |
| `wire_bytes` | serialized payload size |
| `payload_bytes` | logical size `N*D*4` |
| **`throughput_roundtrip_MBps`** | `payload_bytes / roundtrip_s`, **the headline metric** |
| `throughput_serialize_MBps`, `throughput_deserialize_MBps` | directional throughput |
| **`memory_copies`** | structural count on deserialize path: arrow_ipc=0, tobytes=1, pickle=1, json=2 (documented) |
| `deserialize_peak_kb` | `tracemalloc` peak during deserialize, **empirical backing** for the copy claim |
| `speedup_vs_json` | derived headline number |

Rigor: `np.random.default_rng(42)` payloads (repo seed convention), `--warmup 3 --reps 20`, report
median + p95. `--runs N` writes `run_{n}/` subdirs for cross-run std (matches runner convention).

## 5. A2 Schema Compatibility

Define the 5 operators as `(pattern, schema_in, schema_out)` per the proposal contracts, build the
pipeline DAG, and check every edge for `schema_out(Oi)` Arrow-compatibility with `schema_in(Oj)`.
Output: a compatibility matrix (zero-copy-eligible edges / total) plus which edges need serialization.

## 6. Architecture and Code Reuse

Findings from the codebase study drive these decisions:

- **`runner.py` is hardwired** to the Bedrock/context-reuse workflow → **do NOT extend it.** Write a
  standalone entry script, reusing only the generic `ExperimentBenchmark` (start/stop/record/save +
  S3). This is the exact precedent of the Node.js SIMD benchmarks.
- **Charts: new function, not the cost/reuse ones** (none are metric-agnostic). Keep the existing
  cost/reuse aggregator + `chart_generator.py` **untouched**. A foreign metric family should not be
  forced through them. New `chart_zerocopy.py` matches the existing style constants
  (`FONT_SIZE=12`, `figsize=(10,6)`, `dpi=300`, svg, the palette dicts).
- **The compiled `ContextTable` extension runs locally** (confirmed) with `LD_LIBRARY_PATH=$CONDA/lib` +
  `PYTHONPATH=cylon-armada/python`. Develop and validate locally in `cylon_dev`, then run the same
  harness on AWS Lambda to obtain measurements under the 10 GB / Firecracker configuration specified
  in the proposal.

### Deliverables

| File | Purpose |
|------|---------|
| `target/shared/scripts/experiment/exp_a_zerocopy.py` | Benchmark A + A2. Importable `run(config)->list[dict]`; `__main__` for local. Reuses `ExperimentBenchmark` |
| `target/shared/scripts/experiment/examples/run_exp_a.sh` | Local wrapper: sets the two env exports, runs in `cylon_dev` |
| `target/shared/scripts/results/chart_zerocopy.py` | New charts (throughput-by-format, memory-copies, schema-compat matrix) in the existing style |
| `target/aws/scripts/lambda/python/armada_benchmark.py` | Thin Lambda handler → `exp_a_zerocopy.run()` → S3. Single-node, no Map. Applies defaults for every event field; records the configured Lambda memory as provenance |
| `target/aws/scripts/step_functions/workflow_benchmark.asl.json` | Benchmark state machine: a single Task (no Map) invoking the benchmark Lambda, `STANDARD` type. Forwards the whole input (no `Parameters` block) so optional fields never trip the SFN missing-field failure |
| Driver: `cloud_sweep.py --arch benchmark` | The existing sweep driver, extended with a `benchmark` arch (kept out of `--arch all`) that skips the scenario/world-size loop, fires `cylon-armada-benchmark-workflow`, and polls S3 for each run's results CSV. Reuses `fire_execution` + `poll_s3_results` |
| Terraform block: `cylon-armada-benchmark` Lambda + state machine | Dedicated fn, `HANDLER_MODULE=armada_benchmark`, **10240 MB** via `benchmark_memory_mb`, `benchmark_timeout` 900 s; plus the `benchmark_workflow` state machine and its SFN invoke permission. *User applies.* |

Output dir: fresh `results/exp_a_zerocopy/`. It **does not touch** the protected domain data
(seismology / mixed_scientific / epidemiology / hydrology).

## 7. Success criteria

1. `arrow_ipc` round-trips through the compiled `ContextTable` (`to_ipc`/`from_ipc`), and the
   embedding column returns as `FixedSizeList<Float32>` without a copy, verified locally in `cylon_dev`.
2. Measured `throughput_roundtrip_MBps`: `arrow_ipc` > `pickle`/`tobytes` > `json`, with a clear,
   chartable gap; `memory_copies` and `deserialize_peak_kb` corroborate (Arrow lowest).
3. A2: schema-compat matrix shows dense-embedding edges are zero-copy-eligible; variable `LargeUtf8`
   edges are where Arrow's advantage narrows.
4. The same harness runs on AWS Lambda (10 GB), results are written to S3, and the charts regenerate
   from the collected data.

## 8. Explicit assumptions

- Titan-v2 dims {256,512,1024} are the right sweep (consistent with all prior data). 768 optional.
- Protobuf is **included**. The `.proto` is compiled once and the generated `_pb2.py` committed, so
  the benchmark harness has no run-time `protoc` dependency (only the `protobuf` runtime, added to
  `cylon_dev`).
- One dedicated 10 GB benchmark Lambda (one `terraform apply`) is acceptable; it's reused for Exp J's
  L3 test later.
- The benchmark is orchestrated through **AWS Step Functions**, consistent with every other
  experiment in the framework: a dedicated benchmark state machine invokes the benchmark Lambda
  (single-node, so a plain Task state rather than a Map), and a `cloud_sweep`-style driver starts the
  execution and collects results from S3. This keeps Experiment A on the same deployed orchestration
  layer as Experiments B/E rather than on a one-off invocation path.

## 9. How to run

Four stages: validate locally, deploy, fire on AWS, collect and chart. All paths write the
same CSVs, so the same `results.pipeline` step charts local and cloud runs alike.

### 9.1 Local (develop and validate in `cylon_dev`)

Two env exports (dynamic loader + package path), then run the benchmark as a module:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cylon_dev
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH            # finds libcylon_armada.so
export PYTHONPATH=/home/parallels/cylon-armada/python:$PYTHONPATH   # finds the cylon_armada pkg
cd target/shared/scripts

python -m experiment.exp_a_zerocopy \
    --batch-sizes 100 500 1000 5000 --dims 256 512 1024 \
    --warmup 3 --reps 20 --runs 3 \
    --output experiment/results/exp_a_zerocopy
```

Locally `protobuf` and `flatbuffers` are skipped unless installed (`pip install protobuf
flatbuffers`) — 5 formats vs the 7 the Docker image runs. `arrow_ipc`/`contexttable_ipc` need
the compiled extension (the two exports above); a path miss is never a "rebuild required" problem.

Charts + tweakable notebook from the local results:

```bash
python -m results.pipeline --experiment zerocopy \
    --local-dir experiment/results/exp_a_zerocopy --chart-format png
```

### 9.2 Deploy (make it live on AWS)

A code fix is not live until the artifact path is updated (Code Change Rule 5). The C++/Cython
IPC changes and the `protobuf`/`flatbuffers` deps travel in the image; Python-only edits
hot-reload from S3.

```bash
# 1. Rebuild + push the Python image (C++/Cython IPC changes + new serialization deps)
#    --platform=linux/amd64 is required: the image is x86_64 by design (x86_64
#    Miniconda; Lambda/ECS run cpu_architecture = X86_64). On an arm64 build host
#    (Apple Silicon / Parallels) the plain build pulls an arm64 base and the
#    x86_64 Miniconda installer fails with "rosetta error ... ld-linux-x86-64.so.2";
#    the flag forces the amd64 base so it builds (emulated on arm64, native on x86).
docker build --platform=linux/amd64 -t cylon-armada-python -f docker/Dockerfile.python .
#    docker tag + push to ECR :python-latest  (per your ECR login)

# 2. Sync shared scripts + the benchmark handler to S3 (hot-reload; no rebuild for py-only edits)
aws s3 sync target/shared/scripts/            s3://<scripts-bucket>/<prefix>/
aws s3 sync target/aws/scripts/lambda/python/ s3://<scripts-bucket>/<prefix>/lambda/

# 3. Create the 10 GB benchmark Lambda + benchmark state machine
cd target/aws/scripts/terraform && terraform apply
```

Verify the benchmark Lambda's `S3_SCRIPTS_PREFIX` matches the prefix you synced to — prefix drift
has silently served stale code before (CLAUDE.md, Debugging table).

### 9.3 Cloud run (fire the benchmark state machine)

Always dry-run first (prints the payloads and the S3 completion keys, fires nothing):

```bash
cd target/aws/scripts/experiment
python cloud_sweep.py --arch benchmark --runs 4 \
    --batch-sizes 100 500 1000 5000 --dims 256 512 1024 \
    --warmup 3 --reps 20 --dry-run

python cloud_sweep.py --arch benchmark --runs 4 \
    --batch-sizes 100 500 1000 5000 --dims 256 512 1024 --warmup 3 --reps 20
```

`--runs N` fires N separate executions (cold-Lambda variance for error bars); `--matrix-runs M`
repeats the whole matrix inside one Lambda (writes `run_M/` keys). The driver polls S3 for each
run's `..._exp_a_zerocopy_results.csv`. Results land under
`s3://<results-bucket>/results/benchmark/exp_a/` (override with `--results-prefix`).

### 9.4 Collect and chart the cloud results

```bash
aws s3 sync s3://<results-bucket>/results/benchmark/exp_a/ \
    experiment/results/exp_a_zerocopy_cloud/
python -m results.pipeline --experiment zerocopy \
    --local-dir experiment/results/exp_a_zerocopy_cloud --chart-format png
```

### 9.5 Cost note

The benchmark Lambda bills only per invocation — no idle cost — so it can stay applied (it is
reused for Experiment J's L3 test). Experiment A adds no GPU/EC2, so there is nothing here to
tear down; the continuously billing `terraform-gpu` / `terraform-ec2` modules are separate.

## 10. Serverless allocator sensitivity (empirical finding)

The first AWS Lambda run (10 GB, 3 runs, all 7 formats) surfaced a result the local runs did not:
`arrow_ipc` and `contexttable_ipc` throughput **collapses at the largest batch** (N=5000, D=1024,
a ~20 MB payload), falling ~3.8x below their mid-range peak and, with the default allocator, below
pickle. In the mid-range (N=500 to 1000) Arrow wins clearly, so the effect is a crossover, not a
flat loss.

**Root cause.** The benchmark reserializes every rep, allocating a fresh ~20 MB Arrow IPC buffer;
the default arena allocator classifies that as a huge allocation and returns it to the OS on free
(`munmap`), so each rep re-`mmap`s and first-touch-faults the whole buffer. That page-fault churn
is memory-bandwidth-bound and dominates on Lambda's constrained bandwidth (~18 GB/s); the local
machine (~40 GB/s) hides it. The deserialize timing is the tell: at N=1000 it runs at ~120 GB/s
(genuine zero-copy view), but at N=5000 it drops to memcpy speed, i.e. the churn taxes even the
zero-copy read path. The structural memory-copies claim (arrow 0, pickle 1, json 2) is unaffected.

**Allocator sweep** (arrow_ipc round-trip MB/s at N=5000, D=1024):

| Allocator configuration | MB/s | Fixes collapse |
|---|---|---|
| default jemalloc (naive) | 1,373 | no |
| jemalloc with `ARROW_JEMALLOC_DECAY_MS=-1` | 1,337 | no |
| mimalloc | 1,353 | no |
| `system` pool + `MALLOC_MMAP_MAX_=0` + `MALLOC_TRIM_THRESHOLD_=-1` | 4,556 | yes (3.3x) |

Both arena allocators (jemalloc, mimalloc) `munmap` huge blocks on free regardless of their decay
or retention knobs, so their tuning does nothing here. Only forcing every allocation onto the
retained glibc heap (system pool, no `mmap`, no trim) keeps the freed buffer resident for reuse and
removes the churn. `exp_a_zerocopy.py` exposes `ARROW_JEMALLOC_DECAY_MS` as an optional tuning lever
(calls `pyarrow.jemalloc_set_decay_ms`); it is retained as a documented knob but does not help this
workload for the reason above.

**Tuned result.** With the system allocator config, the crossover disappears and Arrow wins across
the whole grid. Full sweep, N=5000, D=1024, median of 3 runs, naive to tuned:

| Format | naive | tuned | factor |
|---|---|---|---|
| arrow_ipc | 1,373 | 5,258 | 3.8x |
| contexttable_ipc | 884 | 3,376 | 3.8x |
| pickle | 2,355 | 2,478 | 1.0x |

arrow vs pickle at N=5000 goes from 0.58x (Arrow loses) to 2.12x (Arrow wins). The tradeoff is real
and unavoidable: the glibc allocator is ~20 percent slower for frequent mid-range allocations
(arrow N=1000, D=1024: 8,910 naive vs 6,856 tuned), so the config trades a modest mid-range cost for
large-payload robustness. There is no allocator-tuning best of both.

**How to report it.** Present naive and tuned as a before and after. The honest framing is that the
zero-copy data-plane advantage is real and large, but realizing it on bandwidth-limited serverless
requires pinning allocations to the heap to avoid huge-allocation `munmap` churn; the naive default
shows a large-payload cliff. That is a systems contribution, config-only, no change to the data
plane. Tuned config (Lambda env, or the equivalent on any host):

```bash
ARROW_DEFAULT_MEMORY_POOL=system
MALLOC_MMAP_MAX_=0
MALLOC_TRIM_THRESHOLD_=-1
```

**Distributed implication.** The cost is on the serialize (send) side, so it carries to multi-worker
Arrow transfers in Experiments B/E independent of the FMI channel (`direct` / `redis` / `s3`) or the
TCPunch rendezvous server (which brokers connection setup only and never touches the payload). A
real transfer serializes once rather than in a tight rep loop, so the effect is smaller there, but
the allocator config should travel with the data-plane code. Data-scale runs (Section 2) that push
the payload past 20 MB will re-encounter this and should use the tuned config.

Artifacts: naive `results/exp_a_zerocopy_cloud/`, tuned `results/exp_a_zerocopy_tuned/` (each with a
median CSV, four charts, and a notebook); S3 prefixes `results/benchmark/exp_a/` and
`.../exp_a_tuned/`.

## Placeholder to keep the cost note intact
tear down; the continuously billing `terraform-gpu` / `terraform-ec2` modules are separate.