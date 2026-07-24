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
| **`arrow_ipc`** | `ContextTable.to_ipc()` / `ContextTable.from_ipc()` (C++ Arrow IPC stream) | The proposed zero-copy data plane; deserialization reads directly into the FixedSizeList buffer |
| **`base64_tobytes`** | `base64(np.tobytes)` → `np.frombuffer` | The framework's current FMI transfer encoding (`run_action.py:90`); included so the comparison reflects the existing implementation |
| **`json`** | `json.dumps(list)` → `json.loads` → `np.array` | JSON-over-HTTP baseline (proposal); corresponds to the metadata path (`manager.py:245`) |
| **`pickle`** | `pickle.dumps(arr, HIGHEST_PROTOCOL)` → `pickle.loads` | Pickle-over-gRPC baseline (proposal) |
| **`protobuf`** | `EmbeddingBatch { int32 n; int32 dim; repeated float values [packed]; }` | Binary-serialization baseline (proposal); expected smaller but measurable gain vs Arrow. `_pb2.py` pre-generated and committed so the harness needs no `protoc` at run time |

Secondary isolation variant: `arrow_ipc_pyarrow` (pure `pa.ipc.new_stream` on a RecordBatch) to
separate Arrow-IPC itself from ContextTable overhead.

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
| `target/aws/scripts/lambda/python/armada_benchmark.py` | Thin Lambda handler → `exp_a_zerocopy.run()` → S3 |
| `target/aws/scripts/step_functions/workflow_benchmark.asl.json` | Benchmark state machine: Task state invoking the benchmark Lambda (single-node), consistent with the existing `workflow*.asl.json` set |
| Driver: `cloud_sweep`-style starter | Starts the benchmark state-machine execution and polls S3 for results (mirrors `target/aws/scripts/experiment/cloud_sweep.py`) |
| Terraform block: `cylon-armada-benchmark` Lambda + state machine | Dedicated fn, `HANDLER_MODULE=armada_benchmark`, **10240 MB** (proposal's canonical config), plus the benchmark state machine. *User applies.* |

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