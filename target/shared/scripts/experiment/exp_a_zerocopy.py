"""Experiment A / A2: Zero-Copy Data Plane Benchmark.

Measures serialization throughput and memory-copy behavior for the embedding
batch payload the framework moves between operators (an Arrow
`FixedSizeList<Float32>[D]` column, the ContextTable schema), comparing the
zero-copy Arrow IPC path against serialization baselines.

Formats. The proposal (Experiment A) names four serialization baselines,
JSON over HTTP, Pickle over gRPC, Protobuf over gRPC, and FlatBuffers over gRPC,
compared against the Arrow IPC zero-copy data plane:

    arrow_ipc          Arrow RecordBatch IPC stream (the zero-copy data-plane edge)
    json               json.dumps(list) -> json.loads -> np.array          (JSON over HTTP)
    pickle             pickle HIGHEST_PROTOCOL                              (Pickle over gRPC)
    protobuf           packed repeated float (proto/embedding_batch.proto)  (Protobuf over gRPC)
    flatbuffers        zero-copy [float] vector (fbs/embedding_batch.fbs)   (FlatBuffers over gRPC)

FlatBuffers is the zero-copy binary baseline: it isolates Arrow's columnar-layout
contribution from the general zero-copy-binary advantage. Supplementary formats,
kept for transparency but not among the proposal's Exp A baselines:

    base64_tobytes     base64(np.tobytes) -> np.frombuffer (the framework's current FMI encoding)
    contexttable_ipc   ContextTable.to_ipc / from_ipc (store snapshot; the ContextTable
                       itself is evaluated by Experiment J, not the data-plane edge)

Each format is timed as serialize (native -> wire) + deserialize (wire ->
decoded) + usable_access (decoded -> (N,D) float32 ready for SIMD). The zero-copy
formats (arrow_ipc, flatbuffers, contexttable_ipc) return an O(1) view into the
wire buffer; the serialization baselines materialize a numpy array.

A2 validates the Arrow schema-compatibility edge condition across the five
canonical operators.

Runs locally in `cylon_dev` (needs LD_LIBRARY_PATH + PYTHONPATH per CLAUDE.md)
and inside the Lambda image. The `arrow_ipc` format degrades gracefully to a
skip if the compiled `cylon_armada` extension is not importable, so the
serialization baselines still run.

Usage (local):
    python exp_a_zerocopy.py --batch-sizes 100 500 1000 --dims 256 512 1024 \
        --reps 20 --warmup 3 --runs 3 --output results/exp_a_zerocopy
"""

import argparse
import base64
import csv
import io
import json
import logging
import os
import pickle
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

# Make sibling packages importable when run as a script from the scripts root.
_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

logger = logging.getLogger(__name__)

SEED = 42  # repo convention (matches runner.py / ExperimentConfig)
FLOAT_BYTES = 4


# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully so the baselines run even when a
# component is missing (e.g. the compiled extension or protobuf runtime).
# ---------------------------------------------------------------------------

def _load_context_table():
    """Return the compiled ContextTable class, or None if unavailable."""
    try:
        from cylon_armada.context_table import ContextTable
        return ContextTable
    except Exception as e:  # ImportError, or libcylon_armada.so not found
        logger.warning(
            "cylon_armada.context_table unavailable (%s); skipping arrow_ipc. "
            "Set LD_LIBRARY_PATH and PYTHONPATH per CLAUDE.md to enable it.",
            type(e).__name__,
        )
        return None


def _load_pyarrow():
    try:
        import pyarrow as pa
        return pa
    except Exception as e:
        logger.warning("pyarrow unavailable (%s); skipping arrow_ipc_pyarrow.", type(e).__name__)
        return None


def _load_protobuf():
    try:
        from proto import embedding_batch_pb2 as pb
        return pb
    except Exception as e:
        logger.warning("protobuf pb2 unavailable (%s); skipping protobuf.", type(e).__name__)
        return None


def _load_flatbuffers():
    """Return (flatbuffers, EmbeddingBatch module) or (None, None) if unavailable."""
    try:
        import flatbuffers
        from expa_fbs import EmbeddingBatch as eb_mod
        return flatbuffers, eb_mod
    except Exception as e:
        logger.warning("flatbuffers unavailable (%s); skipping flatbuffers.", type(e).__name__)
        return None, None


# ---------------------------------------------------------------------------
# Codecs. Each codec turns an (N, D) float32 array into wire bytes and back to
# a SIMD-ready (N, D) float32 array, split into the three timed stages.
#
#   prepare(arr)        -> native in-memory form (NOT timed in the round trip;
#                          reported separately as build_ms, since it is a
#                          one-time producer-side cost the framework amortizes)
#   serialize(native)   -> wire bytes
#   deserialize(wire)   -> decoded intermediate
#   to_ndarray(decoded) -> (N, D) float32
#
# `memory_copies` is the documented structural count of full-array copies on
# the deserialize path; `deserialize_peak_kb` (tracemalloc) is the empirical
# backing measurement.
# ---------------------------------------------------------------------------

class Codec:
    name = "base"
    memory_copies = 0
    available = True

    def __init__(self, n, d):
        self.n = n
        self.d = d

    def prepare(self, arr):
        return arr

    def serialize(self, native):
        raise NotImplementedError

    def deserialize(self, wire):
        raise NotImplementedError

    def to_ndarray(self, decoded):
        raise NotImplementedError


class JsonCodec(Codec):
    name = "json"
    memory_copies = 2  # parse to python lists, then np.array materializes

    def serialize(self, native):
        return json.dumps(native.tolist()).encode("utf-8")

    def deserialize(self, wire):
        return json.loads(wire)

    def to_ndarray(self, decoded):
        return np.asarray(decoded, dtype=np.float32).reshape(self.n, self.d)


class PickleCodec(Codec):
    name = "pickle"
    memory_copies = 1  # unpickle reconstructs the ndarray

    def serialize(self, native):
        return pickle.dumps(native, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, wire):
        return pickle.loads(wire)

    def to_ndarray(self, decoded):
        return decoded.reshape(self.n, self.d)


class Base64TobytesCodec(Codec):
    """The framework's current FMI transfer encoding (run_action.py:90)."""
    name = "base64_tobytes"
    memory_copies = 1  # base64 decode allocates raw bytes; frombuffer is a view

    def serialize(self, native):
        return base64.b64encode(native.tobytes())

    def deserialize(self, wire):
        return base64.b64decode(wire)

    def to_ndarray(self, decoded):
        return np.frombuffer(decoded, dtype=np.float32).reshape(self.n, self.d)


class ProtobufCodec(Codec):
    # Python protobuf marshals repeated floats element-by-element, so it is slow
    # for large float arrays. The proposal notes this is a "Python GIL artifact"
    # and proposes a native Go gRPC/HTTP2 sensitivity comparison; that caveat
    # applies to this row's throughput.
    name = "protobuf"
    memory_copies = 2  # repeated-float container, then np.array copy

    def __init__(self, n, d, pb):
        super().__init__(n, d)
        self._pb = pb

    def serialize(self, native):
        msg = self._pb.EmbeddingBatch(n=self.n, dim=self.d, values=native.ravel())
        return msg.SerializeToString()

    def deserialize(self, wire):
        msg = self._pb.EmbeddingBatch()
        msg.ParseFromString(wire)
        return msg

    def to_ndarray(self, decoded):
        return np.asarray(decoded.values, dtype=np.float32).reshape(self.n, self.d)


class FlatBuffersCodec(Codec):
    """Zero-copy binary baseline (FlatBuffers over gRPC). Reads the [float]
    vector directly from the wire buffer without deserialization, isolating
    Arrow's columnar-layout advantage from generic zero-copy binary encoding."""
    name = "flatbuffers"
    memory_copies = 0  # ValuesAsNumpy() is a zero-copy view into the buffer

    def __init__(self, n, d, fb, eb_mod):
        super().__init__(n, d)
        self._fb = fb
        self._eb = eb_mod  # the generated EmbeddingBatch module

    def serialize(self, native):
        eb = self._eb
        builder = self._fb.Builder(self.n * self.d * FLOAT_BYTES + 64)
        vec = builder.CreateNumpyVector(native.ravel())
        eb.EmbeddingBatchStart(builder)
        eb.EmbeddingBatchAddN(builder, self.n)
        eb.EmbeddingBatchAddDim(builder, self.d)
        eb.EmbeddingBatchAddValues(builder, vec)
        builder.Finish(eb.EmbeddingBatchEnd(builder))
        return bytes(builder.Output())

    def deserialize(self, wire):
        return self._eb.EmbeddingBatch.GetRootAs(wire, 0)

    def to_ndarray(self, decoded):
        return decoded.ValuesAsNumpy().reshape(self.n, self.d)


class ArrowIpcCodec(Codec):
    """The proposed zero-copy data plane: an Arrow RecordBatch IPC stream.

    This is what crosses an operator edge in cylon-armada. The ContextTable
    (contexttable_ipc) is a heavier store snapshot evaluated by Experiment J,
    not the data-plane transfer format measured here.
    """
    name = "arrow_ipc"
    memory_copies = 0  # zero-copy view into the IPC buffer

    def __init__(self, n, d, pa):
        super().__init__(n, d)
        self._pa = pa

    def prepare(self, arr):
        pa = self._pa
        fsl = pa.FixedSizeListArray.from_arrays(
            pa.array(arr.ravel(), type=pa.float32()), self.d)
        return pa.record_batch({"embedding": fsl})

    def serialize(self, native):
        pa = self._pa
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, native.schema)
        writer.write_batch(native)
        writer.close()
        return sink.getvalue()

    def deserialize(self, wire):
        reader = self._pa.ipc.open_stream(wire)
        return reader.read_next_batch()

    def to_ndarray(self, decoded):
        return decoded.column(0).values.to_numpy(zero_copy_only=True).reshape(self.n, self.d)


class ContextTableIpcCodec(Codec):
    """ContextTable store snapshot: to_ipc / from_ipc. The full 10-column store
    (context_id, workflow_id, response, tokens, costs, timestamps, reuse_count,
    plus the embedding) transferred zero-copy: to_ipc returns a pyarrow.Buffer
    (pyarrow_wrap_buffer) and from_ipc holds that buffer by reference
    (MakeFromIpcBuffer) with the O(N) context_id index build deferred to first
    keyed access. The measured transfer path is therefore index-free, matching
    how a broadcast/persist actually moves the store; a receiver pays the index
    cost only on the first keyed get/search. At the representative D=1024 this
    lands above pickle and near the bare RecordBatch (arrow_ipc); at small D the
    10-column serialize overhead dominates. The ContextTable is the subject of
    Experiment J."""
    name = "contexttable_ipc"
    memory_copies = 0

    def __init__(self, n, d, ctx_table_cls):
        super().__init__(n, d)
        self._cls = ctx_table_cls

    def prepare(self, arr):
        table = self._cls(embedding_dim=self.d)
        for i in range(self.n):
            table.put(f"c{i}", arr[i], workflow_id="w", response="")
        return table

    def serialize(self, native):
        return native.to_ipc()

    def deserialize(self, wire):
        return self._cls.from_ipc(wire)

    def to_ndarray(self, decoded):
        batch = decoded.to_arrow()
        col = batch.column(batch.schema.get_field_index("embedding"))
        return col.values.to_numpy(zero_copy_only=True).reshape(self.n, self.d)


def build_codecs(n, d):
    """Instantiate every available codec for an (n, d) cell."""
    codecs = [JsonCodec(n, d), PickleCodec(n, d), Base64TobytesCodec(n, d)]

    pb = _load_protobuf()
    if pb is not None:
        codecs.append(ProtobufCodec(n, d, pb))

    fb, eb_mod = _load_flatbuffers()
    if fb is not None:
        codecs.append(FlatBuffersCodec(n, d, fb, eb_mod))

    pa = _load_pyarrow()
    if pa is not None:
        codecs.append(ArrowIpcCodec(n, d, pa))

    ctx_cls = _load_context_table()
    if ctx_cls is not None:
        codecs.append(ContextTableIpcCodec(n, d, ctx_cls))

    return codecs


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _percentile(values, pct):
    """Simple nearest-rank percentile (avoids a scipy/numpy-version dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def measure_cell(codec, arr, warmup, reps):
    """Time one (codec, N, D) cell. Returns a metrics dict or None on failure."""
    n, d = codec.n, codec.d
    payload_bytes = n * d * FLOAT_BYTES

    try:
        t0 = time.perf_counter()
        native = codec.prepare(arr)
        build_ms = (time.perf_counter() - t0) * 1e3

        # Correctness check before timing — a fast benchmark of a wrong
        # round trip is worthless.
        wire0 = codec.serialize(native)
        out0 = codec.to_ndarray(codec.deserialize(wire0))
        if not np.allclose(arr, out0, atol=1e-5):
            logger.error("%s: round-trip mismatch at n=%d d=%d — skipping", codec.name, n, d)
            return None
        wire_bytes = len(wire0)

        for _ in range(warmup):
            codec.to_ndarray(codec.deserialize(codec.serialize(native)))

        ser_ms, deser_ms, access_ms = [], [], []
        for _ in range(reps):
            a = time.perf_counter()
            wire = codec.serialize(native)
            b = time.perf_counter()
            decoded = codec.deserialize(wire)
            c = time.perf_counter()
            _ = codec.to_ndarray(decoded)
            e = time.perf_counter()
            ser_ms.append((b - a) * 1e3)
            deser_ms.append((c - b) * 1e3)
            access_ms.append((e - c) * 1e3)

        # Empirical peak allocation on the deserialize + access path only.
        # Serialize happens OUTSIDE the window so the peak reflects the copies
        # incurred when reading the wire bytes into a usable array, which is the
        # memory_copies claim under test.
        wire_for_peak = codec.serialize(native)
        tracemalloc.start()
        decoded = codec.deserialize(wire_for_peak)
        _ = codec.to_ndarray(decoded)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except Exception as e:
        logger.exception("%s failed at n=%d d=%d: %s", codec.name, n, d, e)
        return None

    ser = statistics.median(ser_ms)
    deser = statistics.median(deser_ms)
    access = statistics.median(access_ms)
    roundtrip = ser + deser + access
    to_mbps = lambda ms: (payload_bytes / (1024 * 1024)) / (ms / 1e3) if ms > 0 else 0.0

    return {
        "format": codec.name,
        "n": n,
        "d": d,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "wire_ratio": round(wire_bytes / payload_bytes, 4),
        "build_ms": round(build_ms, 4),
        "serialize_ms": round(ser, 4),
        "deserialize_ms": round(deser, 4),
        "usable_access_ms": round(access, 6),
        "roundtrip_ms": round(roundtrip, 4),
        "serialize_ms_p95": round(_percentile(ser_ms, 95), 4),
        "deserialize_ms_p95": round(_percentile(deser_ms, 95), 4),
        "roundtrip_ms_p95": round(
            _percentile([s + d_ + a for s, d_, a in zip(ser_ms, deser_ms, access_ms)], 95), 4),
        "throughput_serialize_MBps": round(to_mbps(ser), 2),
        "throughput_deserialize_MBps": round(to_mbps(deser), 2),
        "throughput_roundtrip_MBps": round(to_mbps(roundtrip), 2),
        "memory_copies": codec.memory_copies,
        "deserialize_peak_kb": round(peak / 1024, 2),
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# A2 — Arrow schema compatibility across the five canonical operators
# ---------------------------------------------------------------------------

# Operator = (pattern, schema_in, schema_out) per proposal Table II. Schemas are
# named by their Arrow logical type so edges can be checked for the zero-copy
# condition schema_out(Oi) == schema_in(Oj).
CANONICAL_OPERATORS = [
    # name, pattern, schema_in, schema_out
    ("Preprocess", "SCATTER", "raw_text:large_utf8", "chunked_text:large_utf8"),
    ("Embed", "SCATTER-GATHER", "chunked_text:large_utf8", "embedding:fixed_size_list<float32>"),
    ("Retrieve", "REDUCE", "query_embedding:fixed_size_list<float32>", "ranked_docs:struct"),
    ("Reason", "POINT_TO_POINT", "context:struct", "response:large_utf8"),
    ("MemoryUpsert", "BROADCAST", "kv_pairs:struct", "ack:bool"),
]

# The pipeline DAG edges under test (producer -> consumer).
PIPELINE_EDGES = [
    ("Preprocess", "Embed"),
    ("Embed", "Retrieve"),
    ("Retrieve", "Reason"),
    ("Reason", "MemoryUpsert"),
]

# A schema is "dense zero-copy friendly" if it is a fixed-size list of a
# primitive — the case where Arrow's advantage is strongest.
_DENSE = "fixed_size_list"
_VARIABLE = "large_utf8"


def schema_compatibility():
    """Return the A2 edge-compatibility rows for the pipeline DAG."""
    out_by_op = {name: sout for name, _, _, sout in CANONICAL_OPERATORS}
    in_by_op = {name: sin for name, _, sin, _ in CANONICAL_OPERATORS}

    rows = []
    for producer, consumer in PIPELINE_EDGES:
        s_out = out_by_op[producer].split(":", 1)[1]
        s_in = in_by_op[consumer].split(":", 1)[1]
        compatible = s_out == s_in
        payload_class = (
            "dense" if _DENSE in s_out else "variable" if _VARIABLE in s_out else "nested"
        )
        rows.append({
            "edge": f"{producer}->{consumer}",
            "schema_out": s_out,
            "schema_in": s_in,
            "arrow_compatible": compatible,
            "zero_copy_eligible": compatible and _DENSE in s_out,
            "payload_class": payload_class,
        })
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _rows_to_csv(rows):
    """Serialize a list[dict] (uniform keys) to CSV text. Empty -> ''."""
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _env_rows(meta=None):
    """Machine/environment provenance as CSV rows (one key,value pair per row).

    Captures the hardware the microbenchmark ran on, which is essential context
    for a serialization benchmark. Kept as CSV for consistency with every other
    artifact this experiment emits.
    """
    info = {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if meta:
        info.update(meta)
    return [{"key": k, "value": v} for k, v in info.items()]


def _add_speedup_vs_json(rows):
    """Annotate each row with speedup_vs_json within its (n, d) cell."""
    json_rt = {
        (r["n"], r["d"]): r["roundtrip_ms"]
        for r in rows if r["format"] == "json"
    }
    for r in rows:
        base = json_rt.get((r["n"], r["d"]))
        r["speedup_vs_json"] = round(base / r["roundtrip_ms"], 3) if base and r["roundtrip_ms"] > 0 else None


def _measure_matrix(config):
    """Run the throughput matrix. Returns list[dict] metric rows (internal)."""
    batch_sizes = config.get("batch_sizes", [100, 500, 1000, 5000])
    dims = config.get("dims", [256, 512, 1024])
    warmup = config.get("warmup", 3)
    reps = config.get("reps", 20)
    seed = config.get("seed", SEED)
    rng = np.random.default_rng(seed)

    rows = []
    for d in dims:
        for n in batch_sizes:
            arr = rng.standard_normal((n, d), dtype=np.float32)
            for codec in build_codecs(n, d):
                row = measure_cell(codec, arr, warmup, reps)
                if row is not None:
                    rows.append(row)
                    logger.info(
                        "%-18s n=%-5d d=%-5d  roundtrip=%8.3fms  %10.1f MB/s  copies=%d",
                        row["format"], n, d, row["roundtrip_ms"],
                        row["throughput_roundtrip_MBps"], row["memory_copies"],
                    )

    _add_speedup_vs_json(rows)
    return rows


def run(config):
    """Run Experiment A / A2 and return results as CSV text.

    Returns a dict mapping filename -> CSV text, so the local path and the Lambda
    handler persist identical artifacts (write to disk vs upload to S3). Every
    result the experiment produces is CSV, consistent with the rest of the
    framework. Pure computation: no file or network I/O.

    config keys: batch_sizes, dims, warmup, reps, seed, meta.
    """
    rows = _measure_matrix(config)
    a2_rows = schema_compatibility()
    logger.info("A2 schema compatibility: %d/%d edges zero-copy-eligible",
                sum(r["zero_copy_eligible"] for r in a2_rows), len(a2_rows))

    return {
        "exp_a_zerocopy_results.csv": _rows_to_csv(rows),
        "exp_a2_schema_compat.csv": _rows_to_csv(a2_rows),
        "exp_a_zerocopy_env.csv": _rows_to_csv(_env_rows(config.get("meta"))),
    }


def persist(csv_map, output_dir, s3_bucket=None, s3_prefix=None):
    """Write each CSV in the map to disk and, when configured, upload to S3."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for filename, text in csv_map.items():
        if not text:
            continue
        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="") as f:
            f.write(text)
        written.append(path)

    if s3_bucket:
        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = (s3_prefix or "experiments").rstrip("/")
            for path in written:
                key = f"{prefix}/exp_a_zerocopy/{os.path.basename(path)}"
                s3.upload_file(path, s3_bucket, key)
                logger.info("Uploaded s3://%s/%s", s3_bucket, key)
        except Exception as e:
            logger.warning("S3 upload skipped: %s", e)

    return written


def main():
    parser = argparse.ArgumentParser(description="Experiment A / A2 zero-copy data plane benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 500, 1000, 5000])
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1, help="Repeat the whole matrix; run_{n}/ subdirs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="results/exp_a_zerocopy")
    parser.add_argument("--s3-bucket", type=str, default=os.environ.get("S3_RESULTS_BUCKET"))
    parser.add_argument("--s3-prefix", type=str, default=os.environ.get("S3_RESULTS_PREFIX", "experiments"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for run_idx in range(1, args.runs + 1):
        config = {
            "batch_sizes": args.batch_sizes, "dims": args.dims,
            "warmup": args.warmup, "reps": args.reps, "seed": args.seed + run_idx - 1,
            "meta": {"run": run_idx, "warmup": args.warmup, "reps": args.reps},
        }
        csv_map = run(config)
        out_dir = args.output if args.runs == 1 else os.path.join(args.output, f"run_{run_idx}")
        written = persist(csv_map, out_dir, s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix)
        logger.info("run %d: wrote %d CSV(s) to %s", run_idx, len(written), out_dir)


if __name__ == "__main__":
    main()