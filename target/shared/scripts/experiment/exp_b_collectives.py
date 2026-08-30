# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment B — per-rank collective-communication benchmark (C++/pycylon runtime).

Measures per-collective latency (P50/P99) and throughput for the collectives the
five agentic operators map onto — scatter / scatterv / gather / allgather / reduce
/ broadcast / allreduce / barrier — over a message-size grid, on one FMI/UCC
channel. Each rank runs the same sweep; rank 0 writes the per-run CSV.

Design for testability: the timing core (`measure_collective`, `run`) is decoupled
from Arrow/pycylon. The collective is invoked through `_invoke`, and payloads are
built by a `payload_builder` callable that defaults to the pycylon implementation
(`build_cylon_payload`) but can be injected in tests with a mock communicator. Only
the payload builder and `main()`'s communicator construction import pycylon, so the
timing logic is unit-testable without the native stack.

See docs/EXP_B_DESIGN.md and docs/superpowers/plans/2026-08-09-expB-sp4-collective-benchmark.md.
"""

# Disable mpi4py's auto MPI_Init-on-import and atexit MPI_Finalize BEFORE anything
# can import mpi4py.MPI (pycylon does, transitively). Otherwise MPI comes up under the
# UCC/redis-OOB path and Cylon's UCXUCCCommunicator::Finalize issues an MPI_Barrier
# that races mpi4py's own MPI_Finalize at exit -> SIGSEGV teardown artifact. This is
# the same guard cylon's scaling.py uses; it must run at module top, pre-pycylon.
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

import argparse
import logging
import os
import statistics
import time

logger = logging.getLogger("exp_b_collectives")

# The collectives Experiment B measures, in operator-relevant order. `barrier`
# carries no payload (pure synchronization); the rest move `msg_size` bytes.
DEFAULT_COLLECTIVES = [
    "scatter",
    "scatterv",
    "gather",
    "allgather",
    "reduce",
    "broadcast",
    "allreduce",
    "barrier",
]

# Message-size grid (bytes): powers of two 8 B – 1 MB, the comm_microbenchmark grid.
DEFAULT_MSG_SIZES = [8, 64, 512, 4096, 32768, 262144, 1048576]

# Collectives that move data (throughput is meaningful); barrier does not.
_DATA_MOVING = {
    "scatter",
    "scatterv",
    "gather",
    "allgather",
    "reduce",
    "broadcast",
    "allreduce",
}

# Table-moving collectives whose payload scales with the message-size grid.
_SIZE_SWEPT = {"scatter", "scatterv", "gather", "allgather", "broadcast"}

# SP1's reduce/allreduce operate on a single Arrow scalar, not a column, so they do
# not sweep the size grid — one fixed-size measurement each.
_SCALAR_COLLECTIVES = {"reduce", "allreduce"}
SCALAR_BYTES = 8


def _percentile(values, pct):
    """Simple nearest-rank percentile (avoids a scipy/numpy-version dependency).

    Copied from exp_a_zerocopy.py to keep this module import-light.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


class UnsupportedCollective(Exception):
    """Raised when a (channel, collective) pair is a known stub / unavailable.

    Carries enough context to record an honest `unsupported` row rather than a
    fabricated timing (e.g. redis-channel gatherv/allgather are empty stubs).
    """


def _invoke(comm, ctx, name, payload, root, reduce_op):
    """Invoke one collective on the communicator with its prebuilt payload.

    The pycylon SP1 collective API is not uniform (scatter takes a list of tables +
    context, reduce takes an op + root, barrier is on the context), so this is the
    single place that maps a collective name to its call. `payload` is whatever
    `payload_builder` produced for this collective; `reduce_op` is the pycylon
    `ReduceOp` (only read by reduce/allreduce).
    """
    if name == "barrier":
        return ctx.barrier()
    if name == "broadcast":
        return comm.broadcast(payload, root, ctx)
    if name == "scatter":
        return comm.scatter(payload, root, ctx)
    if name == "scatterv":
        # SP1 exposes a single scatter entry point; the uneven case differs only in
        # the shard sizes the payload builder produced, so it dispatches the same.
        return comm.scatter(payload, root, ctx)
    if name == "gather":
        return comm.gather(payload, root)
    if name == "allgather":
        return comm.allgather(payload)
    if name == "reduce":
        return comm.reduce(payload, reduce_op, root)
    if name == "allreduce":
        return comm.allreduce(payload, reduce_op)
    raise ValueError(f"unknown collective '{name}'")


def measure_collective(
    comm,
    ctx,
    name,
    msg_size,
    warmup,
    reps,
    root=0,
    reduce_op=None,
    payload_builder=None,
    world_size=1,
    rank=0,
):
    """Time one collective at one message size. Returns a metrics dict.

    `warmup` untimed reps prime the path (the SP1 lib caches, JITs the transport),
    then `reps` timed reps are measured, each preceded by a barrier so ranks enter
    together and slow-rank skew does not leak into a fast rank's timing. Central
    latency is the median (P50); spread is reported as P99 and mean. Throughput uses
    the median so a single GC/scheduler outlier rep cannot distort it.

    `payload_builder(name, msg_size, ctx, world_size, rank, root)` returns
    `(payload, payload_bytes)` or `None` for an unsupported (channel, collective)
    pair. Raising `UnsupportedCollective` mid-measure is also honored. Either way the
    caller records an `unsupported` row rather than a fabricated timing.
    """
    if payload_builder is None:
        payload_builder = build_cylon_payload

    built = payload_builder(name, msg_size, ctx, world_size, rank, root)
    if built is None:
        return _unsupported_metrics(name, msg_size)
    payload, payload_bytes = built

    try:
        for _ in range(warmup):
            _invoke(comm, ctx, name, payload, root, reduce_op)

        lat_ms = []
        for _ in range(reps):
            if name != "barrier":
                # Enter the timed region together; for barrier itself the invoke IS
                # the synchronization, so an extra barrier would double-count.
                ctx.barrier()
            t0 = time.perf_counter()
            _invoke(comm, ctx, name, payload, root, reduce_op)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
    except UnsupportedCollective:
        return _unsupported_metrics(name, msg_size)

    p50 = statistics.median(lat_ms)
    p99 = _percentile(lat_ms, 99)
    mean = statistics.fmean(lat_ms)
    median_s = p50 / 1e3
    is_data = name in _DATA_MOVING
    throughput = (
        (payload_bytes / (1024 * 1024)) / median_s
        if (is_data and median_s > 0 and payload_bytes > 0)
        else 0.0
    )

    return {
        "collective": name,
        "msg_size": msg_size,
        "payload_bytes": payload_bytes,
        "latency_p50_ms": round(p50, 6),
        "latency_p99_ms": round(p99, 6),
        "latency_mean_ms": round(mean, 6),
        "throughput_MBps": round(throughput, 4),
        "barrier_latency_ms": round(p50, 6) if name == "barrier" else 0.0,
        "reps": reps,
        "unsupported": False,
    }


def _unsupported_metrics(name, msg_size):
    """A recorded N/A cell — honest placeholder for a stub/unavailable collective."""
    return {
        "collective": name,
        "msg_size": msg_size,
        "payload_bytes": 0,
        "latency_p50_ms": 0.0,
        "latency_p99_ms": 0.0,
        "latency_mean_ms": 0.0,
        "throughput_MBps": 0.0,
        "barrier_latency_ms": 0.0,
        "reps": 0,
        "unsupported": True,
    }


def run(config):
    """Run the collective sweep on this rank and return a list of metric rows.

    config keys:
      comm, ctx        - the pycylon communicator + context (or mocks in tests)
      collectives      - list of collective names (default DEFAULT_COLLECTIVES)
      msg_sizes        - list of byte sizes (default DEFAULT_MSG_SIZES)
      warmup, reps     - timing knobs
      root             - collective root rank (default 0)
      channel          - channel label for the rows (e.g. 'ucc', 'fmi-direct')
      rank, world_size - this rank's identity
      run_id           - which warmed run this is (for cross-run std later)
      reduce_op        - pycylon ReduceOp for reduce/allreduce
      payload_builder  - override the payload builder (tests inject a mock)

    Pure per-rank computation: no file I/O (the launcher persists rank-0 rows).
    """
    comm = config["comm"]
    ctx = config["ctx"]
    collectives = config.get("collectives", DEFAULT_COLLECTIVES)
    msg_sizes = config.get("msg_sizes", DEFAULT_MSG_SIZES)
    warmup = config.get("warmup", 3)
    reps = config.get("reps", 20)
    root = config.get("root", 0)
    channel = config.get("channel", "unknown")
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)
    run_id = config.get("run_id", 1)
    reduce_op = config.get("reduce_op")
    payload_builder = config.get("payload_builder")

    rows = []
    for name in collectives:
        if name in _SIZE_SWEPT:
            sizes = msg_sizes
        elif name in _SCALAR_COLLECTIVES:
            sizes = [SCALAR_BYTES]
        else:  # barrier — pure synchronization, no payload
            sizes = [0]
        for msg_size in sizes:
            metrics = measure_collective(
                comm,
                ctx,
                name,
                msg_size,
                warmup,
                reps,
                root=root,
                reduce_op=reduce_op,
                payload_builder=payload_builder,
                world_size=world_size,
                rank=rank,
            )
            metrics.update(
                {
                    "channel": channel,
                    "rank": rank,
                    "world_size": world_size,
                    "N": world_size,
                    "run_id": run_id,
                }
            )
            rows.append(metrics)
            if metrics["unsupported"]:
                logger.info("collective %s @ %dB: unsupported on %s (N/A)", name, msg_size, channel)
            else:
                logger.info(
                    "collective %s @ %dB: p50=%.4fms p99=%.4fms tput=%.2fMB/s",
                    name,
                    msg_size,
                    metrics["latency_p50_ms"],
                    metrics["latency_p99_ms"],
                    metrics["throughput_MBps"],
                )
    return rows


def build_cylon_payload(name, msg_size, ctx, world_size, rank, root):
    """Build the pycylon payload for one collective/message-size cell.

    Returns `(payload, payload_bytes)` or `None` if the collective is unsupported on
    this build. Imports pycylon/pyarrow lazily so the timing core stays testable
    without the native stack. `payload_bytes` is the per-rank message size used for
    throughput (msg_size), independent of the table's exact on-wire footprint.
    """
    import numpy as np
    import pyarrow as pa
    from pycylon import Table

    # float32 column sized so N*4 == msg_size (min one element for the tiny sizes).
    n_elems = max(1, msg_size // 4)

    def _table(fill):
        arr = np.full(n_elems, fill, dtype=np.float32)
        return Table.from_arrow(ctx, pa.table({"v": pa.array(arr)}))

    if name == "barrier":
        return (None, 0)

    if name in ("reduce", "allreduce"):
        # SP1's reduce/allreduce reduce a single Arrow scalar (root-delivering /
        # all-delivering respectively), not a column — one 8-byte value per rank.
        return (float(rank + 1), SCALAR_BYTES)

    if name in ("broadcast", "gather", "allgather"):
        return (_table(float(rank + 1)), msg_size)

    if name == "scatter":
        # Root supplies world_size equal shards; non-root ranks supply an empty list.
        if rank == root:
            return ([_table(float(i + 1)) for i in range(world_size)], msg_size)
        return ([], msg_size)

    if name == "scatterv":
        # Uneven shards: shard i has (i+1) base elements. Throughput uses this rank's
        # own shard size for a fair per-rank number.
        base = max(1, n_elems // max(1, world_size))
        if rank == root:
            shards = []
            for i in range(world_size):
                arr = np.full(base * (i + 1), float(i + 1), dtype=np.float32)
                shards.append(Table.from_arrow(ctx, pa.table({"v": pa.array(arr)})))
            return (shards, base * (rank + 1) * 4)
        return ([], base * (rank + 1) * 4)

    return None


# Stable CSV column order for the per-run results.
_CSV_FIELDS = [
    "channel", "collective", "msg_size", "payload_bytes",
    "latency_p50_ms", "latency_p99_ms", "latency_mean_ms",
    "throughput_MBps", "barrier_latency_ms", "reps", "unsupported",
    "rank", "world_size", "N", "run_id",
]


def _rows_to_csv(rows):
    """Render metric rows to CSV text with a stable header (empty string if no rows)."""
    import csv
    import io

    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def build_env(channel, world_size, redis_addr, comm_name):
    """Construct a distributed CylonEnv for the given channel.

    UCC uses the redis-OOB path validated in SP1 (redis INCR assigns ranks, so no
    per-process rank env is needed). FMI redis/direct build an FMIConfig; the FMI
    path is wired from the SP1 bindings but its multi-rank launch is validated
    separately (it needs a rendezvous server for `fmi-direct`). Imports pycylon
    lazily so the module stays importable without the native stack.
    """
    from pycylon.frame import CylonEnv

    if channel == "ucc":
        from pycylon.net.ucc_config import UCCConfig
        from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext

        oob = UCCRedisOOBContext(world_size, f"tcp://{redis_addr}")
        cfg = UCCConfig(oob)
        return CylonEnv(config=cfg, distributed=True)

    if channel in ("fmi-redis", "fmi-direct"):
        from pycylon.net.fmi_config import FMIConfig

        channel_type = "redis" if channel == "fmi-redis" else "direct"
        redis_host, redis_port = redis_addr.split(":")
        rank = int(os.environ.get("RANK", "0"))
        fmi_cfg = FMIConfig(
            rank,
            world_size,
            os.environ.get("TCPUNCH_HOST", "127.0.0.1"),
            int(os.environ.get("TCPUNCH_PORT", "10000")),
            int(os.environ.get("FMI_MAX_TIMEOUT", "60000")),
            True,  # resolveip
            comm_name,
            True,  # nonblocking
            redis_host,
            int(redis_port),
            os.environ.get("FMI_REDIS_NAMESPACE", "cylon_armada"),
            channel_type=channel_type,
        )
        return CylonEnv(config=fmi_cfg, distributed=True)

    if channel == "fmi-direct-redis":
        from pycylon.net.fmi_config import FMIConfig

        redis_host, redis_port = redis_addr.split(":")
        rank = int(os.environ.get("RANK", "0"))
        fmi_cfg = FMIConfig(
            rank,
            world_size,
            "",  # host unused by direct-redis — see advertise_host below
            int(os.environ.get("FMI_LISTEN_PORT", "50055")),
            int(os.environ.get("FMI_MAX_TIMEOUT", "60000")),
            False,  # resolveip — no host to resolve
            comm_name,
            True,  # nonblocking
            redis_host,
            int(redis_port),
            os.environ.get("FMI_REDIS_NAMESPACE", "cylon_armada"),
            channel_type="direct-redis",
            advertise_host=os.environ.get("ADVERTISE_HOST", ""),
        )
        return CylonEnv(config=fmi_cfg, distributed=True)

    raise ValueError(f"unknown channel '{channel}': expected ucc|fmi-redis|fmi-direct|fmi-direct-redis")


def main():
    parser = argparse.ArgumentParser(description="Experiment B collective benchmark (per rank)")
    parser.add_argument("--channel", default="ucc",
                        choices=["ucc", "fmi-redis", "fmi-direct", "fmi-direct-redis"])
    parser.add_argument("--collectives", nargs="+", default=DEFAULT_COLLECTIVES)
    parser.add_argument("--msg-sizes", type=int, nargs="+", default=DEFAULT_MSG_SIZES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1, help="Warmed runs; each writes run_{n}/ CSV")
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument(
        "--world-size", type=int,
        default=int(os.environ.get("CYLON_UCX_OOB_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))),
    )
    parser.add_argument(
        "--redis-addr",
        default=os.environ.get("CYLON_UCX_OOB_REDIS_ADDR", os.environ.get("REDIS_ADDR", "127.0.0.1:6379")),
    )
    parser.add_argument("--output", default="results/exp_b_collectives")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from pycylon.net.reduce_op import ReduceOp

    # Base session id from the launcher (shared across this run's rank processes). Each
    # run derives a distinct CYLON_SESSION_ID from it so the redis OOB INCR counter does
    # not carry over between runs (which makes ranks accumulate). The derivation is
    # deterministic, so all rank processes compute the same id for a given run and still
    # rendezvous.
    base_session = os.environ.get("CYLON_SESSION_ID", f"exp_b_{os.getpid()}")

    for run_idx in range(1, args.runs + 1):
        os.environ["CYLON_SESSION_ID"] = f"{base_session}_run{run_idx}"
        # comm_name must also be unique per run so redis INCR rank assignment does not
        # collide across concurrent/sequential runs (the CLAUDE.md rank-reuse rule).
        comm_name = f"exp_b_{args.channel}_ws{args.world_size}_run{run_idx}"
        env = build_env(args.channel, args.world_size, args.redis_addr, comm_name)
        ctx = env.context
        comm = ctx.get_communicator()
        rank = env.rank
        world_size = env.world_size if hasattr(env, "world_size") else args.world_size

        config = {
            "comm": comm,
            "ctx": ctx,
            "collectives": args.collectives,
            "msg_sizes": args.msg_sizes,
            "warmup": args.warmup,
            "reps": args.reps,
            "root": args.root,
            "channel": args.channel,
            "rank": rank,
            "world_size": world_size,
            "run_id": run_idx,
            "reduce_op": ReduceOp.SUM,
        }
        rows = run(config)
        env.barrier()

        # Rank 0 persists this run's rows. Multi-run writes flat, per-run-prefixed
        # CSVs (run{n}_exp_b_collectives_results.csv) so aggregate_collectives_runs
        # can glob them and compute the cross-run std; a single run writes the
        # canonical name directly.
        if rank == args.root:
            os.makedirs(args.output, exist_ok=True)
            fname = ("exp_b_collectives_results.csv" if args.runs == 1
                     else f"run{run_idx}_exp_b_collectives_results.csv")
            path = os.path.join(args.output, fname)
            with open(path, "w", newline="") as f:
                f.write(_rows_to_csv(rows))
            logger.info("run %d: wrote %d rows to %s", run_idx, len(rows), path)

        # Clean teardown: the Cylon C++ fix (UCXCommunicator::Finalize skips the MPI
        # barrier/finalize on the redis-OOB path, where there is no valid MPI comm)
        # makes this safe — it no longer segfaults/aborts. The mpi4py guard above keeps
        # MPI out of the pure-UCC path entirely.
        env.finalize()

        config["comm"] = None
        config["ctx"] = None
        comm = None
        ctx = None
        del env


if __name__ == "__main__":
    main()