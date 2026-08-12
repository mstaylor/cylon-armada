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

"""HTTP round-trip baseline for the collectives — the Experiment B H1 comparator.

Models stateless-HTTP orchestration (the LangChain-style path): every collective is
a round-trip to a central coordinator that fans in all workers' contributions and
fans out the result, over JSON — no Arrow zero-copy. This is the load-bearing
comparison for the ">=90% latency reduction vs HTTP" claim; it deliberately pays the
serialize + HTTP round-trip cost the Cylon collectives avoid.

The coordinator (`Coordinator` + `make_server`) buffers per (collective, round) until
all `world_size` workers arrive, computes the result, and returns each worker its
slice. The `HTTPCollective` client exposes the same collective set as the Cylon
communicator, and `run(config)` sweeps them producing `measure_collective`-shaped
rows so the charts overlay HTTP against Cylon.

Both the coordinator and the client are plain stdlib (http.server + urllib), so this
runs multi-process for real N-sweeps and multi-thread in tests.
"""

import json
import statistics
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # reuse the percentile helper when importable as a package
    from experiment.exp_b_collectives import _percentile
except Exception:  # pragma: no cover - fallback for standalone import
    def _percentile(values, pct):
        if not values:
            return 0.0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return ordered[k]


# Collectives moving data (JSON float lists) whose payload scales with msg_size.
_SIZE_SWEPT = {"scatter", "scatterv", "gather", "allgather", "broadcast"}
_SCALAR_COLLECTIVES = {"reduce", "allreduce"}
SCALAR_BYTES = 8
DEFAULT_COLLECTIVES = [
    "scatter", "scatterv", "gather", "allgather", "reduce", "broadcast", "allreduce", "barrier",
]
DEFAULT_MSG_SIZES = [8, 64, 512, 4096, 32768, 262144, 1048576]


class Coordinator:
    """Central fan-in / fan-out state for one HTTP collective benchmark.

    Thread-safe: each worker request calls `contribute`, which blocks on a condition
    until all `world_size` contributions for that (collective, round) arrive, then
    every waiter is released with its slice of the computed result.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self._cv = threading.Condition()
        self._buffers = {}   # (collective, round) -> {rank: data}
        self._results = {}   # (collective, round) -> full result

    def contribute(self, collective, rnd, rank, root, data):
        key = (collective, rnd)
        with self._cv:
            buf = self._buffers.setdefault(key, {})
            buf[rank] = data
            if len(buf) == self.world_size:
                self._results[key] = self._compute(collective, root, buf)
                self._cv.notify_all()
            else:
                while key not in self._results:
                    self._cv.wait()
            return self._slice(collective, rank, root, self._results[key])

    @staticmethod
    def _compute(collective, root, buf):
        if collective == "barrier":
            return None
        if collective in ("allreduce", "reduce"):
            # element-wise sum across ranks
            n = len(next(iter(buf.values())))
            return [sum(buf[r][i] for r in buf) for i in range(n)]
        if collective == "broadcast":
            return buf[root]
        if collective in ("gather", "allgather"):
            return [buf[r] for r in sorted(buf)]
        if collective in ("scatter", "scatterv"):
            # root supplies a list of per-rank shards
            return buf[root]
        raise ValueError(f"unknown collective '{collective}'")

    @staticmethod
    def _slice(collective, rank, root, result):
        if collective == "barrier":
            return None
        if collective == "reduce":
            return result if rank == root else None
        if collective == "gather":
            return result if rank == root else None
        if collective in ("scatter", "scatterv"):
            return result[rank]
        return result  # allreduce / broadcast / allgather deliver to all


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        result = self.server.coordinator.contribute(
            req["collective"], req["round"], req["rank"], req.get("root", 0), req.get("data"),
        )
        body = json.dumps({"result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


def make_server(world_size, host="127.0.0.1", port=0):
    """Create (but do not start) a threaded coordinator server. port=0 picks a free port."""
    server = ThreadingHTTPServer((host, port), _Handler)
    server.coordinator = Coordinator(world_size)
    return server


class HTTPCollective:
    """Client with the Cylon collective set, each a JSON round-trip to the coordinator."""

    def __init__(self, base_url, rank, world_size, root=0, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.rank = rank
        self.world_size = world_size
        self.root = root
        self.timeout = timeout
        self._rounds = {}  # collective -> next round number

    def _post(self, collective, data, root=None):
        rnd = self._rounds.get(collective, 0)
        self._rounds[collective] = rnd + 1
        payload = json.dumps({
            "collective": collective, "round": rnd, "rank": self.rank,
            "root": self.root if root is None else root, "data": data,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/collective", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["result"]

    def barrier(self):
        return self._post("barrier", None)

    def allreduce(self, vec):
        return self._post("allreduce", vec)

    def reduce(self, vec, root=0):
        return self._post("reduce", vec, root=root)

    def broadcast(self, vec, root=0):
        return self._post("broadcast", vec, root=root)

    def gather(self, vec, root=0):
        return self._post("gather", vec, root=root)

    def allgather(self, vec):
        return self._post("allgather", vec)

    def scatter(self, shards, root=0):
        # non-root ranks send [] (ignored); root sends world_size shards
        return self._post("scatter", shards, root=root)

    def scatterv(self, shards, root=0):
        return self._post("scatterv", shards, root=root)


def _invoke(client, name, payload, root):
    if name == "barrier":
        return client.barrier()
    if name == "allreduce":
        return client.allreduce(payload)
    if name == "reduce":
        return client.reduce(payload, root)
    if name == "broadcast":
        return client.broadcast(payload, root)
    if name == "gather":
        return client.gather(payload, root)
    if name == "allgather":
        return client.allgather(payload)
    if name == "scatter":
        return client.scatter(payload, root)
    if name == "scatterv":
        return client.scatterv(payload, root)
    raise ValueError(name)


def _build_payload(name, msg_size, world_size, rank, root):
    """JSON float-list payload for one cell (mirrors the Cylon builder's shapes)."""
    n_elems = max(1, msg_size // 8)  # 8 bytes per double, the JSON T_stateless model
    if name == "barrier":
        return (None, 0)
    if name in ("reduce", "allreduce"):
        return ([float(rank + 1)], SCALAR_BYTES)
    if name in ("broadcast", "gather", "allgather"):
        return ([float(rank + 1)] * n_elems, msg_size)
    if name == "scatter":
        return ([[float(i + 1)] * n_elems for i in range(world_size)] if rank == root else [], msg_size)
    if name == "scatterv":
        base = max(1, n_elems // max(1, world_size))
        if rank == root:
            return ([[float(i + 1)] * (base * (i + 1)) for i in range(world_size)], base * (rank + 1) * 8)
        return ([], base * (rank + 1) * 8)
    raise ValueError(name)


def measure(client, name, msg_size, warmup, reps, root=0, world_size=1, rank=0):
    """Time one collective over the HTTP coordinator; returns a metrics row."""
    payload, payload_bytes = _build_payload(name, msg_size, world_size, client.rank, root)

    for _ in range(warmup):
        _invoke(client, name, payload, root)

    lat_ms = []
    for _ in range(reps):
        if name != "barrier":
            client.barrier()
        t0 = time.perf_counter()
        _invoke(client, name, payload, root)
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    p50 = statistics.median(lat_ms)
    median_s = p50 / 1e3
    is_data = name != "barrier"
    tput = (payload_bytes / (1024 * 1024)) / median_s if (is_data and median_s > 0 and payload_bytes > 0) else 0.0
    return {
        "baseline": "http",
        "collective": name,
        "msg_size": msg_size,
        "payload_bytes": payload_bytes,
        "latency_p50_ms": round(p50, 6),
        "latency_p99_ms": round(_percentile(lat_ms, 99), 6),
        "latency_mean_ms": round(statistics.fmean(lat_ms), 6),
        "throughput_MBps": round(tput, 4),
        "barrier_latency_ms": round(p50, 6) if name == "barrier" else 0.0,
        "reps": reps,
    }


def run(config):
    """Sweep collectives × msg_sizes for one worker; returns metric rows.

    config: client, collectives, msg_sizes, warmup, reps, root, rank, world_size, run_id.
    """
    client = config["client"]
    collectives = config.get("collectives", DEFAULT_COLLECTIVES)
    msg_sizes = config.get("msg_sizes", DEFAULT_MSG_SIZES)
    warmup = config.get("warmup", 3)
    reps = config.get("reps", 20)
    root = config.get("root", 0)
    rank = config.get("rank", client.rank)
    world_size = config.get("world_size", client.world_size)
    run_id = config.get("run_id", 1)

    rows = []
    for name in collectives:
        if name in _SIZE_SWEPT:
            sizes = msg_sizes
        elif name in _SCALAR_COLLECTIVES:
            sizes = [SCALAR_BYTES]
        else:
            sizes = [0]
        for msg_size in sizes:
            row = measure(client, name, msg_size, warmup, reps, root=root, world_size=world_size, rank=rank)
            row.update({"rank": rank, "world_size": world_size, "N": world_size, "run_id": run_id})
            rows.append(row)
    return rows