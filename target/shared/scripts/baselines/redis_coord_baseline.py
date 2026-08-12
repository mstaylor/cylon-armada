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

"""Redis-coordination baseline for the collectives — the Experiment B zero-copy isolator.

Every collective coordinates through redis instead of Arrow zero-copy: each worker
writes its JSON-serialized contribution to a per-(collective, round) redis hash,
synchronizes with an INCR barrier, then reads the hash back and computes its slice.
Data therefore round-trips through storage and is serialized both ways — this is what
isolates the contribution of the zero-copy Arrow data plane (vs. storage-mediated
coordination) in the H1 comparison. It reuses the HTTP coordinator's collective math
(`Coordinator._compute`/`_slice`) so both baselines agree on semantics.

`run(config)` produces `measure_collective`-shaped rows so the charts overlay redis
against Cylon and HTTP.
"""

import json
import statistics
import time

try:
    from experiment.exp_b_collectives import _percentile
except Exception:  # pragma: no cover
    def _percentile(values, pct):
        if not values:
            return 0.0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return ordered[k]

from baselines.http_collective_baseline import (
    Coordinator,
    _build_payload,
    DEFAULT_COLLECTIVES,
    DEFAULT_MSG_SIZES,
    SCALAR_BYTES,
    _SIZE_SWEPT,
    _SCALAR_COLLECTIVES,
)

_KEY_TTL = 120  # seconds; stale coordination keys self-expire


class RedisCollective:
    """Client exposing the Cylon collective set over redis coordination (no zero-copy)."""

    def __init__(self, redis_client, rank, world_size, namespace, root=0, poll_s=0.0005):
        self.r = redis_client
        self.rank = rank
        self.world_size = world_size
        self.ns = namespace
        self.root = root
        self._poll = poll_s
        self._rounds = {}  # collective -> next round number

    def _sync(self, key):
        """INCR barrier: block until all world_size workers have arrived at `key`."""
        cnt_key = f"{key}:cnt"
        self.r.incr(cnt_key)
        self.r.expire(cnt_key, _KEY_TTL)
        while int(self.r.get(cnt_key) or 0) < self.world_size:
            time.sleep(self._poll)

    def collective(self, name, data, root=None):
        root = self.root if root is None else root
        rnd = self._rounds.get(name, 0)
        self._rounds[name] = rnd + 1
        key = f"{self.ns}:{name}:{rnd}"

        if name != "barrier" and data is not None:
            self.r.hset(key, self.rank, json.dumps(data))
            self.r.expire(key, _KEY_TTL)

        self._sync(key)

        if name == "barrier":
            return None

        raw = self.r.hgetall(key)  # {rank(bytes): json(bytes)}
        contrib = {int(k): json.loads(v) for k, v in raw.items()}
        result = Coordinator._compute(name, root, contrib)
        return Coordinator._slice(name, self.rank, root, result)

    # Named methods for parity with the other clients / _invoke below.
    def barrier(self):
        return self.collective("barrier", None)

    def allreduce(self, vec):
        return self.collective("allreduce", vec)

    def reduce(self, vec, root=0):
        return self.collective("reduce", vec, root=root)

    def broadcast(self, vec, root=0):
        return self.collective("broadcast", vec, root=root)

    def gather(self, vec, root=0):
        return self.collective("gather", vec, root=root)

    def allgather(self, vec):
        return self.collective("allgather", vec)

    def scatter(self, shards, root=0):
        return self.collective("scatter", shards, root=root)

    def scatterv(self, shards, root=0):
        return self.collective("scatterv", shards, root=root)


def measure(client, name, msg_size, warmup, reps, root=0, world_size=1, rank=0):
    """Time one collective over redis coordination; returns a metrics row."""
    payload, payload_bytes = _build_payload(name, msg_size, world_size, client.rank, root)

    for _ in range(warmup):
        client.collective(name, payload, root=root)

    lat_ms = []
    for _ in range(reps):
        if name != "barrier":
            client.barrier()
        t0 = time.perf_counter()
        client.collective(name, payload, root=root)
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    p50 = statistics.median(lat_ms)
    median_s = p50 / 1e3
    is_data = name != "barrier"
    tput = (payload_bytes / (1024 * 1024)) / median_s if (is_data and median_s > 0 and payload_bytes > 0) else 0.0
    return {
        "baseline": "redis",
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