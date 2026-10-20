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

"""ray-collective's executor: local operator calls, with MemoryUpsert's
AllGather carried out by ray.util.collective's allgather.

The loop is ray-native's and Armada's: every pattern except AllGather is a
plain local call. At the AllGather every rank packs its contribution to Arrow
IPC bytes, its exchanger actor allgathers them with Ray's collective, and the
peers' buffers are read back into tables. The collective is itself a
synchronisation point every epoch, so this arm needs no barrier after the
last epoch and its compared time is run_s.

Packing, the exchange and unpacking are timed apart and accumulated in
ConversionTimings. exchange_s is what the driver waits for the actor call,
and collective_s is the part of it spent inside the two allgathers. Ray is
imported on first use so this module stays importable in images without Ray.
"""

import time
from dataclasses import dataclass

from cylon_armada.dag_compiler import CollectivePattern

from armada.executor import ArmadaExecutor, _operators, lower
from armada.ray_native_executor import align_start


@dataclass
class ConversionTimings:
    """Per-rank totals across a run's epochs."""

    pack_s: float = 0.0
    unpack_s: float = 0.0
    exchange_s: float = 0.0
    collective_s: float = 0.0
    packed_bytes: int = 0

    def add(self, pack_s, unpack_s, exchange_s, collective_s, packed_bytes):
        self.pack_s += pack_s
        self.unpack_s += unpack_s
        self.exchange_s += exchange_s
        self.collective_s += collective_s
        self.packed_bytes += packed_bytes

    def as_record(self):
        return {"pack_s": round(self.pack_s, 6), "unpack_s": round(self.unpack_s, 6),
                "exchange_s": round(self.exchange_s, 6),
                "collective_s": round(self.collective_s, 6),
                "packed_bytes": self.packed_bytes}


class RayCollectiveExecutor:
    def __init__(self, exchanger, rank, timings):
        self.exchanger = exchanger
        self.rank = rank
        self.timings = timings

    def start(self, barrier, timeout_s):
        align_start(barrier, timeout_s)

    def _exchange(self, contribution):
        import ray

        from ray_arm.collective import pack_table, unpack_buffer

        t_pack = time.perf_counter()
        payload = pack_table(contribution)
        t_sent = time.perf_counter()
        buffers, collective_s = ray.get(self.exchanger.allgather.remote(payload))
        t_received = time.perf_counter()
        peers = [unpack_buffer(buffer) for rank, buffer in enumerate(buffers) if rank != self.rank]
        t_unpacked = time.perf_counter()
        self.timings.add(pack_s=t_sent - t_pack, unpack_s=t_unpacked - t_received,
                         exchange_s=t_received - t_sent, collective_s=collective_s,
                         packed_bytes=int(payload.size))
        return [contribution, *peers]

    def run(self, seq, input_tables):
        plan = lower(seq)
        current = input_tables
        for op in _operators(seq):
            if plan.assignments[op.name] == CollectivePattern.AllGather:
                gathered = self._exchange(current)
                current = op.fn(ArmadaExecutor._concat_or_empty(gathered, current.schema))
            else:
                current = op.fn(current)
        return current
