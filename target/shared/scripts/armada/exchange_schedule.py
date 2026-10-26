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

"""Overlap one epoch's model calls with the previous epoch's exchange (Experiment E).

The synchronous executors run each epoch to its AllGather and block there, so
a rank waits for the slowest peer's model calls every epoch. OverlappedEpochs
keeps the same blocking collective and moves it: during epoch e the calling
thread exchanges epoch e-1's contribution and ingests the peers' contexts while
a worker makes epoch e's model calls. A rank's own contexts are stored as soon
as its epoch finishes, and the last epoch is exchanged at drain.

Cylon is single-threaded apart from its non-blocking I/O, so every Cylon call
stays on the calling thread: the operators before the offloaded block (Retrieve
reads the ContextTable), the exchange and MemoryUpsert. The offloaded operators
must make only model calls and touch no Cylon state. The exchange must release
the GIL while it waits, or the worker cannot run.

The worker and the calling thread overlap only between an epoch's offload and
its result, when the calling thread runs nothing but the exchange and
MemoryUpsert. RunMetrics fields written on each side in that window are
disjoint, so it needs no lock.

Retrieve in epoch e sees this rank's contexts through e-1 and its peers'
through e-2.
"""

from concurrent.futures import ThreadPoolExecutor

from cylon_armada.dag_compiler import CollectivePattern

from armada.executor import ArmadaExecutor, _operators, lower

LOCAL_PATTERNS = (CollectivePattern.Scatter, CollectivePattern.PointToPoint)


class OverlappedEpochs:
    def __init__(self, exchange_peers, offloaded_operators):
        self._exchange_peers = exchange_peers
        self._offloaded = tuple(offloaded_operators)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="armada-model-calls")
        self._pending = None

    def run(self, seq, input_tables):
        """One epoch. Returns the ack tables of every upsert it performed."""
        leading, offloaded, upsert = self._split(seq)
        current = input_tables
        for op in leading:
            current = op.fn(current)
        model_calls = self._pool.submit(_apply, offloaded, current)
        acks = self._ingest_pending()
        contribution = model_calls.result()
        acks.append(upsert.fn(contribution))
        self._pending = (upsert, contribution)
        return acks

    def drain(self):
        """Exchange the last epoch's contribution and ingest the peers' contexts."""
        try:
            return self._ingest_pending()
        finally:
            self.close()

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _ingest_pending(self):
        if self._pending is None:
            return []
        upsert, contribution = self._pending
        self._pending = None
        peers = self._exchange_peers(contribution)
        return [upsert.fn(ArmadaExecutor._concat_or_empty(peers, contribution.schema))]

    def _split(self, seq):
        operators = _operators(seq)
        plan = lower(seq)
        upsert = operators[-1]
        if plan.assignments[upsert.name] != CollectivePattern.AllGather:
            raise ValueError(f"the last operator {upsert.name!r} must be an AllGather")
        names = [op.name for op in operators[:-1]]
        start = len(names) - len(self._offloaded)
        if start < 0 or tuple(names[start:]) != self._offloaded:
            raise ValueError(
                f"offloaded operators {list(self._offloaded)} must sit directly before "
                f"{upsert.name!r}; the sequence is {names}")
        leading = operators[:start]
        for op in leading + operators[start:-1]:
            if plan.assignments[op.name] not in LOCAL_PATTERNS:
                raise ValueError(
                    f"operator {op.name!r} has pattern {plan.assignments[op.name]!r}; only "
                    f"local operators may precede the final AllGather")
        return leading, operators[start:-1], upsert


def _apply(operators, table):
    for op in operators:
        table = op.fn(table)
    return table