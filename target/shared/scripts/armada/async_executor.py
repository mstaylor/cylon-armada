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

"""Publish on completion over Cylon (Experiment E, armada-async).

Shares contexts the way RayNativeExecutor does: at the end of each epoch a rank
publishes its contribution and upserts it together with every peer table that
has arrived, and there is no barrier inside the epochs. The exchange is pycylon's
AsyncTableExchange, reached through FMIBridge.async_exchange.

Cylon moves bytes only while it is progressed, where Ray has background threads,
so the model-call operators run on a worker while the calling thread polls the
exchange. Every Cylon call (the leading operators, the exchange and MemoryUpsert)
stays on the calling thread, as in armada.exchange_schedule. After publishing, a
rank keeps progressing until its own sends have completed, no table is half
received and a poll brings nothing new, bounded by settle_s. ray-native's read
returns everything published so far, and this is its counterpart: without it a
fast epoch, a run of cache hits say, would see peers' tables an epoch late.

The exchange is created from the first contribution's schema, because
passthrough columns can ride on the rows. Every rank reaches the end of its first
epoch, so every rank creates it. finish waits up to timeout_s for every peer to
finish and runs the late tables through MemoryUpsert, the counterpart of
ray-native's end-of-run barrier.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait

from armada.exchange_schedule import split_at_final_allgather
from armada.executor import ArmadaExecutor, run_from_allgather


class NoPeersExchange:
    """The exchange of a single rank: nothing to send and nothing to receive."""

    def publish(self, table, reference):
        return None

    def drain(self):
        return []

    def idle(self):
        return True

    def finish_and_drain(self, timeout_s):
        return []


class ArmadaAsyncExecutor:
    def __init__(self, exchange_factory, offloaded_operators, poll_s, settle_s):
        self._exchange_factory = exchange_factory
        self._offloaded = tuple(offloaded_operators)
        self._poll_s = poll_s
        self._settle_s = settle_s
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="armada-async-model-calls")
        self._exchange = None
        self._arrived = []
        self._epoch = 0

    def run(self, seq, input_tables):
        """One epoch. Returns its MemoryUpsert result."""
        leading, offloaded, upsert = split_at_final_allgather(seq, self._offloaded)
        current = input_tables
        for op in leading:
            current = op.fn(current)
        model_calls = self._pool.submit(_apply, offloaded, current)
        while not wait([model_calls], timeout=self._poll_s).done:
            self._collect()
        contribution = model_calls.result()
        if self._exchange is None:
            self._exchange = self._exchange_factory(contribution.schema)
        if contribution.num_rows:
            self._exchange.publish(contribution, self._epoch)
        self._settle()
        gathered = ([contribution] if contribution.num_rows else []) + self._arrived
        self._arrived = []
        self._epoch += 1
        return [upsert.fn(ArmadaExecutor._concat_or_empty(gathered, contribution.schema))]

    def drain(self):
        """Nothing is pending at the end of an epoch; finish collects the late tables."""
        return []

    def finish(self, seq, timeout_s):
        """Wait for every peer, then store what arrived after this rank's last epoch."""
        try:
            if self._exchange is None:
                return None
            late = self._arrived + [table for _, _, table
                                    in self._exchange.finish_and_drain(timeout_s)]
            self._arrived = []
            return run_from_allgather(seq, late) if late else None
        finally:
            self.close()

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _collect(self):
        if self._exchange is None:
            return 0
        arrived = [table for _, _, table in self._exchange.drain()]
        self._arrived.extend(arrived)
        return len(arrived)

    def _settle(self):
        deadline = time.monotonic() + self._settle_s
        while self._collect() or not self._exchange.idle():
            if time.monotonic() >= deadline:
                return


def _apply(operators, table):
    for op in operators:
        table = op.fn(table)
    return table