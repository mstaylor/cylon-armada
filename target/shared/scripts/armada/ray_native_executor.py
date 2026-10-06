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

"""Arm A's executor: local operator calls, with MemoryUpsert's AllGather
replaced by a Ray object-store publish and an epoch-scoped read.

Every pattern except AllGather is already a plain local call under the
compiled plan (armada/executor.py's own dispatch loop treats PointToPoint,
and Scatter under PreDistributed placement, the same way) — this executor
exists only because MemoryUpsert is the one operator this arm takes a
genuinely different path for, sharing contexts through Plasma instead of a
collective, per the design's rejection of ray.util.collective.

There is no per-epoch barrier: a rank reads whatever has been published when
it gets there, and anything published later arrives on its next read. The one
barrier is at the end of the run (finish), so the last epochs' late arrivals
still reach every rank and no rank tears down objects a peer is still reading.
Ray is imported on first use so this module stays importable in images
without Ray.
"""

from cylon_armada.dag_compiler import CollectivePattern

from armada.executor import ArmadaExecutor, _operators, lower

PUBLISHED_PHASE = "published"
DRAINED_PHASE = "drained"


class RayNativeExecutor:
    def __init__(self, shard_actor):
        self.shard_actor = shard_actor

    def _publish_and_drain(self, contribution):
        import ray

        if not contribution.num_rows:
            return ray.get(self.shard_actor.drain_new.remote())
        self.shard_actor.publish.remote([ray.put(contribution)])
        return [contribution, *ray.get(self.shard_actor.drain_new.remote())]

    def run(self, seq, input_tables):
        plan = lower(seq)
        current = input_tables
        for op in _operators(seq):
            if plan.assignments[op.name] == CollectivePattern.AllGather:
                gathered = self._publish_and_drain(current)
                current = op.fn(ArmadaExecutor._concat_or_empty(gathered, current.schema))
            else:
                current = op.fn(current)
        return current

    def finish(self, seq, barrier, timeout_s):
        """End-of-run drain, between two barriers on the run's RunBarrier.

        The first barrier means every rank has published its last epoch, so
        the drain after it sees every contribution this rank has not yet
        read; those rows run through MemoryUpsert and the operators after
        it, exactly as an epoch's gathered rows do. The second barrier means
        every rank has finished reading, so none can exit while a peer still
        resolves objects it owns. Returns that final result, or None when
        nothing arrived late.
        """
        import ray

        ray.get(barrier.wait.remote(PUBLISHED_PHASE), timeout=timeout_s)
        late = ray.get(self.shard_actor.drain_new.remote())
        result = self._upsert_late(seq, late) if late else None
        ray.get(barrier.wait.remote(DRAINED_PHASE), timeout=timeout_s)
        return result

    @staticmethod
    def _upsert_late(seq, late):
        plan = lower(seq)
        operators = _operators(seq)
        start = next((index for index, op in enumerate(operators)
                      if plan.assignments[op.name] == CollectivePattern.AllGather), None)
        if start is None:
            return None
        current = ArmadaExecutor._concat_or_empty(late, late[0].schema)
        for op in operators[start:]:
            current = op.fn(current)
        return current
