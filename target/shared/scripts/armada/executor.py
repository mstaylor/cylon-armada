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

"""Lowering pass and executor for cylon-armada (Experiment E).

lower() turns an ArmadaSequence (LCEL graph) into a compiled ExecutionPlan by
building a cylon_armada.dag_compiler.WorkflowDAG and running it through
compile_workflow, which also enforces the A2 Arrow schema-compatibility check
on every edge.

ArmadaExecutor.run() drives that plan over a FMIBridge: for each operator, it
runs the operator's fn locally, then moves the result to the next stage via
the collective assigned to the operator's own CollectivePattern. There is no
in-process fallback — an unavailable bridge raises rather than silently
executing locally, so the executor can never mask a broken distributed run.
"""

from cylon_armada.dag_compiler import AgentOperator, CollectivePattern, WorkflowDAG, compile_workflow

from armada.operator import ArmadaOperator, ArmadaSequence
from armada.topology import format_peer_map, peer_map


def _operators(seq):
    if isinstance(seq, ArmadaOperator):
        return [seq]
    return seq.operators


def lower(seq):
    operators = _operators(seq)
    agent_operators = [
        AgentOperator(op.name, op.pattern, op.schema_in, op.schema_out)
        for op in operators
    ]
    edges = [(operators[i].name, operators[i + 1].name) for i in range(len(operators) - 1)]
    dag = WorkflowDAG(agent_operators, edges)
    return compile_workflow(dag)


def required_peer_map(seq, world_size, roots=(0,)):
    """Serialized FMI_REQUIRED_PEERS map for the collectives this sequence compiles to.

    Pass to FMIBridge(required_peers=...) so the channel connects only the peers
    the plan will use rather than the full N(N-1)/2 mesh. It has to be handed to
    the bridge's constructor, not to run(): the channel establishes its
    connections while the communicator is being built, which is over by the time
    an executor exists.

        peers = required_peer_map(seq, world_size)
        bridge = FMIBridge(world_size=world_size, rank=rank, required_peers=peers, ...)
        ArmadaExecutor(bridge).run(seq, ...)

    Returns "" for world_size <= 1, which leaves the channel unrestricted —
    there is nothing to connect at a single rank anyway.
    """
    if world_size <= 1:
        return ""
    patterns = set(lower(seq).assignments.values())
    return format_peer_map(peer_map(world_size, patterns, roots))


class ArmadaExecutor:
    """Drives a compiled ExecutionPlan over a FMIBridge.

    Scatter moves data before computing (each rank must have its own shard
    before it can run fn on it); every other pattern computes first, then
    moves the result (each rank computes its own contribution, then that
    contribution is consolidated/broadcast). Getting this order right matters
    beyond correctness at world_size > 1: at world_size <= 1 there is exactly
    one participant, so every pattern collapses to the same thing — call fn,
    do nothing else — which only equals plain Runnable.invoke() chaining
    (ArmadaSequence.invoke, no distribution at all) if the *real* dispatch
    order for every pattern already treats fn as a pure, uniformly-shaped
    per-rank transform. For world_size > 1 an unavailable bridge always
    raises; there is no silent local execution.
    """

    def __init__(self, bridge):
        self.bridge = bridge

    def run(self, seq, input_tables, ctx, root=0, reduce_op="sum"):
        single_rank = self.bridge.world_size <= 1
        if not single_rank and not self.bridge.available:
            raise RuntimeError(
                f"ArmadaExecutor requires an available FMIBridge for "
                f"world_size={self.bridge.world_size} — no in-process fallback"
            )

        operators = _operators(seq)
        plan = lower(seq)
        current = input_tables

        for op in operators:
            pattern = plan.assignments[op.name]

            if single_rank:
                current = op.fn(current)
                continue

            if pattern == CollectivePattern.Scatter:
                current = self.bridge.scatter(current, root)
                current = op.fn(current)
            elif pattern == CollectivePattern.ScatterGather:
                local_result = op.fn(current)
                current = self.bridge.gather(local_result, root)
            elif pattern == CollectivePattern.Reduce:
                local_result = op.fn(current)
                current = self.bridge.reduce_table(local_result, reduce_op, root)
            elif pattern == CollectivePattern.PointToPoint:
                current = op.fn(current)
            elif pattern == CollectivePattern.Broadcast:
                local_result = op.fn(current)
                current = self.bridge.broadcast(local_result, root)
            else:
                raise ValueError(f"unknown collective pattern {pattern!r} for operator {op.name!r}")

        return current