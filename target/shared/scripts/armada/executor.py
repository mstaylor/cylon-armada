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


class ArmadaExecutor:
    """Drives a compiled ExecutionPlan over a FMIBridge.

    At world_size <= 1 every collective is mathematically an identity
    operation with a single participant, so the executor computes that
    identity directly rather than calling the bridge — this is not an
    in-process fallback masking a broken distributed run, it is what the
    same collective would produce for a single rank. For world_size > 1 an
    unavailable bridge always raises; there is no silent local execution.
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
            local_result = op.fn(current)
            pattern = plan.assignments[op.name]

            if pattern == CollectivePattern.Scatter:
                current = local_result[root] if single_rank else self.bridge.scatter(local_result, root)
            elif pattern == CollectivePattern.ScatterGather:
                if single_rank:
                    current = [local_result]
                else:
                    scattered = self.bridge.scatter(local_result, root)
                    current = self.bridge.gather(scattered, root)
            elif pattern == CollectivePattern.Reduce:
                current = local_result if single_rank else self.bridge.reduce_table(local_result, reduce_op, root)
            elif pattern == CollectivePattern.PointToPoint:
                current = local_result
            elif pattern == CollectivePattern.Broadcast:
                current = local_result if single_rank else self.bridge.broadcast(local_result, root)
            else:
                raise ValueError(f"unknown collective pattern {pattern!r} for operator {op.name!r}")

        return current