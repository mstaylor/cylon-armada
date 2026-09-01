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
"""

from cylon_armada.dag_compiler import AgentOperator, WorkflowDAG, compile_workflow

from armada.operator import ArmadaOperator, ArmadaSequence


def lower(seq):
    if isinstance(seq, ArmadaOperator):
        operators = [seq]
    else:
        operators = seq.operators

    agent_operators = [
        AgentOperator(op.name, op.pattern, op.schema_in, op.schema_out)
        for op in operators
    ]
    edges = [(operators[i].name, operators[i + 1].name) for i in range(len(operators) - 1)]
    dag = WorkflowDAG(agent_operators, edges)
    return compile_workflow(dag)