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

"""Pattern-carrying LCEL operators for the cylon-armada executor (Experiment E).

Each ArmadaOperator is a LangChain Runnable that also carries the
CollectivePattern and Arrow schema pair cylon_armada.dag_compiler needs to lower
the composed graph to a collective execution plan. Composing operators with `|`
yields an ArmadaSequence, runnable either by LangChain's native runtime (via
Runnable.invoke) or by the cylon-armada executor's lowering pass.
"""

from typing import Any, Callable, List, Optional

import pyarrow as pa
from langchain_core.runnables import Runnable, RunnableConfig

from cylon_armada.dag_compiler import CollectivePattern


class ArmadaOperator(Runnable):
    def __init__(self, name: str, pattern: CollectivePattern,
                 schema_in: pa.Schema, schema_out: pa.Schema,
                 fn: Callable[[Any], Any]):
        if not isinstance(schema_in, pa.Schema) or not isinstance(schema_out, pa.Schema):
            raise TypeError("schema_in/schema_out must be pyarrow.Schema instances")
        self.name = name
        self.pattern = pattern
        self.schema_in = schema_in
        self.schema_out = schema_out
        self.fn = fn

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        return self.fn(input)

    def __or__(self, other: "ArmadaOperator") -> "ArmadaSequence":
        return ArmadaSequence([self, other])

    def __repr__(self) -> str:
        return f"ArmadaOperator({self.name!r}, pattern={self.pattern})"


class ArmadaSequence(Runnable):
    def __init__(self, operators: List[ArmadaOperator]):
        self.operators = list(operators)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        result = input
        for op in self.operators:
            result = op.invoke(result, config, **kwargs)
        return result

    def __or__(self, other: ArmadaOperator) -> "ArmadaSequence":
        return ArmadaSequence(self.operators + [other])

    def __repr__(self) -> str:
        names = " | ".join(op.name for op in self.operators)
        return f"ArmadaSequence({names})"