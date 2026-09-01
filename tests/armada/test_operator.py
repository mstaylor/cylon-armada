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

"""Unit tests for ArmadaOperator / ArmadaSequence (Experiment E SP1 Task 1).

Run: pytest tests/armada/test_operator.py -x
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pyarrow as pa
import pytest
from langchain_core.runnables import Runnable

from armada.operator import ArmadaOperator, ArmadaSequence
from cylon_armada.dag_compiler import CollectivePattern


def _schema():
    return pa.schema([pa.field("v", pa.int64())])


def test_operator_is_runnable_and_composes():
    s = _schema()
    a = ArmadaOperator("A", CollectivePattern.Scatter, s, s, fn=lambda x: x)
    b = ArmadaOperator("B", CollectivePattern.Reduce, s, s, fn=lambda x: x)
    seq = a | b
    assert [op.name for op in seq.operators] == ["A", "B"]
    assert a.invoke(123) == 123


def test_operator_is_a_langchain_runnable():
    s = _schema()
    a = ArmadaOperator("A", CollectivePattern.Scatter, s, s, fn=lambda x: x)
    assert isinstance(a, Runnable)


def test_operator_invoke_calls_fn_with_input():
    s = _schema()
    seen = []
    a = ArmadaOperator("A", CollectivePattern.PointToPoint, s, s, fn=lambda x: seen.append(x) or x * 2)
    assert a.invoke(21) == 42
    assert seen == [21]


def test_three_way_compose_preserves_order():
    s = _schema()
    a = ArmadaOperator("A", CollectivePattern.Scatter, s, s, fn=lambda x: x)
    b = ArmadaOperator("B", CollectivePattern.ScatterGather, s, s, fn=lambda x: x)
    c = ArmadaOperator("C", CollectivePattern.Reduce, s, s, fn=lambda x: x)
    seq = a | b | c
    assert isinstance(seq, ArmadaSequence)
    assert [op.name for op in seq.operators] == ["A", "B", "C"]


def test_sequence_or_operator_appends():
    s = _schema()
    a = ArmadaOperator("A", CollectivePattern.Scatter, s, s, fn=lambda x: x)
    b = ArmadaOperator("B", CollectivePattern.Reduce, s, s, fn=lambda x: x)
    c = ArmadaOperator("C", CollectivePattern.Broadcast, s, s, fn=lambda x: x)
    seq = ArmadaSequence([a, b])
    seq2 = seq | c
    assert [op.name for op in seq2.operators] == ["A", "B", "C"]


def test_operator_carries_schema_and_pattern():
    s_in = pa.schema([pa.field("text", pa.large_utf8())])
    s_out = pa.schema([pa.field("chunks", pa.large_utf8())])
    a = ArmadaOperator("Preprocess", CollectivePattern.Scatter, s_in, s_out, fn=lambda x: x)
    assert a.pattern == CollectivePattern.Scatter
    assert a.schema_in == s_in
    assert a.schema_out == s_out