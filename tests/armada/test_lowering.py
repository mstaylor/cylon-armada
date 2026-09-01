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

"""Unit tests for the LCEL-graph-to-execution-plan lowering pass (SP1 Task 3).

Run: pytest tests/armada/test_lowering.py -x
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pyarrow as pa
import pytest

from armada.executor import lower
from armada.operator import ArmadaOperator
from cylon_armada.dag_compiler import CollectivePattern, SchemaMismatchError


def _emb_schema(d, name="embedding"):
    return pa.schema([pa.field(name, pa.list_(pa.field("item", pa.float32()), d))])


def _text_schema(name="t"):
    return pa.schema([pa.field(name, pa.large_utf8())])


def test_lower_compiles_and_rejects_mismatch():
    emb = _emb_schema(8, "e")
    txt = _text_schema("t")
    ok = ArmadaOperator("Embed", CollectivePattern.ScatterGather, txt, emb, fn=lambda x: x) \
         | ArmadaOperator("Retrieve", CollectivePattern.Reduce, emb, emb, fn=lambda x: x)
    plan = lower(ok)
    assert plan.assignments["Embed"] == CollectivePattern.ScatterGather

    bad = ArmadaOperator("Embed", CollectivePattern.ScatterGather, txt, emb, fn=lambda x: x) \
          | ArmadaOperator("Retrieve", CollectivePattern.Reduce, txt, emb, fn=lambda x: x)
    with pytest.raises(SchemaMismatchError):
        lower(bad)


def test_lower_records_every_operators_pattern():
    txt = _text_schema()
    emb = _emb_schema(8)
    seq = (
        ArmadaOperator("Preprocess", CollectivePattern.Scatter, txt, txt, fn=lambda x: x)
        | ArmadaOperator("Embed", CollectivePattern.ScatterGather, txt, emb, fn=lambda x: x)
        | ArmadaOperator("Retrieve", CollectivePattern.Reduce, emb, emb, fn=lambda x: x)
    )
    plan = lower(seq)
    assert plan.assignments["Preprocess"] == CollectivePattern.Scatter
    assert plan.assignments["Embed"] == CollectivePattern.ScatterGather
    assert plan.assignments["Retrieve"] == CollectivePattern.Reduce


def test_lower_builds_linear_edges_in_sequence_order():
    txt = _text_schema()
    seq = (
        ArmadaOperator("A", CollectivePattern.Scatter, txt, txt, fn=lambda x: x)
        | ArmadaOperator("B", CollectivePattern.Broadcast, txt, txt, fn=lambda x: x)
        | ArmadaOperator("C", CollectivePattern.Reduce, txt, txt, fn=lambda x: x)
    )
    plan = lower(seq)
    edge_pairs = [(e.producer, e.consumer) for e in plan.edges]
    assert edge_pairs == [("A", "B"), ("B", "C")]


def test_lower_single_operator_has_no_edges():
    txt = _text_schema()
    seq = ArmadaOperator("Solo", CollectivePattern.Broadcast, txt, txt, fn=lambda x: x)
    plan = lower(seq)
    assert plan.assignments["Solo"] == CollectivePattern.Broadcast
    assert list(plan.edges) == []