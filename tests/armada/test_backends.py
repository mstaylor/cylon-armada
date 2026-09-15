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

"""Backend selection for the Experiment E arms.

The two arms differ only in how workers share state: Arm A moves contexts with
an AllGather, Arm B lets every rank read and write one Redis. The workflow
object, the operators and the Bedrock calls are identical, which is what makes
any wall-clock difference attributable to the data plane.

Run: pytest tests/armada/test_backends.py -v
"""

import os
import sys

import pyarrow as pa
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.operator import ArmadaOperator
from armada.run_cosmic_local import ExecutionBackend, context_store_for, run_epochs
from cylon_armada.dag_compiler import CollectivePattern

TEXT = pa.schema([pa.field("raw_text", pa.large_utf8())])


def _sequence(recorder):
    def shout(table):
        recorder.append(table.num_rows)
        return pa.table({"raw_text": [t.upper() for t in table.column(0).to_pylist()]},
                        schema=TEXT)

    return ArmadaOperator("Preprocess", CollectivePattern.Scatter, TEXT, TEXT, fn=shout)


class _Bridge:
    world_size = 2
    available = True
    channel_type = "spy"
    _ctx = None
    rank = 0

    def __init__(self):
        self.calls = []

    def allgather(self, table):
        self.calls.append("allgather")
        return [table]


def _shards():
    return [pa.table({"raw_text": ["a", "b"]}, schema=TEXT),
            pa.table({"raw_text": ["c"]}, schema=TEXT)]


def test_langchain_backend_uses_native_invoke_and_no_collectives():
    seen = []
    bridge = _Bridge()

    results = run_epochs(_sequence(seen), _shards(), ExecutionBackend.LangChain,
                         bridge=bridge, ctx=None, root=0)

    assert bridge.calls == []
    assert seen == [2, 1]
    assert [r.column("raw_text").to_pylist() for r in results] == [["A", "B"], ["C"]]


def test_langchain_backend_needs_no_bridge_at_all():
    """Spec 4: Arm B shares state through the store, so it is handed no bridge.
    Charging it FMI connection setup it never uses would bias the comparison."""
    seen = []

    results = run_epochs(_sequence(seen), _shards(), ExecutionBackend.LangChain,
                         bridge=None, ctx=None, root=0)

    assert seen == [2, 1]
    assert len(results) == 2


def test_armada_backend_runs_through_the_executor():
    seen = []
    bridge = _Bridge()

    results = run_epochs(_sequence(seen), _shards(), ExecutionBackend.Armada,
                         bridge=bridge, ctx=None, root=0)

    assert seen == [2, 1]
    assert [r.column("raw_text").to_pylist() for r in results] == [["A", "B"], ["C"]]


def test_both_backends_produce_identical_results():
    """The correctness gate in miniature: same workflow, same input, same output.
    Only timing may differ."""
    armada = run_epochs(_sequence([]), _shards(), ExecutionBackend.Armada,
                        bridge=_Bridge(), ctx=None, root=0)
    langchain = run_epochs(_sequence([]), _shards(), ExecutionBackend.LangChain,
                           bridge=None, ctx=None, root=0)

    assert [r.to_pydict() for r in armada] == [r.to_pydict() for r in langchain]


def test_no_shards_means_no_epochs_and_no_error():
    """A rank whose shard is empty at a large world size runs zero epochs."""
    assert run_epochs(_sequence([]), [], ExecutionBackend.Armada,
                      bridge=_Bridge(), ctx=None, root=0) == []


def test_backend_resolves_from_its_string_name():
    assert ExecutionBackend("langchain") is ExecutionBackend.LangChain
    assert ExecutionBackend("armada") is ExecutionBackend.Armada
    with pytest.raises(ValueError):
        ExecutionBackend("ray")


def test_each_arm_is_bound_to_its_context_store(monkeypatch):
    """The arm IS its store: Armada on the Arrow ContextTable, LangChain on Redis."""
    monkeypatch.delenv("CONTEXT_BACKEND", raising=False)
    assert context_store_for(ExecutionBackend.Armada) == "cylon"
    assert context_store_for(ExecutionBackend.LangChain) == "redis"


def test_a_conflicting_context_backend_in_the_environment_is_refused(monkeypatch):
    """Fail fast at the boundary rather than silently overriding an operator's
    explicit setting with the arm's."""
    monkeypatch.setenv("CONTEXT_BACKEND", "redis")
    with pytest.raises(ValueError, match="CONTEXT_BACKEND"):
        context_store_for(ExecutionBackend.Armada)


def test_a_matching_context_backend_in_the_environment_is_accepted(monkeypatch):
    monkeypatch.setenv("CONTEXT_BACKEND", "redis")
    assert context_store_for(ExecutionBackend.LangChain) == "redis"


def test_an_unknown_backend_fails_loudly_instead_of_running_as_armada():
    """The dispatch is exhaustive: a backend that is neither member must not
    fall through to the collectives path by default."""
    with pytest.raises(ValueError, match="unknown execution backend"):
        run_epochs(_sequence([]), _shards(), "ray", bridge=_Bridge(), ctx=None, root=0)
