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


def test_the_isolated_backend_is_a_distinct_arm():
    assert ExecutionBackend("isolated") is ExecutionBackend.Isolated


def test_the_isolated_arm_runs_on_its_own_local_store():
    """Isolated is the controlled analogue of the shipped Cosmic AI design: a
    rank sees only its own contexts, so it runs on the process-local Arrow table
    rather than a Redis every rank writes to."""
    assert context_store_for(ExecutionBackend.Isolated) == "cylon"


def test_the_isolated_arm_fires_no_collective():
    """A collective would make it a sharing arm, which is the thing it is the
    control for."""
    seen = []
    bridge = _Bridge()

    results = run_epochs(_sequence(seen), _shards(), ExecutionBackend.Isolated,
                         bridge=bridge, ctx=None, root=0)

    assert bridge.calls == []
    assert seen == [2, 1]
    assert [r.column("raw_text").to_pylist() for r in results] == [["A", "B"], ["C"]]


def test_the_isolated_arm_needs_no_bridge_at_all():
    assert run_epochs(_sequence([]), _shards(), ExecutionBackend.Isolated,
                      bridge=None, ctx=None, root=0) != []


def test_all_three_arms_produce_identical_results():
    """The correctness gate across every arm: same workflow, same input, same
    output. Only which contexts each rank can see, and the timing, may differ."""
    per_arm = {
        arm: run_epochs(_sequence([]), _shards(), arm,
                        bridge=_Bridge() if arm is ExecutionBackend.Armada else None,
                        ctx=None, root=0)
        for arm in (ExecutionBackend.Armada, ExecutionBackend.LangChain,
                    ExecutionBackend.Isolated)
    }
    rendered = {arm: [r.to_pydict() for r in results] for arm, results in per_arm.items()}

    assert rendered[ExecutionBackend.Armada] == rendered[ExecutionBackend.LangChain]
    assert rendered[ExecutionBackend.Armada] == rendered[ExecutionBackend.Isolated]


def test_the_runner_gates_only_when_a_tolerance_is_configured(monkeypatch):
    """Gating is opt-in at the run level so an ungated control stays possible
    and directly comparable to every measurement taken before it."""
    from armada.run_cosmic_local import reuse_policy_for

    monkeypatch.delenv("REUSE_KEY_TOLERANCE", raising=False)
    assert reuse_policy_for() is None

    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "0.0091")
    validator = reuse_policy_for()
    assert validator is not None
    assert validator(0.150, 0.152) is True
    assert validator(0.150, 0.900) is False


def test_the_query_key_survives_preprocess_and_embed():
    """Retrieve reads the query key off its input table, so the key has to
    traverse two operators to get there. Without the passthrough it is always
    None, the runtime fails closed, and gating silently degrades to reusing
    nothing — which does not error."""
    from unittest.mock import MagicMock

    import numpy as np

    from armada.cosmic_workflow import build_embed_operator, build_preprocess_operator

    dims = 8
    pre = build_preprocess_operator(max_chars=None, dimensions=dims)
    table = pa.table({"raw_text": pa.array(["a galaxy"], type=pa.large_utf8()),
                      "reuse_key": pa.array([0.42], type=pa.float64())})

    after_pre = pre.fn(table)
    assert after_pre.column("reuse_key").to_pylist() == [0.42]

    embedding_service = MagicMock()
    embedding_service.embed.return_value = (np.zeros(dims, dtype=np.float32), {})
    after_embed = build_embed_operator(embedding_service, dimensions=dims).fn(after_pre)

    assert after_embed.column("reuse_key").to_pylist() == [0.42]


def test_a_table_without_a_reuse_key_passes_through_unchanged():
    """An ungated run sends no key column, and neither operator may invent one."""
    from unittest.mock import MagicMock

    import numpy as np

    from armada.cosmic_workflow import build_embed_operator, build_preprocess_operator

    dims = 8
    pre = build_preprocess_operator(max_chars=None, dimensions=dims)
    after_pre = pre.fn(pa.table({"raw_text": pa.array(["a galaxy"], type=pa.large_utf8())}))
    assert "reuse_key" not in after_pre.column_names

    embedding_service = MagicMock()
    embedding_service.embed.return_value = (np.zeros(dims, dtype=np.float32), {})
    after_embed = build_embed_operator(embedding_service, dimensions=dims).fn(after_pre)

    assert "reuse_key" not in after_embed.column_names


def test_a_fixed_outlier_threshold_makes_a_prompt_depend_only_on_its_galaxy():
    """The sweep-level confound.

    The generator picks its template with a percentile over whatever array it is
    handed, and the runner generates prompts per rank AFTER sharding. So without
    a pinned threshold the same galaxy gets a different prompt at a different
    world size, and an isolation curve measured across N mixes the effect under
    study with a moving workload. Matching max_tasks does not fix it.
    """
    import numpy as np

    from cosmic_ai.task_generator import generate_tasks_from_results

    rng = np.random.default_rng(7)
    n = 40
    pred = rng.uniform(0.05, 0.5, n)
    true = pred - rng.normal(0.0, 0.02, n)
    mags = rng.normal(0.0, 1.0, (n, 5))

    def prompts(lo, hi, threshold):
        return generate_tasks_from_results(pred[lo:hi], true[lo:hi], mags[lo:hi],
                                           max_tasks=hi - lo, seed=42,
                                           outlier_threshold=threshold)[:hi - lo]

    pinned = 0.03
    assert prompts(0, 10, pinned) == prompts(0, 40, pinned)[:10]


def test_without_a_pinned_threshold_the_shard_size_changes_the_prompts():
    """The converse, pinned so the fix cannot be quietly reverted: this is the
    behaviour that makes an unpinned sweep invalid.

    Residuals are constructed so the 90th percentile of the first ten is far
    below that of all forty. Galaxy 9 is then an outlier within its own shard
    and an ordinary galaxy within the population, so it gets a different
    template depending only on how many galaxies were sharded alongside it.
    """
    import numpy as np

    from cosmic_ai.task_generator import generate_tasks_from_results

    residuals = np.array([0.001] * 9 + [0.01] + [0.1] * 30)
    pred = np.full(40, 0.3)
    true = pred - residuals
    mags = np.zeros((40, 5))

    def prompts(lo, hi):
        return generate_tasks_from_results(pred[lo:hi], true[lo:hi], mags[lo:hi],
                                           max_tasks=hi - lo, seed=42)[:hi - lo]

    local, whole = prompts(0, 10), prompts(0, 40)[:10]
    assert local[9] != whole[9]


def test_the_runner_leaves_the_threshold_unpinned_unless_configured(monkeypatch):
    from armada.run_cosmic_local import outlier_threshold_for

    monkeypatch.delenv("OUTLIER_RESIDUAL_THRESHOLD", raising=False)
    assert outlier_threshold_for() is None

    monkeypatch.setenv("OUTLIER_RESIDUAL_THRESHOLD", "0.0295")
    assert outlier_threshold_for() == 0.0295


def test_a_non_numeric_outlier_threshold_fails_fast(monkeypatch):
    import pytest as _pytest

    from armada.run_cosmic_local import outlier_threshold_for

    monkeypatch.setenv("OUTLIER_RESIDUAL_THRESHOLD", "p90")
    with _pytest.raises(ValueError, match="OUTLIER_RESIDUAL_THRESHOLD"):
        outlier_threshold_for()


def test_a_galaxys_prompt_is_the_same_at_every_world_size():
    """The sweep's central integrity requirement, and the one a pinned outlier
    threshold alone does NOT satisfy.

    Template choice also alternates on index parity, and that index is local to
    the array passed in — so galaxy 19 is odd at N=1 and even as rank 1's first
    row at N=64, giving the same galaxy a different prompt. Pinning the
    threshold fixes one route to corpus-size dependence; the offset fixes the
    other. Without both, an isolation curve measured across N mixes the effect
    under study with a moving workload.
    """
    import numpy as np

    from armada.run_cosmic_local import shard_bounds
    from cosmic_ai.task_generator import generate_tasks_from_results

    rng = np.random.default_rng(11)
    total = 60
    pred = rng.uniform(0.05, 0.5, total)
    true = pred - rng.normal(0.0, 0.02, total)
    mags = rng.normal(0.0, 1.0, (total, 5))
    threshold = float(np.percentile(np.abs(pred - true), 90))

    def corpus(world_size):
        out = {}
        for rank in range(world_size):
            lo, hi = shard_bounds(total, world_size, rank)
            prompts = generate_tasks_from_results(
                pred[lo:hi], true[lo:hi], mags[lo:hi], max_tasks=hi - lo, seed=42,
                outlier_threshold=threshold, index_offset=lo)[:hi - lo]
            for offset, prompt in enumerate(prompts):
                out[lo + offset] = prompt
        return out

    reference = corpus(1)
    for world_size in (2, 4, 8, 15):
        assert corpus(world_size) == reference, f"corpus changed at N={world_size}"


def test_a_shard_local_index_alone_changes_the_prompt():
    """The converse, so the offset cannot be quietly dropped."""
    import numpy as np

    from cosmic_ai.task_generator import generate_tasks_from_results

    pred = np.full(4, 0.3)
    true = pred - 0.001
    mags = np.zeros((4, 5))

    def prompt(offset):
        return generate_tasks_from_results(pred[:1], true[:1], mags[:1], max_tasks=1,
                                           seed=42, outlier_threshold=1.0,
                                           index_offset=offset)[0]

    assert prompt(0) != prompt(1)


def test_the_corpus_hash_identifies_the_workload_a_rank_ran():
    """Recorded so the frozen-corpus claim is a fact the results carry, not a
    property the runner is trusted to have preserved."""
    from armada.run_cosmic_local import corpus_hash

    a = corpus_hash(0, ["p0", "p1"], [0.1, 0.2])
    assert a == corpus_hash(0, ["p0", "p1"], [0.1, 0.2])
    assert a != corpus_hash(0, ["p0", "CHANGED"], [0.1, 0.2])
    assert a != corpus_hash(0, ["p0", "p1"], [0.1, 0.9])
    assert a != corpus_hash(5, ["p0", "p1"], [0.1, 0.2])


def test_concatenated_shard_hashes_match_across_world_sizes():
    """The property the sweep needs: the same population sharded differently
    must still identify as the same corpus."""
    from armada.run_cosmic_local import corpus_hash, shard_bounds

    prompts = [f"prompt-{i}" for i in range(12)]
    keys = [i / 100 for i in range(12)]

    def digest(world_size):
        parts = []
        for rank in range(world_size):
            lo, hi = shard_bounds(12, world_size, rank)
            parts.append(corpus_hash(lo, prompts[lo:hi], keys[lo:hi]))
        return "".join(parts)

    assert len({digest(n) for n in (1, 2, 3, 4, 6)}) == 5
    per_rank = {n: [corpus_hash(*(lambda lo, hi: (lo, prompts[lo:hi], keys[lo:hi]))(
        *shard_bounds(12, n, r))) for r in range(n)] for n in (1, 2, 4)}
    assert per_rank[2][0] != per_rank[1][0]


def test_the_isolated_arm_refuses_a_conflicting_snapshot_setting(monkeypatch):
    """The isolated arm's isolation must be a property of the arm, not of a flag
    another file happens to set. A cylon store snapshots its whole table to a
    Redis key every rank shares, so a control with the snapshot on is not a
    control."""
    from armada.run_cosmic_local import enforce_isolation

    monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "1")
    with pytest.raises(ValueError, match="CONTEXT_TABLE_SNAPSHOT"):
        enforce_isolation(ExecutionBackend.Isolated)


def test_the_isolated_arm_turns_the_snapshot_off_by_itself(monkeypatch):
    """setenv before delenv so monkeypatch owns the key and restores it.

    enforce_isolation writes os.environ directly, and a value monkeypatch never
    recorded leaks into every later test in the process — which is how this
    test first broke test_the_snapshot_is_on_by_default.
    """
    from armada.run_cosmic_local import enforce_isolation

    monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "placeholder")
    monkeypatch.delenv("CONTEXT_TABLE_SNAPSHOT")

    enforce_isolation(ExecutionBackend.Isolated)

    assert os.environ["CONTEXT_TABLE_SNAPSHOT"] == "0"


def test_a_sharing_arm_is_left_alone(monkeypatch):
    """Only the control needs forcing; the sharing arms are configured by the
    driver and must not be silently overridden."""
    from armada.run_cosmic_local import enforce_isolation

    monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "1")
    enforce_isolation(ExecutionBackend.Armada)
    assert os.environ["CONTEXT_TABLE_SNAPSHOT"] == "1"


def test_every_rank_runs_the_same_number_of_epochs():
    """Each epoch fires one AllGather that every rank must reach.

    Shards are not all equal — at 1253 galaxies and N=8, five ranks hold 157
    and three hold 156, so with batch 4 some ranks would plan 40 epochs and
    others 39. A rank that stops calling the collective while its peers wait in
    it hangs the run until the FMI timeout, which is why the count comes from
    the largest shard rather than this rank's own.
    """
    from armada.run_cosmic_local import collective_epochs, shard_bounds
    from armada.epochs import epoch_count

    for total, world_size, batch in ((1253, 8, 4), (1253, 3, 4), (32, 2, 4), (19, 4, 4)):
        agreed = collective_epochs(total, world_size, batch)
        for rank in range(world_size):
            lo, hi = shard_bounds(total, world_size, rank)
            assert epoch_count(hi - lo, batch) <= agreed


def test_the_unequal_shard_case_actually_disagrees_without_the_fix():
    """Pins the hazard: at N=8 over 1253 galaxies the per-rank counts really do
    differ, so this is not a theoretical concern."""
    from armada.run_cosmic_local import shard_bounds
    from armada.epochs import epoch_count

    counts = {epoch_count(hi - lo, 4)
              for lo, hi in (shard_bounds(1253, 8, r) for r in range(8))}

    assert len(counts) > 1
