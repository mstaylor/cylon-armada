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

"""Tests for the offline reuse replay.

The replay is going to be used to decide whether the headline claim survives,
so its arithmetic has to be pinned against cases whose answer is known by
construction rather than by running it.

Run: pytest tests/test_reuse_replay.py -v
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from experiment.reuse_replay import ISOLATED, SHARED, normalize, replay, sweep, totals

D = 8


def _identical(n):
    """n copies of one vector: every galaxy matches every other."""
    return np.tile(np.ones(D, dtype=np.float32), (n, 1))


def _orthogonal(n):
    """n mutually orthogonal vectors: nothing ever matches anything."""
    return np.eye(n, D, dtype=np.float32) if n <= D else None


def test_identical_galaxies_reuse_everything_after_the_first_epoch():
    """With one epoch of 4 identical galaxies, the first epoch is all misses
    (nothing is visible yet) and every later epoch is all hits."""
    summaries = replay(_identical(12), world_size=1, topology=ISOLATED,
                       threshold=0.85, epoch_batch_size=4)

    assert summaries[0]["llm_calls"] == 4
    assert summaries[0]["cache_hits"] == 8
    assert summaries[0]["retrievals"] == 12


def test_a_galaxy_cannot_reuse_a_context_from_its_own_epoch():
    """Within-epoch blindness. Retrieve runs over the whole epoch before
    MemoryUpsert stores any of it, so 4 identical galaxies in one epoch are 4
    misses, not 1 miss and 3 hits."""
    summaries = replay(_identical(4), world_size=1, topology=ISOLATED,
                       threshold=0.85, epoch_batch_size=4)

    assert summaries[0]["llm_calls"] == 4
    assert summaries[0]["cache_hits"] == 0


def test_a_smaller_epoch_finds_more_reuse():
    """The epoch size is the visibility granularity, so it bounds reuse. This
    is the knob that would explain a shared arm underperforming."""
    coarse = totals(replay(_identical(12), 1, ISOLATED, 0.85, epoch_batch_size=12))
    fine = totals(replay(_identical(12), 1, ISOLATED, 0.85, epoch_batch_size=1))

    assert coarse["llm_calls"] == 12
    assert fine["llm_calls"] == 1


def test_orthogonal_galaxies_never_reuse():
    summaries = replay(_orthogonal(8), world_size=1, topology=ISOLATED,
                       threshold=0.85, epoch_batch_size=1)

    assert summaries[0]["cache_hits"] == 0
    assert summaries[0]["llm_calls"] == 8


def test_sharing_beats_isolation_when_ranks_meet_different_kinds_first():
    """The headline mechanism, and the condition it actually needs.

    Sharing pays only when ranks encounter DIFFERENT kinds first and then
    converge. Catalogue [A, B, B, A] at N=2 gives rank0 [A, B] and rank1
    [B, A], so each rank's second galaxy is a kind the other already published.
    Isolated, nobody can reuse; shared, both reuse.

    Symmetric shards move in lockstep and sharing gains nothing there — which
    is why this test does not use identical galaxies.
    """
    a = np.zeros(D, dtype=np.float32); a[0] = 1.0
    b = np.zeros(D, dtype=np.float32); b[1] = 1.0
    embeddings = np.vstack([a, b, b, a])

    iso = totals(replay(embeddings, 2, ISOLATED, 0.85, epoch_batch_size=1))
    shr = totals(replay(embeddings, 2, SHARED, 0.85, epoch_batch_size=1))

    assert iso["llm_calls"] == 4
    assert shr["llm_calls"] == 2
    assert shr["galaxies"] == iso["galaxies"] == 4


def test_symmetric_shards_gain_nothing_from_sharing():
    """The converse, stated so the replay's limits are explicit: when every
    rank holds the same mix, isolation already answers every query locally and
    the two topologies coincide. A result showing no sharing benefit may mean
    the catalogue is partitioned this way, not that sharing is useless."""
    embeddings = _identical(8)

    iso = totals(replay(embeddings, 2, ISOLATED, 0.85, epoch_batch_size=2))
    shr = totals(replay(embeddings, 2, SHARED, 0.85, epoch_batch_size=2))

    assert iso["llm_calls"] == shr["llm_calls"]


def test_at_world_size_one_the_topologies_are_identical():
    """There is nothing to share with one rank, so the curves must start
    together. A divergence at N=1 would mean the replay is modelling the
    topology rather than the visibility."""
    embeddings = _identical(16)
    iso = totals(replay(embeddings, 1, ISOLATED, 0.85, epoch_batch_size=4))
    shr = totals(replay(embeddings, 1, SHARED, 0.85, epoch_batch_size=4))

    assert iso == shr


def test_every_galaxy_is_accounted_for_on_both_topologies():
    """calls + hits == galaxies, per rank — the same invariant the live gate
    asserts, so a replay bug cannot quietly drop work."""
    embeddings = np.random.default_rng(0).normal(size=(37, D)).astype(np.float32)
    for topology in (ISOLATED, SHARED):
        for n in (1, 2, 4, 8):
            for r in replay(embeddings, n, topology, 0.85, 4):
                assert r["cache_hits"] + r["llm_calls"] == r["galaxies"]


def test_the_population_is_tiled_exactly_across_ranks():
    """Shards must cover the catalogue once, with no gap or overlap, at world
    sizes that do not divide it evenly."""
    embeddings = np.random.default_rng(1).normal(size=(19, D)).astype(np.float32)
    for n in (1, 2, 4, 8, 16):
        summaries = replay(embeddings, n, SHARED, 0.85, 4)
        covered = sorted(tuple(r["shard"]) for r in summaries)
        cursor = 0
        for start, stop in covered:
            assert start == cursor
            cursor = stop
        assert cursor == 19
        assert sum(r["galaxies"] for r in summaries) == 19


def test_the_threshold_decides_how_near_is_near_enough():
    """Two vectors ~0.92 apart: reused at 0.85, not reused at 0.95. The
    threshold is the single knob the whole reuse rate turns on, so its
    direction is pinned rather than assumed."""
    near = np.zeros(D, dtype=np.float32); near[0] = 1.0
    other = np.zeros(D, dtype=np.float32); other[0] = 1.0; other[1] = 0.42
    embeddings = np.vstack([near, other])

    assert float(normalize(embeddings)[0] @ normalize(embeddings)[1]) == pytest.approx(0.922, abs=0.01)
    assert totals(replay(embeddings, 1, ISOLATED, 0.85, 1))["cache_hits"] == 1
    assert totals(replay(embeddings, 1, ISOLATED, 0.95, 1))["cache_hits"] == 0


def test_normalize_leaves_a_zero_row_at_zero_rather_than_dividing():
    """A zero embedding must not produce NaN, and must match nothing — the
    same answer ContextRouter's scalar fallback gives."""
    matrix = normalize(np.vstack([np.ones((1, D)), np.zeros((1, D))]).astype(np.float32))

    assert not np.isnan(matrix).any()
    assert float(matrix[1] @ matrix[0]) == 0.0


def test_shuffling_the_catalogue_changes_nothing_when_all_galaxies_are_alike():
    """The permutation must move galaxies, not invent or drop them."""
    embeddings = _identical(16)
    straight = totals(replay(embeddings, 4, ISOLATED, 0.85, 4))
    shuffled = totals(replay(embeddings, 4, ISOLATED, 0.85, 4,
                             order=np.random.default_rng(2).permutation(16)))

    assert straight["galaxies"] == shuffled["galaxies"] == 16
    assert straight["llm_calls"] == shuffled["llm_calls"]


def test_an_unknown_topology_is_refused():
    with pytest.raises(ValueError, match="unknown topology"):
        replay(_identical(4), 1, "gossip", 0.85, 4)


def test_sweep_covers_both_topologies_at_every_world_size():
    result = sweep(_identical(16), [1, 2, 4], threshold=0.85, epoch_batch_size=4)

    assert set(result) == {ISOLATED, SHARED}
    for topology in (ISOLATED, SHARED):
        assert [row["world_size"] for row in result[topology]] == [1, 2, 4]
        assert all(row["galaxies"] == 16 for row in result[topology])

def test_the_gate_refuses_a_near_identical_prompt_about_a_different_quantity():
    """The validity problem in one test: two prompts whose embeddings are
    near-identical but which describe very different redshifts. Ungated the
    second reuses the first; gated it must not."""
    a = np.zeros(D, dtype=np.float32); a[0] = 1.0
    b = a.copy(); b[1] = 0.05
    embeddings = np.vstack([a, b])
    redshifts = np.array([0.15, 0.94])

    ungated = totals(replay(embeddings, 1, ISOLATED, 0.85, 1))
    gated = totals(replay(embeddings, 1, ISOLATED, 0.85, 1,
                          gate_values=redshifts, gate_tolerance=0.04))

    assert ungated["cache_hits"] == 1
    assert gated["cache_hits"] == 0


def test_the_gate_admits_a_reuse_that_is_actually_valid():
    a = np.zeros(D, dtype=np.float32); a[0] = 1.0
    b = a.copy(); b[1] = 0.05
    embeddings = np.vstack([a, b])
    redshifts = np.array([0.15, 0.16])

    gated = totals(replay(embeddings, 1, ISOLATED, 0.85, 1,
                          gate_values=redshifts, gate_tolerance=0.04))

    assert gated["cache_hits"] == 1


def test_the_gate_searches_candidates_not_only_the_single_best_match():
    """A slightly worse cosine match that is valid must still be reused.
    Gating only the argmax would throw away reuse the domain permits."""
    a = np.zeros(D, dtype=np.float32); a[0] = 1.0
    best = a.copy(); best[1] = 0.02
    ok = a.copy(); ok[1] = 0.10
    embeddings = np.vstack([best, ok, a])
    redshifts = np.array([0.90, 0.15, 0.15])

    summaries = replay(embeddings, 1, ISOLATED, 0.85, 1,
                       gate_values=redshifts, gate_tolerance=0.04)

    assert summaries[0]["cache_hits"] == 1
    assert summaries[0]["worst_gate_error"] == pytest.approx(0.0, abs=1e-9)


def test_a_gate_rejection_is_counted_and_costs_an_llm_call():
    a = np.zeros(D, dtype=np.float32); a[0] = 1.0
    b = a.copy(); b[1] = 0.05
    embeddings = np.vstack([a, b])

    summaries = replay(embeddings, 1, ISOLATED, 0.85, 1,
                       gate_values=np.array([0.15, 0.94]), gate_tolerance=0.04)

    assert summaries[0]["gate_rejections"] == 1
    assert summaries[0]["llm_calls"] == 2
    assert summaries[0]["cache_hits"] + summaries[0]["llm_calls"] == summaries[0]["galaxies"]


def test_no_gate_leaves_the_deployed_behaviour_untouched():
    """The gate is opt-in: without gate_values the replay must reproduce the
    ungated numbers exactly, or the tolerance sweep is not comparable to the
    pipeline as it stands."""
    embeddings = np.random.default_rng(3).normal(size=(40, D)).astype(np.float32)
    plain = totals(replay(embeddings, 4, SHARED, 0.85, 4))
    explicit_none = totals(replay(embeddings, 4, SHARED, 0.85, 4,
                                  gate_values=None, gate_tolerance=0.04))

    assert plain == explicit_none


def test_worst_gate_error_never_exceeds_the_tolerance():
    """The number the validity claim rests on: every accepted reuse is within
    tolerance, so the reported worst case bounds the error."""
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(60, D)).astype(np.float32)
    values = rng.uniform(0.1, 0.9, size=60)

    for r in replay(embeddings, 2, SHARED, 0.5, 4,
                    gate_values=values, gate_tolerance=0.05):
        assert r["worst_gate_error"] <= 0.05 + 1e-9
