"""Correctness gate for an Experiment E paired run.

Response text is deliberately not compared: LLM output is nondeterministic
across arms, so equality is asserted on coverage, count and provenance. A gate
that compared text would fail every run for a reason that says nothing about
the data plane.

Run: pytest tests/test_exp_e_gate.py -v
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from results.exp_e_gate import check_run


def _records(galaxies_per_rank, ranks=4, failed=0, throttles=0):
    return [
        {"rank": r, "world_size": ranks, "galaxies": galaxies_per_rank,
         "shard": [r * galaxies_per_rank, (r + 1) * galaxies_per_rank],
         "records_written": galaxies_per_rank, "records_failed": failed,
         "throttle_events": throttles, "run_s": 1.0}
        for r in range(ranks)
    ]


def test_a_matched_pair_passes():
    result = check_run(_records(19), _records(19))
    assert result.passed, result.failures


def test_differing_galaxy_coverage_fails():
    result = check_run(_records(19), _records(18))
    assert not result.passed
    assert any("coverage" in f for f in result.failures)


def test_any_failed_record_fails_the_run():
    result = check_run(_records(19), _records(19, failed=1))
    assert not result.passed
    assert any("failed" in f for f in result.failures)


def test_throttling_invalidates_the_run():
    """A 429 is indistinguishable from a slow rank in timing data."""
    result = check_run(_records(19), _records(19, throttles=3))
    assert not result.passed
    assert any("throttl" in f for f in result.failures)


def test_overlapping_shards_fail():
    dup = _records(19)
    dup[1]["shard"] = dup[0]["shard"]
    result = check_run(dup, _records(19))
    assert not result.passed
    assert any("overlap" in f or "coverage" in f for f in result.failures)


def test_a_missing_rank_fails():
    result = check_run(_records(19)[:-1], _records(19))
    assert not result.passed
    assert any("rank" in f for f in result.failures)


def test_a_rank_that_errored_fails_even_with_counts_intact():
    """A rank whose run raised still writes its record, marked with the error."""
    broken = _records(19)
    broken[2]["error"] = "RuntimeError: boom"
    result = check_run(broken, _records(19))
    assert not result.passed
    assert any("rank 2" in f and "error" in f for f in result.failures)


def test_a_galaxy_that_produced_no_record_fails():
    """Every galaxy has to come out the far end as a stored record."""
    short = _records(19)
    short[0]["records_written"] = 18
    result = check_run(short, _records(19))
    assert not result.passed
    assert any("accounting for 18 of 19 galaxies" in f for f in result.failures)


def test_failure_messages_name_arm_and_rank():
    """A gate failure has to be debuggable from the message alone."""
    result = check_run(_records(19), _records(19, throttles=1))
    assert all(f.startswith("langchain rank ") for f in result.failures)


def test_an_empty_arm_fails_rather_than_passing_vacuously():
    result = check_run([], _records(19))
    assert not result.passed
    assert any("no rank records" in f for f in result.failures)


def test_a_reversed_shard_is_rejected_even_when_its_counts_are_self_consistent():
    """Defense in depth: the tiling check advances by max(cursor, stop), so a
    reversed shard that starts at the cursor would otherwise be silently
    treated as zero width."""
    forged = _records(19)
    forged[1]["shard"] = [38, 19]
    forged[1]["galaxies"] = -19
    forged[1]["records_written"] = -19
    result = check_run(forged, _records(19))
    assert not result.passed
    assert any("malformed shard" in f for f in result.failures)


def test_arms_at_different_world_sizes_are_not_a_paired_run():
    """Same total coverage at different N is not the comparison the experiment
    makes — 4 ranks of 19 versus 2 ranks of 38 must fail."""
    result = check_run(_records(19, ranks=4), _records(38, ranks=2))
    assert not result.passed
    assert any("world_size differs" in f for f in result.failures)


def test_a_rank_that_reused_a_context_still_accounts_for_every_galaxy():
    """Once a hit avoids a store, records_written alone no longer covers the
    shard; written plus hits must. Without this the gate would fail every run
    that achieved any reuse — i.e. exactly the runs H4 is about."""
    records = _records(19)
    records[0]["records_written"] = 15
    records[0]["cache_hits"] = 4

    assert check_run(records, _records(19)).passed


def test_a_rank_that_lost_records_still_fails_even_with_hits_counted():
    """The relaxation must not become a hole: written + hits short of the
    shard is a lost record, and the gate has to keep catching it."""
    records = _records(19)
    records[0]["records_written"] = 15
    records[0]["cache_hits"] = 2

    result = check_run(records, _records(19))

    assert not result.passed
    assert any("19 galaxies" in f for f in result.failures)


def test_the_gate_accepts_a_third_arm():
    """Three arms now run per point; a gate that only knew two would silently
    ignore the control."""
    assert check_run(_records(19), _records(19), isolated=_records(19)).passed


def test_a_mismatched_control_fails_the_run():
    """An uncompared control is not a control: if the isolated arm covered a
    different population, the isolation penalty computed from it is meaningless."""
    result = check_run(_records(19), _records(19), isolated=_records(18))

    assert not result.passed
    assert any("coverage differs" in f for f in result.failures)


def test_a_control_at_a_different_world_size_fails():
    result = check_run(_records(19), _records(19), isolated=_records(19, ranks=2))

    assert not result.passed
    assert any("world_size differs" in f for f in result.failures)


def test_two_arms_still_gate_without_a_control():
    assert check_run(_records(19), _records(19)).passed
