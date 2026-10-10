import json
import threading

import numpy as np
import pytest

from cosmic_campaign.agent import assign_requests, fmi_share, rounds_needed, run_agent
from cosmic_campaign.grid import CampaignSettings, Configuration, run_slots
from cosmic_campaign.plan_cache import CacheMode, PlanCache
from cosmic_campaign.planner import PlanOutcome
from cosmic_campaign.requests import execution_requests

SETTINGS = CampaignSettings("b", "b", "Anomaly Detection", "folder", {"A": "/tmp/a.py"}, "p/exp2")


class _RefPlanner:
    source = "llm"

    def __init__(self):
        self.calls = 0

    def plan(self, request):
        self.calls += 1
        return PlanOutcome(list(request.references), self.source, 1000.0, 10, 5, None)


class _ToolSourcePlanner(_RefPlanner):
    source = "llm_tool"


class _ThrottledPlanner(_RefPlanner):
    def plan(self, request):
        raise RuntimeError("ThrottlingException: rate exceeded")


class _Embedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return np.array([1.0, 0.0]), {"token_count": 7, "cache_hit": False}


class _AllGather:
    """Thread barrier standing in for FMI AllGather across in-process agents."""

    def __init__(self, n):
        self.n, self.lock, self.slots, self.barrier = n, threading.Lock(), {}, threading.Barrier(n, timeout=10)

    def for_rank(self, rank):
        def share(table):
            with self.lock:
                self.slots[rank] = table
            self.barrier.wait()
            tables = [self.slots[r] for r in range(self.n)]
            self.barrier.wait()
            return tables
        return share


def _requests(n_runs):
    return execution_requests(run_slots([Configuration("scaling", 100, 1, 512, 11)], 0, n_runs),
                              "A", SETTINGS, seed=0)


def _run(tmp_path, requests, planners, mode=CacheMode.STRUCTURAL, embedders=None):
    n, gather, results = len(planners), _AllGather(len(planners)), {}
    embedders = embedders or [_Embedder() for _ in planners]

    def agent(rank):
        cache = PlanCache(mode, 0.9, SETTINGS, "A")
        results[rank] = run_agent(rank, n, requests, planners[rank], cache, embedders[rank],
                                  gather.for_rank(rank), str(tmp_path / f"plans_{rank}.jsonl"),
                                  str(tmp_path / f"agent_{rank}.json"))

    threads = [threading.Thread(target=agent, args=(r,)) for r in range(n)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    assert not any(t.is_alive() for t in threads)
    return results


def test_requests_are_split_round_robin_and_rounds_cover_the_largest_share():
    requests = _requests(5)
    assert [len(assign_requests(requests, r, 2)) for r in (0, 1)] == [3, 2]
    assert rounds_needed(5, 2) == 3


def test_agents_share_plans_so_later_requests_hit_the_cache(tmp_path):
    planners = [_RefPlanner(), _RefPlanner()]
    results = _run(tmp_path, _requests(6), planners)
    rows = results[0] + results[1]
    assert len(rows) == 6
    assert sum(p.calls for p in planners) == 2
    assert {r["source"] for r in rows} == {"llm", "cache_structural"}
    assert all(r["mismatches"] == [[]] for r in rows)
    hits = [r for r in rows if r["source"] == "cache_structural"]
    assert all(r["similarity"] == 1.0 and r["matched_request"] for r in hits)


def test_an_agent_with_fewer_requests_still_joins_every_round(tmp_path):
    results = _run(tmp_path, _requests(3), [_RefPlanner(), _RefPlanner()], mode=CacheMode.OFF)
    assert len(results[0]) + len(results[1]) == 3
    assert len(open(tmp_path / "plans_1.jsonl").read().splitlines()) == 1


def test_tool_planner_results_are_cached_and_shared_like_plain_llm_ones(tmp_path):
    planners = [_ToolSourcePlanner(), _ToolSourcePlanner()]
    _run(tmp_path, _requests(4), planners)
    assert sum(p.calls for p in planners) == 2


def test_off_mode_never_embeds(tmp_path):
    embedders = [_Embedder(), _Embedder()]
    results = _run(tmp_path, _requests(4), [_RefPlanner(), _RefPlanner()], mode=CacheMode.OFF,
                   embedders=embedders)
    assert sum(e.calls for e in embedders) == 0
    assert all(r["embed_ms"] == 0.0 and r["embed_tokens"] == 0 for r in results[0] + results[1])


def test_rows_record_embedding_cost_and_the_round_allgather_time(tmp_path):
    results = _run(tmp_path, _requests(4), [_RefPlanner(), _RefPlanner()])
    rows = results[0] + results[1]
    assert all(r["embed_tokens"] == 7 and r["embed_ms"] >= 0.0 and r["share_ms"] >= 0.0 for r in rows)
    summary = json.load(open(tmp_path / "agent_0.json"))
    assert summary["rank"] == 0 and len(summary["share_ms"]) == 2 and summary["wall_ms"] > 0


def test_a_planner_exception_is_recorded_and_no_agent_hangs(tmp_path):
    results = _run(tmp_path, _requests(4), [_ThrottledPlanner(), _RefPlanner()], mode=CacheMode.OFF)
    failed = results[0]
    assert len(failed) == 2 and all("ThrottlingException" in r["error"] and r["plans"] == [] for r in failed)
    assert all(r["source"] == "llm" for r in failed)
    assert len(results[1]) == 2


class _Bridge:
    available = False


def test_sharing_refuses_to_run_alone_when_other_agents_expect_it():
    with pytest.raises(RuntimeError, match="FMI"):
        fmi_share(_Bridge(), world_size=2)
    assert fmi_share(_Bridge(), world_size=1)("t") == ["t"]