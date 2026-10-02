"""run_epochs dispatches ray-native and ray-cylon instead of refusing them.

Ray cluster formation and the real executors are exercised in
tests/ray_arm/ and tests/armada/test_ray_native_executor.py; this file
checks only that run_cosmic_local wires the right thing to the right
backend, with everything below it faked.
"""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

import armada.run_cosmic_local as mod
from armada.run_cosmic_local import ExecutionBackend, form_ray_cluster, run_epochs


def _seq_of_one_passthrough():
    seq = MagicMock()
    seq.invoke.side_effect = lambda shard: shard
    return seq


def test_ray_cylon_runs_through_armada_executor_with_the_given_bridge(monkeypatch):
    captured = {}

    class FakeArmadaExecutor:
        def __init__(self, bridge):
            captured["bridge"] = bridge

        def run(self, seq, input_tables, ctx, root, placement):
            captured["ran"] = True
            captured["placement"] = placement
            return input_tables

    monkeypatch.setattr(mod, "ArmadaExecutor", FakeArmadaExecutor)

    bridge = object()
    shards = [pa.table({"value": [1]})]
    results = run_epochs(_seq_of_one_passthrough(), shards,
                         ExecutionBackend.RayCylon, bridge=bridge, ctx=None)

    assert captured["bridge"] is bridge
    assert captured["ran"] is True
    assert captured["placement"] is mod.InputPlacement.PreDistributed
    assert len(results) == 1


def test_ray_native_runs_through_the_ray_native_executor(monkeypatch):
    captured = {}

    class FakeRayNativeExecutor:
        def __init__(self, shard_actor):
            captured["shard_actor"] = shard_actor

        def run(self, seq, input_tables):
            captured["ran"] = True
            return input_tables

    monkeypatch.setattr(mod, "RayNativeExecutor", FakeRayNativeExecutor)

    shard_actor = object()
    shards = [pa.table({"value": [1]})]
    results = run_epochs(_seq_of_one_passthrough(), shards,
                         ExecutionBackend.RayNative, shard_actor=shard_actor)

    assert captured["shard_actor"] is shard_actor
    assert captured["ran"] is True
    assert len(results) == 1


def test_an_unrecognised_backend_still_raises():
    with pytest.raises(ValueError):
        run_epochs(_seq_of_one_passthrough(), [pa.table({"value": [1]})],
                   "not-a-real-backend", bridge=None, ctx=None)


class _FakeRay:
    def __init__(self, alive):
        self.alive = alive
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def nodes(self):
        return [{"Alive": flag} for flag in self.alive]


def _record_cluster_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "start_head",
                        lambda port, client, comm: calls.append(("head", port, comm)) or "h:1")
    monkeypatch.setattr(mod, "join_by_discovery",
                        lambda client, comm, timeout_s: calls.append(("join", comm, timeout_s)) or "h:1")
    monkeypatch.setattr(mod, "wait_for_nodes",
                        lambda expected, timeout_s, nodes_fn: calls.append(
                            ("wait", expected, timeout_s, nodes_fn())))
    return calls


def test_rank_zero_starts_the_head_on_the_configured_port(monkeypatch):
    monkeypatch.setenv("RAY_PORT", "7001")
    monkeypatch.setenv("RAY_CLUSTER_TIMEOUT_S", "12")
    calls = _record_cluster_calls(monkeypatch)
    fake_ray = _FakeRay([True, True])

    form_ray_cluster(0, 2, "run-x", redis_client=object(), ray_api=fake_ray)

    assert calls[0] == ("head", 7001, "run-x")
    assert fake_ray.init_calls == [{"address": "auto", "ignore_reinit_error": True}]
    assert calls[-1] == ("wait", 2, 12.0, 2)


def test_other_ranks_join_by_discovery_and_count_only_alive_nodes(monkeypatch):
    monkeypatch.setenv("RAY_RENDEZVOUS_TIMEOUT_S", "7")
    calls = _record_cluster_calls(monkeypatch)
    fake_ray = _FakeRay([True, False, True])

    form_ray_cluster(1, 3, "run-y", redis_client=object(), ray_api=fake_ray)

    assert calls[0] == ("join", "run-y", 7.0)
    assert not any(call[0] == "head" for call in calls)
    assert calls[-1][0] == "wait" and calls[-1][1] == 3 and calls[-1][3] == 2


def test_ray_native_shard_actor_is_pinned_to_this_node_with_a_run_scoped_registry():
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        shard = mod.ray_native_shard_actor(0, 1, "run-z", ray_api=ray)
        registry = ray.get_actor("cosmic_registry_run-z")
        ray.get(shard.publish.remote(["ctx"]))
        ray.get(shard.flush.remote())
        assert ray.get(registry.count.remote()) == 1
        ray.kill(registry)
    finally:
        ray.shutdown()
