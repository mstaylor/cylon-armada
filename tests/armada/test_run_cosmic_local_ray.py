"""run_epochs dispatches ray-native and ray-cylon instead of refusing them.

Ray cluster formation and the real executors are exercised in
tests/ray_arm/ and tests/armada/test_ray_native_executor.py; this file
checks only that run_cosmic_local wires the right thing to the right
backend, with everything below it faked.
"""

import json
from types import SimpleNamespace
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


def test_ray_native_actors_pin_the_shard_and_name_registry_and_barrier_per_run(monkeypatch):
    """The named actors live in one configured namespace, not the driver's
    own: every rank is a separate driver with its own anonymous namespace,
    so a name lookup scoped to that would give each rank a private registry
    and barrier and the ranks would never meet."""
    ray = pytest.importorskip("ray")
    monkeypatch.setenv("RAY_NAMESPACE", "test-ns")
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        handles = mod.ray_native_actors(0, 1, "run-z", ray_api=ray)
        assert ray.get_actor("cosmic_registry_run-z", namespace="test-ns")._actor_id == \
            handles.registry._actor_id
        assert ray.get_actor("cosmic_barrier_run-z", namespace="test-ns")._actor_id == \
            handles.barrier._actor_id
        ray.get(handles.shard_actor.publish.remote(["ctx"]))
        ray.get(handles.shard_actor.flush.remote())
        assert ray.get(handles.registry.count.remote()) == 1
        mod.release_ray_native_run(handles, rank=0, world_size=1, ray_api=ray, timeout_s=10.0)
        with pytest.raises(ValueError):
            ray.get_actor("cosmic_registry_run-z", namespace="test-ns")
        with pytest.raises(ValueError):
            ray.get_actor("cosmic_barrier_run-z", namespace="test-ns")
    finally:
        ray.shutdown()


class _FakeActorDied(Exception):
    pass


def _fake_release_ray(calls, depart_raises=False):
    def remote_call(name, *args):
        calls.append((name,) + args)
        return name

    barrier = SimpleNamespace(
        wait_departed=SimpleNamespace(remote=lambda count: remote_call("wait_departed", count)),
        depart=SimpleNamespace(remote=lambda: remote_call("depart")),
    )

    def get(ref, timeout=None):
        calls.append(("get", ref, timeout))
        if depart_raises and ref == "depart":
            raise _FakeActorDied()

    ray_api = SimpleNamespace(
        get=get,
        kill=lambda actor: calls.append(("kill", actor)),
        exceptions=SimpleNamespace(RayActorError=_FakeActorDied),
    )
    handles = mod.RayNativeHandles(shard_actor="shard", registry="registry", barrier=barrier)
    return handles, ray_api


def test_rank_zero_kills_the_named_actors_only_after_every_other_rank_departs():
    calls = []
    handles, ray_api = _fake_release_ray(calls)

    mod.release_ray_native_run(handles, rank=0, world_size=4, ray_api=ray_api, timeout_s=5.0)

    assert calls[0] == ("wait_departed", 3)
    assert calls[1] == ("get", "wait_departed", 5.0)
    assert ("kill", "registry") in calls[2:]
    assert ("kill", handles.barrier) in calls[2:]


def test_other_ranks_depart_and_never_kill():
    calls = []
    handles, ray_api = _fake_release_ray(calls)

    mod.release_ray_native_run(handles, rank=2, world_size=4, ray_api=ray_api, timeout_s=5.0)

    assert ("depart",) in calls
    assert not any(call[0] == "kill" for call in calls)


def test_a_departure_whose_reply_is_lost_to_the_cleanup_kill_is_not_an_error():
    calls = []
    handles, ray_api = _fake_release_ray(calls, depart_raises=True)

    mod.release_ray_native_run(handles, rank=1, world_size=2, ray_api=ray_api, timeout_s=5.0)


def _main_env(monkeypatch, tmp_path, comm):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("COMM_NAME", comm)
    monkeypatch.setenv("CONTEXT_BACKEND", "")
    pricing = MagicMock(source="stub")
    pricing.get_llm_cost.return_value = 0.0
    pricing.get_embedding_cost.return_value = 0.0
    monkeypatch.setattr(mod, "pricing_for", lambda: pricing)
    monkeypatch.setattr(mod, "_redis_client_for_rendezvous", lambda: None)
    return str(tmp_path / "record.json")


class _FakeClock:
    """perf_counter that moves only when a test phase advances it, so each
    recorded duration is exactly what that phase injected and never races
    real work such as Ray actor startup."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


CLUSTER_S = 4.0
ESTABLISH_S = 0.5
RUN_S = 2.0
TEARDOWN_S = 8.0


def _install_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(mod, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    return clock


def test_ray_native_records_cluster_formation_and_teardown_barrier_apart(monkeypatch, tmp_path):
    ray = pytest.importorskip("ray")
    out = _main_env(monkeypatch, tmp_path, "rn-timing")
    clock = _install_clock(monkeypatch)
    real_actors = mod.ray_native_actors
    real_run_epochs = mod.run_epochs

    def timed_cluster(rank, world_size, comm_name, redis_client, ray_api):
        ray_api.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
        clock.advance(CLUSTER_S)

    def timed_actors(*args, **kwargs):
        handles = real_actors(*args, **kwargs)
        clock.advance(ESTABLISH_S)
        return handles

    def timed_run_epochs(*args, **kwargs):
        results = real_run_epochs(*args, **kwargs)
        clock.advance(RUN_S)
        return results

    class TimedFinish(mod.RayNativeExecutor):
        def finish(self, seq, barrier, timeout_s):
            result = super().finish(seq, barrier, timeout_s)
            clock.advance(TEARDOWN_S)
            return result

    monkeypatch.setattr(mod, "form_ray_cluster", timed_cluster)
    monkeypatch.setattr(mod, "ray_native_actors", timed_actors)
    monkeypatch.setattr(mod, "run_epochs", timed_run_epochs)
    monkeypatch.setattr(mod, "RayNativeExecutor", TimedFinish)

    assert mod.main(["--backend", "ray-native", "--galaxies", "6", "--result-path", out]) == 0
    record = json.load(open(out))

    assert record["ray_cluster_s"] == CLUSTER_S
    assert record["establish_s"] == ESTABLISH_S
    assert record["run_s"] == RUN_S
    assert record["teardown_barrier_s"] == TEARDOWN_S
    assert record["records_written"] == 6
    assert "error" not in record
    assert not ray.is_initialized()


def test_ray_cylon_charges_cluster_formation_to_ray_cluster_s_not_establish_s(monkeypatch, tmp_path):
    out = _main_env(monkeypatch, tmp_path, "rc-timing")
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(mod, "form_ray_cluster",
                        lambda rank, world_size, comm_name, redis_client, ray_api:
                        clock.advance(CLUSTER_S))

    bridge = MagicMock(rank=0, world_size=1, available=True, channel_type="direct-redis", _ctx=None)

    class FakeCylonRayActor:
        def __init__(self, rank, world_size, comm_name, required_peers, nonblocking):
            self.rank = rank
            self._bridge = None

        def start_env(self):
            self._bridge = bridge
            clock.advance(ESTABLISH_S)
            return True

    monkeypatch.setattr(mod, "CylonRayActor", FakeCylonRayActor)

    assert mod.main(["--backend", "ray-cylon", "--galaxies", "6", "--result-path", out]) == 0
    record = json.load(open(out))

    assert record["ray_cluster_s"] == CLUSTER_S
    assert record["establish_s"] == ESTABLISH_S
    assert "teardown_barrier_s" not in record
    bridge.finalize.assert_called_once()
