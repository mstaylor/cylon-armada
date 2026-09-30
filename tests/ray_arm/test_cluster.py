"""Cluster formation, with the Ray CLI and node census injected.

Forming the cluster is measured as establish time and compared against FMI
rendezvous, so it must fail loudly rather than proceed with a short cluster:
a sweep that silently runs at N=3 when it recorded N=4 produces a scaling
curve that cannot be trusted anywhere.
"""

import subprocess

import pytest

from ray_arm.cluster import (
    ClusterFormationTimeout, RayStartFailed, _own_address, join_by_discovery,
    join_cluster, start_head, wait_for_nodes,
)


class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.store.get(key)


def test_start_head_publishes_the_address_it_started_on(monkeypatch):
    monkeypatch.setenv("ADVERTISE_HOST", "10.0.0.5")
    redis = FakeRedis()
    calls = []
    address = start_head(6379, redis, "run_a",
                         runner=lambda cmd, **kw: calls.append(cmd))
    assert address.endswith(":6379")
    from ray_arm.rendezvous import discover_head
    assert discover_head(redis, "run_a", timeout_s=1.0) == address


def test_start_head_invokes_ray_start_with_head_flag(monkeypatch):
    monkeypatch.setenv("ADVERTISE_HOST", "10.0.0.5")
    redis = FakeRedis()
    calls = []
    start_head(6379, redis, "run_a", runner=lambda cmd, **kw: calls.append(cmd))
    assert calls, "ray start was never invoked"
    assert "--head" in calls[0]


def test_start_head_refuses_to_publish_a_loopback_address(monkeypatch):
    """A head that advertises 127.0.0.1 is reachable only by itself, so every
    other rank joins nothing and the run silently measures a one-node cluster.
    Detection is forced here rather than left to the host's routing table: a
    unit test that opens a UDP socket passes or fails on the network the
    machine happens to have."""
    monkeypatch.delenv("ADVERTISE_HOST", raising=False)
    monkeypatch.setattr("ray_arm.cluster._detect_routable_address", lambda: "127.0.0.1")
    redis = FakeRedis()

    with pytest.raises(RuntimeError):
        start_head(6379, redis, "run_a", runner=lambda cmd, **kw: None)

    assert redis.store == {}


def test_own_address_honours_advertise_host_override(monkeypatch):
    monkeypatch.setenv("ADVERTISE_HOST", "10.1.2.3")
    assert _own_address(6379) == "10.1.2.3:6379"


def test_own_address_raises_when_resolved_host_is_loopback(monkeypatch):
    monkeypatch.delenv("ADVERTISE_HOST", raising=False)
    monkeypatch.setattr("ray_arm.cluster._detect_routable_address", lambda: "127.0.0.1")
    with pytest.raises(RuntimeError):
        _own_address(6379)


def test_join_cluster_passes_the_address_to_ray_start():
    calls = []
    join_cluster("10.0.0.5:6379", runner=lambda cmd, **kw: calls.append(cmd))
    assert "--address=10.0.0.5:6379" in calls[0]


def test_start_head_wraps_process_failure_as_ray_start_failed():
    redis = FakeRedis()

    def failing_runner(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(RayStartFailed) as excinfo:
        start_head(6379, redis, "run_a", runner=failing_runner)
    assert "run_a" in str(excinfo.value)


def test_join_cluster_wraps_process_failure_as_ray_start_failed():
    def failing_runner(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(RayStartFailed) as excinfo:
        join_cluster("10.0.0.5:6379", runner=failing_runner)
    assert "10.0.0.5:6379" in str(excinfo.value)


def test_wait_for_nodes_returns_once_the_census_is_complete():
    census = iter([1, 2, 4])
    wait_for_nodes(4, timeout_s=10.0, nodes_fn=lambda: next(census),
                   sleep=lambda s: None)


def test_wait_for_nodes_raises_when_the_cluster_stays_short():
    with pytest.raises(ClusterFormationTimeout) as excinfo:
        wait_for_nodes(4, timeout_s=1.0, nodes_fn=lambda: 3,
                       sleep=lambda s: None)
    assert "3" in str(excinfo.value) and "4" in str(excinfo.value)


def test_wait_for_nodes_poll_s_changes_call_count_before_timeout():
    fast_calls = {"n": 0}

    def fast_nodes_fn():
        fast_calls["n"] += 1
        return 3

    with pytest.raises(ClusterFormationTimeout):
        wait_for_nodes(4, timeout_s=1.0, nodes_fn=fast_nodes_fn,
                       sleep=lambda s: None, poll_s=0.5)

    slow_calls = {"n": 0}

    def slow_nodes_fn():
        slow_calls["n"] += 1
        return 3

    with pytest.raises(ClusterFormationTimeout):
        wait_for_nodes(4, timeout_s=1.0, nodes_fn=slow_nodes_fn,
                       sleep=lambda s: None, poll_s=1.0)

    assert fast_calls["n"] != slow_calls["n"]


def test_join_by_discovery_discovers_and_joins_the_published_head():
    redis = FakeRedis()
    calls = []
    from ray_arm.rendezvous import publish_head
    publish_head(redis, "run_a", "10.0.0.5:6379")

    address = join_by_discovery(redis, "run_a", timeout_s=1.0,
                                runner=lambda cmd, **kw: calls.append(cmd))

    assert address == "10.0.0.5:6379"
    assert calls, "ray start was never invoked"
    assert "--address=10.0.0.5:6379" in calls[0]

def test_address_detection_failure_is_a_domain_error(monkeypatch):
    """In a network-isolated container the UDP probe raises OSError. Surfacing
    that bare, from inside cluster formation, gives no hint that ADVERTISE_HOST
    is the way out."""
    import socket as socket_module

    from ray_arm.cluster import AddressDetectionFailed, _detect_routable_address

    class DeadSocket:
        def connect(self, addr):
            raise OSError("Network is unreachable")

        def getsockname(self):
            raise AssertionError("should not be reached")

        def close(self):
            pass

    monkeypatch.setattr(socket_module, "socket", lambda *a, **kw: DeadSocket())

    with pytest.raises(AddressDetectionFailed) as excinfo:
        _detect_routable_address()
    assert "ADVERTISE_HOST" in str(excinfo.value)
