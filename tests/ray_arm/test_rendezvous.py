"""Ray head rendezvous, tested against a fake Redis.

The head address is the one piece of state every rank must agree on, and a
rank that polls forever is indistinguishable from a rank that never started,
so the timeout path is as important as the success path.
"""

import pytest

from ray_arm.rendezvous import (
    RendezvousTimeout, discover_head, head_key, publish_head,
)


class FakeRedis:
    def __init__(self, initial=None):
        self.store = dict(initial or {})
        self.expirations = {}

    def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value
        if ex is not None:
            self.expirations[key] = ex

    def get(self, key):
        return self.store.get(key)


def test_head_key_is_scoped_to_the_comm_name():
    """Two concurrent runs must not read each other's head address, which is
    the same collision the FMI path avoids by scoping on comm_name."""
    assert head_key("run_a") != head_key("run_b")
    assert "run_a" in head_key("run_a")


def test_published_head_is_discoverable():
    redis = FakeRedis()
    publish_head(redis, "run_a", "10.0.0.5:6379")
    assert discover_head(redis, "run_a", timeout_s=1.0) == "10.0.0.5:6379"


def test_publish_sets_a_ttl_so_a_dead_run_does_not_poison_the_next():
    redis = FakeRedis()
    publish_head(redis, "run_a", "10.0.0.5:6379", ttl_s=120)
    assert redis.expirations[head_key("run_a")] == 120


def test_discover_polls_until_the_head_appears():
    redis = FakeRedis()
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) == 3:
            publish_head(redis, "run_a", "10.0.0.9:6379")

    assert discover_head(redis, "run_a", timeout_s=10.0, poll_s=0.5,
                         sleep=fake_sleep) == "10.0.0.9:6379"
    assert len(calls) == 3


def test_discover_raises_rather_than_hanging_when_the_head_never_appears():
    """A silent hang here looks exactly like a slow head node, and the task
    would burn its whole timeout before anyone learned the head died."""
    redis = FakeRedis()
    with pytest.raises(RendezvousTimeout) as excinfo:
        discover_head(redis, "run_a", timeout_s=1.0, poll_s=0.5,
                      sleep=lambda s: None)
    assert "run_a" in str(excinfo.value)