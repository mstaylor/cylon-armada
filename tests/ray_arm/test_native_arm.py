"""Arm A actors on a local Ray instance.

These run a real local Ray rather than a mock: the property under test is that
context published by one rank becomes visible to another through the object
store, and a mock would assert our own assumptions about Ray rather than Ray's
behaviour.
"""

import pytest

ray = pytest.importorskip("ray")
np = pytest.importorskip("numpy")

from ray_arm.native import ContextRegistry, ShardActor


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_a_rank_sees_another_ranks_published_context(local_ray):
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 2, registry)
    rank1 = ShardActor.remote(1, 2, registry)

    ray.get(rank0.publish.remote(["ctx-from-0"]))
    ray.get(rank0.flush.remote())
    visible = ray.get(rank1.visible_contexts.remote())

    assert "ctx-from-0" in visible


def test_publish_returns_the_number_stored(local_ray):
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 1, registry)
    assert ray.get(rank0.publish.remote(["a", "b", "c"])) == 3


def test_visible_contexts_is_empty_before_anything_is_published(local_ray):
    """Cold start: an empty store must return an empty list, never None."""
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 1, registry)
    assert ray.get(rank0.visible_contexts.remote()) == []


def test_second_read_only_fetches_new_contexts_via_watermark(local_ray):
    """A read must fetch only what changed, not the whole run's history, or
    the arm becomes quadratic in the number of reads and distorts the scaling
    measurement it produces. The property belongs to the ShardActor read path,
    so it is asserted through this rank's watermark: it advances exactly to
    what the registry holds, and a read with nothing published in between
    fetches nothing and leaves it where it was."""
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 2, registry)
    rank1 = ShardActor.remote(1, 2, registry)

    ray.get(rank0.publish.remote(["first"]))
    ray.get(rank0.flush.remote())
    first_read = ray.get(rank1.visible_contexts.remote())
    watermark_after_first = ray.get(registry.count.remote())

    ray.get(rank0.publish.remote(["second"]))
    ray.get(rank0.flush.remote())
    second_read = ray.get(rank1.visible_contexts.remote())

    new_since_first, total_after_second = ray.get(
        registry.all_refs.remote(since=watermark_after_first)
    )

    assert first_read == ["first"]
    assert second_read == ["first", "second"]
    assert len(new_since_first) == 1
    assert total_after_second == 2
    assert ray.get(rank1.watermark.remote()) == total_after_second

    third_read = ray.get(rank1.visible_contexts.remote())
    assert third_read == ["first", "second"]
    assert ray.get(rank1.watermark.remote()) == total_after_second


def test_numpy_payload_survives_the_object_store_round_trip(local_ray):
    """Plasma's zero-copy path is for numpy/Arrow buffers; the other tests use
    plain strings for simplicity, so this keeps that path exercised."""
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 2, registry)
    rank1 = ShardActor.remote(1, 2, registry)

    array = np.arange(8, dtype=np.float32)
    ray.get(rank0.publish.remote([array]))
    ray.get(rank0.flush.remote())

    visible = ray.get(rank1.visible_contexts.remote())

    assert len(visible) == 1
    np.testing.assert_array_equal(visible[0], array)


def test_drain_new_returns_only_contexts_since_the_last_drain(local_ray):
    """MemoryUpsert must see each epoch's new contributions exactly once. A
    second drain_new with nothing published in between must return empty,
    not the same contexts again — that would double count store/ingest."""
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 2, registry)
    rank1 = ShardActor.remote(1, 2, registry)

    ray.get(rank0.publish.remote(["epoch1-from-0"]))
    ray.get(rank0.flush.remote())
    first = ray.get(rank1.drain_new.remote())
    assert first == ["epoch1-from-0"]

    again = ray.get(rank1.drain_new.remote())
    assert again == []

    ray.get(rank0.publish.remote(["epoch2-from-0"]))
    ray.get(rank0.flush.remote())
    second = ray.get(rank1.drain_new.remote())
    assert second == ["epoch2-from-0"]


def test_drain_new_and_visible_contexts_share_one_watermark(local_ray):
    """Both reads consume the same log position, so mixing them on one
    shard must not replay or skip anything."""
    registry = ContextRegistry.remote()
    rank0 = ShardActor.remote(0, 2, registry)
    rank1 = ShardActor.remote(1, 2, registry)

    ray.get(rank0.publish.remote(["a"]))
    ray.get(rank0.flush.remote())
    assert ray.get(rank1.drain_new.remote()) == ["a"]

    ray.get(rank0.publish.remote(["b"]))
    ray.get(rank0.flush.remote())
    assert ray.get(rank1.visible_contexts.remote()) == ["a", "b"]
    assert ray.get(rank1.drain_new.remote()) == []
