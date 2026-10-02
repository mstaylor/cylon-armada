"""RayNativeExecutor against a real local Ray instance.

Mirrors ArmadaExecutor's dispatch loop (armada/executor.py): every pattern
except AllGather is a plain local call, and AllGather is the one place this
executor differs, reading the object store instead of a collective.
"""

import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")

from armada.operator import ArmadaOperator
from armada.ray_native_executor import RayNativeExecutor
from cylon_armada.dag_compiler import CollectivePattern
from ray_arm.native import ContextRegistry, ShardActor


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


VALUE_SCHEMA = pa.schema([("value", pa.int64())])


def _passthrough(name, pattern):
    return ArmadaOperator(name, pattern, VALUE_SCHEMA, VALUE_SCHEMA, fn=lambda t: t)


def test_point_to_point_operators_run_locally_with_no_ray_traffic(local_ray):
    registry = ContextRegistry.remote()
    shard = ShardActor.remote(0, 1, registry)
    executor = RayNativeExecutor(shard)
    seq = _passthrough("Reason", CollectivePattern.PointToPoint)

    result = executor.run(seq, pa.table({"value": [1, 2, 3]}))

    assert result.column("value").to_pylist() == [1, 2, 3]
    assert ray.get(registry.count.remote()) == 0


def test_allgather_operator_publishes_and_reads_back_other_ranks(local_ray):
    """No barrier: rank 0 reads before rank 1 publishes, so only the later
    reader sees both contributions within the same epoch."""
    registry = ContextRegistry.remote()
    shard0 = ShardActor.remote(0, 2, registry)
    shard1 = ShardActor.remote(1, 2, registry)
    seq = _passthrough("MemoryUpsert", CollectivePattern.AllGather)

    result0 = RayNativeExecutor(shard0).run(seq, pa.table({"value": [1]}))
    result1 = RayNativeExecutor(shard1).run(seq, pa.table({"value": [2]}))

    assert result0.column("value").to_pylist() == [1]
    assert sorted(result1.column("value").to_pylist()) == [1, 2]


def test_a_second_epoch_does_not_see_the_first_epochs_contribution_again(local_ray):
    """The double-count this executor exists to avoid: MemoryUpsert's fn
    must not run twice over rank 0's first-epoch row during rank 1's second
    epoch read."""
    registry = ContextRegistry.remote()
    shard0 = ShardActor.remote(0, 2, registry)
    shard1 = ShardActor.remote(1, 2, registry)

    seen = []

    def upsert_fn(table):
        seen.append(table.column("value").to_pylist())
        return table

    seq = ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather,
                         VALUE_SCHEMA, VALUE_SCHEMA, fn=upsert_fn)

    RayNativeExecutor(shard0).run(seq, pa.table({"value": [1]}))
    RayNativeExecutor(shard1).run(seq, pa.table({"value": [10]}))

    RayNativeExecutor(shard0).run(seq, pa.table({"value": [2]}))
    RayNativeExecutor(shard1).run(seq, pa.table({"value": [20]}))

    rank1_second_epoch_input = seen[3]
    assert 1 not in rank1_second_epoch_input
    assert 10 not in rank1_second_epoch_input


def test_an_epoch_where_every_rank_contributed_nothing_reaches_upsert_as_empty(local_ray):
    """Trailing epochs carry zero-row tables; MemoryUpsert must still run,
    on an empty table of the input schema, as it does under ArmadaExecutor."""
    registry = ContextRegistry.remote()
    shard = ShardActor.remote(0, 1, registry)
    seen = []

    def upsert_fn(table):
        seen.append(table)
        return table

    seq = ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather,
                         VALUE_SCHEMA, VALUE_SCHEMA, fn=upsert_fn)

    RayNativeExecutor(shard).run(seq, VALUE_SCHEMA.empty_table())

    assert seen[0].num_rows == 0
    assert seen[0].schema == VALUE_SCHEMA
