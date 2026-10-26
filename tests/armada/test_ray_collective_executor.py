"""RayCollectiveExecutor against a real two-rank ray.util.collective group.

Every pattern except AllGather is a plain local call, as under the other
executors. MemoryUpsert's AllGather goes through Ray's collective, and the
Arrow-to-bytes conversion either side of it is timed apart from the exchange.
"""

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")

from armada.exchange_schedule import OverlappedEpochs
from armada.operator import ArmadaOperator
from armada.ray_collective_executor import ConversionTimings, RayCollectiveExecutor
from cylon_armada.dag_compiler import CollectivePattern
from ray_arm.collective import form_collective_group

VALUE_SCHEMA = pa.schema([("value", pa.int64())])
NAMESPACE = "coll-exec-test"


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, namespace=NAMESPACE)
    yield
    ray.shutdown()


def _group(name, world_size=2):
    exchangers = {}
    for rank in reversed(range(world_size)):
        exchangers[rank] = form_collective_group(rank, world_size, name, NAMESPACE, ray,
                                                 timeout_s=30.0, poll_s=0.05)
    return exchangers


def _recording_upsert(seen):
    def upsert_fn(table):
        seen.append(sorted(table.column("value").to_pylist()))
        return table

    return ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather,
                          VALUE_SCHEMA, VALUE_SCHEMA, fn=upsert_fn)


def test_point_to_point_operators_run_locally_and_never_touch_the_group(local_ray):
    timings = ConversionTimings()
    executor = RayCollectiveExecutor(exchanger=None, rank=0, timings=timings)
    seq = ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                         fn=lambda t: t)

    result = executor.run(seq, pa.table({"value": [1, 2]}))

    assert result.column("value").to_pylist() == [1, 2]
    assert timings.as_record()["packed_bytes"] == 0


def test_each_rank_sees_all_rows_and_its_own_rows_exactly_once(local_ray):
    exchangers = _group("exec-once")
    seen = {0: [], 1: []}
    timings = {0: ConversionTimings(), 1: ConversionTimings()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(RayCollectiveExecutor(exchangers[r], r, timings[r]).run,
                               _recording_upsert(seen[r]), pa.table({"value": values}))
                   for r, values in ((0, [1]), (1, [2, 3]))]
        [future.result(timeout=60) for future in futures]

    assert seen[0] == [[1, 2, 3]]
    assert seen[1] == [[1, 2, 3]]


def test_a_rank_with_nothing_new_still_joins_the_exchange(local_ray):
    """A trailing epoch carries a zero-row table, but its peers are still in
    the collective, so the rank must arrive or the epoch hangs."""
    exchangers = _group("exec-empty")
    seen = {0: [], 1: []}
    timings = {0: ConversionTimings(), 1: ConversionTimings()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(RayCollectiveExecutor(exchangers[r], r, timings[r]).run,
                               _recording_upsert(seen[r]), table)
                   for r, table in ((0, VALUE_SCHEMA.empty_table()), (1, pa.table({"value": [7]})))]
        [future.result(timeout=60) for future in futures]

    assert seen[0] == [[7]]
    assert seen[1] == [[7]]


def test_conversion_and_exchange_are_timed_apart(local_ray):
    """The Arrow-to-bytes copy is the cost the zero-copy claim is about, so it
    is reported on its own rather than folded into the exchange."""
    exchangers = _group("exec-timed")
    timings = {0: ConversionTimings(), 1: ConversionTimings()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(RayCollectiveExecutor(exchangers[r], r, timings[r]).run,
                               _recording_upsert([]), pa.table({"value": [r]}))
                   for r in (0, 1)]
        [future.result(timeout=60) for future in futures]

    record = timings[0].as_record()
    assert set(record) == {"pack_s", "unpack_s", "exchange_s", "collective_s", "packed_bytes"}
    assert record["packed_bytes"] > 0
    assert record["exchange_s"] >= record["collective_s"] >= 0.0
    assert record["pack_s"] >= 0.0 and record["unpack_s"] >= 0.0


def test_timings_accumulate_across_epochs():
    timings = ConversionTimings()
    timings.add(pack_s=0.5, unpack_s=0.25, exchange_s=2.0, collective_s=1.0, packed_bytes=10)
    timings.add(pack_s=0.5, unpack_s=0.25, exchange_s=2.0, collective_s=1.0, packed_bytes=6)

    assert timings.as_record() == {"pack_s": 1.0, "unpack_s": 0.5, "exchange_s": 4.0,
                                   "collective_s": 2.0, "packed_bytes": 16}


def _overlapped_sequence(seen):
    reason = ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                            fn=lambda t: t)
    return reason | _recording_upsert(seen)


def test_overlapped_epochs_deliver_every_row_once_to_both_ranks(local_ray):
    """Two ranks run three epochs with the exchange deferred one epoch. A rank
    with an empty trailing epoch still joins every exchange, and after drain
    each rank has stored every row exactly once."""
    exchangers = _group("exec-overlap")
    seen = {0: [], 1: []}
    epochs = {0: [[1], [2], [3]], 1: [[10, 11], [12], []]}

    def run_rank(rank):
        executor = RayCollectiveExecutor(exchangers[rank], rank, ConversionTimings())
        schedule = OverlappedEpochs(executor.exchange_peers, ("Reason",))
        seq = _overlapped_sequence(seen[rank])
        for values in epochs[rank]:
            schedule.run(seq, pa.table({"value": values}, schema=VALUE_SCHEMA))
        schedule.drain()

    with ThreadPoolExecutor(max_workers=2) as pool:
        [future.result(timeout=60) for future in [pool.submit(run_rank, r) for r in (0, 1)]]

    everything = [1, 2, 3, 10, 11, 12]
    assert sorted(v for batch in seen[0] for v in batch) == everything
    assert sorted(v for batch in seen[1] for v in batch) == everything
