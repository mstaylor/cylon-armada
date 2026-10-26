"""OverlappedEpochs overlaps one epoch's model calls with the previous epoch's exchange.

Every Cylon call (the operators before the offloaded block, the exchange and
MemoryUpsert) stays on the calling thread. Only the offloaded operators, which
make model calls and touch no Cylon state, run on a worker. While the worker
runs epoch e's model calls, the calling thread exchanges epoch e-1's
contribution and ingests the peers' contexts.
"""

import threading
import time

import pyarrow as pa
import pytest

from armada.exchange_schedule import OverlappedEpochs
from armada.operator import ArmadaOperator
from cylon_armada.dag_compiler import CollectivePattern

VALUE_SCHEMA = pa.schema([("value", pa.int64())])
OFFLOADED = ("Reason",)


def _values(table):
    return sorted(table.column("value").to_pylist())


def _sequence(log, reason_s=0.0):
    def retrieve(table):
        log.append(("retrieve", _values(table), threading.get_ident()))
        return table

    def reason(table):
        log.append(("reason", _values(table), threading.get_ident()))
        time.sleep(reason_s)
        return table

    def upsert(table):
        log.append(("upsert", _values(table), threading.get_ident()))
        return table

    return (ArmadaOperator("Retrieve", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                           fn=retrieve)
            | ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                             fn=reason)
            | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, VALUE_SCHEMA,
                             VALUE_SCHEMA, fn=upsert))


def _epoch(*values):
    return pa.table({"value": list(values)}, schema=VALUE_SCHEMA)


def _recording_exchange(log, exchange_s=0.0):
    def exchange(table):
        log.append(("exchange", _values(table), threading.get_ident()))
        time.sleep(exchange_s)
        return [pa.table({"value": [v + 100 for v in table.column("value").to_pylist()]},
                         schema=VALUE_SCHEMA)]

    return exchange


def test_peers_of_an_epoch_are_ingested_while_the_next_epoch_reasons():
    log = []
    schedule = OverlappedEpochs(_recording_exchange(log), OFFLOADED)
    seq = _sequence(log)

    schedule.run(seq, _epoch(1))
    schedule.run(seq, _epoch(2))
    schedule.drain()

    caller = threading.get_ident()
    assert [(step, values) for step, values, thread in log if thread == caller] == [
        ("retrieve", [1]), ("upsert", [1]),
        ("retrieve", [2]), ("exchange", [1]), ("upsert", [101]), ("upsert", [2]),
        ("exchange", [2]), ("upsert", [102])]
    assert [(step, values) for step, values, thread in log if thread != caller] == [
        ("reason", [1]), ("reason", [2])]


def test_every_cylon_step_stays_on_the_calling_thread():
    log = []
    schedule = OverlappedEpochs(_recording_exchange(log), OFFLOADED)
    seq = _sequence(log)
    for value in (1, 2, 3):
        schedule.run(seq, _epoch(value))
    schedule.drain()

    caller = threading.get_ident()
    assert {thread for step, _, thread in log if step != "reason"} == {caller}
    assert {thread for step, _, thread in log if step == "reason"} != {caller}


def test_the_exchange_overlaps_the_next_epochs_model_calls():
    step_s = 0.3
    log = []
    schedule = OverlappedEpochs(_recording_exchange(log, exchange_s=step_s), OFFLOADED)
    seq = _sequence(log, reason_s=step_s)

    started = time.perf_counter()
    for value in (1, 2, 3):
        schedule.run(seq, _epoch(value))
    schedule.drain()
    elapsed = time.perf_counter() - started

    assert elapsed < 3 * 2 * step_s - step_s


def test_every_upsert_ack_is_returned_exactly_once():
    log = []
    schedule = OverlappedEpochs(_recording_exchange(log), OFFLOADED)
    seq = _sequence(log)

    returned = schedule.run(seq, _epoch(1, 2)) + schedule.run(seq, _epoch(3)) + schedule.drain()

    assert sorted(v for t in returned for v in t.column("value").to_pylist()) == [
        1, 2, 3, 101, 102, 103]


def test_drain_with_nothing_run_returns_no_acks():
    assert OverlappedEpochs(_recording_exchange([]), OFFLOADED).drain() == []


def test_an_exchange_failure_surfaces_on_the_calling_thread():
    def failing_exchange(table):
        raise RuntimeError("peer timed out")

    schedule = OverlappedEpochs(failing_exchange, OFFLOADED)
    seq = _sequence([])
    schedule.run(seq, _epoch(1))

    with pytest.raises(RuntimeError, match="peer timed out"):
        schedule.run(seq, _epoch(2))
    schedule.close()


def test_a_model_call_failure_surfaces_on_the_calling_thread():
    def failing_reason(table):
        raise RuntimeError("throttled")

    seq = (ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                          fn=failing_reason)
           | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, VALUE_SCHEMA,
                            VALUE_SCHEMA, fn=lambda t: t))
    schedule = OverlappedEpochs(_recording_exchange([]), OFFLOADED)

    with pytest.raises(RuntimeError, match="throttled"):
        schedule.run(seq, _epoch(1))
    schedule.close()


def test_offloaded_operators_must_sit_directly_before_the_final_allgather():
    seq = (ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                          fn=lambda t: t)
           | ArmadaOperator("Retrieve", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                            fn=lambda t: t)
           | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, VALUE_SCHEMA,
                            VALUE_SCHEMA, fn=lambda t: t))
    schedule = OverlappedEpochs(_recording_exchange([]), OFFLOADED)

    with pytest.raises(ValueError, match="Reason"):
        schedule.run(seq, _epoch(1))
    schedule.close()


def test_a_collective_before_the_final_allgather_is_refused():
    seq = (ArmadaOperator("Retrieve", CollectivePattern.Reduce, VALUE_SCHEMA, VALUE_SCHEMA,
                          fn=lambda t: t)
           | ArmadaOperator("Reason", CollectivePattern.PointToPoint, VALUE_SCHEMA, VALUE_SCHEMA,
                            fn=lambda t: t)
           | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, VALUE_SCHEMA,
                            VALUE_SCHEMA, fn=lambda t: t))
    schedule = OverlappedEpochs(_recording_exchange([]), OFFLOADED)

    with pytest.raises(ValueError, match="Retrieve"):
        schedule.run(seq, _epoch(1))
    schedule.close()