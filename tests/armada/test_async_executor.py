"""ArmadaAsyncExecutor shares contexts as ray-native does, over a Cylon exchange.

At the end of each epoch a rank publishes its contribution and upserts it with
every peer table that has arrived, with no barrier inside the epochs. While the
model calls run on a worker, the calling thread keeps the channel moving, and
every Cylon step stays on the calling thread.
"""

import threading
import time

import pyarrow as pa
import pytest

from armada.async_executor import ArmadaAsyncExecutor, NoPeersExchange
from armada.operator import ArmadaOperator
from cylon_armada.dag_compiler import CollectivePattern

VALUE_SCHEMA = pa.schema([("value", pa.int64())])
OFFLOADED = ("Reason",)


def _values(table):
    return sorted(table.column("value").to_pylist())


def _epoch(*values):
    return pa.table({"value": list(values)}, schema=VALUE_SCHEMA)


def _sequence(log, reason_s=0.0, reason_error=None):
    def retrieve(table):
        log.append(("retrieve", _values(table), threading.get_ident()))
        return table

    def reason(table):
        log.append(("reason", _values(table), threading.get_ident()))
        time.sleep(reason_s)
        if reason_error is not None:
            raise reason_error
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


class FakeExchange:
    def __init__(self, log, late=(), busy_polls=0):
        self.log = log
        self.queued = []
        self.late = list(late)
        self.publishes = []
        self.busy_polls = busy_polls
        self.finish_timeouts = []

    def publish(self, table, reference):
        self.log.append(("publish", _values(table), threading.get_ident()))
        self.publishes.append((_values(table), reference))

    def drain(self):
        self.log.append(("drain", None, threading.get_ident()))
        arrived, self.queued = self.queued, []
        return [(1, 0, table) for table in arrived]

    def idle(self):
        self.log.append(("idle", None, threading.get_ident()))
        if self.busy_polls:
            self.busy_polls -= 1
            return False
        return True

    def finish_and_drain(self, timeout_s):
        self.finish_timeouts.append(timeout_s)
        self.log.append(("finish", None, threading.get_ident()))
        late, self.late = self.late, []
        return [(1, 0, table) for table in late]


class Factory:
    def __init__(self, exchange):
        self.exchange = exchange
        self.schemas = []

    def __call__(self, schema):
        self.schemas.append(schema)
        return self.exchange


def _upserts(log):
    return [values for step, values, _ in log if step == "upsert"]


def test_own_rows_and_arrived_peer_tables_reach_upsert_together_at_epoch_end():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    executor.run(seq, _epoch(1))
    exchange.queued.append(_epoch(101))
    executor.run(seq, _epoch(2))
    executor.close()

    assert _upserts(log) == [[1], [2, 101]]


def test_the_channel_is_progressed_while_model_calls_run():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log, reason_s=0.2)

    executor.run(seq, _epoch(1))
    log.clear()
    executor.run(seq, _epoch(2))
    executor.close()

    steps = [step for step, _, _ in log]
    assert steps[:steps.index("publish")].count("drain") >= 5


def test_publish_once_per_epoch_with_rows_and_never_for_an_empty_epoch():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    for epoch in (_epoch(1), _epoch(), _epoch(3)):
        executor.run(seq, epoch)
    executor.close()

    assert exchange.publishes == [([1], 0), ([3], 2)]


def test_every_cylon_step_stays_on_the_calling_thread():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log, reason_s=0.05)

    executor.run(seq, _epoch(1))
    executor.run(seq, _epoch(2))
    executor.finish(seq, timeout_s=5.0)

    caller = threading.get_ident()
    assert {step for step, _, thread in log if thread == caller} == {
        "retrieve", "drain", "idle", "publish", "upsert", "finish"}
    assert {step for step, _, thread in log if thread != caller} == {"reason"}


def test_every_peer_table_reaches_upsert_exactly_once():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    original_publish = exchange.publish

    def publish_then_receive(table, reference):
        original_publish(table, reference)
        exchange.queued.append(_epoch(200 + reference))

    exchange.publish = publish_then_receive
    seq = _sequence(log)

    exchange.queued.append(_epoch(100))
    executor.run(seq, _epoch(1))
    exchange.queued.append(_epoch(110))
    executor.run(seq, _epoch(2))
    executor.run(seq, _epoch(3))
    exchange.late.append(_epoch(300))
    executor.finish(seq, timeout_s=5.0)

    upserted = sorted(v for values in _upserts(log) for v in values)
    assert upserted == [1, 2, 3, 100, 110, 200, 201, 202, 300]


def test_the_exchange_is_created_once_from_the_first_contribution_schema():
    log = []
    factory = Factory(FakeExchange(log))
    executor = ArmadaAsyncExecutor(factory, OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    executor.run(seq, _epoch(1))
    executor.run(seq, _epoch(2))
    executor.close()

    assert factory.schemas == [VALUE_SCHEMA]


def test_a_rank_with_no_rows_still_creates_the_exchange_and_finishes():
    log = []
    exchange = FakeExchange(log)
    factory = Factory(exchange)
    executor = ArmadaAsyncExecutor(factory, OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    executor.run(seq, _epoch())
    executor.run(seq, _epoch())
    executor.finish(seq, timeout_s=5.0)

    assert factory.schemas == [VALUE_SCHEMA]
    assert exchange.publishes == []
    assert ("finish", None) in [(step, values) for step, values, _ in log]


def test_finish_runs_late_tables_through_upsert_and_returns_none_when_nothing_is_late():
    log = []
    exchange = FakeExchange(log, late=[_epoch(7)])
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)
    executor.run(seq, _epoch(1))

    result = executor.finish(seq, timeout_s=5.0)

    assert _values(result) == [7]
    assert _upserts(log)[-1] == [7]

    quiet_log = []
    quiet = ArmadaAsyncExecutor(Factory(FakeExchange(quiet_log)), OFFLOADED, poll_s=0.01, settle_s=1.0)
    quiet.run(_sequence(quiet_log), _epoch(1))
    assert quiet.finish(_sequence(quiet_log), timeout_s=5.0) is None


def test_a_model_call_failure_surfaces_on_the_calling_thread():
    log = []
    executor = ArmadaAsyncExecutor(Factory(FakeExchange(log)), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log, reason_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        executor.run(seq, _epoch(1))
    executor.close()


def test_no_peers_exchange_runs_a_single_rank_with_no_traffic():
    log = []
    executor = ArmadaAsyncExecutor(lambda schema: NoPeersExchange(), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    executor.run(seq, _epoch(1, 2))
    assert executor.finish(seq, timeout_s=5.0) is None
    assert _upserts(log) == [[1, 2]]


def test_drain_has_nothing_pending_for_run_epochs():
    executor = ArmadaAsyncExecutor(Factory(FakeExchange([])), OFFLOADED, poll_s=0.01, settle_s=1.0)
    assert executor.drain() == []
    executor.close()

def test_epoch_end_keeps_progressing_until_the_exchange_is_idle():
    log = []
    exchange = FakeExchange(log, busy_polls=4)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)

    executor.run(seq, _epoch(1))
    executor.close()

    steps = [step for step, _, _ in log]
    after_publish = steps[steps.index("publish"):steps.index("upsert")]
    assert after_publish.count("drain") >= 5


def test_a_table_arriving_while_settling_is_upserted_in_that_epoch():
    log = []
    exchange = FakeExchange(log, busy_polls=2)
    original_idle = exchange.idle

    def idle_then_deliver():
        if exchange.busy_polls == 1:
            exchange.queued.append(_epoch(150))
        return original_idle()

    exchange.idle = idle_then_deliver
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)

    executor.run(_sequence(log), _epoch(1))
    executor.close()

    assert _upserts(log) == [[1, 150]]


def test_settling_gives_up_at_the_settle_bound():
    log = []
    exchange = FakeExchange(log, busy_polls=10 ** 9)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=0.05)

    started = time.monotonic()
    executor.run(_sequence(log), _epoch(1))
    executor.close()

    assert time.monotonic() - started < 1.0
    assert _upserts(log) == [[1]]


def test_finish_passes_its_timeout_to_the_exchange():
    log = []
    exchange = FakeExchange(log)
    executor = ArmadaAsyncExecutor(Factory(exchange), OFFLOADED, poll_s=0.01, settle_s=1.0)
    seq = _sequence(log)
    executor.run(seq, _epoch(1))

    executor.finish(seq, timeout_s=7.5)

    assert exchange.finish_timeouts == [7.5]
