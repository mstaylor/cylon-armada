import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

ray = pytest.importorskip("ray")
pa = pytest.importorskip("pyarrow")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from experiment import exp_b_collectives as ebc  # noqa: E402
from experiment import exp_b_ray  # noqa: E402
from experiment.exp_b_payloads import arrow_table, payload_table_builder  # noqa: E402

NAMESPACE = "expb-ray-test"


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, namespace=NAMESPACE)
    yield
    ray.shutdown()


def _open_all(channel, world_size, name):
    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = [pool.submit(exp_b_ray.open_channel, channel, rank, world_size, name,
                               NAMESPACE, ray) for rank in reversed(range(world_size))]
        return list(reversed([f.result(timeout=60) for f in futures]))


def _sweep(channels, collectives, msg_sizes, payload="float32"):
    world_size = len(channels)

    def one(rank):
        channel = channels[rank]
        return ebc.run({
            "comm": channel.comm, "ctx": channel.ctx, "collectives": collectives,
            "msg_sizes": msg_sizes, "warmup": 2, "reps": 3, "root": 0, "rank": rank,
            "world_size": world_size, "channel": "test", "payload": payload,
            "payload_builder": payload_table_builder(payload),
            "inner_seconds": channel.inner_seconds,
        })

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        return [f.result(timeout=120) for f in [pool.submit(one, r) for r in range(world_size)]]


def _by_collective(rows):
    return {row["collective"]: row for row in rows}


def test_gloo_rows_carry_driver_and_inner_latency(local_ray):
    channels = _open_all("ray-gloo", 2, "gloo-rows")

    rows = _by_collective(_sweep(channels, ["allgather", "broadcast", "allreduce", "reduce",
                                            "barrier"], [4096])[0])

    for name, row in rows.items():
        assert row["unsupported"] is False, name
        assert row["latency_inner_p50_ms"] != "", name
        assert row["latency_p50_ms"] >= row["latency_inner_p50_ms"], name


def test_a_rank_that_opens_first_waits_for_the_group_before_its_first_collective(local_ray):
    import time

    def rank_one():
        channel = exp_b_ray.open_channel("ray-gloo", 1, 2, "gloo-early", NAMESPACE, ray)
        return channel.comm.allreduce(2.0, None)

    def rank_zero():
        time.sleep(1.5)
        channel = exp_b_ray.open_channel("ray-gloo", 0, 2, "gloo-early", NAMESPACE, ray)
        return channel.comm.allreduce(1.0, None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        early = pool.submit(rank_one)
        late = pool.submit(rank_zero)
        assert [early.result(timeout=60), late.result(timeout=60)] == [3.0, 3.0]


def test_gloo_scatter_and_gather_rows_are_unsupported(local_ray):
    channels = _open_all("ray-gloo", 2, "gloo-unsupported")

    rows = _by_collective(_sweep(channels, ["scatter", "scatterv", "gather"], [64])[0])

    assert all(row["unsupported"] is True for row in rows.values())


def test_gloo_allgather_returns_every_ranks_table(local_ray):
    channels = _open_all("ray-gloo", 2, "gloo-values")
    tables = [arrow_table("kv_pairs", 512, float(rank + 1)) for rank in range(2)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda r: channels[r].comm.allgather(tables[r]), range(2)))

    for gathered in results:
        assert [t.equals(expected) for t, expected in zip(gathered, tables)] == [True, True]


def test_gloo_broadcast_hands_every_rank_the_roots_table(local_ray):
    channels = _open_all("ray-gloo", 2, "gloo-bcast")
    tables = [arrow_table("float32", 256, float(rank + 1)) for rank in range(2)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda r: channels[r].comm.broadcast(tables[r], 0, None), range(2)))

    assert all(table.equals(tables[0]) for table in results)


def test_objectstore_reduce_and_allreduce_rows_are_unsupported(local_ray):
    channels = _open_all("ray-objectstore", 2, "os-unsupported")

    rows = _by_collective(_sweep(channels, ["reduce", "allreduce", "scatter"], [64])[0])

    assert all(row["unsupported"] is True for row in rows.values())


def test_objectstore_rows_are_timed_with_blank_inner_columns(local_ray):
    channels = _open_all("ray-objectstore", 2, "os-rows")

    rows = _by_collective(_sweep(channels, ["allgather", "broadcast", "gather", "barrier"],
                                 [4096], payload="kv_pairs")[0])

    for name, row in rows.items():
        assert row["unsupported"] is False, name
        assert row["latency_p50_ms"] > 0.0, name
        assert row["latency_inner_p50_ms"] == "", name


def test_objectstore_allgather_returns_every_ranks_table(local_ray):
    channels = _open_all("ray-objectstore", 3, "os-values")
    tables = [arrow_table("kv_pairs", 512, float(rank + 1)) for rank in range(3)]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda r: channels[r].comm.allgather(tables[r]), range(3)))

    for gathered in results:
        assert all(t.equals(expected) for t, expected in zip(gathered, tables))


def test_objectstore_gather_delivers_every_table_to_root_only(local_ray):
    channels = _open_all("ray-objectstore", 2, "os-gather")
    tables = [arrow_table("float32", 64, float(rank + 1)) for rank in range(2)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda r: channels[r].comm.gather(tables[r], 0), range(2)))

    assert all(t.equals(e) for t, e in zip(results[0], tables))
    assert results[1] is None


@pytest.mark.parametrize("channel", ["ray-gloo", "ray-objectstore"])
def test_a_single_rank_world_still_times_every_supported_collective(local_ray, channel):
    channels = _open_all(channel, 1, f"solo-{channel}")

    rows = _sweep(channels, ["allgather", "broadcast", "barrier"], [64])[0]

    assert all(row["unsupported"] is False and row["latency_p50_ms"] >= 0.0 for row in rows)


class _FakeChannel:
    def __init__(self):
        self.comm = None
        self.ctx = self
        self.inner_seconds = None
        self.run_actors = ("board",)
        self.barrier_actor = "barrier"
        self.barriers = 0

    def barrier(self):
        self.barriers += 1


def _args(tmp_path, runs=2):
    return argparse.Namespace(channel="ray-objectstore", payload="float32", collectives=["barrier"],
                              msg_sizes=[64], warmup=0, reps=1, runs=runs, root=0,
                              output=str(tmp_path))


def _run_rank(tmp_path, rank, world_size, calls, monkeypatch):
    channel = _FakeChannel()
    monkeypatch.setattr(exp_b_ray.ebc, "run", lambda config: [
        {"channel": config["channel"], "payload": config["payload"], "collective": "barrier",
         "msg_size": 0, "N": config["world_size"], "run_id": config["run_id"]}])
    paths = exp_b_ray.run_rank(
        _args(tmp_path), rank, world_size, "cn", redis_client=None, ray_api=None,
        form_cluster=lambda *a: calls.append(("form",) + a[:3]),
        release=lambda *a: calls.append(("release",) + a[:4]),
        open_channel=lambda *a: calls.append(("open",) + a[:4]) or channel,
        namespace=NAMESPACE)
    return paths, channel


def test_run_rank_forms_the_cluster_once_and_writes_one_csv_per_run(tmp_path, monkeypatch):
    calls = []

    paths, channel = _run_rank(tmp_path, 0, 2, calls, monkeypatch)

    assert [c[0] for c in calls] == ["form", "open", "release"]
    assert calls[1] == ("open", "ray-objectstore", 0, 2, "expb_cn")
    assert calls[2] == ("release", "barrier", ("board",), 0, 2)
    assert [os.path.basename(p) for p in paths] == ["run1_exp_b_collectives_results.csv",
                                                    "run2_exp_b_collectives_results.csv"]
    assert channel.barriers == 2
    with open(paths[1]) as f:
        row = next(csv.DictReader(f))
    assert (row["channel"], row["payload"], row["run_id"]) == ("ray-objectstore", "float32", "2")


def test_only_the_root_rank_writes_results(tmp_path, monkeypatch):
    paths, _ = _run_rank(tmp_path, 1, 2, [], monkeypatch)

    assert paths == []
    assert os.listdir(tmp_path) == []