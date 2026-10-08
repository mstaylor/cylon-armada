"""Cosmic AI Track 1 Arm B (FMI direct) Lambda scripts, faked at the FMIBridge seam.

The fake replaces FMIBridge only: tables cross it as real pycylon Tables built in a
local CylonContext, so the Arrow and pycylon conversion in inference_FMI.py runs for
real across concurrent rank threads, without a rendezvous server or AWS.

Needs the pycylon in-place build on PYTHONPATH (CLAUDE.md environment setup).
Run: pytest tests/cosmic_ai_lambda/test_arm_b.py -v
"""

import json
import os
import sys
import threading
import types

import pytest

pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pycylon = pytest.importorskip("pycylon")
from pycylon import CylonContext, Table  # noqa: E402

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_LAMBDA_SRC = os.path.join(_REPO, "target", "aws", "scripts", "terraform-cosmic-ai", "lambda_src")
_SHARED_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
sys.path.insert(0, os.path.abspath(_SHARED_SCRIPTS))
sys.path.insert(0, os.path.abspath(_LAMBDA_SRC))

import inference  # noqa: E402
import inference_FMI  # noqa: E402
import initializer  # noqa: E402
import initializer_FMI  # noqa: E402
import summarizer_FMI  # noqa: E402
from communicator.fmi_bridge import FMIBridge  # noqa: E402


class _World:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=10)
        self.slot = None
        self.parts = {}


_WORLDS = {}
_WORLDS_LOCK = threading.Lock()


class _FakeBridge:
    instances = []

    def __init__(self, world_size, rank, channel_type, rendezvous_host, rendezvous_port,
                 comm_name, maxtimeout, nonblocking, assigned_rank=None, initialises=True):
        self.world_size = world_size
        self.rank = rank if assigned_rank is None else assigned_rank
        self.settings = dict(channel_type=channel_type, rendezvous_host=rendezvous_host,
                             rendezvous_port=rendezvous_port, comm_name=comm_name,
                             maxtimeout=maxtimeout, nonblocking=nonblocking)
        self.available = world_size > 1 and initialises
        self.context = CylonContext(config=None, distributed=False) if self.available else None
        self.finalized = False
        with _WORLDS_LOCK:
            self.world = _WORLDS.setdefault(comm_name, _World(world_size))
            _FakeBridge.instances.append(self)

    def barrier(self):
        if self.available:
            self.world.barrier.wait()

    def broadcast(self, table, root):
        assert isinstance(table, Table)
        if self.rank == root:
            self.world.slot = table.to_arrow()
        self.world.barrier.wait()
        received = Table.from_arrow(self.context, self.world.slot)
        self.world.barrier.wait()
        return received

    def gather(self, table, root):
        assert isinstance(table, Table)
        self.world.parts[self.rank] = table.to_arrow()
        self.world.barrier.wait()
        result = []
        if self.rank == root:
            result = [Table.from_arrow(self.context, self.world.parts[r]) for r in range(self.world.size)]
        self.world.barrier.wait()
        return result

    def finalize(self):
        self.finalized = True


class _Body:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.lock = threading.Lock()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        with self.lock:
            self.objects[(Bucket, Key)] = Body.encode() if isinstance(Body, str) else Body


def _files(prefix, n):
    return [f"{prefix}/{i}.pt" for i in range(1, n + 1)]


@pytest.mark.parametrize("n_files,world_size,expected", [
    (2, 1, {0: ["10MB/1.pt", "10MB/2.pt"]}),
    (3, 2, {0: ["10MB/1.pt", "10MB/2.pt"], 1: "10MB/3.pt"}),
    (2, 2, {0: "10MB/1.pt", 1: "10MB/2.pt"}),
    (1, 3, {0: "10MB/1.pt", 1: "10MB/1.pt", 2: "10MB/1.pt"}),
    (2, 3, {0: "10MB/1.pt", 1: "10MB/2.pt", 2: "10MB/1.pt"}),
])
def test_assign_partitions_matches_published_mapping(n_files, world_size, expected):
    assert initializer.assign_partitions(_files("10MB", n_files), world_size) == expected


@pytest.mark.parametrize("data_map,world_size", [
    ({"0": "10MB/1.pt"}, 1),
    ({"0": ["10MB/1.pt", "10MB/12.pt"], "1": "10MB/3.pt"}, 2),
    ({"0": "10MB/7.pt", "1": "10MB/7.pt", "2": ["10MB/8.pt", "10MB/9.pt", "10MB/10.pt"]}, 3),
])
def test_partition_map_round_trips_through_an_arrow_table(data_map, world_size):
    table = inference_FMI.encode_data_map(data_map, world_size, "10MB")
    assert table.schema == inference_FMI.PARTITION_MAP_SCHEMA
    for rank in range(world_size):
        assert inference_FMI.decode_rank_paths(table, rank, "10MB") == data_map[str(rank)]


def test_partition_key_outside_prefix_is_rejected():
    with pytest.raises(ValueError):
        inference_FMI.encode_data_map({"0": "25MB/1.pt"}, 1, "10MB")


def test_rank_without_a_partition_is_rejected():
    table = inference_FMI.encode_data_map({"0": "10MB/1.pt"}, 1, "10MB")
    with pytest.raises(ValueError):
        inference_FMI.decode_rank_paths(table, 1, "10MB")


def test_real_fmibridge_at_world_size_one_is_unavailable_with_no_context():
    bridge = FMIBridge(world_size=1, rank=0, channel_type="direct")
    assert not bridge.available
    assert bridge.context is None


@pytest.mark.parametrize("env,expected", [
    ({}, {"code_fetch_s": None, "init_s": None, "cold_start": None}),
    ({"LAMBDA_ENTRY_TS": "100.0"}, {"code_fetch_s": None, "init_s": 10.0, "cold_start": None}),
    ({"LAMBDA_ENTRY_TS": "100.0", "CODE_FETCH_END_TS": "104.0"},
     {"code_fetch_s": 4.0, "init_s": 6.0, "cold_start": None}),
    ({"LAMBDA_ENTRY_TS": "100.0", "CODE_FETCH_END_TS": "104.0", "CONTAINER_COLD_START": "1"},
     {"code_fetch_s": 4.0, "init_s": 6.0, "cold_start": 1}),
    ({"LAMBDA_ENTRY_TS": "100.0", "CODE_FETCH_END_TS": "104.0", "CONTAINER_COLD_START": "0"},
     {"code_fetch_s": 4.0, "init_s": 6.0, "cold_start": 0}),
])
def test_startup_timings_split_code_fetch_out_of_init(monkeypatch, env, expected):
    monkeypatch.delenv("LAMBDA_ENTRY_TS", raising=False)
    monkeypatch.delenv("CODE_FETCH_END_TS", raising=False)
    monkeypatch.delenv("CONTAINER_COLD_START", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert inference.startup_timings(110.0) == expected


def _fake_run_inference(model, dataloader, device, batch_size, rank, result_path, data_path):
    return {
        "inference_s": 1.0 + rank,
        "num_samples": 511,
        "num_batches": 1,
        "total_cpu_time (seconds)": 2.0,
        "total_cpu_memory (MB)": 10.0,
        "execution_time (seconds/batch)": 2.0,
        "throughput_bps": 5.0,
        "sample_persec": 255.5,
    }


def _fake_load(args, stage_timings):
    stage_timings["scatter_s"] = 0.25
    stage_timings["model_load_s"] = 0.5
    return None, None


def _rank_args(rank, world_size, comm_name):
    return types.SimpleNamespace(
        rank=rank, world_size=world_size, batch_size=512, device="cpu",
        data_bucket="bkt", data_prefix="10MB", result_path="res/run1",
        payload_bucket="bkt", payload_key="temp-results/p.json", comm_name=comm_name,
        fmi_channel_type="direct", fmi_options="nonblocking", fmi_max_timeout_ms=1000,
        rendezvous_host="rdv", rendezvous_port=10000,
    )


def _run_world(world_size, data_map, monkeypatch, comm_name, code_fetch_end_ts="2.0", bridge_cls=_FakeBridge):
    payload = {"data_map": data_map}
    s3 = _FakeS3({("bkt", "temp-results/p.json"): json.dumps(payload).encode()})
    monkeypatch.setattr(inference_FMI, "s3_client", s3)
    monkeypatch.setattr(inference_FMI, "run_inference", _fake_run_inference)
    monkeypatch.setattr(inference_FMI, "load_partition_and_model", _fake_load)
    monkeypatch.setattr(inference_FMI, "FMIBridge", bridge_cls)
    monkeypatch.setenv("LAMBDA_ENTRY_TS", "1.0")
    if code_fetch_end_ts is None:
        monkeypatch.delenv("CODE_FETCH_END_TS", raising=False)
    else:
        monkeypatch.setenv("CODE_FETCH_END_TS", code_fetch_end_ts)

    seen_paths, errors = {}, []

    def rank_main(rank):
        args = _rank_args(rank, world_size, comm_name)
        try:
            inference_FMI.run_rank(args, 3.0)
            seen_paths[rank] = args.data_path
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    return s3, seen_paths, errors


@pytest.mark.parametrize("world_size,data_map", [
    (1, {"0": "10MB/1.pt"}),
    (2, {"0": "10MB/1.pt", "1": "10MB/10.pt"}),
    (3, {"0": ["10MB/1.pt", "10MB/2.pt"], "1": "10MB/3.pt", "2": "10MB/4.pt"}),
])
def test_ranks_receive_partitions_by_broadcast_and_root_gathers_every_rank(world_size, data_map, monkeypatch):
    s3, seen_paths, errors = _run_world(world_size, data_map, monkeypatch, f"t-{world_size}")
    assert not errors, errors

    assert seen_paths == {r: data_map[str(r)] for r in range(world_size)}

    records = json.loads(s3.objects[("bkt", "res/run1/combined_data.json")])
    metrics = json.loads(s3.objects[("bkt", "res/run1/aggregate_metrics.json")])
    assert metrics["ranks_aggregated"] == world_size
    assert metrics["aggregate_s"] >= 0
    assert [r["rank"] for r in records] == list(range(world_size))
    for rank, record in enumerate(records):
        for field in inference.STAGE_FIELDS:
            assert record[field] is not None, field
        assert record["code_fetch_s"] == pytest.approx(1.0)
        assert record["init_s"] == pytest.approx(1.0)
        assert record["inference_s"] == pytest.approx(1.0 + rank)
        assert record["data_path"] == data_map[str(rank)]
        assert record["num_samples"] == 511
        staged = sum(record[f] for f in ("comm_init_s", "payload_fetch_s", "scatter_s", "model_load_s",
                                         "barrier_s", "publish_s"))
        assert record["barrier_s"] >= 0 and record["publish_s"] >= 0 and record["comm_init_s"] >= 0
        assert staged <= record["total_s"] + 1e-6


def test_missing_code_fetch_timestamp_travels_as_null_not_zero(monkeypatch):
    s3, _, errors = _run_world(1, {"0": "10MB/1.pt"}, monkeypatch, "t-null", code_fetch_end_ts=None)
    assert not errors, errors
    record = json.loads(s3.objects[("bkt", "res/run1/combined_data.json")])[0]
    assert record["code_fetch_s"] is None
    assert record["init_s"] == pytest.approx(2.0)


def test_bridge_is_built_from_execution_settings_and_finalized(monkeypatch):
    _FakeBridge.instances.clear()
    _, _, errors = _run_world(2, {"0": "10MB/1.pt", "1": "10MB/2.pt"}, monkeypatch, "t-settings")
    assert not errors, errors
    assert len(_FakeBridge.instances) == 2
    for bridge in _FakeBridge.instances:
        assert bridge.settings == dict(channel_type="direct", rendezvous_host="rdv", rendezvous_port=10000,
                                       comm_name="t-settings", maxtimeout=1000, nonblocking=True)
        assert bridge.finalized


@pytest.mark.parametrize("bridge_kwargs", [{"initialises": False}, {"assigned_rank": 5}])
def test_rank_fails_loudly_when_the_communicator_is_unusable(monkeypatch, bridge_kwargs):
    broken = lambda **kw: _FakeBridge(**kw, **bridge_kwargs)
    monkeypatch.setattr(inference_FMI, "FMIBridge", broken)
    with pytest.raises(inference_FMI.FMIStageError):
        inference_FMI.connect(_rank_args(0, 2, f"t-broken-{len(bridge_kwargs)}-{list(bridge_kwargs)[0]}"))


def test_initializer_items_carry_fmi_settings_and_no_data_path(monkeypatch):
    s3 = _FakeS3()
    monkeypatch.setattr(initializer_FMI, "s3_client", s3)
    monkeypatch.setattr(initializer_FMI, "get_file_list", lambda bucket, prefix: _files("10MB", 5))
    event = {
        "bucket": "bkt", "object_type": "folder", "script": "/tmp/x.py", "S3_object_name": "Anomaly Detection",
        "result_path": "res/run2", "file_limit": "2", "world_size": 2, "batch_size": 512,
        "data_bucket": "bkt", "data_prefix": "10MB", "fmi_channel_type": "direct", "fmi_options": "nonblocking",
        "fmi_max_timeout": 900000, "rendezvous_host": "rdv", "rendezvous_port": 10000,
    }
    first = initializer_FMI.lambda_handler(dict(event), None)
    second = initializer_FMI.lambda_handler(dict(event), None)

    items = json.loads(s3.objects[("bkt", first["body"]["S3_KEY"])])
    assert len(items) == 2
    assert all("DATA_PATH" not in item for item in items)
    assert {item["RANK"] for item in items} == {"0", "1"}
    payload = json.loads(s3.objects[("bkt", items[0]["PAYLOAD_KEY"])])
    assert payload["data_map"] == {"0": "10MB/1.pt", "1": "10MB/2.pt"}
    assert items[0]["PAYLOAD_KEY"] != "payload.json"

    other = json.loads(s3.objects[("bkt", second["body"]["S3_KEY"])])
    assert items[0]["FMI_COMM_NAME"] != other[0]["FMI_COMM_NAME"]
    assert len({item["FMI_COMM_NAME"] for item in items}) == 1
    assert items[0]["FMI_CHANNEL_TYPE"] == "direct" and items[0]["FMI_OPTIONS"] == "nonblocking"


FMI_ENVIRONMENT = {
    "FMI_CHANNEL_TYPE": "direct", "FMI_OPTIONS": "nonblocking", "FMI_MAX_TIMEOUT": "300000",
    "RENDEZVOUS_HOST": "rdv.env", "RENDEZVOUS_PORT": "10000",
}

BASE_EVENT = {
    "bucket": "bkt", "object_type": "folder", "script": "/tmp/x.py", "S3_object_name": "Anomaly Detection",
    "result_path": "res/env", "file_limit": "1", "world_size": 1, "batch_size": 512,
    "data_bucket": "bkt", "data_prefix": "10MB",
}


def _clear_fmi_environment(monkeypatch):
    for name in FMI_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _initialize(monkeypatch, event):
    s3 = _FakeS3()
    monkeypatch.setattr(initializer_FMI, "s3_client", s3)
    monkeypatch.setattr(initializer_FMI, "get_file_list", lambda bucket, prefix: _files("10MB", 1))
    response = initializer_FMI.lambda_handler(dict(event), None)
    return json.loads(s3.objects[("bkt", response["body"]["S3_KEY"])])


def test_initializer_reads_fmi_settings_from_environment_when_event_omits_them(monkeypatch):
    _clear_fmi_environment(monkeypatch)
    for name, value in FMI_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    item = _initialize(monkeypatch, BASE_EVENT)[0]

    assert item["FMI_CHANNEL_TYPE"] == "direct" and item["FMI_OPTIONS"] == "nonblocking"
    assert item["FMI_MAX_TIMEOUT"] == "300000"
    assert item["RENDEZVOUS_HOST"] == "rdv.env" and item["RENDEZVOUS_PORT"] == "10000"


def test_initializer_environment_takes_precedence_over_event(monkeypatch):
    _clear_fmi_environment(monkeypatch)
    for name, value in FMI_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    event = {**BASE_EVENT, "rendezvous_host": "rdv.event", "fmi_max_timeout": 1000}

    item = _initialize(monkeypatch, event)[0]

    assert item["RENDEZVOUS_HOST"] == "rdv.env"
    assert item["FMI_MAX_TIMEOUT"] == "300000"


def test_initializer_rejects_event_without_rendezvous(monkeypatch):
    _clear_fmi_environment(monkeypatch)
    with pytest.raises(initializer_FMI.InvalidFMIEvent):
        initializer_FMI.lambda_handler({"bucket": "bkt", "result_path": "r"}, None)


def test_initializer_reports_settings_missing_from_both_environment_and_event(monkeypatch):
    _clear_fmi_environment(monkeypatch)
    monkeypatch.setenv("RENDEZVOUS_HOST", "rdv.env")
    with pytest.raises(initializer_FMI.InvalidFMIEvent, match="RENDEZVOUS_PORT"):
        initializer_FMI.lambda_handler(dict(BASE_EVENT), None)


def _summarize(monkeypatch, metrics, world_size=2):
    items = [{"DATA_BUCKET": "bkt", "RESULT_PATH": "res/run3/", "WORLD_SIZE": str(world_size)}]
    objects = {("bkt", "temp-results/i.json"): json.dumps(items).encode()}
    if metrics is not None:
        objects[("bkt", "res/run3/aggregate_metrics.json")] = json.dumps(metrics).encode()
    s3 = _FakeS3(objects)
    monkeypatch.setattr(summarizer_FMI.boto3, "client", lambda name: s3)
    return summarizer_FMI.lambda_handler({"body": {"S3_BUCKET": "bkt", "S3_KEY": "temp-results/i.json"}}, None)


def test_summarizer_passes_rank0_metrics_through(monkeypatch):
    result = _summarize(monkeypatch, {"aggregate_s": 0.1, "ranks_aggregated": 2, "bytes_aggregated": 64})
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["ranks_aggregated"] == 2 and "summarize_s" in body


def test_summarizer_fails_the_execution_when_a_rank_is_missing(monkeypatch):
    with pytest.raises(summarizer_FMI.IncompleteGatherError, match="ranks_aggregated=1 != world_size=2"):
        _summarize(monkeypatch, {"aggregate_s": 0.1, "ranks_aggregated": 1, "bytes_aggregated": 64})


def test_summarizer_fails_the_execution_when_root_never_wrote_metrics(monkeypatch):
    with pytest.raises(summarizer_FMI.IncompleteGatherError, match="did not complete the gather"):
        _summarize(monkeypatch, None)


def _combine(monkeypatch, world_size):
    s3 = _FakeS3()
    monkeypatch.setattr(inference_FMI, "s3_client", s3)
    data_map = {str(r): f"10MB/{r + 1}.pt" for r in range(world_size)}
    partition_map = inference_FMI.encode_data_map(data_map, world_size, "10MB")
    gathered = [inference_FMI.row_table(r, inference_FMI.GATHERED_FIELDS,
                                        {"num_samples": 511, "num_batches": 1, "inference_s": 1.5,
                                         "cold_start": 1})
                for r in range(world_size)]
    trailing = [inference_FMI.row_table(r, inference_FMI.TRAILING_FIELDS,
                                        {"publish_s": 0.1, "total_s": 7.0})
                for r in range(world_size)]
    args = types.SimpleNamespace(world_size=world_size, rank=0, comm_name="t", batch_size=512,
                                 device="cpu", result_path="res/combine", data_prefix="10MB",
                                 data_bucket="bkt")
    inference_FMI.write_combined_result(args, partition_map, gathered, trailing,
                                        __import__("time").time(), 0.25)
    records = json.loads(s3.objects[("bkt", "res/combine/combined_data.json")])
    metrics = json.loads(s3.objects[("bkt", "res/combine/aggregate_metrics.json")])
    return records, metrics


def test_combined_counts_are_integers_like_arm_a(monkeypatch):
    records, _ = _combine(monkeypatch, 2)
    for record in records:
        assert record["num_samples"] == 511 and isinstance(record["num_samples"], int)
        assert isinstance(record["num_batches"], int)
        assert record["cold_start"] == 1 and isinstance(record["cold_start"], int)
        assert isinstance(record["inference_s"], float)


def test_aggregate_metrics_time_the_trailing_gather_separately(monkeypatch):
    _, metrics = _combine(monkeypatch, 2)
    assert metrics["trailing_gather_s"] == 0.25
    assert 0 <= metrics["aggregate_s"] < 1.0
    assert metrics["ranks_aggregated"] == 2
