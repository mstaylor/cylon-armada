"""--epoch-schedule overlapped wires OverlappedEpochs to the arm's own exchange.

Armada and ray-cylon exchange over the FMI bridge, ray-collective over Ray's
allgather. ray-native and langchain have no per-epoch exchange to overlap, so
asking for it is refused rather than silently ignored.
"""

import json
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

import armada.run_cosmic_local as mod
from armada.executor import ArmadaExecutor
from armada.run_cosmic_local import ExecutionBackend, run_epochs


def _table(*values):
    return pa.table({"value": list(values)})


def test_armada_exchange_returns_every_rank_but_this_one():
    tables = [_table(0), _table(1), _table(2)]
    bridge = MagicMock(world_size=3, rank=1, available=True)
    bridge.allgather.return_value = tables

    peers = ArmadaExecutor(bridge).exchange_peers(tables[1], ctx=None)

    assert peers == [tables[0], tables[2]]


def test_armada_exchange_at_one_rank_has_no_peers_and_no_collective():
    bridge = MagicMock(world_size=1, rank=0, available=True)

    assert ArmadaExecutor(bridge).exchange_peers(_table(0), ctx=None) == []
    bridge.allgather.assert_not_called()


class _RecordingSchedule:
    def __init__(self):
        self.ran = []
        self.drained = False

    def run(self, seq, epoch_input):
        self.ran.append(epoch_input)
        return [pa.table({"ack": [True]})]

    def drain(self):
        self.drained = True
        return [pa.table({"ack": [True, True]})]


def test_run_epochs_hands_every_epoch_to_the_schedule_then_drains():
    schedule = _RecordingSchedule()
    shards = [_table(1), _table(2)]

    results = run_epochs(MagicMock(), shards, ExecutionBackend.Armada, schedule=schedule)

    assert schedule.ran == shards
    assert schedule.drained
    assert sum(len(t) for t in results) == 4


def _main_env(monkeypatch, tmp_path, comm):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("COMM_NAME", comm)
    monkeypatch.setenv("CONTEXT_BACKEND", "")
    pricing = MagicMock(source="stub")
    pricing.get_llm_cost.return_value = 0.0
    pricing.get_embedding_cost.return_value = 0.0
    monkeypatch.setattr(mod, "pricing_for", lambda: pricing)
    return str(tmp_path / "record.json")


def test_armada_overlapped_runs_through_the_schedule_and_records_it(monkeypatch, tmp_path):
    out = _main_env(monkeypatch, tmp_path, "armada-overlap")
    bridge = MagicMock(rank=0, world_size=1, available=True, channel_type="direct-redis",
                       _ctx=None)
    monkeypatch.setattr(mod, "FMIBridge", lambda **kwargs: bridge)
    monkeypatch.setattr(mod, "required_peer_map", lambda seq, world_size: "")
    captured = {}
    real_run_epochs = mod.run_epochs

    def capturing_run_epochs(*args, **kwargs):
        captured["schedule"] = kwargs.get("schedule")
        return real_run_epochs(*args, **kwargs)

    monkeypatch.setattr(mod, "run_epochs", capturing_run_epochs)

    assert mod.main(["--backend", "armada", "--epoch-schedule", "overlapped", "--galaxies", "6",
                     "--result-path", out]) == 0
    record = json.load(open(out))

    assert captured["schedule"] is not None
    assert record["epoch_schedule"] == "overlapped"
    assert record["records_written"] == 6
    assert record["acks_visible"] == 6
    assert "error" not in record


def test_synchronous_is_the_default_and_uses_no_schedule(monkeypatch, tmp_path):
    out = _main_env(monkeypatch, tmp_path, "armada-sync")
    bridge = MagicMock(rank=0, world_size=1, available=True, channel_type="direct-redis",
                       _ctx=None)
    monkeypatch.setattr(mod, "FMIBridge", lambda **kwargs: bridge)
    monkeypatch.setattr(mod, "required_peer_map", lambda seq, world_size: "")
    captured = {}
    real_run_epochs = mod.run_epochs

    def capturing_run_epochs(*args, **kwargs):
        captured["schedule"] = kwargs.get("schedule")
        return real_run_epochs(*args, **kwargs)

    monkeypatch.setattr(mod, "run_epochs", capturing_run_epochs)

    assert mod.main(["--backend", "armada", "--galaxies", "6", "--result-path", out]) == 0

    assert captured["schedule"] is None
    assert json.load(open(out))["epoch_schedule"] == "synchronous"


@pytest.mark.parametrize("backend", ["ray-native", "langchain", "isolated"])
def test_overlap_is_refused_where_there_is_no_per_epoch_exchange(monkeypatch, tmp_path, backend):
    out = _main_env(monkeypatch, tmp_path, f"{backend}-overlap")

    with pytest.raises(SystemExit):
        mod.main(["--backend", backend, "--epoch-schedule", "overlapped", "--galaxies", "6",
                  "--result-path", out])



def test_the_record_carries_the_effective_reuse_settings_and_launch_attempt(monkeypatch, tmp_path):
    out = _main_env(monkeypatch, tmp_path, "armada-provenance")
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "0.0005")
    monkeypatch.setenv("OUTLIER_RESIDUAL_THRESHOLD", "0.027677")
    monkeypatch.setenv("LAUNCH_ATTEMPT", "2")
    bridge = MagicMock(rank=0, world_size=1, available=True, channel_type="direct-redis",
                       _ctx=None)
    monkeypatch.setattr(mod, "FMIBridge", lambda **kwargs: bridge)
    monkeypatch.setattr(mod, "required_peer_map", lambda seq, world_size: "")

    assert mod.main(["--backend", "armada", "--galaxies", "6", "--result-path", out]) == 0
    record = json.load(open(out))

    assert record["reuse_key_tolerance"] == 0.0005
    assert record["outlier_threshold"] == 0.027677
    assert record["launch_attempt"] == 2
