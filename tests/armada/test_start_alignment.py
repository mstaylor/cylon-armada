"""Every synchronized arm aligns its ranks after inference and before the clock.

A rank whose Fargate task came up early, or whose AstroMAE shard finished
first, would otherwise wait for its peers inside run_s. The wait is recorded
as start_barrier_s, apart from both establish_s and run_s.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import armada.run_cosmic_local as mod

ESTABLISH_S = 0.5
START_ALIGN_S = 3.0
RUN_S = 2.0


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _main_env(monkeypatch, tmp_path, comm):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("COMM_NAME", comm)
    monkeypatch.setenv("CONTEXT_BACKEND", "")
    pricing = MagicMock(source="stub")
    pricing.get_llm_cost.return_value = 0.0
    pricing.get_embedding_cost.return_value = 0.0
    monkeypatch.setattr(mod, "pricing_for", lambda: pricing)
    monkeypatch.setattr(mod, "_redis_client_for_rendezvous", lambda: None)
    clock = _Clock()
    monkeypatch.setattr(mod, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    return str(tmp_path / "record.json"), clock


def _record_inference_and_run(monkeypatch, clock, events):
    def fake_inference(data_path, model_path, world_size, rank, batch_size, device, n_galaxies):
        events.append("inference")
        predictions, true_redshifts, magnitudes = mod.astromae_fixture(n_galaxies)
        return predictions, true_redshifts, magnitudes, None, (0, n_galaxies)

    def fake_run_epochs(*args, **kwargs):
        events.append("run")
        clock.advance(RUN_S)
        return []

    monkeypatch.setattr(mod, "astromae_inference", fake_inference)
    monkeypatch.setattr(mod, "run_epochs", fake_run_epochs)


INFERENCE_ARGS = ["--galaxies", "6", "--real-inference", "--data-path", "d.pt",
                  "--model-path", "m.pt"]


def test_armada_aligns_after_inference_and_outside_run_s(monkeypatch, tmp_path):
    out, clock = _main_env(monkeypatch, tmp_path, "armada-align")
    events = []

    class FakeBridge:
        rank = 0
        world_size = 1
        available = True
        channel_type = "direct-redis"
        _ctx = None

        def __init__(self, **kwargs):
            clock.advance(ESTABLISH_S)

        def barrier(self):
            events.append("barrier")
            clock.advance(START_ALIGN_S)

        def finalize(self):
            pass

    monkeypatch.setattr(mod, "FMIBridge", FakeBridge)
    monkeypatch.setattr(mod, "required_peer_map", lambda seq, world_size: "")
    _record_inference_and_run(monkeypatch, clock, events)

    assert mod.main(["--backend", "armada", *INFERENCE_ARGS, "--result-path", out]) == 0
    record = json.load(open(out))

    assert events[:3] == ["inference", "barrier", "run"]
    assert record["establish_s"] == ESTABLISH_S
    assert record["start_barrier_s"] == START_ALIGN_S
    assert record["run_s"] == RUN_S


@pytest.mark.parametrize("backend, executor_name, actors_name", [
    ("ray-native", "RayNativeExecutor", "ray_native_actors"),
    ("ray-collective", "RayCollectiveExecutor", "ray_collective_actors"),
])
def test_ray_arms_align_after_inference_and_outside_establish_s(
        monkeypatch, tmp_path, backend, executor_name, actors_name):
    out, clock = _main_env(monkeypatch, tmp_path, f"{backend}-align")
    events = []

    def fake_actors(*args, **kwargs):
        clock.advance(ESTABLISH_S)
        if backend == "ray-native":
            return mod.RayNativeHandles(shard="shard", registry="registry", barrier="barrier")
        return mod.RayCollectiveHandles(exchanger="exchanger", barrier="barrier", rank=0,
                                        timings=mod.ConversionTimings(),
                                        gloo={"interface": "eth0", "node_ip": "10.0.0.1",
                                              "source": "derived"})

    class FakeExecutor:
        def __init__(self, *args):
            pass

        def start(self, barrier, timeout_s):
            events.append("barrier")
            clock.advance(START_ALIGN_S)

        def finish(self, seq, barrier, timeout_s):
            return None

    fake_ray = SimpleNamespace(shutdown=lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    monkeypatch.setattr(mod, "form_ray_cluster", lambda *args: None)
    monkeypatch.setattr(mod, actors_name, fake_actors)
    monkeypatch.setattr(mod, executor_name, FakeExecutor)
    monkeypatch.setattr(mod, "release_ray_native_run", lambda *args: None)
    monkeypatch.setattr(mod, "release_ray_collective_run", lambda *args: None)
    _record_inference_and_run(monkeypatch, clock, events)

    assert mod.main(["--backend", backend, *INFERENCE_ARGS, "--result-path", out]) == 0
    record = json.load(open(out))

    assert events[:3] == ["inference", "barrier", "run"]
    assert record["establish_s"] == ESTABLISH_S
    assert record["start_barrier_s"] == START_ALIGN_S
    assert record["run_s"] == RUN_S