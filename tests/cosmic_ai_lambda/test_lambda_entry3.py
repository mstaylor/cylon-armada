"""lambda_entry3 tells the inference script whether this invocation is the container's first."""

import importlib.util
import os

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docker", "cosmic-ai-lambda", "lambda_entry3.py")


def _load():
    spec = importlib.util.spec_from_file_location("lambda_entry3_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_first_invocation_in_a_container_is_a_cold_start(monkeypatch):
    monkeypatch.delenv("CONTAINER_COLD_START", raising=False)
    entry = _load()
    entry.mark_container_invocation()
    assert os.environ["CONTAINER_COLD_START"] == "1"
    entry.mark_container_invocation()
    assert os.environ["CONTAINER_COLD_START"] == "0"
    assert _load() is not entry
    fresh = _load()
    fresh.mark_container_invocation()
    assert os.environ["CONTAINER_COLD_START"] == "1"
