"""lambda_entry3 tells the inference script whether this invocation is the container's first."""

import importlib.util
import json
import os
import sys
import time

import pytest
from botocore.exceptions import ClientError

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


class _FakeS3:
    def __init__(self, present=(), appear_after_heads=None):
        self.objects = {key: {} for key in present}
        self.heads = 0
        self.appear_after_heads = appear_after_heads

    def head_object(self, Bucket, Key):
        self.heads += 1
        if self.appear_after_heads is not None and self.heads >= self.appear_after_heads:
            self.objects.setdefault(Key, {"source": "peer"})
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = json.loads(Body)


def _python(code):
    return [sys.executable, "-c", code]


def test_a_child_that_exits_cleanly_returns_zero_and_writes_no_marker():
    entry = _load()
    s3 = _FakeS3()
    assert entry.run_watching_for_abort(_python("pass"), s3, "bucket", "run/abort.json", 0.05) == 0
    assert s3.objects == {}


def test_a_child_that_fails_writes_the_abort_marker_for_its_peers(monkeypatch):
    monkeypatch.setenv("RANK", "222")
    entry = _load()
    s3 = _FakeS3()
    assert entry.run_watching_for_abort(_python("raise SystemExit(3)"), s3, "bucket", "run/abort.json", 0.05) == 3
    assert s3.objects["run/abort.json"]["rank"] == "222"
    assert s3.objects["run/abort.json"]["returncode"] == 3


def test_a_marker_from_a_peer_stops_a_hung_child_within_a_poll():
    entry = _load()
    s3 = _FakeS3(appear_after_heads=2)
    started = time.monotonic()
    with pytest.raises(entry.RunAborted, match="run/abort.json"):
        entry.run_watching_for_abort(_python("import time; time.sleep(60)"), s3, "bucket", "run/abort.json", 0.05)
    assert time.monotonic() - started < 10


def test_execute_script_without_an_abort_key_runs_the_child_unwatched(monkeypatch, tmp_path):
    monkeypatch.delenv("ABORT_KEY", raising=False)
    entry = _load()
    script = tmp_path / "ok.py"
    script.write_text("pass\n")
    monkeypatch.setattr(entry, "get_file", lambda **kw: object())
    entry.execute_script({"s3_object_type": "file", "script": str(script), "s3_bucket": "b",
                          "s3_object_name": "o", "args": None})


def test_execute_script_with_an_abort_key_requires_a_poll_interval(monkeypatch, tmp_path):
    monkeypatch.setenv("ABORT_KEY", "run/abort.json")
    monkeypatch.delenv("ABORT_POLL_S", raising=False)
    entry = _load()
    script = tmp_path / "ok.py"
    script.write_text("pass\n")
    monkeypatch.setattr(entry, "get_file", lambda **kw: object())
    with pytest.raises(entry.MissingAbortPollInterval):
        entry.execute_script({"s3_object_type": "file", "script": str(script), "s3_bucket": "b",
                              "s3_object_name": "o", "args": None})
