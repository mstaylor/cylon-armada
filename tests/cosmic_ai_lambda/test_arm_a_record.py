"""Arm A per-rank record written by the instrumented inference.py, with S3 and the model faked.

Run: pytest tests/cosmic_ai_lambda/test_arm_a_record.py -v
"""

import io
import json
import types

import pytest

torch = pytest.importorskip("torch")

from tests.cosmic_ai_lambda.test_arm_b import _FakeS3, inference


class _PartitionS3(_FakeS3):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def download_fileobj(self, Bucket, Key, Fileobj):
        torch.save(self.dataset, Fileobj)


def test_engine_publishes_every_stage_field_once_per_rank(monkeypatch):
    dataset = torch.utils.data.TensorDataset(torch.zeros(5, 3), torch.zeros(5, 2), torch.zeros(5))
    s3 = _PartitionS3(dataset)
    monkeypatch.setattr(inference, "s3_client", s3)
    monkeypatch.setattr(inference, "load_model", lambda path, device: (lambda batch: batch[0].sum()))

    args = types.SimpleNamespace(data_path="10MB/1.pt", data_bucket="bkt", batch_size=2,
                                 model_path="unused", device="cpu", rank=0, result_path="res/a")
    inference.engine(args, stage_timings={
        "process_start_ts": __import__("time").time(),
        "code_fetch_s": 4.0, "init_s": 6.0, "payload_fetch_s": 0.1, "cold_start": 1,
    })

    record = json.loads(s3.objects[("bkt", "res/a/0.json")])
    for field in inference.STAGE_FIELDS:
        assert record[field] is not None, field
    assert record["code_fetch_s"] == 4.0 and record["init_s"] == 6.0
    assert record["cold_start"] == 1
    assert record["barrier_s"] == 0.0
    assert record["num_samples"] == 5 and record["num_batches"] == 3
    staged = sum(record[f] for f in ("payload_fetch_s", "scatter_s", "model_load_s",
                                     "inference_s", "barrier_s", "publish_s"))
    assert staged <= record["total_s"] + 0.1
