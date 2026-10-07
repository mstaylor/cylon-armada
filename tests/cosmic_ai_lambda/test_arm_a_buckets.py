"""Arm A Lambda code takes its buckets from the execution, never from a hardcoded name.

The cosmicai account's bucket is `cosmicai`; the cylon account's is `cosmicai-data-cylon`.
"""

import json

import pytest

from tests.cosmic_ai_lambda.test_arm_b import _FakeS3, inference

import summarizer  # noqa: E402


class _ListingS3(_FakeS3):
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for b, k in objects if b == Bucket and k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]}

        return _Paginator()


def test_summarizer_reads_and_writes_the_executions_data_bucket(monkeypatch):
    items = [{"RESULT_PATH": "res/run1", "DATA_BUCKET": "cosmicai", "S3_BUCKET": "cosmicai"}]
    s3 = _ListingS3({
        ("cosmicai", "temp-results/items.json"): json.dumps(items).encode(),
        ("cosmicai", "res/run1/0.json"): json.dumps({"rank": 0}).encode(),
        ("cosmicai", "res/run1/1.json"): json.dumps({"rank": 1}).encode(),
        ("cosmicai-data-cylon", "res/run1/9.json"): json.dumps({"rank": 9}).encode(),
    })
    monkeypatch.setattr(summarizer.boto3, "client", lambda name: s3)

    response = summarizer.lambda_handler(
        {"body": {"S3_BUCKET": "cosmicai", "S3_KEY": "temp-results/items.json"}}, None)

    assert response["statusCode"] == 200
    assert json.loads(s3.objects[("cosmicai", "res/run1/combined_data.json")]) == [{"rank": 0}, {"rank": 1}]
    assert json.loads(s3.objects[("cosmicai", "res/run1/aggregate_metrics.json")])["ranks_aggregated"] == 2


def test_worker_reads_payload_from_its_s3_bucket_environment(monkeypatch):
    s3 = _FakeS3({("cosmicai", "payload.json"): json.dumps({"batch_size": 512}).encode()})
    monkeypatch.setattr(inference, "s3_client", s3)
    monkeypatch.setenv("S3_BUCKET", "cosmicai")
    assert inference.fetch_payload() == {"batch_size": 512}


def test_worker_without_s3_bucket_fails_with_a_clear_error(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(KeyError, match="S3_BUCKET"):
        inference.fetch_payload()