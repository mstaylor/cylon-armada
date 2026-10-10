import json

import pytest

from cosmic_campaign.plan_check import (PLAN_FIELDS, PlanParseError, mismatched_fields,
                                        parse_plans, validate_plan)

FENCE = "`" * 3
REF = {"bucket": "b", "file_limit": "11", "world_size": 11, "batch_size": 512, "object_type": "folder",
       "S3_object_name": "Anomaly Detection", "script": "/tmp/a.py",
       "result_path": "p/A/result-partition-100MB/1GB/run1", "data_bucket": "b", "data_prefix": "100MB"}


@pytest.mark.parametrize("text", [json.dumps(REF), f"{FENCE}json\n{json.dumps(REF)}\n{FENCE}",
                                  f"Here is the input:\n{json.dumps([REF])}\nDone."])
def test_plans_are_parsed_from_json_with_or_without_fences_and_prose(text):
    assert parse_plans(text) == [REF]


def test_output_without_json_is_a_parse_error_not_a_crash():
    with pytest.raises(PlanParseError):
        parse_plans("I cannot help with that.")


def test_a_complete_plan_is_valid_and_a_short_one_names_what_is_missing():
    assert validate_plan(REF) == []
    problems = validate_plan({k: v for k, v in REF.items() if k != "data_prefix"})
    assert any("data_prefix" in p for p in problems)
    assert set(PLAN_FIELDS) == set(REF)


def test_mismatches_are_reported_per_field_with_light_type_coercion():
    assert mismatched_fields({**REF, "file_limit": 11, "world_size": "11"}, REF) == []
    assert mismatched_fields({**REF, "file_limit": "10"}, REF) == ["file_limit"]


def test_a_key_outside_the_plan_fields_is_a_mismatch():
    assert mismatched_fields({**REF, "fmi_channel_type": "s3"}, REF) == ["extra fmi_channel_type"]