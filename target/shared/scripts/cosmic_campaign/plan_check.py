"""Turn LLM output into Step Functions inputs and check them against the reference."""

import json
import re

PLAN_FIELDS = ("bucket", "file_limit", "world_size", "batch_size", "object_type", "S3_object_name",
               "script", "result_path", "data_bucket", "data_prefix")


class PlanParseError(ValueError):
    pass


def parse_plans(text):
    for candidate in re.findall(r"(\[.*\]|\{.*\})", text, re.S):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        plans = value if isinstance(value, list) else [value]
        if plans and all(isinstance(p, dict) for p in plans):
            return plans
    raise PlanParseError(f"no JSON plan in model output: {text[:200]!r}")


def validate_plan(plan):
    return [f"missing {field}" for field in PLAN_FIELDS if plan.get(field) in (None, "")]


def mismatched_fields(plan, reference):
    wrong = [field for field in PLAN_FIELDS if str(plan.get(field)) != str(reference.get(field))]
    return wrong + [f"extra {key}" for key in plan if key not in PLAN_FIELDS]
