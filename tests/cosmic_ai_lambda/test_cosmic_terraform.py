"""terraform-cosmic-ai deploys both Cosmic AI Lambda arms to either account by variables alone.

Terraform is not run by pytest, so this reads the module's .tf source as text.
"""

import glob
import os
import re

_MODULE = os.path.join(os.path.dirname(__file__), "..", "..", "target", "aws", "scripts", "terraform-cosmic-ai")


def _source():
    return "\n".join(open(p).read() for p in sorted(glob.glob(os.path.join(_MODULE, "*.tf"))))


def _block(kind, type_name, name):
    match = re.search(rf'{kind} "{type_name}" "{name}" \{{(.*?)\n\}}', _source(), re.S)
    assert match, f"{kind} {type_name}.{name} is missing"
    return match.group(1)


def _default(variable):
    match = re.search(r"default\s*=\s*(.+)", _block("variable", "", variable).strip()) if False else None
    body = re.search(rf'variable "{variable}" \{{(.*?)\n\}}', _source(), re.S)
    assert body, variable
    match = re.search(r"default\s*=\s*(.+)", body.group(1))
    return match.group(1).strip() if match else None


def test_executor_runs_the_lambda_runtime_client_not_the_bare_handler():
    executor = _block("resource", "aws_lambda_function", "executor")
    assert '"python", "-m", "awslambdaric", "lambda_entry3.handler"' in executor


def test_executor_is_deployed_by_digest_resolved_from_the_tag():
    executor = _block("resource", "aws_lambda_function", "executor")
    assert "image_digest" in executor or "local.executor_image_uri" in executor
    assert 'data "aws_ecr_image" "executor"' in _source()
    assert "executor_image_uri" not in re.findall(r'variable "(\w+)"', _source())


def test_executor_matches_the_measured_configuration():
    assert _default("executor_memory_mb") == "10240"
    assert _default("executor_timeout") == "900"


def test_names_match_the_cylon_account_so_the_campaign_driver_works_in_both():
    assert _default("project_name") == '"cylon-armada-cosmic-ai"'
    assert '"${var.project_name}-workflow"' in _source()
    assert '"${var.project_name}-fmi-workflow"' in _source()


def test_arm_b_is_deployed_with_throttle_only_retries_and_fmi_environment():
    _block("resource", "aws_lambda_function", "fmi_init")
    _block("resource", "aws_lambda_function", "fmi_summarize")
    workflow = _block("resource", "aws_sfn_state_machine", "fmi_workflow")
    assert "fmi_executor_retry_errors" in workflow or "local.fmi_asl_vars" in workflow
    assert _default("fmi_executor_retry_errors").startswith('["Lambda.TooManyRequestsException"]')
    fmi_init = _block("resource", "aws_lambda_function", "fmi_init")
    for name in ("FMI_CHANNEL_TYPE", "FMI_OPTIONS", "FMI_MAX_TIMEOUT", "RENDEZVOUS_HOST", "RENDEZVOUS_PORT"):
        assert name in fmi_init, name


def test_scripts_the_executor_downloads_are_uploaded_unconditionally_and_fail_loud():
    scripts = _block("resource", "aws_s3_object", "scripts")
    assert not re.search(r"^\s*count\s*=", scripts, re.M)
    assert re.search(r"precondition\s*\{[^}]*fileexists\(", scripts, re.S)
    source = _source()
    for script in ("inference.py", "inference_FMI.py", "communicator/__init__.py", "communicator/fmi_bridge.py"):
        assert script in source, script


def test_the_model_must_already_be_in_the_data_bucket():
    _block("data", "aws_s3_object", "model")


def test_plan_fails_when_credentials_belong_to_another_account():
    source = _source()
    assert 'data "aws_caller_identity" "current"' in source
    assert re.search(r"data\.aws_caller_identity\.current\.account_id\s*==\s*var\.account_id", source)


def test_state_machines_wait_for_the_role_policy_that_grants_log_delivery():
    """Created alongside the policy, a state machine fails with 'The state machine IAM Role is not
    authorized to access the Log Destination' because IAM has not applied the policy yet."""
    for name in ("workflow", "fmi_workflow"):
        block = _block("resource", "aws_sfn_state_machine", name)
        assert re.search(r"depends_on\s*=\s*\[[^\]]*aws_iam_role_policy\.step_functions_policy", block), name
