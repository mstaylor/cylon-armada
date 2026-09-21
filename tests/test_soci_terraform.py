"""The SOCI index builder's Terraform config, checked against the spec.

Terraform is not exercised by pytest, so these assertions read the .tf source
as text. They exist because the spec fixes several values whose drift would be
silent: an image that is not indexed still runs, just slowly, so a wrong tag or
a missing ephemeral storage setting produces no error anywhere.

Run: pytest tests/test_soci_terraform.py -v
"""

import os
import re

_TF = os.path.join(os.path.dirname(__file__), "..", "target", "aws", "scripts", "terraform")


def _read(name):
    with open(os.path.join(_TF, name)) as handle:
        return handle.read()


def test_soci_is_on_by_default():
    """Indexing is deliberately on by default so a plain `terraform apply`
    deploys it on any checkout. An off default is worse than it looks here:
    terraform.tfvars is gitignored, so the opt-in does not travel between
    machines, and the gate then creates nothing while reporting success."""
    body = _read("variables.tf")
    block = re.search(r'variable "enable_soci_indexing".*?\n}', body, re.S)
    assert block, "enable_soci_indexing variable is missing"
    assert re.search(r"^\s*default\s*=\s*true\s*$", block.group(0), re.M)


def test_required_deployment_values_have_defaults():
    """These three had no default, so `terraform apply` prompted for them and
    could not run unattended. They are fixed facts about this deployment."""
    body = _read("variables.tf")
    expected = {
        "account_id": '"448324707516"',
        "ecr_repository_name": '"cylon-armada"',
        "results_bucket_name": '"staylor.dev2"',
    }
    for name, value in expected.items():
        block = re.search(r'variable "%s".*?\n}' % name, body, re.S)
        assert block, "%s variable is missing" % name
        assert re.search(
            r"^\s*default\s*=\s*%s\s*$" % re.escape(value), block.group(0), re.M
        ), "%s must default to %s" % (name, value)


def test_generator_memory_clears_the_measured_peak():
    """A real run against the 1.87 GB cosmic image peaked at 1017 MB with a
    1024 MB limit, 7 MB of headroom. An OOM here is silent in the way that
    matters: the index is simply absent and the pull time is unchanged, which
    reads as "SOCI did not help" rather than as a failure."""
    body = _read("variables.tf")
    block = re.search(r'variable "soci_lambda_memory_mb".*?\n}', body, re.S)
    assert block, "soci_lambda_memory_mb variable is missing"
    default = re.search(r"^\s*default\s*=\s*(\d+)\s*$", block.group(0), re.M)
    assert default, "soci_lambda_memory_mb has no default"
    assert int(default.group(1)) >= 2048, "must clear the 1017 MB measured peak"


def test_only_the_cosmic_tag_is_indexed_by_default():
    """The FMI base image is pulled by Lambda, not Fargate, so indexing it
    spends Lambda time for no gain."""
    body = _read("variables.tf")
    block = re.search(r'variable "soci_indexed_tag".*?\n}', body, re.S)
    assert block, "soci_indexed_tag variable is missing"
    assert re.search(
        r'^\s*default\s*=\s*"cylon-armada-cosmic-python"\s*$', block.group(0), re.M
    )


def test_generator_lambda_matches_the_spec_settings():
    """These four values are fixed by the spec. Ephemeral storage is the one
    that matters most: the generator materialises the image under /tmp and the
    cosmic image is 1.88 GB, so the 512 MB default would fail every index.

    Assertions are anchored to the actual assignment (with a trailing \\b) so
    that, e.g., "1024" cannot be satisfied by matching inside "10240"."""
    body = _read("soci.tf")
    block = re.search(r'resource "aws_lambda_function" "soci_index_generator".*?\n}', body, re.S)
    assert block, "soci_index_generator lambda is missing"
    text = block.group(0)
    assert re.search(r'runtime\s*=\s*"provided\.al2023"', text)
    assert re.search(
        r"memory_size\s*=\s*var\.soci_lambda_memory_mb\b", text
    ), "memory must come from the variable, not a literal"
    assert re.search(r"timeout\s*=\s*900\b", text), "timeout must be 900 seconds"
    assert re.search(r"size\s*=\s*10240\b", text), "ephemeral storage must be 10240 MB"


def test_event_rule_filters_to_push_success_and_one_tag():
    """A rule that fires on every ECR event would index every image on every
    push, including failed pushes, and spend Lambda time doing it. Assertions
    are anchored on the actual key/value pairs of the event pattern rather than
    loose token membership."""
    body = _read("soci.tf")
    block = re.search(r'resource "aws_cloudwatch_event_rule" "soci_image_pushed".*?\n}', body, re.S)
    assert block, "soci_image_pushed rule is missing"
    text = block.group(0)
    assert re.search(r'source\s*=\s*\["aws\.ecr"\]', text)
    assert re.search(r'"action-type"\s*=\s*\["PUSH"\]', text)
    assert re.search(r'result\s*=\s*\["SUCCESS"\]', text)
    assert re.search(
        r'"image-tag"\s*=\s*\[var\.soci_indexed_tag\]', text
    ), "the rule must filter on the indexed tag"


def test_ecr_write_permissions_are_scoped_to_the_repository():
    """GetAuthorizationToken has to be unscoped because ECR requires it. The
    layer and image writes do not, and an unscoped PutImage would let this
    Lambda overwrite any image in the registry. The policy is an HCL
    jsonencode() call, not literal JSON (unquoted references, trailing
    commas), so json.loads cannot parse it without evaluating Terraform
    expressions; instead the Resource assignment immediately following
    ecr:PutImage is asserted to equal the repository ARN reference exactly."""
    body = _read("soci.tf")
    assert "ecr:PutImage" in body
    after_put_image = body.split("ecr:PutImage", 1)[1]
    resource_match = re.search(r"Resource\s*=\s*(\S+)", after_put_image)
    assert resource_match, "no Resource assignment found after ecr:PutImage"
    assert resource_match.group(1) == "data.aws_ecr_repository.main.arn"