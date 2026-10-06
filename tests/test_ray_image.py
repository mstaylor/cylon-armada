"""Ray in the cosmic image, checked as text.

Docker is not available in this environment, so these assertions read the
Dockerfile and Terraform. They exist because the failure they guard against is
silent: an image without Ray launches, runs, and fails inside the task after a
pull and a provision have already been paid for.
"""

import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
COSMIC_DOCKERFILE = "docker/Dockerfile.cosmic.python"


def _read(path):
    with open(os.path.join(ROOT, path)) as handle:
        return handle.read()


def _read_tree(path, suffix=".tf"):
    body = []
    for entry in sorted(os.listdir(os.path.join(ROOT, path))):
        if entry.endswith(suffix):
            body.append(_read(os.path.join(path, entry)))
    return "\n".join(body)


def _resource_block(body, kind, name):
    match = re.search(rf'resource "{kind}" "{name}" \{{(.*?)\n\}}\n', body, re.S)
    assert match, f"{kind}.{name} is missing"
    return match.group(1)


def test_ray_version_is_pinned():
    """An unpinned Ray makes the baseline a moving target: a rerun months later
    would not be comparing against the same system."""
    body = _read(COSMIC_DOCKERFILE)
    assert re.search(r"^ARG\s+RAY_VERSION\s*=\s*\d+\.\d+", body, re.M)
    assert re.search(r"ray(\[[a-z,]+\])?==\$\{RAY_VERSION\}", body)


def test_ray_install_cannot_move_pyarrow_out_from_under_pycylon():
    """pycylon is a compiled extension bound to the conda env's pyarrow ABI.
    An unconstrained `pip install ray[default]` may bump pyarrow, protobuf or
    grpcio, and the break would surface inside a Fargate task. Constraining the
    install and importing both at build time moves that failure to the build."""
    body = _read(COSMIC_DOCKERFILE)
    assert re.search(r"pip install[^\n]*-c\s+/tmp/constraints\.txt", body), \
        "ray is installed without a constraints file"
    assert re.search(r"import pycylon,\s*ray", body), \
        "no build-time import check for pycylon and ray together"


def test_the_build_proves_ray_fmi_the_context_table_and_the_arm_modules():
    """_import_fmi degrades to a warning when FMI is missing, so the build has
    to assert it. The arm modules are imported so stale scripts fail the build,
    not every task."""
    body = _read(COSMIC_DOCKERFILE)
    assert "assert ray.__version__ == '${RAY_VERSION}'" in body
    assert "assert all(_import_fmi())" in body
    assert "from cylon_armada.context_table import ContextTable" in body
    for module in ("armada.run_cosmic_local", "armada.ray_native_executor",
                   "ray_arm.native", "ray_arm.cluster", "ray_arm.cylon_on_ray"):
        assert module in body, f"no build-time import of {module}"
    assert "ray --version" in body


def test_every_arm_runs_on_the_one_cosmic_task_definition():
    """One image for every Exp E arm, so image size and pull time cannot differ
    between them. The Ray settings ride on that one task definition."""
    terraform = _read_tree("target/aws/scripts/terraform")
    assert 'resource "aws_ecs_task_definition" "ray_armada"' not in terraform
    assert "ray_image_tag" not in terraform

    cosmic = _resource_block(terraform, "aws_ecs_task_definition", "cosmic_armada")
    for name in ("RAY_PORT", "RAY_NAMESPACE", "RAY_RENDEZVOUS_TIMEOUT_S",
                 "RAY_CLUSTER_TIMEOUT_S", "RAY_BARRIER_TIMEOUT_S"):
        assert f'name = "{name}"' in cosmic, name

    driver = _read("target/aws/scripts/experiment/fargate_cosmic_poc.py")
    assert "RAY_TASK_DEFINITION" not in driver


def test_no_lambda_image_carries_ray():
    """Ray belongs only in the Fargate cosmic image; the Lambda images stay
    small for the 10 GB limit and cold starts."""
    for path in ("docker/Dockerfile.fmi.python", "docker/cosmic-ai-lambda/Dockerfile"):
        assert not re.search(r"pip install[^\n]*\bray\b", _read(path)), path
