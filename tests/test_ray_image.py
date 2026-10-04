"""The Ray image, checked as text.

Docker is not available in this environment, so these assertions read the
Dockerfile. They exist because the failure they guard against is silent: an
image without Ray launches, runs, and fails inside the task after a pull and a
provision have already been paid for.
"""

import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(path):
    with open(os.path.join(ROOT, path)) as handle:
        return handle.read()


def _read_tree(path, suffix=".tf"):
    body = []
    for entry in sorted(os.listdir(os.path.join(ROOT, path))):
        if entry.endswith(suffix):
            body.append(_read(os.path.join(path, entry)))
    return "\n".join(body)


def test_ray_image_builds_on_the_cosmic_image():
    """Arm B needs pycylon with FMI, which the cosmic image already carries.
    Building from a thinner base would mean rebuilding that toolchain."""
    body = _read("docker/Dockerfile.ray.python")
    assert re.search(r"^ARG\s+BASE_IMAGE\s*=\s*cylon-armada-cosmic-python", body, re.M)
    assert re.search(r"^FROM\s+.*\$\{BASE_IMAGE\}", body, re.M)


def test_ray_version_is_pinned():
    """An unpinned Ray makes the baseline a moving target: a rerun months later
    would not be comparing against the same system."""
    body = _read("docker/Dockerfile.ray.python")
    assert re.search(r"^ARG\s+RAY_VERSION\s*=\s*\d+\.\d+", body, re.M)
    assert re.search(r"ray(\[[a-z,]+\])?==\$\{RAY_VERSION\}", body)


def test_ray_install_cannot_move_pyarrow_out_from_under_pycylon():
    """pycylon is a compiled extension bound to the conda env's pyarrow ABI.
    `pip install ray[default]` with no constraints is free to bump pyarrow,
    protobuf or grpcio, and the break would surface as an ImportError inside a
    Fargate task, after the image was built, pushed and pulled. Constraining
    the install and importing both libraries at build time moves that failure
    to the build, where it costs nothing."""
    body = _read("docker/Dockerfile.ray.python")
    assert re.search(r"pip install[^\n]*(\\\s*\n[^\n]*)*-c\s+/tmp/constraints\.txt", body), \
        "ray is installed without a constraints file"
    assert re.search(r"import pycylon,\s*ray", body), \
        "no build-time import check for pycylon and ray together"


def _resource_block(body, kind, name):
    match = re.search(rf'resource "{kind}" "{name}" \{{(.*?)\n\}}\n', body, re.S)
    assert match, f"{kind}.{name} is missing"
    return match.group(1)


def test_a_ray_task_definition_consumes_the_ray_image_tag():
    """run_task cannot override a container image, so the Ray arms need a
    task definition of their own carrying the Ray image. The driver sends
    both Ray arms to that family; the two must agree or a Ray arm runs on the
    cosmic image, which has no Ray."""
    terraform = _read_tree("target/aws/scripts/terraform")
    block = _resource_block(terraform, "aws_ecs_task_definition", "ray_armada")
    assert 'family                   = "${var.project_name}-ray"' in block
    assert "var.ray_image_tag" in block

    driver = _read("target/aws/scripts/experiment/fargate_cosmic_poc.py")
    assert re.search(r'RAY_TASK_DEFINITION\s*=\s*"cylon-armada-ray"', driver)
    assert re.search(r"ARMS_WITHOUT_EXECUTOR\s*=\s*\(\)", driver)


def test_ray_and_cosmic_tasks_get_identical_resources_and_environment():
    """Armada runs on the cosmic family and the Ray arms on the ray family. A
    larger task or a different base environment on one family would be a
    difference the comparison silently measures."""
    terraform = _read_tree("target/aws/scripts/terraform")
    cosmic = _resource_block(terraform, "aws_ecs_task_definition", "cosmic_armada")
    ray = _resource_block(terraform, "aws_ecs_task_definition", "ray_armada")

    for field in ("cpu", "memory", "task_role_arn", "execution_role_arn",
                  "requires_compatibilities", "network_mode", "entryPoint"):
        cosmic_line = re.search(rf"^\s*{field}\s*=.*$", cosmic, re.M).group(0).strip()
        ray_line = re.search(rf"^\s*{field}\s*=.*$", ray, re.M).group(0).strip()
        assert re.sub(r"\s+", " ", cosmic_line) == re.sub(r"\s+", " ", ray_line), field
    assert "local.ecs_env" in ray


def test_the_ray_image_build_fails_on_a_stale_base():
    """The shared scripts are baked into the fmi image and inherited through
    cosmic. A Ray image built on a cosmic image older than the Ray arm code
    would start, form a cluster and then fail on import inside every task.
    Importing the arm's modules at build time moves that to the build."""
    body = _read("docker/Dockerfile.ray.python")
    for module in ("armada.ray_native_executor", "ray_arm.native", "ray_arm.cluster",
                   "armada.run_cosmic_local"):
        assert module in body, f"no build-time import of {module}"
    assert re.search(r"ray --version|bin/ray\b", body), "no check that the ray CLI is installed"


def test_the_post_build_check_compares_both_images_scripts_and_starts_ray():
    """Armada runs on the cosmic image and the Ray arms on the Ray image. If
    the two carry different scripts, the arms run different code. The check
    hashes the script tree in both images against the working tree, and
    starts and stops a Ray head inside the Ray image."""
    body = _read("docker/check_ray_images.sh")
    assert body.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in body
    assert "COSMIC_IMAGE" in body and "RAY_IMAGE" in body
    assert "sha256sum" in body and "target/shared/scripts" in body
    assert "ray start --head" in body and "ray stop" in body
    assert "import pycylon" in body


def test_the_ray_image_build_pins_ray_and_proves_fmi_and_the_context_table():
    """ray-cylon needs pycylon's FMI transport and every arm the ContextTable.
    _import_fmi degrades to a warning when FMI is missing, so the build has to
    assert it rather than merely import it."""
    body = _read("docker/Dockerfile.ray.python")
    assert "assert ray.__version__ == '${RAY_VERSION}'" in body
    assert "assert all(_import_fmi())" in body
    assert "from cylon_armada.context_table import ContextTable" in body


def test_the_post_build_check_verifies_versions_and_that_the_push_landed():
    """A push that silently failed, or a SOCI tag older than its base, leaves
    an arm running the previous image with nothing in the results to show it."""
    body = _read("docker/check_ray_images.sh")
    assert "ray.__version__ == '$RAY_VERSION'" in body
    assert "assert all(_import_fmi())" in body
    assert "ContextTable(8)" in body
    assert "PREVIOUS_RAY_DIGEST" in body and "PREVIOUS_COSMIC_DIGEST" in body
    assert "aws ecr describe-images" in body and "RepoDigests" in body
    assert "SOCI_SUFFIX" in body and "imagePushedAt" in body
