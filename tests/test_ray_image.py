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


def test_the_ray_image_tag_stays_gated_until_something_consumes_it():
    """The variable exists, but no Terraform resource reads it and the sweep
    driver's task definition is fixed — run_task cannot override a container
    image, so a Ray arm launched today would run the cosmic image without Ray.
    Asserting the variable alone would be a green test for an unreachable
    image. This ties the two together: whoever wires a Ray task definition has
    to take the arms out of ARMS_WITHOUT_EXECUTOR in the same change, and
    until then the gate must hold."""
    terraform = _read("target/aws/scripts/terraform/variables.tf")
    assert re.search(r'variable "ray_image_tag"', terraform), "ray_image_tag variable is missing"

    consumed = re.search(r"var\.ray_image_tag", _read_tree("target/aws/scripts/terraform"))
    driver = _read("target/aws/scripts/experiment/fargate_cosmic_poc.py")
    gated = re.search(r'ARMS_WITHOUT_EXECUTOR\s*=\s*\("ray-native",\s*"ray-cylon"\)', driver)

    assert consumed or gated, (
        "no Terraform resource consumes ray_image_tag, so both Ray arms must "
        "stay in ARMS_WITHOUT_EXECUTOR"
    )
