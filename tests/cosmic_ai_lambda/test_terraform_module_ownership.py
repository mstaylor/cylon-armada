"""The Cosmic AI Lambda workflow belongs to terraform-cosmic-ai alone.

Its project_name defaults to cylon-armada-cosmic-ai, so its functions and state
machines are named cylon-armada-cosmic-ai-*. If the main module declares the
same names, whichever module applies second in an account fails with
ResourceConflictException. Terraform is not run by pytest, so this reads the
.tf source as text.

Run: pytest tests/cosmic_ai_lambda/test_terraform_module_ownership.py -v
"""

import glob
import os
import re

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "target", "aws", "scripts")
MAIN_MODULE = os.environ.get("MAIN_TERRAFORM_DIR", os.path.join(_SCRIPTS, "terraform"))
COSMIC_AI_MODULE = os.path.join(_SCRIPTS, "terraform-cosmic-ai")
NAMED_RESOURCES = ("aws_lambda_function", "aws_sfn_state_machine")


def _named_resources(module_dir):
    found = []
    for path in glob.glob(os.path.join(module_dir, "*.tf")):
        body = open(path).read()
        for kind in NAMED_RESOURCES:
            for match in re.finditer(rf'resource "{kind}" "(\w+)" \{{.*?\n\}}', body, re.S):
                found.append((kind, match.group(1), match.group(0)))
    return found


def test_the_cosmic_ai_module_still_declares_the_workflow():
    kinds = {kind for kind, _, _ in _named_resources(COSMIC_AI_MODULE)}
    assert kinds == set(NAMED_RESOURCES)


def test_the_main_module_declares_no_cosmic_ai_function_or_state_machine():
    offenders = [f"{kind}.{name}" for kind, name, block in _named_resources(MAIN_MODULE)
                 if "cosmic_ai" in name or "cosmic-ai" in block]
    assert offenders == []