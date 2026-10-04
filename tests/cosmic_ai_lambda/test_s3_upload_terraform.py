"""Every aws_s3_object upload in the main Terraform module fails the plan on a missing source.

A count-gated upload that evaluates to zero leaves the previous S3 object live
and reports success; that served an uninstrumented inference.py to the
2026-10-02 Arm A smoke run. Terraform is not run by pytest, so this reads the
.tf source as text.

Run: pytest tests/cosmic_ai_lambda/test_s3_upload_terraform.py -v
"""

import glob
import os
import re

_TF = os.path.join(os.path.dirname(__file__), "..", "..", "target", "aws", "scripts", "terraform")


def _s3_object_blocks():
    for path in glob.glob(os.path.join(_TF, "*.tf")):
        body = open(path).read()
        for match in re.finditer(r'resource "aws_s3_object" "(\w+)" \{.*?\n\}', body, re.S):
            yield match.group(1), match.group(0)


def test_there_are_uploads_to_check():
    assert {name for name, _ in _s3_object_blocks()} >= {
        "cosmic_ai_model", "cosmic_ai_data", "cosmic_ai_inference", "cosmic_ai_fmi_scripts",
    }


def test_no_upload_is_count_gated():
    for name, block in _s3_object_blocks():
        assert not re.search(r"^\s*count\s*=", block, re.M), name


def test_every_upload_has_a_source_exists_precondition():
    for name, block in _s3_object_blocks():
        assert re.search(r"precondition\s*\{[^}]*fileexists\(", block, re.S), name
