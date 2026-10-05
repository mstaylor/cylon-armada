"""No aws_s3_object upload in the main Terraform module can skip itself silently.

A count gated on the source file existing evaluates to zero on a machine without
that file, leaves the previous S3 object live and reports success. Repo code is
uploaded unconditionally; the large model and data artifacts upload only when
their source variable is set and otherwise must already exist in S3. Terraform
is not run by pytest, so this reads the .tf source as text.

Run: pytest tests/cosmic_ai_lambda/test_s3_upload_terraform.py -v
"""

import glob
import os
import re

_TF = os.path.join(os.path.dirname(__file__), "..", "..", "target", "aws", "scripts", "terraform")

OPTIONAL_ARTIFACT_UPLOADS = {"cosmic_ai_model_upload", "cosmic_ai_data_upload"}
UNCONDITIONAL_CODE_UPLOADS = {"cosmic_ai_inference", "cosmic_ai_fmi_scripts"}


def _blocks(kind, type_name):
    for path in glob.glob(os.path.join(_TF, "*.tf")):
        body = open(path).read()
        for match in re.finditer(rf'{kind} "{type_name}" "(\w+)" \{{.*?\n\}}', body, re.S):
            yield match.group(1), match.group(0)


def _uploads():
    return dict(_blocks("resource", "aws_s3_object"))


def test_there_are_uploads_to_check():
    assert set(_uploads()) >= OPTIONAL_ARTIFACT_UPLOADS | UNCONDITIONAL_CODE_UPLOADS


def test_no_upload_is_gated_on_its_source_file_existing():
    for name, block in _uploads().items():
        count = re.search(r"^\s*count\s*=(.*)$", block, re.M)
        assert not (count and "fileexists" in count.group(1)), name


def test_repo_code_uploads_are_unconditional():
    uploads = _uploads()
    for name in UNCONDITIONAL_CODE_UPLOADS:
        assert not re.search(r"^\s*count\s*=", uploads[name], re.M), name


def test_every_upload_has_a_source_exists_precondition():
    for name, block in _uploads().items():
        assert re.search(r"precondition\s*\{[^}]*fileexists\(", block, re.S), name


def test_optional_artifacts_fall_back_to_requiring_the_object_already_in_s3():
    existing = dict(_blocks("data", "aws_s3_object"))
    for name in OPTIONAL_ARTIFACT_UPLOADS:
        artifact = name.removesuffix("_upload")
        assert artifact in existing, artifact
        assert f'var.{artifact}_source == ""' in existing[artifact], artifact