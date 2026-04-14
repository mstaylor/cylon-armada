"""lambda_entry — thin entry point for all cylon-armada Lambda functions.

Follows the cylon lambda_entry1.py pattern: downloads scripts from S3 at
cold-start time, then delegates to the target handler module. This eliminates
Docker image rebuilds for script changes — deploy bug fixes in seconds via:

    aws s3 sync target/shared/scripts/ s3://<bucket>/scripts/
    aws s3 sync target/aws/scripts/lambda/python/ s3://<bucket>/scripts/lambda/

Configuration (resolved in priority order: event payload > env var > default):
    S3_SCRIPTS_BUCKET   — S3 bucket containing the scripts folder
    S3_SCRIPTS_PREFIX   — Key prefix (default: "scripts/")
    HANDLER_MODULE      — Python module name to import and call .handler()
                          (e.g., "armada_init", "armada_executor",
                          "armada_aggregate", "rendezvous_test")
                          (env var only — not overridable from event)

The bucket and prefix can be set three ways:
    1. Event payload fields (per-invocation override) — highest priority
    2. Lambda environment variables (Terraform-managed) — default
    3. Unset — falls back to baked-in image scripts (local dev, Rivanna)

S3 layout (mirrors the local project structure):

    s3://<bucket>/scripts/
    ├── communicator/           <- target/shared/scripts/communicator/
    ├── context/                <- target/shared/scripts/context/
    ├── coordinator/            <- target/shared/scripts/coordinator/
    ├── cost/                   <- target/shared/scripts/cost/
    ├── chain/                  <- target/shared/scripts/chain/
    ├── simd/                   <- target/shared/scripts/simd/
    └── lambda/                 <- target/aws/scripts/lambda/python/
        ├── armada_init.py
        ├── armada_executor.py
        ├── armada_aggregate.py
        └── rendezvous_test.py

Warm-start behaviour:
    After the first invocation (cold start), scripts are cached in /tmp and
    modules are cached in sys.modules.  Subsequent warm invocations skip the
    S3 download entirely — zero overhead.
"""

import importlib
import logging
import os
import sys

# Suppress OpenMPI MPI_Init before any pycylon import. Lambda has no HOME
# directory, causing opal_init to fail. The FMI communicator uses TCP/TCPunch
# and does not need MPI at runtime, but pycylon links against OpenMPI and
# mpi4py would trigger MPI_Init on import unless suppressed here.
try:
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
except ImportError:
    pass

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

_LOCAL_DIR = "/tmp/armada_scripts"

# Module-level flag — tracks whether scripts have been downloaded this
# cold start so warm invocations skip S3 entirely.
_downloaded = False


def _download_scripts(s3_bucket, s3_prefix):
    """Download the scripts folder from S3 to /tmp/armada_scripts/.

    Follows the cylon lambda_entry1.py get_file(use_folder=True) pattern:
    recursively lists the S3 prefix and downloads every object, preserving
    the directory structure under /tmp/armada_scripts/.
    """
    global _downloaded

    if _downloaded:
        return True

    s3 = boto3.client("s3")
    try:
        paginator = s3.get_paginator("list_objects_v2")
        count = 0

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                # Strip the prefix to get the relative path
                relative = key[len(s3_prefix):] if key.startswith(s3_prefix) else key
                dest = os.path.join(_LOCAL_DIR, relative)

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                s3.download_file(s3_bucket, key, dest)
                count += 1

        if count == 0:
            logger.warning("No scripts found at s3://%s/%s", s3_bucket, s3_prefix)
            return False

        logger.info(
            "Downloaded %d scripts from s3://%s/%s to %s",
            count, s3_bucket, s3_prefix, _LOCAL_DIR,
        )

    except ClientError as exc:
        logger.error("S3 script download failed: %s", exc)
        return False

    # Prepend download paths so S3 versions override baked-in image scripts.
    # lambda/ subdir → handler modules (armada_init, armada_executor, etc.)
    # root dir → shared scripts (communicator/, context/, coordinator/, etc.)
    lambda_dir = os.path.join(_LOCAL_DIR, "lambda")
    for d in (lambda_dir, _LOCAL_DIR):
        abs_d = os.path.abspath(d)
        if os.path.isdir(abs_d) and abs_d not in sys.path:
            sys.path.insert(0, abs_d)

    _downloaded = True
    return True


def handler(event, context):
    """Lambda entry point — download scripts from S3 and delegate to target handler.

    All Lambda functions (init, executor, aggregate, rendezvous) use this
    same entry point.  Terraform sets HANDLER_MODULE per-function to control
    which module is imported and invoked.
    """
    handler_module = os.environ.get("HANDLER_MODULE", "")
    if not handler_module:
        raise ValueError(
            "HANDLER_MODULE environment variable not set — "
            "configure it in Terraform image_config or Lambda environment"
        )

    # Event payload overrides env vars (per-invocation configurability)
    s3_bucket = event.get("S3_SCRIPTS_BUCKET") or os.environ.get("S3_SCRIPTS_BUCKET", "")
    s3_prefix = event.get("S3_SCRIPTS_PREFIX") or os.environ.get("S3_SCRIPTS_PREFIX", "scripts/")

    # Ensure prefix ends with /
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    # Download scripts from S3 (skipped on warm starts and when no bucket set)
    if s3_bucket:
        ok = _download_scripts(s3_bucket, s3_prefix)
        if ok:
            logger.info(
                "S3 scripts loaded — delegating to %s.handler", handler_module
            )
        else:
            logger.warning(
                "S3 script load failed — falling back to baked-in %s",
                handler_module,
            )
    else:
        logger.info(
            "S3_SCRIPTS_BUCKET not set — using baked-in %s", handler_module
        )

    # Import the target handler module and delegate
    mod = importlib.import_module(handler_module)
    return mod.handler(event, context)