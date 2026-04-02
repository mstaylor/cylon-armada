"""S3 script loader — download shared scripts from a payload-supplied S3 path.

Follows the cylon_init.py / lambda_entry1.py pattern:
  - S3 coordinates (bucket, prefix) travel in the Step Functions event payload
  - armada_init reads them from the top-level input and embeds them in each
    per-task payload it builds for the Map state
  - armada_executor / armada_aggregate call load_scripts() at the top of their
    handler functions — before any lazy imports — to download the scripts folder
    and prepend the download dir to sys.path

This means script changes deploy in seconds via:
    aws s3 sync target/shared/scripts/ s3://<bucket>/scripts/

No image rebuild or ECR push required. The baked-in image scripts are the
fallback when no S3 coordinates are supplied (local dev, smoke tests, Rivanna).

Warm-start behaviour:
    Python caches imported modules in sys.modules after the first invocation.
    Subsequent warm invocations reuse the already-imported modules with zero
    S3 overhead. A new cold start re-downloads when S3 coordinates are present.
"""

import logging
import os
import sys

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_DIR = "/tmp/armada_scripts"

# Module-level flag — tracks whether a download has already completed this
# cold start so warm invocations skip the S3 call entirely.
_downloaded = False


def load_scripts(
    s3_bucket: str,
    s3_prefix: str = "scripts/",
    local_dir: str = _DEFAULT_LOCAL_DIR,
) -> bool:
    """Download shared scripts from S3 and prepend to sys.path.

    Call this at the top of each handler() function, before any lazy imports
    from the shared scripts package.

    Args:
        s3_bucket:  S3 bucket name (from event payload).
        s3_prefix:  S3 key prefix for the scripts folder. Default: "scripts/".
        local_dir:  Local /tmp destination. Default: "/tmp/armada_scripts".

    Returns:
        True if scripts were downloaded (or already downloaded this cold start).
        False if the download failed — caller should fall back to baked-in path.
    """
    global _downloaded

    if _downloaded:
        # Already downloaded this cold start — warm invocation, skip S3.
        return True

    s3 = boto3.client("s3")
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

        count = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                relative = key[len(s3_prefix):] if key.startswith(s3_prefix) else key
                dest = os.path.join(local_dir, relative)

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                s3.download_file(s3_bucket, key, dest)
                count += 1

        if count == 0:
            logger.warning("S3 loader: no files found at s3://%s/%s", s3_bucket, s3_prefix)
            return False

        logger.info(
            "S3 loader: downloaded %d files from s3://%s/%s → %s",
            count, s3_bucket, s3_prefix, local_dir,
        )

    except ClientError as exc:
        logger.error("S3 loader: download failed — %s", exc)
        return False

    # Prepend download dir to sys.path so lazy imports resolve here first.
    abs_dir = os.path.abspath(local_dir)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)

    _downloaded = True
    return True