"""
S3 downloader for cylon-armada experiment result files.

Downloads summary CSVs and stopwatch CSVs from S3 using prefix-based
discovery. Follows the cylon results pipeline pattern.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def download_from_s3(
    bucket: str,
    prefix: str,
    download_dir: str,
) -> List[str]:
    """Download all result files from S3 matching a prefix.

    Returns list of local file paths downloaded.
    """
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client("s3")
    downloaded = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                filename = os.path.basename(key)

                # Download summary CSVs and stopwatch CSVs
                if not (filename.endswith("_summary.csv")
                        or filename.endswith("_stopwatch.csv")):
                    continue

                rel_path = key[len(prefix):].lstrip("/")
                local_path = os.path.join(download_dir, rel_path) if rel_path != filename else os.path.join(download_dir, filename)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
                s3_client.download_file(bucket, key, local_path)
                downloaded.append(local_path)

    except ClientError as e:
        logger.error("S3 error: %s", e)
    except Exception as e:
        logger.error("Download error: %s", e)

    logger.info("Downloaded %d files from s3://%s/%s", len(downloaded), bucket, prefix)
    return downloaded


def download_experiment_results(config) -> None:
    """Download all result files for configured experiments from S3.

    Populates local_data_dir on each experiment config after download.
    """
    for exp in config.experiments:
        if exp.local_data_dir:
            logger.info(
                "Skipping S3 download for %s (using local: %s)",
                exp.label, exp.local_data_dir,
            )
            continue

        if not exp.s3_prefix_pattern or not config.s3_bucket:
            logger.warning("No S3 config for %s, skipping download", exp.label)
            continue

        prefix = exp.s3_prefix_pattern.format(
            platform=exp.platform,
            instance_label=exp.instance_label,
        )

        download_dir = os.path.join(
            config.download_dir,
            exp.platform,
            exp.instance_label,
        )

        files = download_from_s3(
            bucket=config.s3_bucket,
            prefix=prefix,
            download_dir=download_dir,
        )

        if files:
            exp.local_data_dir = download_dir
            logger.info("Set local_data_dir for %s: %s", exp.label, download_dir)