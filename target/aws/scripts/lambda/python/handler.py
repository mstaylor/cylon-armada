"""Lambda handler — S3 script runner for context-reuse workers.

Follows Cylon's lambda_entry1.py pattern: downloads scripts from S3 at
runtime rather than baking them into the Docker image.  This lets us
update the context-reuse logic (context/, chain/, cost/, simd/, coordinator/)
independently of the Lambda container build.

The Step Functions state machine invokes this handler for each state
(prepare_tasks, route_task, aggregate_results).  The event payload
carries both:
  1. S3 coordinates — where to download the scripts from
  2. Action-specific data — forwarded as environment variables + JSON file
"""

import json
import logging
import os
import sys
import subprocess
import tempfile

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 download helpers (adapted from cylon/docker/aws/lambda/lambda_entry1.py)
# ---------------------------------------------------------------------------

def get_file(file_name, bucket, prefix=None, use_folder=False):
    """Download a file or folder from S3."""
    if prefix is None:
        prefix = os.path.basename(file_name)

    s3_client = boto3.client("s3")
    try:
        if use_folder:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if "Contents" not in response:
                logger.error("No files found in s3://%s/%s", bucket, prefix)
                return None

            for s3_object in response["Contents"]:
                s3_key = s3_object["Key"]
                path, filename = os.path.split(s3_key)
                path = f"/tmp/{path}"
                if path and not os.path.exists(path):
                    os.makedirs(path)
                if not s3_key.endswith("/"):
                    download_to = f"{path}/{filename}" if path else filename
                    logger.info("Downloading s3://%s/%s → %s", bucket, s3_key, download_to)
                    s3_client.download_file(bucket, s3_key, download_to)
            return True
        else:
            with open(file_name, "wb") as f:
                s3_client.download_fileobj(bucket, prefix, f)
            return f
    except ClientError as e:
        logging.error(e)
        return None


def execute_script(data):
    """Execute a downloaded Python script via subprocess."""
    use_folder = data["s3_object_type"] == "folder"
    result = get_file(
        file_name=data["script"],
        bucket=data["s3_bucket"],
        prefix=data["s3_object_name"],
        use_folder=use_folder,
    )

    if result is None and not use_folder:
        logger.error("Unable to retrieve %s from s3://%s", data["script"], data["s3_bucket"])
        raise RuntimeError(f"Failed to download script from S3: {data['script']}")

    cmd = ["python", data["script"]]
    if data.get("args"):
        cmd += data["args"].split()

    logger.info("Executing: %s", " ".join(cmd))
    returncode = subprocess.call(cmd, shell=False)

    if returncode != 0:
        raise RuntimeError(f"Script {data['script']} exited with code {returncode}")


# ---------------------------------------------------------------------------
# Environment variable propagation
# ---------------------------------------------------------------------------

# Fields that are always expected in the Step Functions event payload
_REQUIRED_FIELDS = ["S3_BUCKET", "S3_OBJECT_NAME", "SCRIPT", "S3_OBJECT_TYPE"]

# Optional fields propagated as environment variables
_OPTIONAL_ENV_FIELDS = [
    # Bedrock / model config
    "BEDROCK_LLM_MODEL_ID",
    "BEDROCK_EMBEDDING_MODEL_ID",
    "BEDROCK_EMBEDDING_DIMENSIONS",
    "SIMILARITY_THRESHOLD",
    "BEDROCK_PRICING_CONFIG",
    "AWS_DEFAULT_REGION",
    # Infrastructure
    "REDIS_HOST",
    "REDIS_PORT",
    "DYNAMO_ENDPOINT_URL",
    "DYNAMO_TABLE_NAME",
    # Workflow / task identity
    "WORKFLOW_ID",
    "RANK",
    "WORLD_SIZE",
    "ACTION",
    # Cylon FMI communicator
    "RENDEZVOUS_HOST",
    "RENDEZVOUS_PORT",
    "FMI_CHANNEL_TYPE",
    "FMI_CHANNEL",
    "FMI_MAX_TIMEOUT",
    # Context backend
    "CONTEXT_BACKEND",
    # Cost tracking
    "ENABLE_COST_TRACKING",
    # Logging
    "LOG_LEVEL",
]


def _set_env_from_event(event):
    """Propagate event fields to environment variables.

    Follows Cylon's pattern: each field in the event payload is set as an
    environment variable so that downloaded scripts can read config via
    os.environ without parsing the Lambda event directly.
    """
    for key in _REQUIRED_FIELDS:
        value = event.get(key)
        if value is not None:
            os.environ[key] = str(value)
        else:
            raise ValueError(f"Required field '{key}' missing from event payload")

    for key in _OPTIONAL_ENV_FIELDS:
        if key in event and event[key] is not None:
            os.environ[key] = str(event[key])

    # Write the full action payload to a temp file so the script can
    # read structured data (embeddings, task lists, etc.) beyond what
    # fits in flat env vars.
    if "action_payload" in event:
        payload_path = "/tmp/action_payload.json"
        with open(payload_path, "w") as f:
            json.dump(event["action_payload"], f)
        os.environ["ACTION_PAYLOAD_PATH"] = payload_path


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def handler(event, context):
    """Lambda handler — download scripts from S3 and execute.

    Invoked by Step Functions for each state:
      - PrepareTasks  → downloads coordinator scripts, runs prepare_tasks
      - RouteTask     → downloads context-reuse scripts, runs route_task
      - AggregateResults → downloads coordinator scripts, runs aggregate_results

    Event payload structure:
        {
            "S3_BUCKET": "cylon-armada-scripts",
            "S3_OBJECT_NAME": "target/shared/scripts",
            "S3_OBJECT_TYPE": "folder",
            "SCRIPT": "/tmp/target/shared/scripts/run_action.py",
            "ACTION": "route_task",
            "BEDROCK_LLM_MODEL_ID": "...",
            "REDIS_HOST": "...",
            ...
            "action_payload": { ... task-specific data ... }
        }
    """
    logger.info("Handler invoked with action: %s", event.get("ACTION", "unknown"))

    # Step 1: Propagate event fields to environment variables
    _set_env_from_event(event)

    # Step 2: Download scripts from S3 and execute
    data = {
        "s3_bucket": event["S3_BUCKET"],
        "s3_object_name": event["S3_OBJECT_NAME"],
        "s3_object_type": event["S3_OBJECT_TYPE"],
        "script": event["SCRIPT"],
        "args": event.get("EXEC_ARGS"),
    }

    execute_script(data)

    # Step 3: Read result from the output file written by the script
    result_path = os.environ.get("ACTION_RESULT_PATH", "/tmp/action_result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    return {
        "status": "completed",
        "action": event.get("ACTION", "unknown"),
        "message": f"Executed via Python {sys.version}",
    }