import sys
import json
import time
import argparse
import subprocess

import boto3
from botocore.exceptions import ClientError
import os

import logging

_invocations_in_container = 0


def mark_container_invocation():
    """Tell the inference script whether this is the container's first invocation (a cold start)."""
    global _invocations_in_container
    _invocations_in_container += 1
    os.environ['CONTAINER_COLD_START'] = '1' if _invocations_in_container == 1 else '0'


def environ_or_required(key, required: bool = True):

    return  (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': required}
    )


def get_file(file_name, bucket, prefix=None, use_folder=False):
    # If S3 object_name was not specified, use file_name
    if prefix is None:
        prefix = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:

        if use_folder:
            list_kwargs = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            response = s3_client.list_objects_v2(**list_kwargs)

            if 'Contents' not in response:
                print("No files found in the specified directory.")
                return

            for s3_object in response['Contents']:
                s3_key = s3_object["Key"]
                path, filename = os.path.split(s3_key)
                # bucket folders root does not include a /
                path = f"/tmp/{path}"
                print(f's3key: {s3_key} path: {path} filename: {filename}')
                if len(path) != 0 and not os.path.exists(path):
                    print(f'creating os path: {path}')
                    os.makedirs(path)
                if not s3_key.endswith("/"):
                    download_to = f'{path}/{filename}' if path else filename
                    print(f'downloading key: {s3_key} to {download_to}')
                    s3_client.download_file(bucket, s3_key, download_to)

        else:
            with open(file_name, 'wb') as f:
                s3_client.download_fileobj(bucket, prefix, f)
            return f
    except ClientError as e:
        logging.error(e)
        return None

class RunAborted(RuntimeError):
    pass


class MissingAbortPollInterval(ValueError):
    pass


def _run_context():
    return (f"rank={os.environ.get('RANK')}, world_size={os.environ.get('WORLD_SIZE')}, "
            f"result_path={os.environ.get('RESULT_PATH')}")


def abort_marker_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') not in ('404', 'NoSuchKey', 'NotFound'):
            logging.warning(f"abort marker check for s3://{bucket}/{key} failed: {e}")
        return False


def write_abort_marker(s3_client, bucket, key, returncode):
    s3_client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps({'source': 'worker', 'rank': os.environ.get('RANK'),
                         'returncode': returncode, 'time': time.time()}),
        ContentType='application/json')


def run_watching_for_abort(command, s3_client, bucket, key, poll_s):
    process = subprocess.Popen(command, shell=False)
    while True:
        try:
            returncode = process.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            if abort_marker_exists(s3_client, bucket, key):
                process.kill()
                process.wait()
                raise RunAborted(f"stopped by abort marker s3://{bucket}/{key} ({_run_context()})")
    if returncode != 0:
        write_abort_marker(s3_client, bucket, key, returncode)
    return returncode


def execute_script(data=None):
    usefolder = data['s3_object_type'] == 'folder'
    script = get_file(file_name=data['script'], bucket=data['s3_bucket'],
                      prefix=data['s3_object_name'], use_folder=usefolder)
    os.environ['CODE_FETCH_END_TS'] = str(time.time())

    if script is None and not usefolder:
        print(f"unable to retrieve file {data['script']} from AWS S3")

    scriptargs = data['args']
    command = ['python', data['script']] + (scriptargs.split() if scriptargs is not None else [])

    abort_key = os.environ.get('ABORT_KEY')
    if abort_key:
        poll_s = os.environ.get('ABORT_POLL_S')
        if not poll_s:
            raise MissingAbortPollInterval(f"ABORT_KEY {abort_key} is set without ABORT_POLL_S ({_run_context()})")
        returncode = run_watching_for_abort(command, boto3.client('s3'), data['s3_bucket'], abort_key,
                                            float(poll_s))
    else:
        returncode = subprocess.call(command, shell=False)

    if returncode != 0:
        raise RuntimeError(
            f"script {data['script']} exited with code {returncode} "
            f"(rank={os.environ.get('RANK')}, world_size={os.environ.get('WORLD_SIZE')}, "
            f"result_path={os.environ.get('RESULT_PATH')})"
        )

def handler(event, context):

    handler_entry_ts = time.time()

    print("received: ", event)

    for key in event.keys():
        os.environ[key] = f"{event[key]}"

    mark_container_invocation()

    os.environ['LAMBDA_ENTRY_TS'] = str(handler_entry_ts)

    parser = argparse.ArgumentParser(description="run S3 script")

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))
    parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_OBJECT_NAME'))
    parser.add_argument('-s', dest='script', type=str, help="script to execute",
                        **environ_or_required('SCRIPT'))
    parser.add_argument('-t', dest='s3_object_type', type=str, **environ_or_required('S3_OBJECT_TYPE', required=False),
                        choices=['file', 'folder'],
                        help="file or folder for S3 pull")  # w
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS', required=False))

    print ("parsing args")
    args, unknown = parser.parse_known_args()

    #execute

    data = vars(args)

    print(f"executing script: {data['s3_object_name']}")
    execute_script(data)
    print("executed script")



    return f'Executed Serverless FMI using Python{sys.version}! environment: {os.environ["S3_BUCKET"]}'