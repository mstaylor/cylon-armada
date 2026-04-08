"""redis_dns_updater — updates Route 53 when a Redis ECS task reaches RUNNING.

Triggered by EventBridge on ECS Task State Change events. Reads the public IP
from the task's ENI and upserts the Route 53 A record so clients always resolve
to the current Fargate task.

Environment variables:
    ROUTE53_ZONE_ID  — hosted zone ID (e.g. Z1D633PJN98FT9)
    REDIS_HOSTNAME   — fully-qualified record name (e.g. redis.example.com)
    REDIS_DNS_TTL    — TTL in seconds (default: 30)
"""

import logging
import os

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

ec2 = boto3.client("ec2")
route53 = boto3.client("route53")


def handler(event, context):
    detail = event.get("detail", {})
    status = detail.get("lastStatus")
    task_arn = detail.get("taskArn", "")

    if status != "RUNNING":
        logger.info("Ignoring task %s in status %s", task_arn, status)
        return

    eni_id = _find_eni(detail)
    if not eni_id:
        logger.error("No ENI attachment found for task %s", task_arn)
        return

    public_ip = _get_public_ip(eni_id)
    if not public_ip:
        logger.error("ENI %s has no public IP — is assign_public_ip enabled?", eni_id)
        return

    zone_id = os.environ["ROUTE53_ZONE_ID"]
    hostname = os.environ["REDIS_HOSTNAME"]
    ttl = int(os.environ.get("REDIS_DNS_TTL", "30"))

    _upsert_record(zone_id, hostname, public_ip, ttl)
    logger.info("Updated %s → %s (TTL %ds)", hostname, public_ip, ttl)
    return {"hostname": hostname, "public_ip": public_ip}


def _find_eni(detail):
    for attachment in detail.get("attachments", []):
        if attachment.get("type") == "eni":
            for item in attachment.get("details", []):
                if item.get("name") == "networkInterfaceId":
                    return item.get("value")
    return None


def _get_public_ip(eni_id):
    resp = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    iface = resp["NetworkInterfaces"][0]
    return iface.get("Association", {}).get("PublicIp")


def _upsert_record(zone_id, hostname, ip, ttl):
    route53.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": hostname,
                    "Type": "A",
                    "TTL": ttl,
                    "ResourceRecords": [{"Value": ip}],
                },
            }]
        },
    )