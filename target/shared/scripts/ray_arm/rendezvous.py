"""Agreeing on one Ray head address across independently launched ranks.

Every rank is its own Fargate task, so there is no launcher that knows the
head's address before the head exists. Redis already serves this role for FMI
rank assignment, so the same coordinator carries the Ray head address rather
than introducing a second discovery mechanism.
"""

import time

HEAD_KEY_PREFIX = "cylon-armada:ray-head:"


class RendezvousTimeout(RuntimeError):
    """No head address appeared before the deadline."""


def head_key(comm_name):
    return f"{HEAD_KEY_PREFIX}{comm_name}"


def publish_head(redis_client, comm_name, address, ttl_s=3600):
    redis_client.set(head_key(comm_name), address, ex=ttl_s)


def discover_head(redis_client, comm_name, timeout_s, poll_s=0.5, sleep=time.sleep):
    key = head_key(comm_name)
    deadline = timeout_s
    waited = 0.0
    while True:
        value = redis_client.get(key)
        if value is not None:
            return value.decode() if isinstance(value, bytes) else value
        if waited >= deadline:
            raise RendezvousTimeout(
                f"no Ray head published for comm_name={comm_name!r} within "
                f"{timeout_s}s at key {key!r}"
            )
        sleep(poll_s)
        waited += poll_s