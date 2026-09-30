"""Forming a Ray cluster from independently launched ranks.

Rank 0 starts the head and publishes its address; every other rank discovers
that address and joins. The census wait exists because ray start returns as
soon as the local daemon is up, which is well before the cluster has the node
count the measurement assumes.
"""

import ipaddress
import os
import socket
import subprocess
import time

from ray_arm.rendezvous import discover_head, publish_head

ADDRESS_PROBE_TARGET = os.environ.get("ADDRESS_PROBE_TARGET", "8.8.8.8")
ADDRESS_PROBE_PORT = int(os.environ.get("ADDRESS_PROBE_PORT", 80))


class ClusterFormationTimeout(RuntimeError):
    """The cluster never reached the expected node count."""


class RayStartFailed(RuntimeError):
    """`ray start` failed while forming the cluster."""


class AddressDetectionFailed(RuntimeError):
    """This host's routable address could not be determined."""


def _is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _detect_routable_address():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((ADDRESS_PROBE_TARGET, ADDRESS_PROBE_PORT))
        return sock.getsockname()[0]
    except OSError as exc:
        raise AddressDetectionFailed(
            f"could not determine this host's routable address by probing "
            f"{ADDRESS_PROBE_TARGET}:{ADDRESS_PROBE_PORT}: {exc}. Set "
            f"ADVERTISE_HOST to the address peers should dial."
        ) from exc
    finally:
        sock.close()


def _own_address(port):
    host = os.environ.get("ADVERTISE_HOST") or _detect_routable_address()
    if _is_loopback(host):
        raise RuntimeError(
            f"resolved loopback address {host!r} for Ray head advertisement; "
            "set ADVERTISE_HOST to override"
        )
    return f"{host}:{port}"


def start_head(port, redis_client, comm_name, runner=subprocess.run):
    try:
        runner(["ray", "start", "--head", f"--port={port}"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RayStartFailed(
            f"ray start --head failed for comm_name={comm_name!r} on port {port}: {exc}"
        ) from exc
    address = _own_address(port)
    publish_head(redis_client, comm_name, address)
    return address


def join_cluster(address, runner=subprocess.run):
    try:
        runner(["ray", "start", f"--address={address}"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RayStartFailed(
            f"ray start --address={address} failed: {exc}"
        ) from exc


def join_by_discovery(redis_client, comm_name, timeout_s, runner=subprocess.run):
    address = discover_head(redis_client, comm_name, timeout_s)
    join_cluster(address, runner=runner)
    return address


def wait_for_nodes(expected, timeout_s, nodes_fn, sleep=time.sleep, poll_s=1.0):
    waited = 0.0
    seen = nodes_fn()
    while seen < expected:
        if waited >= timeout_s:
            raise ClusterFormationTimeout(
                f"cluster reached {seen} of {expected} nodes in {timeout_s}s"
            )
        sleep(poll_s)
        waited += poll_s
        seen = nodes_fn()