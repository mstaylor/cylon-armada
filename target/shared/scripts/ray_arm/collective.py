"""ray-collective: contexts exchanged with Ray's own collective primitive.

ray.util.collective is the collective a Ray user reaches for, so it is the
strongest Ray comparator for a thesis about collectives. Its CPU backend here
is torch_gloo (torch.distributed Gloo, Ray 2.48 and later). Its calls run
only inside actors, so each rank owns one exchanger actor pinned to its own
node, and rank 0 joins the ranks' exchangers into one group.

The collective moves tensors, not Arrow tables, so each contribution is
serialised to an Arrow IPC byte buffer before the exchange and read back
after it. That conversion is the copy Armada's Arrow data plane avoids, and
callers time it separately.

Contributions differ in length, so an exchange is two allgathers: the sizes,
then the payloads zero-padded to the largest.

Gloo otherwise finds its own address by resolving the hostname, which inside
a container can give loopback or the wrong interface, and the group then
hangs on addresses its peers cannot reach. So each exchanger binds Gloo to
the interface holding the node IP Ray already advertises, in its own process
before the group initialises. An explicit GLOO_SOCKET_IFNAME still wins.
"""

import ipaddress
import logging
import os
import socket
import time

import numpy as np
import pyarrow as pa
import ray

COLLECTIVE_BACKEND = os.environ.get("RAY_COLLECTIVE_BACKEND", "torch_gloo")
GLOO_INTERFACE_ENV = "GLOO_SOCKET_IFNAME"

logger = logging.getLogger(__name__)


class GlooInterfaceNotFound(RuntimeError):
    """No usable interface holds the node IP Gloo must bind."""


def resolve_gloo_interface(rank, node_ip, ipv4_by_interface, configured=None):
    """The interface Gloo binds: the configured one if given, else the one
    holding node_ip.

    ipv4_by_interface maps interface name to its IPv4 addresses. A loopback
    node IP is refused, since peers on other tasks cannot reach it.
    """
    if configured:
        return configured
    seen = ", ".join(f"{name}={'/'.join(addresses) or '-'}"
                     for name, addresses in sorted(ipv4_by_interface.items()))
    if ipaddress.ip_address(node_ip).is_loopback:
        raise GlooInterfaceNotFound(
            f"rank {rank}: Ray node IP {node_ip} is loopback, which peers on other tasks "
            f"cannot reach; interfaces seen: {seen}")
    holders = sorted(name for name, addresses in ipv4_by_interface.items() if node_ip in addresses)
    if not holders:
        raise GlooInterfaceNotFound(
            f"rank {rank}: no interface holds Ray node IP {node_ip}; interfaces seen: {seen}")
    return holders[0]


def local_ipv4_by_interface():
    """This host's IPv4 addresses by interface name."""
    import psutil

    return {name: [a.address for a in addresses if a.family == socket.AF_INET]
            for name, addresses in psutil.net_if_addrs().items()}


def exchanger_name(group_name, rank):
    return f"cosmic_exchanger_{group_name}_{rank}"


def pack_table(table):
    """The table as an Arrow IPC stream in a uint8 buffer."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return np.frombuffer(sink.getvalue(), dtype=np.uint8)


def unpack_buffer(buffer):
    """The table a pack_table buffer holds."""
    return pa.ipc.open_stream(pa.py_buffer(buffer)).read_all()


def exchanger_options(rank, group_name, node_id, namespace):
    """Actor options for one rank's exchanger: named for discovery, pinned to
    the rank's node, and given the Gloo interface when one is configured."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    options = {
        "name": exchanger_name(group_name, rank),
        "namespace": namespace,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    }
    interface = os.environ.get(GLOO_INTERFACE_ENV)
    if interface:
        options["runtime_env"] = {"env_vars": {GLOO_INTERFACE_ENV: interface}}
    return options


@ray.remote(num_cpus=0)
class CollectiveExchanger:
    """One rank's member of the run's collective group."""

    def __init__(self, rank, world_size, group_name):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        configured = os.environ.get(GLOO_INTERFACE_ENV)
        node_ip = ray.util.get_node_ip_address()
        interface = resolve_gloo_interface(rank, node_ip, local_ipv4_by_interface(), configured)
        os.environ[GLOO_INTERFACE_ENV] = interface
        self._gloo = {"interface": interface, "node_ip": node_ip,
                      "source": "configured" if configured else "derived"}
        logger.info("rank %d binds Gloo to %s (node IP %s, %s)", rank, interface, node_ip,
                    self._gloo["source"])

    def ready(self):
        return True

    def gloo_interface(self):
        """The interface Gloo binds, the node IP, and whether it was configured or derived."""
        return dict(self._gloo)

    def allgather(self, payload):
        """Every rank's buffer in rank order, and the seconds spent inside the
        two allgathers."""
        import torch
        from ray.util import collective

        started = time.perf_counter()
        size = torch.tensor([payload.size], dtype=torch.int64)
        sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(self.world_size)]
        collective.allgather(sizes, size, group_name=self.group_name)
        lengths = [int(s.item()) for s in sizes]
        padded = torch.zeros(max(lengths), dtype=torch.uint8)
        padded.numpy()[: payload.size] = payload
        gathered = [torch.zeros(max(lengths), dtype=torch.uint8) for _ in range(self.world_size)]
        collective.allgather(gathered, padded, group_name=self.group_name)
        buffers = [g.numpy()[:length] for g, length in zip(gathered, lengths)]
        return buffers, time.perf_counter() - started


def _await_exchangers(world_size, group_name, namespace, ray_api, timeout_s, poll_s):
    deadline = time.monotonic() + timeout_s
    handles = []
    for rank in range(world_size):
        while True:
            try:
                handles.append(ray_api.get_actor(exchanger_name(group_name, rank),
                                                 namespace=namespace))
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"collective group {group_name!r} found {rank} of {world_size} "
                        f"exchangers within {timeout_s}s")
                time.sleep(poll_s)
    return handles


def form_collective_group(rank, world_size, group_name, namespace, ray_api, timeout_s, poll_s):
    """Start this rank's exchanger and, on rank 0, join every rank's into one group.

    Returns this rank's exchanger once it is running. Rank 0 returns only once
    the group exists, so a rank that waits on rank 0 afterwards knows the
    group is ready.
    """
    node_id = ray_api.get_runtime_context().get_node_id()
    exchanger = CollectiveExchanger.options(
        **exchanger_options(rank, group_name, node_id, namespace)
    ).remote(rank, world_size, group_name)
    ray_api.get(exchanger.ready.remote())
    if rank == 0:
        from ray.experimental.collective import create_collective_group

        handles = _await_exchangers(world_size, group_name, namespace, ray_api, timeout_s, poll_s)
        create_collective_group(handles, backend=COLLECTIVE_BACKEND, name=group_name)
    return exchanger
