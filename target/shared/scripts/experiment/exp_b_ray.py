"""Experiment B per-rank collective benchmark over Ray, the Cylon comparator.

Two channels, each an adapter exposing the pycylon communicator and context
interface, so exp_b_collectives.run times them with the same loop, barriers,
reps and CSV as the Cylon channels:

  ray-gloo         ray.util.collective (torch_gloo) through each rank's
                   CollectiveExchanger actor. Tables cross as Arrow IPC buffers.
                   Rows also carry the seconds spent inside Gloo, measured in
                   the actor, as latency_inner_*.
  ray-objectstore  ray.put / ray.get, with refs exchanged per round through an
                   ExchangeBoard actor.

Collectives a channel cannot express are recorded as unsupported rows.
Cluster formation and teardown are Experiment E's own, injected by main().
"""

import argparse
import logging
import os
from dataclasses import dataclass

from experiment import exp_b_collectives as ebc
from experiment.exp_b_payloads import PAYLOAD_KINDS, payload_table_builder

logger = logging.getLogger("exp_b_ray")

CHANNELS = ("ray-gloo", "ray-objectstore")


class RayGlooComm:
    """Collectives through one rank's CollectiveExchanger."""

    def __init__(self, exchanger, rank, ray_api):
        self._exchanger = exchanger
        self._rank = rank
        self._ray = ray_api
        self._inner_s = None

    def last_inner_s(self):
        return self._inner_s

    def _call(self, object_ref):
        result, self._inner_s = self._ray.get(object_ref)
        return result

    def allgather(self, table):
        from ray_arm.collective import pack_table, unpack_buffer

        buffers = self._call(self._exchanger.allgather.remote(pack_table(table)))
        return [unpack_buffer(buffer) for buffer in buffers]

    def broadcast(self, table, root, ctx):
        from ray_arm.collective import pack_table, unpack_buffer

        payload = pack_table(table) if self._rank == root else None
        return unpack_buffer(self._call(self._exchanger.broadcast.remote(payload, root)))

    def reduce(self, value, op, root):
        return self._call(self._exchanger.reduce.remote(value, root))

    def allreduce(self, value, op):
        return self._call(self._exchanger.allreduce.remote(value))

    def barrier(self):
        return self._call(self._exchanger.barrier.remote())

    def gather(self, table, root):
        raise ebc.UnsupportedCollective("ray.util.collective has no gather")

    def scatter(self, tables, root, ctx):
        raise ebc.UnsupportedCollective("ray.util.collective has no scatter")


class RayGlooCtx:
    def __init__(self, comm):
        self._comm = comm

    def barrier(self):
        return self._comm.barrier()


class RayObjectStoreComm:
    """Collectives as ray.put plus ray.get, one ExchangeBoard round per call."""

    def __init__(self, board, rank, world_size, ray_api):
        self._board = board
        self._rank = rank
        self._world_size = world_size
        self._ray = ray_api
        self._round = 0

    def _next_round(self):
        round_number = self._round
        self._round += 1
        return round_number

    def _post(self, round_number, table):
        self._board.post.remote(round_number, self._rank, [self._ray.put(table)])

    def _collect(self, round_number, expected, readers):
        ref_lists = self._ray.get(self._board.collect.remote(round_number, expected, readers))
        return self._ray.get([ref_list[0] for ref_list in ref_lists])

    def allgather(self, table):
        round_number = self._next_round()
        self._post(round_number, table)
        return self._collect(round_number, self._world_size, self._world_size)

    def broadcast(self, table, root, ctx):
        round_number = self._next_round()
        if self._rank == root:
            if self._world_size > 1:
                self._post(round_number, table)
            return table
        return self._collect(round_number, 1, self._world_size - 1)[0]

    def gather(self, table, root):
        round_number = self._next_round()
        self._post(round_number, table)
        if self._rank == root:
            return self._collect(round_number, self._world_size, 1)
        return None

    def reduce(self, value, op, root):
        raise ebc.UnsupportedCollective("the object store channel has no reduce")

    def allreduce(self, value, op):
        raise ebc.UnsupportedCollective("the object store channel has no allreduce")

    def scatter(self, tables, root, ctx):
        raise ebc.UnsupportedCollective("the object store channel has no scatter")


class RayObjectStoreCtx:
    """Barrier through the run's RunBarrier, one phase per call."""

    def __init__(self, barrier_actor, ray_api):
        self._barrier = barrier_actor
        self._ray = ray_api
        self._phase = 0

    def barrier(self):
        phase = f"b{self._phase}"
        self._phase += 1
        self._ray.get(self._barrier.wait.remote(phase))


@dataclass(frozen=True)
class RayChannel:
    comm: object
    ctx: object
    inner_seconds: object
    barrier_actor: object
    run_actors: tuple


def open_channel(channel, rank, world_size, name, namespace, ray_api):
    """This rank's adapters for the channel, plus the run's named actors.

    Returns only once every rank has opened its side. Rank 0 forms the Gloo
    group after the other exchangers exist, so a rank that returned earlier
    would call into a group not yet initialised in its actor.
    """
    from ray_arm.native import RunBarrier

    barrier_actor = RunBarrier.options(
        name=f"{name}_barrier", namespace=namespace, get_if_exists=True, lifetime="detached"
    ).remote(world_size)
    opened = _open_adapters(channel, rank, world_size, name, namespace, ray_api, barrier_actor)
    ray_api.get(barrier_actor.wait.remote("opened"))
    return opened


def _open_adapters(channel, rank, world_size, name, namespace, ray_api, barrier_actor):
    if channel == "ray-gloo":
        from ray_arm.collective import form_collective_group

        exchanger = form_collective_group(
            rank, world_size, name, namespace, ray_api,
            timeout_s=float(os.environ.get("RAY_COLLECTIVE_TIMEOUT_S", 300)),
            poll_s=float(os.environ.get("RAY_COLLECTIVE_POLL_S", 0.5)))
        comm = RayGlooComm(exchanger, rank, ray_api)
        return RayChannel(comm, RayGlooCtx(comm), comm.last_inner_s, barrier_actor, ())
    if channel == "ray-objectstore":
        from ray_arm.exchange_board import ExchangeBoard

        board = ExchangeBoard.options(
            name=f"{name}_board", namespace=namespace, get_if_exists=True, lifetime="detached"
        ).remote()
        return RayChannel(RayObjectStoreComm(board, rank, world_size, ray_api),
                          RayObjectStoreCtx(barrier_actor, ray_api), None, barrier_actor,
                          (board,))
    raise ValueError(f"unknown channel {channel!r}: expected one of {CHANNELS}")


def run_rank(args, rank, world_size, comm_name, redis_client, ray_api, form_cluster, release,
             open_channel=open_channel, namespace=None):
    """Join the cluster, run args.runs measured sweeps, release. Returns the
    CSV paths this rank wrote, which only the root rank does."""
    namespace = namespace or os.environ.get("RAY_NAMESPACE", "cylon-armada")
    form_cluster(rank, world_size, comm_name, redis_client, ray_api)
    channel = open_channel(args.channel, rank, world_size, f"expb_{comm_name}", namespace, ray_api)
    paths = []
    for run_idx in range(1, args.runs + 1):
        rows = ebc.run({
            "comm": channel.comm,
            "ctx": channel.ctx,
            "collectives": args.collectives,
            "msg_sizes": args.msg_sizes,
            "warmup": args.warmup,
            "reps": args.reps,
            "root": args.root,
            "channel": args.channel,
            "payload": args.payload,
            "payload_builder": payload_table_builder(args.payload),
            "inner_seconds": channel.inner_seconds,
            "rank": rank,
            "world_size": world_size,
            "run_id": run_idx,
        })
        channel.ctx.barrier()
        if rank == args.root:
            os.makedirs(args.output, exist_ok=True)
            fname = ("exp_b_collectives_results.csv" if args.runs == 1
                     else f"run{run_idx}_exp_b_collectives_results.csv")
            path = os.path.join(args.output, fname)
            with open(path, "w", newline="") as f:
                f.write(ebc._rows_to_csv(rows))
            logger.info("run %d: wrote %d rows to %s", run_idx, len(rows), path)
            paths.append(path)
    release(channel.barrier_actor, channel.run_actors, rank, world_size, ray_api,
            float(os.environ.get("RAY_RELEASE_TIMEOUT_S", 300)))
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Experiment B collective benchmark over Ray (per rank)")
    parser.add_argument("--channel", required=True, choices=CHANNELS)
    parser.add_argument("--payload", default="float32", choices=PAYLOAD_KINDS)
    parser.add_argument("--collectives", nargs="+", default=ebc.DEFAULT_COLLECTIVES)
    parser.add_argument("--msg-sizes", type=int, nargs="+", default=ebc.DEFAULT_MSG_SIZES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--output", default="results/exp_b_collectives")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
                        format="%(asctime)s %(levelname)s %(message)s")
    import ray

    from armada.run_cosmic_local import _redis_client_for_rendezvous, _release_run, form_ray_cluster

    run_rank(args, int(os.environ["RANK"]), args.world_size, os.environ["COMM_NAME"],
             _redis_client_for_rendezvous(), ray, form_ray_cluster, _release_run)


if __name__ == "__main__":
    main()