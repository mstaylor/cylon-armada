"""Arm B: Ray schedules the workers, Cylon collectives move the data.

Ported from CylonFlow (github.com/cylondata/cylonflow), whose Ray executor
creates a placement group of single-CPU bundles and binds one Cylon actor to
each. Two deliberate departures from that source:

CylonFlow stands Cylon up over Gloo. This arm uses the FMI transport Armada
uses, because the arm exists to isolate the scheduler: on Gloo, the comparison
against Armada would vary transport and scheduler together.

CylonFlow targets Ray 1.x, where a placement group was passed to options()
directly. Current Ray requires binding a placement group through a scheduling
strategy object instead — this module does not create the placement group or
launch the actors, only the bundle spec, the strategy validation, and the
per-rank Cylon environment; whichever module later launches `CylonRayActor`
under a placement group must use that modern scheduling-strategy form, not the
Ray 1.x `options(placement_group=...)` shape.
"""

import os

from communicator.fmi_bridge import FMIBridge

PLACEMENT_STRATEGIES = ("STRICT_SPREAD", "PACK")


def bundle_spec(world_size, cpus_per_worker=1):
    return [{"CPU": cpus_per_worker} for _ in range(world_size)]


def resolve_strategy(requested):
    strategy = requested or os.environ.get("RAY_PLACEMENT_STRATEGY", "STRICT_SPREAD")
    if strategy not in PLACEMENT_STRATEGIES:
        raise ValueError(
            f"unknown Ray placement strategy {strategy!r}; "
            f"valid strategies: {sorted(PLACEMENT_STRATEGIES)}"
        )
    return strategy


def resolve_listen_port(rank, strategy, base=None):
    """The port this rank binds for the direct-redis channel.

    Under STRICT_SPREAD each rank owns a node, so every rank binds the same
    base port — identical to Arm C, which runs one rank per Fargate task and
    never needs anything else.

    PACK exists for the co-location sensitivity check, and there several ranks
    share a node. A single fixed port would make every rank after the first
    fail to bind, and FMIBridge swallows that failure: it logs, sets the
    communicator to None, and reports available False. Nothing downstream
    checks that, so the arm would run with no collectives at all and still
    write a result record, labelling an isolated run as ray-cylon. Offsetting
    by rank keeps each bind distinct.
    """
    if base is None:
        base = int(os.environ.get("FMI_LISTEN_PORT", 10000))
    if strategy == "PACK":
        return base + rank
    return base


class CylonRayActor:
    """One rank, scheduled by Ray, communicating through Cylon.

    rank, world_size, comm_name, and required_peers come from the caller.
    required_peers carries no default on purpose: when FMIBridge receives a
    falsy peer map it pops FMI_REQUIRED_PEERS, and the C++ Direct channel then
    eagerly connects every pair, N(N-1)/2 rendezvous pairings where Arm C
    establishes only the DAG's edges. Since establish_s is compared between
    the arms, a caller that omitted the map would make Arm B slower for a
    reason that is not the scheduler. Arm C always computes it, so this arm
    requires it rather than inviting the omission.

    channel_type defaults to the transport Arm C hardcodes, so a caller that
    omits it matches Arm C rather than having to know the value.

    rank is the rank the caller requests. FMI assigns the authoritative rank
    by Redis INCR during communicator construction, so start_env replaces it
    with the assigned rank the way Arm C does; shard and context attribution
    must both follow the assigned rank or a rank computes one shard while
    participating in the collective as another.

    Every remaining FMIBridge argument is built the way run_cosmic_local.py's
    Armada backend builds it, argument for argument: listen_port, redis_host,
    redis_port, maxtimeout, and advertise_host from the same environment
    variables with the same defaults, and the rendezvous pair from
    FMIBridge's own constructor defaults, which is what Arm C takes because
    it never passes them. rendezvous_host reaches FMIConfig on every channel
    type, so sourcing it from RENDEZVOUS_HOST here would make the two arms
    build different bridges wherever that variable is set. Scheduler is the
    only thing allowed to differ between the arms.
    """

    def __init__(self, rank, world_size, comm_name, required_peers,
                 channel_type="direct-redis", nonblocking=True,
                 placement_strategy=None):
        self.requested_rank = rank
        self.rank = rank
        self.world_size = world_size
        self.channel_type = channel_type
        self.comm_name = comm_name
        self.nonblocking = nonblocking
        self.required_peers = required_peers
        self.placement_strategy = placement_strategy
        self._bridge = None

    def start_env(self):
        self._bridge = FMIBridge(
            world_size=self.world_size,
            rank=self.rank,
            channel_type=self.channel_type,
            comm_name=self.comm_name,
            listen_port=resolve_listen_port(
                self.requested_rank, resolve_strategy(self.placement_strategy)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
            maxtimeout=int(os.environ.get("FMI_MAX_TIMEOUT", 60000)),
            nonblocking=self.nonblocking,
            advertise_host=os.environ.get("ADVERTISE_HOST", ""),
            required_peers=self.required_peers,
        )
        self.rank = self._bridge.rank
        return self._bridge.available

    def shutdown(self):
        if self._bridge is not None:
            self._bridge.finalize()
            self._bridge = None
