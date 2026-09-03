"""FMI Communicator bridge for cylon-armada.

Wraps Cylon's FMI communicator (pycylon.net.fmi_communicator) for
inter-Lambda communication in context reuse workflows.

Available Python-level operations:
  - barrier()        → allreduce(0, SUM) — synchronise all workers
  - reduce_float()   → allreduce(value, SUM/MAX/MIN)
  - reduce_cost()    → reduce_float(cost, SUM)
  - reduce_metrics() → reduce_float per metric key
  - scatter()/gather()/allgather()/broadcast() — Arrow Table collectives
  - reduce_table()   → column-wise reduce/allreduce over an Arrow column

Broadcast operations (broadcast_bytes, broadcast_embeddings, etc.) are NOT
available at the Python level through pycylon. Those paths fall back to Redis
automatically when fmi.available is True but broadcast is not supported.

Usage:
    bridge = FMIBridge(world_size=4, rank=2,
                       channel_type="direct",
                       rendezvous_host="host", rendezvous_port=10000,
                       redis_host="redis-host", redis_port=6379)
    bridge.barrier()
    total_cost = bridge.reduce_cost(local_cost)
"""

import logging
import os

logger = logging.getLogger(__name__)

FMI_CHANNEL_TYPES = frozenset({"direct", "direct-redis", "redis", "s3"})

FMI_CHANNEL_ALIASES = {"tcpunch": "direct"}


def _import_fmi():
    """Lazy-import pycylon FMI — only available in Cylon Lambda containers.

    mpi4py.rc flags must be set before any pycylon import to prevent OpenMPI
    from calling MPI_Init automatically. MPI_Init fails in Lambda (no HOME dir).
    The FMI communicator uses TCP/TCPunch — it does not need MPI at runtime,
    but pycylon links against OpenMPI and mpi4py would trigger MPI_Init on
    import unless suppressed here.
    """
    try:
        import mpi4py
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
    except ImportError:
        pass

    try:
        from pycylon.net.fmi_config import FMIConfig
        from pycylon.net.reduce_op import ReduceOp
        from pycylon.frame import CylonEnv
        return FMIConfig, CylonEnv, ReduceOp
    except ImportError:
        logger.warning("pycylon FMI not available — running without inter-Lambda communication")
        return None, None, None


class FMIBridge:
    """Bridge between cylon-armada and Cylon's FMI communicator.

    Args:
        world_size:       Total number of Lambda workers.
        rank:             This worker's rank (0-indexed). Advisory only when
                          redis_host/redis_port are set — the communicator then
                          assigns the real rank via a Redis INCR counter, and
                          self.rank is updated to match after construction.
        channel_type:     FMI channel type — 'direct' (TCPunch hole punching),
                          'direct-redis' (plain TCP listen/connect within a VPC via
                          Redis-published addresses; Fargate/ECS), 'redis', or 's3'.
                          Matched case-insensitively. Default: 'redis'.
        rendezvous_host:  TCPunch rendezvous server host — 'direct' channel only.
        rendezvous_port:  TCPunch rendezvous server port — 'direct' channel only.
                          Default: 10000.
        listen_port:      This rank's own local TCP listen port — 'direct-redis'
                          channel only. A distinct config axis from
                          rendezvous_port: 'direct' dials a remote server,
                          'direct-redis' binds a local socket and publishes its
                          address via Redis. Default: 10000.
        redis_host:       Redis host (required for 'redis' channel).
        redis_port:       Redis port. Default: 6379.
        comm_name:        Communication group name — all workers in the same run
                          must use the same value. Default: 'cylon_armada'.
        maxtimeout:       FMI max timeout ms. Default: 120000.
        enableping:       Enable FMI ping. Default: False.
        advertise_host:   Explicit address to advertise to peers for the
                          'direct-redis' channel — distinct from rendezvous_host,
                          which is TCPunch-specific. Leave unset (default) on
                          Fargate/ECS to let ECS metadata auto-discovery resolve
                          the task's real address.
        required_peers:   Serialized per-rank connection map from
                          armada.topology.format_peer_map(), restricting the
                          channel to the peers the compiled plan actually uses
                          instead of the full N(N-1)/2 mesh. None keeps the mesh.
    """

    def __init__(self, world_size, rank, channel_type="redis",
                 rendezvous_host="", rendezvous_port=10000, listen_port=10000,
                 redis_host="", redis_port=6379,
                 comm_name="cylon_armada", maxtimeout=120000,
                 enableping=False, nonblocking=True, advertise_host="",
                 required_peers=None):
        self.world_size = int(world_size)
        self.rank = int(rank)

        normalized = channel_type.lower()
        normalized = FMI_CHANNEL_ALIASES.get(normalized, normalized)
        if normalized not in FMI_CHANNEL_TYPES:
            raise ValueError(
                f"unknown FMI channel type {channel_type!r} "
                f"(rank={self.rank}, world_size={self.world_size}, comm_name={comm_name!r}); "
                f"valid types: {sorted(FMI_CHANNEL_TYPES)}"
            )
        self.channel_type = normalized

        self._FMIConfig, self._CylonEnv, self._ReduceOp = _import_fmi()
        self._env = None
        self._comm = None
        self._ctx = None
        self._fmi_config = None

        if self._FMIConfig is None or self.world_size <= 1:
            return

        # Read by Direct::init() in cylon's C++ FMI channel, which otherwise
        # eagerly connects to every peer — N(N-1)/2 rendezvous pairings. Must be
        # set before the communicator is constructed, since that is when the
        # channel establishes its connections. It carries every rank's row
        # ("0:1,2;1:0,3") because the real rank is only assigned by the Redis INCR
        # counter afterwards — see armada.topology.format_peer_map.
        if required_peers:
            os.environ["FMI_REQUIRED_PEERS"] = required_peers
        else:
            os.environ.pop("FMI_REQUIRED_PEERS", None)

        port = int(listen_port) if self.channel_type == "direct-redis" else int(rendezvous_port)

        try:
            self._fmi_config = self._FMIConfig(
                rank=self.rank,
                world_size=self.world_size,
                host=rendezvous_host,
                port=port,
                maxtimeout=int(maxtimeout),
                resolveip=True,
                comm_name=comm_name,
                nonblocking=nonblocking,
                redis_host=redis_host,
                redis_port=int(redis_port),
                redis_namespace=comm_name,
                enableping=enableping,
                channel_type=self.channel_type,
                advertise_host=advertise_host,
            )
            self._env = self._CylonEnv(config=self._fmi_config, distributed=True)
            self._ctx = self._env.context
            self._comm = self._ctx.get_communicator()
            self.rank = self._ctx.get_rank()
            logger.info(
                "FMI communicator initialized: rank=%d world_size=%d channel=%s",
                self.rank, self.world_size, self.channel_type,
            )
        except Exception as e:
            logger.error("FMI communicator init failed: %s", e)
            self._env = None
            self._comm = None
            self._ctx = None
            self._fmi_config = None

    @classmethod
    def from_env(cls):
        """Create FMIBridge from environment variables (Lambda context)."""
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            channel_type=os.environ.get("FMI_CHANNEL_TYPE", "redis"),
            rendezvous_host=os.environ.get("RENDEZVOUS_HOST", ""),
            rendezvous_port=int(os.environ.get("RENDEZVOUS_PORT", 10000)),
            listen_port=int(os.environ.get("FMI_LISTEN_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
            advertise_host=os.environ.get("ADVERTISE_HOST", ""),
        )

    @classmethod
    def from_payload(cls, payload):
        """Create FMIBridge from a Step Functions task payload."""
        # Use experiment_name (unique per run) as comm_name so the Redis INCR rank
        # counter resets for each run. workflow_id is shared across runs 1-4 to
        # enable context reuse, but if used as comm_name the INCR accumulates and
        # run 2+ workers get ranks >= world_size, orphaning them at the rendezvous.
        # experiment_name is unique per run (includes run number in tag).
        experiment_name = payload.get("experiment_name", "")
        workflow_id = payload.get("workflow_id", "")
        comm_name_base = experiment_name or workflow_id
        comm_name = f"cylon_armada_{comm_name_base}" if comm_name_base else "cylon_armada"
        channel_type = payload.get("fmi_channel_type", "direct")
        # nonblocking=True: async connection attempts allow simultaneous SYN
        # exchange — required for TCP hole punching. nonblocking=False causes
        # sequential blocking which breaks the TCPunch timing window.
        fmi_options = payload.get("fmi_options", "nonblocking")
        nonblocking = (fmi_options != "blocking")
        return cls(
            world_size=int(payload.get("world_size", 1)),
            rank=int(payload.get("rank", 0)),
            channel_type=channel_type,
            rendezvous_host=os.environ.get("RENDEZVOUS_HOST", ""),
            rendezvous_port=int(os.environ.get("RENDEZVOUS_PORT", 10000)),
            listen_port=int(os.environ.get("FMI_LISTEN_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
            comm_name=comm_name,
            maxtimeout=300000,
            nonblocking=nonblocking,
            advertise_host=os.environ.get("ADVERTISE_HOST", ""),
        )

    @property
    def available(self):
        """True if FMI communicator is initialised and ready."""
        return self._comm is not None

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def barrier(self):
        """Synchronise all workers."""
        if not self.available:
            return
        self._comm.allreduce(0, self._ReduceOp.SUM)
        logger.debug("Barrier complete (rank=%d)", self.rank)

    def reduce_float(self, value, root=0, op="sum"):
        """Reduce a float value across all workers (result available on all ranks).

        Args:
            value: Local float value.
            root:  Unused — pycylon allreduce delivers result to all ranks.
            op:    'sum', 'max', or 'min'.

        Returns:
            Reduced value on all ranks.
        """
        if not self.available or self.world_size <= 1:
            return value

        reduce_op = {
            "sum": self._ReduceOp.SUM,
            "max": self._ReduceOp.MAX,
            "min": self._ReduceOp.MIN,
        }.get(op, self._ReduceOp.SUM)

        return self._comm.allreduce(float(value), reduce_op)

    def reduce_cost(self, local_cost, root=0):
        """Reduce total cost across all workers."""
        return self.reduce_float(local_cost, root=root, op="sum")

    def reduce_metrics(self, local_metrics, root=0):
        """Reduce multiple float metrics across all workers."""
        if not self.available or self.world_size <= 1:
            return local_metrics

        result = {}
        for key, value in local_metrics.items():
            if isinstance(value, (int, float)):
                result[key] = self.reduce_float(float(value), root=root, op="sum")
            else:
                result[key] = value
        return result

    def scatter(self, tables, root=0):
        if not self.available:
            raise RuntimeError(f"FMI not available (rank={self.rank}, world_size={self.world_size})")
        return self._comm.scatter(tables, root, self._ctx)

    def gather(self, table, root=0):
        if not self.available:
            raise RuntimeError(f"FMI not available (rank={self.rank}, world_size={self.world_size})")
        return self._comm.gather(table, root)

    def allgather(self, table):
        if not self.available:
            raise RuntimeError(f"FMI not available (rank={self.rank}, world_size={self.world_size})")
        return self._comm.allgather(table)

    def broadcast(self, table, root=0):
        if not self.available:
            raise RuntimeError(f"FMI not available (rank={self.rank}, world_size={self.world_size})")
        return self._comm.broadcast(table, root, self._ctx)

    def reduce_table(self, column, op, root=0):
        if not self.available:
            raise RuntimeError(f"FMI not available (rank={self.rank}, world_size={self.world_size})")
        reduce_op = {
            "sum": self._ReduceOp.SUM,
            "max": self._ReduceOp.MAX,
            "min": self._ReduceOp.MIN,
        }.get(op, self._ReduceOp.SUM)
        return self._comm.reduce_column(column, reduce_op, root)

    # ------------------------------------------------------------------
    # Broadcast — not available via pycylon allreduce; callers fall back
    # to Redis when these return None.
    # ------------------------------------------------------------------

    def broadcast_bytes(self, data, root=0):
        """Not supported via pycylon FMI — returns None so callers fall back to Redis."""
        logger.debug("broadcast_bytes not available via pycylon FMI (rank=%d)", self.rank)
        return None

    def broadcast_embeddings(self, embeddings_b64_list, root=0):
        """Not supported via pycylon FMI — returns None so callers fall back to Redis."""
        return None

    def broadcast_context_table(self, context_data, root=0):
        """Not supported via pycylon FMI — returns None so callers fall back to Redis."""
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(self):
        """Clean up FMI communicator."""
        if self._env is not None:
            try:
                self._env.finalize()
            except Exception:
                pass
            self._env = None
            self._comm = None
            logger.info("FMI finalized (rank=%d)", self.rank)