"""FMI Communicator bridge for cylon-armada.

Wraps Cylon's FMI communicator (pycylon.net.fmi_communicator) for
inter-Lambda communication in context reuse workflows.

Available Python-level operations (pycylon exposes only allreduce):
  - barrier()        → allreduce(0, SUM) — synchronise all workers
  - reduce_float()   → allreduce(value, SUM/MAX/MIN)
  - reduce_cost()    → reduce_float(cost, SUM)
  - reduce_metrics() → reduce_float per metric key

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
        rank:             This worker's rank (0-indexed).
        channel_type:     FMI channel type ('direct', 'redis', 's3'). Default: 'redis'.
        rendezvous_host:  Rendezvous server host (required for 'direct' channel).
        rendezvous_port:  Rendezvous server port. Default: 10000.
        redis_host:       Redis host (required for 'redis' channel).
        redis_port:       Redis port. Default: 6379.
        comm_name:        Communication group name — all workers in the same run
                          must use the same value. Default: 'cylon_armada'.
        maxtimeout:       FMI max timeout ms. Default: 30000.
    """

    def __init__(self, world_size, rank, channel_type="redis",
                 rendezvous_host="", rendezvous_port=10000,
                 redis_host="", redis_port=6379,
                 comm_name="cylon_armada", maxtimeout=30000):
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.channel_type = channel_type

        self._FMIConfig, self._CylonEnv, self._ReduceOp = _import_fmi()
        self._env = None
        self._comm = None

        if self._FMIConfig is None or self.world_size <= 1:
            return

        try:
            fmi_config = self._FMIConfig(
                rank=self.rank,
                world_size=self.world_size,
                host=rendezvous_host,
                port=int(rendezvous_port),
                maxtimeout=int(maxtimeout),
                resolveip=True,
                comm_name=comm_name,
                nonblocking=False,
                redis_host=redis_host,
                redis_port=int(redis_port),
                redis_namespace=comm_name,
                channel_type=channel_type,
            )
            self._env = self._CylonEnv(config=fmi_config, distributed=True)
            self._comm = self._env.context.get_communicator()
            logger.info(
                "FMI communicator initialized: rank=%d world_size=%d channel=%s",
                self.rank, self.world_size, channel_type,
            )
        except Exception as e:
            logger.error("FMI communicator init failed: %s", e)
            self._env = None
            self._comm = None

    @classmethod
    def from_env(cls):
        """Create FMIBridge from environment variables (Lambda context)."""
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            channel_type=os.environ.get("FMI_CHANNEL_TYPE", "redis"),
            rendezvous_host=os.environ.get("RENDEZVOUS_HOST", ""),
            rendezvous_port=int(os.environ.get("RENDEZVOUS_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
        )

    @classmethod
    def from_payload(cls, payload):
        """Create FMIBridge from a Step Functions task payload."""
        return cls(
            world_size=int(payload.get("world_size", 1)),
            rank=int(payload.get("rank", 0)),
            channel_type=payload.get("fmi_channel_type", "redis"),
            rendezvous_host=os.environ.get("RENDEZVOUS_HOST", ""),
            rendezvous_port=int(os.environ.get("RENDEZVOUS_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
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