"""FMI Communicator bridge for cylon-armada.

Wraps Cylon's FMI communicator for inter-Lambda communication in
context reuse workflows. Based on the pattern from AI-for-Astronomy's
inference_FMI.py.

Three use cases:
1. Context broadcasting — rank 0 broadcasts embedding cache to all workers
2. Result reduction — workers reduce costs/metrics back to rank 0
3. Model parallelism — tensor exchange between stages

Usage:
    from communicator.fmi_bridge import FMIBridge

    bridge = FMIBridge(world_size=10, rank=3)
    bridge.barrier()
    bridge.broadcast_embeddings(embeddings, root=0)
    bridge.reduce_cost(local_cost, root=0)
"""

import base64
import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _import_fmi():
    """Lazy-import FMI — only available in Cylon Lambda containers."""
    try:
        import fmi
        from fmilib.fmi_operations import fmi_communicator
        return fmi, fmi_communicator
    except ImportError:
        logger.warning("FMI not available — running without inter-Lambda communication")
        return None, None


class FMIBridge:
    """Bridge between cylon-armada and Cylon's FMI communicator.

    Provides context-reuse-specific operations on top of FMI primitives.

    Args:
        world_size: Total number of Lambda workers.
        rank: This worker's rank (0-indexed).
        channel_type: FMI channel type ('redis', 'direct', 's3'). Default: 'redis'.
        hint: FMI performance hint ('fast', 'low_latency'). Default: 'fast'.
    """

    def __init__(self, world_size, rank, channel_type="redis", hint="fast"):
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.channel_type = channel_type
        self.hint = hint

        self._fmi, self._fmi_communicator_cls = _import_fmi()
        self._comm = None

        if self._fmi is not None:
            self._comm = self._fmi_communicator_cls(self.world_size, self.rank)
            fmi_hint = getattr(self._fmi.hints, hint, self._fmi.hints.fast)
            self._comm.hint(fmi_hint)
            logger.info(
                "FMI communicator initialized: rank=%d, world_size=%d, "
                "channel_type=%s, hint=%s",
                self.rank, self.world_size, channel_type, hint,
            )

    @classmethod
    def from_env(cls):
        """Create FMIBridge from environment variables (Lambda context).

        Env vars (matching Cylon's Lambda conventions):
        - RANK, WORLD_SIZE — worker identity
        - FMI_CHANNEL_TYPE — channel type: 'redis', 'direct', 's3' (default: 'redis')
        - FMI_HINT — performance hint: 'fast', 'low_latency' (default: 'fast')
        - RENDEZVOUS_HOST, RENDEZVOUS_PORT — for 'direct' channel (TCPunch)
        - REDIS_HOST, REDIS_PORT — for 'redis' channel
        """
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        channel_type = os.environ.get("FMI_CHANNEL_TYPE", "redis")
        hint = os.environ.get("FMI_HINT", "fast")
        return cls(world_size=world_size, rank=rank,
                   channel_type=channel_type, hint=hint)

    @classmethod
    def from_payload(cls, payload):
        """Create FMIBridge from a Step Functions task payload."""
        return cls(
            world_size=int(payload.get("world_size", 1)),
            rank=int(payload.get("rank", 0)),
            channel_type=payload.get("fmi_channel_type", "redis"),
            hint=payload.get("fmi_hint", "fast"),
        )

    @property
    def available(self):
        """True if FMI is available and initialized."""
        return self._comm is not None

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def barrier(self):
        """Synchronize all workers."""
        if not self.available:
            return
        self._comm.barrier()
        logger.debug("Barrier complete (rank=%d)", self.rank)

    def reduce_float(self, value, root=0, op="sum"):
        """Reduce a float value across all workers to root.

        Args:
            value: Local float value.
            root: Destination rank.
            op: Reduction operation ('sum', 'max', 'min').

        Returns:
            Reduced value on root rank, local value on others.
        """
        if not self.available or self.world_size <= 1:
            return value

        fmi_op = getattr(self._fmi.op, op, self._fmi.op.sum)
        result = self._comm.reduce(
            float(value), root,
            self._fmi.func(fmi_op),
            self._fmi.types(self._fmi.datatypes.double),
        )
        return result

    def broadcast_bytes(self, data, root=0):
        """Broadcast raw bytes from root to all workers.

        Args:
            data: Bytes to broadcast (only needs to be valid on root).
            root: Source rank.

        Returns:
            Received bytes on all ranks.
        """
        if not self.available or self.world_size <= 1:
            return data

        if self.rank == root:
            # Encode length + data
            length = len(data) if data else 0
            payload = length.to_bytes(8, "big") + (data or b"")
        else:
            payload = b""

        # Use FMI broadcast — sends raw bytes
        result = self._comm.broadcast(payload, root)
        if self.rank != root:
            length = int.from_bytes(result[:8], "big")
            return result[8:8 + length]
        return data

    # ------------------------------------------------------------------
    # Context reuse operations
    # ------------------------------------------------------------------

    def broadcast_embeddings(self, embeddings_b64_list, root=0):
        """Broadcast embedding cache from root to all workers.

        Avoids each worker independently loading embeddings from Redis.

        Args:
            embeddings_b64_list: List of (context_id, embedding_b64) tuples.
                Only needs to be valid on root rank.
            root: Source rank.

        Returns:
            List of (context_id, embedding_b64) tuples on all ranks.
        """
        if not self.available or self.world_size <= 1:
            return embeddings_b64_list

        if self.rank == root:
            payload = json.dumps(embeddings_b64_list).encode("utf-8")
        else:
            payload = None

        received = self.broadcast_bytes(payload, root=root)
        return json.loads(received.decode("utf-8"))

    def broadcast_context_table(self, context_data, root=0):
        """Broadcast full context table (embeddings + metadata) from root.

        Args:
            context_data: Serializable dict with context store snapshot.
                Only needs to be valid on root rank.
            root: Source rank.

        Returns:
            Context data dict on all ranks.
        """
        if not self.available or self.world_size <= 1:
            return context_data

        if self.rank == root:
            payload = json.dumps(context_data).encode("utf-8")
        else:
            payload = None

        received = self.broadcast_bytes(payload, root=root)
        return json.loads(received.decode("utf-8"))

    def reduce_cost(self, local_cost, root=0):
        """Reduce total cost across all workers to root.

        Args:
            local_cost: This worker's cost (float).
            root: Destination rank.

        Returns:
            Total cost on root, local cost on others.
        """
        return self.reduce_float(local_cost, root=root, op="sum")

    def reduce_metrics(self, local_metrics, root=0):
        """Reduce multiple metrics across all workers.

        Args:
            local_metrics: Dict with float values to reduce.
            root: Destination rank.

        Returns:
            Dict with summed values on root, local values on others.
        """
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
    # Model parallelism operations
    # ------------------------------------------------------------------

    def exchange_tensor(self, tensor_data, tensor_shape, dest_rank):
        """Send a tensor to another Lambda worker.

        Used for model parallelism — send intermediate activations
        between pipeline stages.

        Args:
            tensor_data: numpy float32 array.
            tensor_shape: Tuple of dimensions.
            dest_rank: Target worker rank.
        """
        if not self.available:
            raise RuntimeError("FMI not available for tensor exchange")

        payload = {
            "data_b64": base64.b64encode(tensor_data.tobytes()).decode("ascii"),
            "shape": list(tensor_shape),
            "dtype": "float32",
            "src_rank": self.rank,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Use point-to-point send (if available) or broadcast pattern
        # FMI doesn't have direct send/recv, so use gather + scatter pattern
        logger.info("Tensor exchange: rank %d → rank %d, shape=%s",
                     self.rank, dest_rank, tensor_shape)

        # For now, use broadcast — all ranks get it, only dest uses it
        self.broadcast_bytes(payload_bytes, root=self.rank)

    def allgather_tensors(self, local_tensor):
        """All-gather tensors from all workers.

        Used for model parallelism — collect outputs from parallel stages
        before fusion.

        Args:
            local_tensor: numpy float32 array (this worker's contribution).

        Returns:
            List of numpy arrays, one per rank.
        """
        if not self.available or self.world_size <= 1:
            return [local_tensor]

        # Serialize local tensor
        local_payload = {
            "data_b64": base64.b64encode(local_tensor.tobytes()).decode("ascii"),
            "shape": list(local_tensor.shape),
            "rank": self.rank,
        }
        local_bytes = json.dumps(local_payload).encode("utf-8")

        # All-gather: each rank broadcasts, all collect
        gathered = []
        for src_rank in range(self.world_size):
            if src_rank == self.rank:
                data = self.broadcast_bytes(local_bytes, root=src_rank)
            else:
                data = self.broadcast_bytes(None, root=src_rank)

            payload = json.loads(data.decode("utf-8"))
            tensor = np.frombuffer(
                base64.b64decode(payload["data_b64"]),
                dtype=np.float32,
            ).reshape(payload["shape"])
            gathered.append(tensor)

        return gathered

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(self):
        """Clean up FMI communicator."""
        if self._comm is not None:
            self.barrier()
            logger.info("FMI finalized (rank=%d)", self.rank)
            self._comm = None