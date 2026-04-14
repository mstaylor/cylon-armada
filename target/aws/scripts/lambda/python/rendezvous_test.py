"""rendezvous_test — validates FMI rendezvous server connectivity and address exchange.

Phase 1: TCP connectivity check — confirms the rendezvous server is reachable.
Phase 2: Address exchange — a single rank registers with the rendezvous server
         via the FMI direct (TCPunch) channel. To complete the exchange, invoke
         two Lambda functions concurrently with the same comm_name but different
         ranks (rank=0, rank=1). allreduce(1, SUM) == world_size confirms both
         workers completed the rendezvous handshake and can communicate.

Single-rank invocation (Phase 2):
    Invoke two Lambdas concurrently with the same comm_name:

    # Terminal 1 (rank 0):
    aws lambda invoke --function-name cylon-armada-rendezvous-test \
      --payload '{"rank":0,"world_size":2,"comm_name":"test_abc"}' /dev/stdout

    # Terminal 2 (rank 1):
    aws lambda invoke --function-name cylon-armada-rendezvous-test \
      --payload '{"rank":1,"world_size":2,"comm_name":"test_abc"}' /dev/stdout

    Both should return success with allreduce result == 2.

TCP-only invocation (Phase 1 only — no rank/comm_name needed):
    aws lambda invoke --function-name cylon-armada-rendezvous-test \
      --payload '{}' /dev/stdout

Returns:
    {
        "success": true | false,
        "rendezvous_host": "...",
        "rendezvous_port": 10000,
        "rank": 0,
        "world_size": 2,
        "comm_name": "test_abc",
        "connectivity_ms": ...,     # Phase 1 TCP connect latency
        "exchange_ms": ...,         # Phase 2 allreduce latency (single rank)
        "allreduce_result": 2.0,    # Should equal world_size
        "error": "..."              # present only on failure
    }
"""

import logging
import os
import socket
import sys
import time
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def handler(event, context):
    """Lambda entry point."""
    rendezvous_host = os.environ.get("RENDEZVOUS_HOST", "").strip()
    rendezvous_port = int(os.environ.get("RENDEZVOUS_PORT", 10000))
    redis_host = os.environ.get("REDIS_HOST", "")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))

    if not rendezvous_host:
        return {"success": False, "error": "RENDEZVOUS_HOST environment variable not set"}

    # ------------------------------------------------------------------
    # Phase 1 — TCP connectivity
    # ------------------------------------------------------------------
    logger.info("Phase 1: TCP connect to %s:%d", rendezvous_host, rendezvous_port)
    t0 = time.monotonic()
    try:
        with socket.create_connection((rendezvous_host, rendezvous_port), timeout=5.0):
            pass
        connectivity_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info("Phase 1 passed in %.1fms", connectivity_ms)
    except Exception as e:
        return {
            "success": False,
            "phase": "tcp_connect",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": round((time.monotonic() - t0) * 1000, 2),
            "error": str(e),
        }

    # ------------------------------------------------------------------
    # Phase 2 — Address exchange via FMI direct channel
    #
    # Requires rank and comm_name in the event payload. If not provided,
    # return Phase 1 results only (TCP connectivity check).
    # ------------------------------------------------------------------
    rank = event.get("rank")
    if rank is None:
        return {
            "success": True,
            "phase": "tcp_only",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": connectivity_ms,
        }

    rank = int(rank)
    world_size = int(event.get("world_size", 2))
    comm_name = event.get("comm_name", f"rendezvous_test_{uuid.uuid4().hex[:8]}")

    logger.info(
        "Phase 2: FMI address exchange (rank=%d, world_size=%d, comm_name=%s, channel=direct)",
        rank, world_size, comm_name,
    )

    shared_scripts = os.environ.get("SHARED_SCRIPTS_PATH", "/cylon-armada/scripts")
    if shared_scripts not in sys.path:
        sys.path.insert(0, shared_scripts)

    try:
        from communicator.fmi_bridge import FMIBridge

        t_start = time.monotonic()
        bridge = FMIBridge(
            world_size=world_size,
            rank=rank,
            channel_type="direct",
            rendezvous_host=rendezvous_host,
            rendezvous_port=rendezvous_port,
            redis_host=redis_host,
            redis_port=redis_port,
            comm_name=comm_name,
        )

        if not bridge.available:
            return {
                "success": False,
                "phase": "address_exchange",
                "rendezvous_host": rendezvous_host,
                "rendezvous_port": rendezvous_port,
                "rank": rank,
                "world_size": world_size,
                "comm_name": comm_name,
                "connectivity_ms": connectivity_ms,
                "error": "pycylon FMI not available in this container",
            }

        logger.info("Rank %d: communicator ready, running allreduce", rank)
        total = bridge.reduce_float(1.0, op="sum")
        exchange_ms = round((time.monotonic() - t_start) * 1000, 2)

        logger.info("Rank %d: allreduce returned %.1f in %.1fms", rank, total, exchange_ms)
        bridge.finalize()

        success = int(total) == world_size
        result = {
            "success": success,
            "phase": "address_exchange",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "rank": rank,
            "world_size": world_size,
            "comm_name": comm_name,
            "connectivity_ms": connectivity_ms,
            "exchange_ms": exchange_ms,
            "allreduce_result": total,
        }
        if not success:
            result["error"] = f"allreduce returned {total}, expected {world_size}"
        return result

    except Exception as e:
        logger.error("Rank %d failed: %s", rank, e)
        return {
            "success": False,
            "phase": "address_exchange",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "rank": rank,
            "world_size": world_size,
            "comm_name": comm_name,
            "connectivity_ms": connectivity_ms,
            "error": str(e),
        }