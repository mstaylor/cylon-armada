"""rendezvous_test — validates FMI rendezvous server connectivity and address exchange.

Phase 1: TCP connectivity check — confirms the rendezvous server is reachable.
Phase 2: Address exchange — two threads simulate rank 0 and rank 1 using the
         FMI direct (TCPunch) channel. A barrier() call exercises the full
         rendezvous handshake and peer address exchange.

Returns:
    {
        "success": true | false,
        "rendezvous_host": "...",
        "rendezvous_port": 10000,
        "connectivity_ms": ...,     # Phase 1 TCP connect latency
        "exchange_ms": {            # Phase 2 barrier latency per rank
            "rank_0": ...,
            "rank_1": ...
        },
        "error": "..."              # present only on failure
    }
"""

import logging
import os
import socket
import sys
import threading
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def handler(event, context):
    """Lambda entry point."""
    rendezvous_host = os.environ.get("RENDEZVOUS_HOST", "").strip()
    rendezvous_port = int(os.environ.get("RENDEZVOUS_PORT", 10000))

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
    # Two threads simulate rank 0 and rank 1. barrier() exercises the
    # full rendezvous handshake and peer address exchange.
    # ------------------------------------------------------------------
    logger.info("Phase 2: FMI address exchange (world_size=2, channel=direct)")

    shared_scripts = os.environ.get("SHARED_SCRIPTS_PATH", "/cylon-armada/scripts")
    if shared_scripts not in sys.path:
        sys.path.insert(0, shared_scripts)

    results = {}
    errors = {}

    def run_rank(rank):
        try:
            from communicator.fmi_bridge import FMIBridge

            t_start = time.monotonic()
            bridge = FMIBridge(
                world_size=2,
                rank=rank,
                channel_type="direct",
                hint="low_latency",
            )

            if not bridge.available:
                errors[rank] = "fmilib not available in this container"
                return

            logger.info("Rank %d: communicator initialized, running barrier", rank)
            bridge.barrier()
            exchange_ms = round((time.monotonic() - t_start) * 1000, 2)
            logger.info("Rank %d: barrier complete in %.1fms", rank, exchange_ms)
            bridge.finalize()
            results[rank] = exchange_ms

        except Exception as e:
            logger.error("Rank %d failed: %s", rank, e)
            errors[rank] = str(e)

    threads = [threading.Thread(target=run_rank, args=(r,), name=f"rank-{r}") for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Check for threads that timed out (still alive after join)
    timed_out = [t.name for t in threads if t.is_alive()]
    if timed_out:
        return {
            "success": False,
            "phase": "address_exchange",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": connectivity_ms,
            "error": f"Threads timed out after 30s: {timed_out}",
        }

    if errors:
        return {
            "success": False,
            "phase": "address_exchange",
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": connectivity_ms,
            "errors": {f"rank_{k}": v for k, v in errors.items()},
        }

    return {
        "success": True,
        "rendezvous_host": rendezvous_host,
        "rendezvous_port": rendezvous_port,
        "connectivity_ms": connectivity_ms,
        "exchange_ms": {
            "rank_0": results.get(0),
            "rank_1": results.get(1),
        },
    }