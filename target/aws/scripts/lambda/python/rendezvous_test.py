"""rendezvous_test — validates FMI rendezvous server connectivity.

Performs a TCP connectivity check against the rendezvous server to confirm
it is reachable and accepting connections. Used as a deployment health check
before running FMI direct-channel experiments.

Returns:
    {
        "success": true | false,
        "rendezvous_host": "...",
        "rendezvous_port": 10000,
        "connectivity_ms": ...,
        "error": "..."          # present only on failure
    }
"""

import logging
import os
import socket
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
        return {
            "success": False,
            "error": "RENDEZVOUS_HOST environment variable not set",
        }

    logger.info("TCP connect to %s:%d", rendezvous_host, rendezvous_port)

    t0 = time.monotonic()
    try:
        with socket.create_connection((rendezvous_host, rendezvous_port), timeout=5.0):
            pass
        connectivity_ms = (time.monotonic() - t0) * 1000
        logger.info("Connected in %.1fms", connectivity_ms)
        return {
            "success": True,
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": round(connectivity_ms, 2),
        }
    except Exception as e:
        connectivity_ms = (time.monotonic() - t0) * 1000
        logger.error("Connect failed: %s", e)
        return {
            "success": False,
            "rendezvous_host": rendezvous_host,
            "rendezvous_port": rendezvous_port,
            "connectivity_ms": round(connectivity_ms, 2),
            "error": str(e),
        }