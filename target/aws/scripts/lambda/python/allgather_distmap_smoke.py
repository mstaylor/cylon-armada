"""allgather_distmap_smoke — tests whether FMI's AllGather collective completes
when its participants are independently scheduled Step Functions Distributed
Map child executions, rather than two Lambdas a single caller invokes directly
(the shape rendezvous_test.py exercises). Step Functions gives no guarantee
that Distributed Map children start together, so a rank could already be
waiting at the collective while another has not been scheduled yet.

Each child builds a one row table holding its own rank and calls AllGather.
Success means every child's row appears in the table every child gets back,
which is only possible if every participant actually joined the collective.
invocation_start_epoch_ms on each result is the figure that answers the
independent start question: how spread apart the children's actual start
times were.

Invoke through the state machine this ships with
(allgather_distmap_smoke.asl.json / aws_sfn_state_machine.allgather_distmap_smoke),
not by calling this handler directly:

    aws stepfunctions start-execution --state-machine-arn <arn> --input '{
      "items": [
        {"rank": 0, "world_size": 2, "comm_name": "distmap_smoke_1"},
        {"rank": 1, "world_size": 2, "comm_name": "distmap_smoke_1"}
      ]
    }'

Returns, per rank:
    {
        "success": true | false,
        "rank": 0,
        "world_size": 2,
        "comm_name": "...",
        "channel_type": "...",
        "invocation_start_epoch_ms": ...,
        "allgather_ms": ...,
        "gathered_ranks": [0, 1],
        "error": "..."   # present only on failure
    }
"""

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def handler(event, context):
    invocation_start_epoch_ms = round(time.time() * 1000, 2)
    base = {"invocation_start_epoch_ms": invocation_start_epoch_ms}

    try:
        rank = int(event["rank"])
        world_size = int(event["world_size"])
        comm_name = str(event["comm_name"])
    except (KeyError, TypeError, ValueError) as e:
        return {**base, "success": False, "error": f"malformed event: {e}"}

    channel_type = event.get("channel_type", os.environ.get("FMI_CHANNEL_TYPE", "redis"))
    base.update({
        "rank": rank,
        "world_size": world_size,
        "comm_name": comm_name,
        "channel_type": channel_type,
    })

    shared_scripts = os.environ.get("SHARED_SCRIPTS_PATH", "/cylon-armada/scripts")
    if shared_scripts not in sys.path:
        sys.path.insert(0, shared_scripts)

    try:
        import pyarrow as pa
        from pycylon import Table as CylonTable

        from communicator.fmi_bridge import FMIBridge

        logger.info(
            "Rank %d: joining AllGather (world_size=%d, comm_name=%s, channel=%s, "
            "started at epoch_ms=%.2f)",
            rank, world_size, comm_name, channel_type, invocation_start_epoch_ms,
        )

        bridge = FMIBridge(
            world_size=world_size,
            rank=rank,
            channel_type=channel_type,
            rendezvous_host=os.environ.get("RENDEZVOUS_HOST", ""),
            rendezvous_port=int(os.environ.get("RENDEZVOUS_PORT", 10000)),
            listen_port=int(os.environ.get("FMI_LISTEN_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
            comm_name=comm_name,
            maxtimeout=int(os.environ.get("FMI_MAX_TIMEOUT", 300000)),
        )

        if not bridge.available:
            return {**base, "success": False, "error": "FMI communicator not available in this container"}

        local_table = CylonTable.from_arrow(bridge._ctx, pa.table({"rank": [rank]}))

        t0 = time.monotonic()
        gathered = bridge.allgather(local_table)
        allgather_ms = round((time.monotonic() - t0) * 1000, 2)

        gathered_ranks = sorted(gathered.to_arrow().column("rank").to_pylist())
        expected_ranks = list(range(world_size))

        bridge.finalize()

        success = gathered_ranks == expected_ranks
        result = {
            **base,
            "success": success,
            "allgather_ms": allgather_ms,
            "gathered_ranks": gathered_ranks,
        }
        if not success:
            result["error"] = f"allgather returned ranks {gathered_ranks}, expected {expected_ranks}"
        return result

    except Exception as e:
        logger.error("Rank %d failed: %s", rank, e)
        return {**base, "success": False, "error": str(e)}