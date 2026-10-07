"""Run the Cosmic AI campaign (Experiment 1) one Step Functions execution at a time.

Usage:
    python cosmic_campaign_run.py --dry-run
    python cosmic_campaign_run.py --series baseline --live
    python cosmic_campaign_run.py --series scaling --max-workers 41 --live
    python cosmic_campaign_run.py --live

Executions run strictly in sequence: Arm A workers share one payload.json at the bucket root.
A local JSONL manifest records every execution so a crashed campaign resumes where it stopped.
"""

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass

from cosmic_campaign.grid import (
    CampaignSettings,
    RunSlot,
    baseline_configurations,
    batch_sweep,
    format_gb,
    reference_input,
    run_slots,
    scaling_grid,
)

logger = logging.getLogger("cosmic_campaign_run")

STATE_MACHINES = {"A": "cylon-armada-cosmic-ai-workflow", "B": "cylon-armada-cosmic-ai-fmi-workflow"}
SCRIPTS = {"A": "/tmp/Anomaly Detection/Inference/inference.py",
           "B": "/tmp/Anomaly Detection/Inference/inference_FMI.py"}


@dataclass(frozen=True)
class PlannedExecution:
    slot: RunSlot
    name: str
    sfn_input: dict
    experiment: str
    arm: str


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(STATE_MACHINES), default="A")
    p.add_argument("--experiment", default="exp1")
    p.add_argument("--series", choices=["all", "scaling", "batch", "baseline"], default="all")
    p.add_argument("--partitions", type=int, nargs="+", default=[25, 50, 75, 100])
    p.add_argument("--data-sizes", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10, 12.6])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    p.add_argument("--batch-partition-mb", type=int, default=100)
    p.add_argument("--batch-data-gb", type=float, default=1)
    p.add_argument("--baseline-workers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--warmup-runs", type=int, default=1,
                   help="cold-start runs per configuration; reported separately, never averaged in")
    p.add_argument("--measured-runs", type=int, default=3,
                   help="measured runs per scaling and baseline configuration (published campaign: 3)")
    p.add_argument("--batch-measured-runs", type=int, default=4,
                   help="measured runs per batch sweep configuration (published campaign: 4)")
    p.add_argument("--bucket", default="cosmicai-data-cylon")
    p.add_argument("--data-bucket", default="cosmicai-data-cylon")
    p.add_argument("--object-name", default="Anomaly Detection")
    p.add_argument("--object-type", default="folder")
    p.add_argument("--result-prefix", default="cylon-armada-track1")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--manifest", default="cosmic_campaign_manifest.jsonl")
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--execution-timeout-s", type=float, default=1800.0)
    p.add_argument("--est-worker-seconds", type=float, default=30.0)
    p.add_argument("--memory-gb", type=float, default=10.0)
    p.add_argument("--gb-second-usd", type=float, default=0.0000166667)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    return p


def _configurations(args):
    configs = []
    if args.series in ("all", "baseline"):
        configs += baseline_configurations(args.batch_partition_mb, args.baseline_workers, args.batch_size)
    if args.series in ("all", "scaling"):
        configs += scaling_grid(args.partitions, args.data_sizes, args.batch_size)
    if args.series in ("all", "batch"):
        configs += batch_sweep(args.batch_partition_mb, args.batch_data_gb, args.batch_sizes)
    if args.max_workers is not None:
        configs = [c for c in configs if c.workers <= args.max_workers]
    return configs


def execution_name(experiment, arm, slot, nonce):
    c = slot.configuration
    size = format_gb(c.data_gb).replace(".", "p")
    phase = "w" if slot.phase == "warmup" else "r"
    return (f"{experiment}-{arm}-{c.series[:4]}-{c.partition_mb}MB-{size}-b{c.batch_size}"
            f"-n{c.workers}-{phase}{slot.index}-{nonce}")


def plan_campaign(args):
    settings = CampaignSettings(
        bucket=args.bucket, data_bucket=args.data_bucket, object_name=args.object_name,
        object_type=args.object_type, scripts=SCRIPTS,
        result_prefix=f"{args.result_prefix}/{args.experiment}",
    )
    slots = []
    for configuration in _configurations(args):
        measured = args.batch_measured_runs if configuration.series == "batch" else args.measured_runs
        slots += run_slots([configuration], args.warmup_runs, measured)
    return [PlannedExecution(slot, execution_name(args.experiment, args.arm, slot, uuid.uuid4().hex[:8]),
                             reference_input(slot, args.arm, settings), args.experiment, args.arm)
            for slot in slots]


def estimate_cost(planned, worker_seconds, memory_gb, gb_second_usd):
    invocations = sum(p.slot.configuration.workers for p in planned)
    gb_seconds = invocations * worker_seconds * memory_gb
    return {"executions": len(planned), "worker_invocations": invocations,
            "gb_seconds": gb_seconds, "usd": gb_seconds * gb_second_usd}


def print_dry_run(planned, args):
    print(f"{'execution':<72} workers  result_path")
    for p in planned:
        print(f"{p.name:<72} {p.slot.configuration.workers:>7}  {p.sfn_input['result_path']}")
    cost = estimate_cost(planned, args.est_worker_seconds, args.memory_gb, args.gb_second_usd)
    print(f"\n{cost['executions']} executions, {cost['worker_invocations']} worker invocations, "
          f"{cost['gb_seconds']:.0f} GB-s, about ${cost['usd']:.2f} of Lambda at "
          f"{args.est_worker_seconds} s per worker; excludes Step Functions and S3")


TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}


class ConcurrentExecutionError(RuntimeError):
    pass


def slot_key(slot):
    c = slot.configuration
    return (c.series, c.partition_mb, c.data_gb, c.batch_size, c.workers, slot.phase, slot.index)


def run_key(planned):
    return (planned.experiment, planned.arm, *slot_key(planned.slot))


def completed_slots(manifest_path):
    done = set()
    try:
        with open(manifest_path) as handle:
            for line in handle:
                row = json.loads(line)
                if row["status"] == "SUCCEEDED":
                    done.add((row["experiment"], row["arm"], *row["slot"]))
    except FileNotFoundError:
        pass
    return done


def _assert_idle(sfn, state_machine_arn, finished, poll_s, sleep, idle_checks):
    """Refuse to start while another execution runs; this driver's own finished
    executions are ignored, since the RUNNING list can lag a completed one."""
    for check in range(idle_checks):
        running = [e["name"] for e in sfn.list_executions(
            stateMachineArn=state_machine_arn, statusFilter="RUNNING", maxResults=100)["executions"]
            if e["name"] not in finished]
        if not running:
            return
        if check < idle_checks - 1:
            sleep(poll_s)
    raise ConcurrentExecutionError(
        f"{state_machine_arn} has RUNNING execution(s) {running}; Arm A executions share "
        f"payload.json and must not overlap")


def _wait(sfn, arn, poll_s, timeout_s, sleep):
    deadline = time.monotonic() + timeout_s
    while True:
        described = sfn.describe_execution(executionArn=arn)
        if described["status"] in TERMINAL:
            return described
        if time.monotonic() > deadline:
            return {**described, "status": "DRIVER_TIMEOUT"}
        sleep(poll_s)


def _summarized_status(described):
    """SUCCEEDED only if the Summarize step also reported success; the Arm A summarizer
    returns statusCode 404 or 500 without failing the execution."""
    if described["status"] != "SUCCEEDED":
        return described["status"]
    try:
        output = json.loads(described.get("output") or "{}")
    except json.JSONDecodeError:
        return "SUMMARY_FAILED"
    code = output.get("statusCode") if isinstance(output, dict) else None
    return "SUCCEEDED" if code in (None, 200) else "SUMMARY_FAILED"


def run_campaign(planned, sfn, state_machine_arn, manifest_path, poll_s, timeout_s, sleep=time.sleep,
                 idle_checks=6):
    done = completed_slots(manifest_path)
    finished = set()
    rows = []
    for p in planned:
        if run_key(p) in done:
            continue
        key = slot_key(p.slot)
        _assert_idle(sfn, state_machine_arn, finished, poll_s, sleep, idle_checks)
        arn = sfn.start_execution(stateMachineArn=state_machine_arn, name=p.name,
                                  input=json.dumps(p.sfn_input))["executionArn"]
        described = _wait(sfn, arn, poll_s, timeout_s, sleep)
        if described["status"] in TERMINAL:
            finished.add(p.name)
        status = _summarized_status(described)
        start, stop = described.get("startDate"), described.get("stopDate")
        row = {"experiment": p.experiment, "arm": p.arm, "slot": list(key), "name": p.name, "arn": arn,
               "status": status,
               "phase": p.slot.phase, "workers": p.slot.configuration.workers,
               "result_path": p.sfn_input["result_path"],
               "start": start.isoformat() if start else None,
               "stop": stop.isoformat() if stop else None,
               "duration_s": (stop - start).total_seconds() if start and stop else None}
        with open(manifest_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        logger.info("%s %s %s", p.name, status, row["duration_s"])
        rows.append(row)
    return rows


def _state_machine_arn(sfn, name):
    for page in sfn.get_paginator("list_state_machines").paginate():
        for machine in page["stateMachines"]:
            if machine["name"] == name:
                return machine["stateMachineArn"]
    raise SystemExit(f"state machine {name!r} not found; run terraform apply")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    planned = plan_campaign(args)
    if args.dry_run:
        print_dry_run(planned, args)
        return 0
    import boto3

    sfn = boto3.client("stepfunctions", region_name=args.region)
    arn = _state_machine_arn(sfn, STATE_MACHINES[args.arm])
    rows = run_campaign(planned, sfn, arn, args.manifest, args.poll_s, args.execution_timeout_s)
    failed = [r for r in rows if r["status"] != "SUCCEEDED"]
    logger.info("%d executions run, %d not SUCCEEDED", len(rows), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
