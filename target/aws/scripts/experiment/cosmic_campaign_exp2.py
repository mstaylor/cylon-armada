"""Experiment 2: the Cosmic AI campaign planned by agents, executed exactly as Experiment 1.

Usage:
    python cosmic_campaign_exp2.py requests --series baseline --out requests.json
    run_agents.sh 2 --requests requests.json --out-dir plans --cache-mode structural --similarity-threshold 0.95
    python cosmic_campaign_exp2.py merge --plans-dir plans --out-dir plans --requests requests.json
    python cosmic_campaign_exp2.py execute --merged plans/plans_merged.json --series baseline --live \
        --stop-on-failure --manifest exp2_cylon.jsonl

A plan that differs from its reference input in any field is recorded as wrong, and the reference
is executed instead, so Experiment 2 runs the same executions as Experiment 1.
"""

import argparse
import dataclasses
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cosmic_campaign_run as driver  # noqa: E402
from cosmic_campaign.grid import CampaignSettings  # noqa: E402
from cosmic_campaign.plan_check import PLAN_FIELDS  # noqa: E402
from cosmic_campaign.requests import execution_requests  # noqa: E402


def write_requests(path, requests, settings, arm):
    with open(path, "w") as handle:
        json.dump({"arm": arm, "settings": dataclasses.asdict(settings),
                   "requests": [{"request_id": r.request_id, "kind": r.kind, "text": r.text,
                                 "references": list(r.references)} for r in requests]}, handle, indent=1)


def _row_is_correct(row):
    return row.get("error") is None and all(not m for m in row["mismatches"])


def _as_checked(plan, reference):
    return {field: type(reference[field])(plan[field]) for field in PLAN_FIELDS}


def merge_plans(plan_rows):
    """The inputs to execute, one per reference, and a per-execution plan record."""
    to_run, records = [], []
    for row in plan_rows:
        correct = _row_is_correct(row) and len(row["plans"]) == len(row["references"])
        for index, reference in enumerate(row["references"]):
            to_run.append(_as_checked(row["plans"][index], reference) if correct else reference)
            records.append({"request_id": row["request_id"], "source": row["source"],
                            "result_path": reference["result_path"], "plan_correct": correct,
                            "mismatches": row["mismatches"], "error": row.get("error")})
    return to_run, records


def check_complete(plan_rows, request_ids):
    seen = Counter(r["request_id"] for r in plan_rows)
    missing = [i for i in request_ids if i not in seen]
    repeated = [i for i, n in seen.items() if n > 1]
    unknown = [i for i in seen if i not in set(request_ids)]
    if missing or repeated or unknown:
        raise SystemExit(f"plan manifests do not cover the requests once each: missing {missing[:5]}, "
                         f"repeated {repeated[:5]}, unknown {unknown[:5]}")


def campaign_order(planned_inputs, merged_inputs):
    """Merged inputs reordered to the campaign's own order, so warm up and run order match Experiment 1."""
    by_path = {i["result_path"]: i for i in merged_inputs}
    planned_paths = {i["result_path"] for i in planned_inputs}
    missing = [p for p in by_path if p not in planned_paths]
    if missing:
        raise SystemExit(f"{len(missing)} merged inputs match no planned execution, e.g. {missing[0]}")
    return [by_path[i["result_path"]] for i in planned_inputs if i["result_path"] in by_path]


def _mean(values):
    return sum(values) / len(values) if values else None


def plan_metrics(plan_rows, agents=()):
    sources = [r["source"] for r in plan_rows]
    wrong = [r for r in plan_rows if not _row_is_correct(r)]
    planning = [r["planning_ms"] for r in plan_rows if r.get("planning_ms") is not None]
    return {
        "requests": len(plan_rows),
        "llm_calls": sources.count("llm"),
        "llm_tool_calls": sources.count("llm_tool"),
        "cache_hits_exact": sources.count("cache_exact"),
        "cache_hits_structural": sources.count("cache_structural"),
        "wrong_plans": len(wrong),
        "wrong_reuse": sum(1 for r in wrong if r["source"].startswith("cache_")),
        "planning_ms_mean": _mean(planning),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in plan_rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in plan_rows),
        "embed_ms_mean": _mean([r["embed_ms"] for r in plan_rows if "embed_ms" in r]),
        "embed_tokens": sum(r.get("embed_tokens") or 0 for r in plan_rows),
        "planning_wall_ms": max((a["wall_ms"] for a in agents), default=None),
        "share_ms_mean": _mean([ms for a in agents for ms in a["share_ms"]]),
    }


def _settings(args):
    return CampaignSettings(bucket=args.bucket, data_bucket=args.data_bucket, object_name=args.object_name,
                            object_type=args.object_type, scripts=driver.SCRIPTS,
                            result_prefix=f"{args.result_prefix}/{args.experiment}")


def _campaign_args(argv):
    p = driver.build_parser()
    p.set_defaults(experiment="exp2")
    p.add_argument("--kind", choices=["execution", "sweep"], default="execution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="requests.json")
    p.add_argument("--merged")
    return p.parse_args(argv)


def cmd_requests(argv):
    args = _campaign_args(["--dry-run", *argv])
    if args.kind == "sweep":
        raise SystemExit("sweep requests are not supported in Experiment 2: use --kind execution")
    settings = _settings(args)
    planned = driver.plan_campaign(args)
    slots = [p.slot for p in planned if p.slot.configuration.series != "baseline"]
    requests = execution_requests(slots, args.arm, settings, args.seed)
    write_requests(args.out, requests, settings, args.arm)
    print(f"{len(requests)} requests for {len(slots)} executions written to {args.out}")


def cmd_merge(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--plans-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--requests", required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for path in sorted(glob.glob(os.path.join(args.plans_dir, "plans_*.jsonl")))
            for line in open(path)]
    check_complete(rows, [r["request_id"] for r in json.load(open(args.requests))["requests"]])
    agents = [json.load(open(path)) for path in sorted(glob.glob(os.path.join(args.plans_dir, "agent_*.json")))]
    to_run, records = merge_plans(rows)
    metrics = plan_metrics(rows, agents)
    json.dump({"inputs": to_run, "records": records}, open(os.path.join(args.out_dir, "plans_merged.json"), "w"),
              indent=1)
    json.dump(metrics, open(os.path.join(args.out_dir, "plan_metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))


def cmd_execute(argv):
    args = _campaign_args(argv)
    merged = json.load(open(args.merged))["inputs"]
    by_path = {p.sfn_input["result_path"]: p for p in driver.plan_campaign(args)}
    ordered = campaign_order([p.sfn_input for p in by_path.values()], merged)
    planned = [dataclasses.replace(by_path[i["result_path"]], sfn_input=i) for i in ordered]
    if args.dry_run:
        driver.print_dry_run(planned, args)
        return 0
    import boto3

    sfn = boto3.Session(profile_name=args.profile).client("stepfunctions", region_name=args.region)
    arn = driver._state_machine_arn(sfn, driver.STATE_MACHINES[args.arm])
    rows = driver.run_campaign(planned, sfn, arn, args.manifest, args.poll_s, args.execution_timeout_s,
                               stop_on_failure=args.stop_on_failure)
    return 1 if any(r["status"] != "SUCCEEDED" for r in rows) else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"requests": cmd_requests, "merge": cmd_merge, "execute": cmd_execute}
    if not argv or argv[0] not in commands:
        raise SystemExit(f"usage: cosmic_campaign_exp2.py {{{','.join(commands)}}} [options]")
    return commands[argv[0]](argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())