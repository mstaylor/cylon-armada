"""Billed Lambda cost of Cosmic AI campaign executions.

Each execution's cost comes from the platform's own REPORT records (billed duration and
memory size) of every function in the workflow, read with CloudWatch Logs Insights over the
execution's start to stop window. Executions run one at a time, so a window holds one
execution; a window whose executor invocation count differs from the worker count also caught
part of another execution and is left out of the summary.

    python -m results.cosmic_lambda_cost --manifest exp2_cylon.jsonl \
        --variant run1=exp2_armB_run1_accept_poll_1s.jsonl \
        --usage-cache results/cosmic_exp2/billed_usage.json \
        --executions-out results/cosmic_exp2/cost_executions.csv \
        --out results/cosmic_exp2/cost_summary.csv --chart-dir results/cosmic_exp2/final_charts
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from results.chart_cosmic_campaign import (
    BAR_EDGE, BAR_LW, FONT_SIZE, TITLE_SIZE, _legend_below, _save, partition_panels, subexperiment,
)
from results.cosmic_lambda_results import CONFIGURATION_FIELDS, REPORTED_PHASE, account_of, write_csv

REPORT_QUERY = ('filter @type="REPORT" | stats count(*) as n, sum(@billedDuration) as billed_ms, '
                'max(@memorySize) as mem by @log')
TERMINAL_QUERY_STATUS = {"Complete", "Failed", "Cancelled", "Timeout"}
BYTES_PER_MB = 1000000
MB_PER_GB = 1024


class MissingBilledUsageError(RuntimeError):
    pass


def _epoch(timestamp):
    return int(datetime.fromisoformat(timestamp).timestamp())


def query_billed_usage(logs, log_groups, start, stop, slack_s, poll_s):
    """Invocations, billed milliseconds and memory per function for one time window."""
    query_id = logs.start_query(logGroupNames=log_groups, startTime=start, endTime=stop + slack_s,
                                queryString=REPORT_QUERY)["queryId"]
    while True:
        result = logs.get_query_results(queryId=query_id)
        if result["status"] in TERMINAL_QUERY_STATUS:
            break
        time.sleep(poll_s)
    if result["status"] != "Complete":
        raise MissingBilledUsageError(f"query {query_id} for {start}-{stop} ended {result['status']}")
    usage = {}
    for row in result["results"]:
        fields = {f["field"]: f["value"] for f in row}
        usage[fields["@log"].rsplit("/", 1)[-1]] = {"invocations": int(fields["n"]),
                                                    "billed_ms": float(fields["billed_ms"]),
                                                    "memory_mb": float(fields["mem"]) / BYTES_PER_MB}
    return usage


def collect_usage(rows, logs, log_groups, cache, slack_s, poll_s):
    """Billed usage per successful execution name, querying only executions not in `cache`."""
    usage = {}
    for row in rows:
        if row["status"] != "SUCCEEDED":
            continue
        if row["name"] not in cache:
            cache[row["name"]] = query_billed_usage(logs, log_groups, _epoch(row["start"]), _epoch(row["stop"]),
                                                    slack_s, poll_s)
        usage[row["name"]] = cache[row["name"]]
    return usage


def collect_usage_by_account(rows, logs_by_account, default_logs, log_groups, cache, slack_s, poll_s):
    """collect_usage with each execution queried in the logs of the account in its arn."""
    groups = defaultdict(list)
    for row in rows:
        groups[account_of(row) if account_of(row) in logs_by_account else None].append(row)
    usage = {}
    for account, group in groups.items():
        logs = logs_by_account[account] if account else default_logs
        usage.update(collect_usage(group, logs, log_groups, cache, slack_s, poll_s))
    return usage


def execution_cost(usage, gb_second_usd, request_usd):
    gb_seconds = sum(u["billed_ms"] / 1000 * u["memory_mb"] / MB_PER_GB for u in usage.values())
    invocations = sum(u["invocations"] for u in usage.values())
    return {"gb_seconds": gb_seconds, "invocations": invocations,
            "usd": gb_seconds * gb_second_usd + invocations * request_usd}


def summarize_costs(rows, usage, executor_function, gb_second_usd, request_usd):
    """Per execution costs, and mean and sample standard deviation per configuration and phase."""
    executions = []
    cells = defaultdict(list)
    for row in rows:
        if row["name"] not in usage:
            continue
        execution_usage = usage[row["name"]]
        executor = execution_usage.get(executor_function, {})
        attributed = executor.get("invocations") == row["workers"]
        cost = execution_cost(execution_usage, gb_second_usd, request_usd)
        variant = row.get("variant", "")
        phase = REPORTED_PHASE[row["phase"]]
        configuration = dict(zip(CONFIGURATION_FIELDS, row["slot"][:5]))
        executions.append({"name": row["name"], "experiment": row["experiment"], "arm": row["arm"],
                           "variant": variant, **configuration, "phase": phase, "run": row["slot"][6],
                           "attributed": attributed,
                           "duration_s": row["duration_s"], "executor_billed_s": executor.get("billed_ms", 0) / 1000,
                           **cost})
        if attributed:
            key = (row["experiment"], row["arm"], variant, *row["slot"][:5], phase)
            cells[key].append((cost, row["duration_s"]))
    summary = []
    for key, runs in sorted(cells.items(), key=lambda item: tuple(str(k) for k in item[0])):
        usd = [c["usd"] for c, _ in runs]
        summary.append({"experiment": key[0], "arm": key[1], "variant": key[2],
                        **dict(zip(CONFIGURATION_FIELDS, key[3:8])), "phase": key[8], "n": len(runs),
                        "usd_mean": statistics.mean(usd),
                        "usd_std": statistics.stdev(usd) if len(usd) >= 2 else None,
                        "gb_seconds_mean": statistics.mean(c["gb_seconds"] for c, _ in runs),
                        "duration_s_mean": statistics.mean(d for _, d in runs)})
    return executions, summary


def cost_ratio_by_run_group(executions):
    """Arm B's cost over Arm A's per configuration, pairing run i of each arm, as the mean and sample
    std across the run groups. Only attributed measured executions of the unlabelled manifests count."""
    by_arm = {"A": {}, "B": {}}
    for e in executions:
        if e["attributed"] and e["phase"] == "measured" and not e["variant"] and e["arm"] in by_arm:
            configuration = (e["experiment"], *(e[f] for f in CONFIGURATION_FIELDS))
            by_arm[e["arm"]][(configuration, e["run"])] = e["usd"]
    ratios = defaultdict(list)
    for (configuration, run), a_usd in by_arm["A"].items():
        if (configuration, run) in by_arm["B"]:
            ratios[configuration].append(by_arm["B"][(configuration, run)] / a_usd)
    return [{"experiment": c[0], **dict(zip(CONFIGURATION_FIELDS, c[1:])), "run_pairs": len(values),
             "cost_ratio_mean": statistics.mean(values),
             "cost_ratio_std": statistics.stdev(values) if len(values) >= 2 else None}
            for c, values in sorted(ratios.items(), key=lambda item: tuple(str(k) for k in item[0]))]


ARM_BAR_COLORS = {("A", ""): "orange", ("B", ""): "green"}
VARIANT_BAR_COLOR = "lightgreen"


def render_cost_chart(executions, summary, output_dir, chart_format, dpi):
    """A cost per execution bar chart per partition size, and one line chart of Arm B's cost over
    Arm A's with the sample std across run groups."""
    scaling = [s for s in summary if s["series"] == "scaling" and s["phase"] == "measured"]
    cell = {(s["arm"], s["variant"], s["partition_mb"], s["workers"]): s for s in scaling}
    arms = sorted({(s["arm"], s["variant"]) for s in scaling}, key=lambda a: (a[0], a[1] != "", a[1]))
    files = []
    width = 0.8 / len(arms)
    for partition in sorted({s["partition_mb"] for s in scaling}):
        configurations = sorted({(s["data_gb"], s["workers"]) for s in scaling if s["partition_mb"] == partition})
        fig, ax = plt.subplots(figsize=(10, 6))
        for k, (arm, variant) in enumerate(arms):
            present = [(i, cell[(arm, variant, partition, w)]) for i, (_, w) in enumerate(configurations)
                       if (arm, variant, partition, w) in cell]
            if not present:
                continue
            ax.bar([i + (k - (len(arms) - 1) / 2) * width for i, _ in present], [c["usd_mean"] for _, c in present],
                   width, color=ARM_BAR_COLORS.get((arm, variant), VARIANT_BAR_COLOR), edgecolor=BAR_EDGE,
                   linewidth=BAR_LW, label=subexperiment(arm) + (f" ({variant})" if variant else ""))
        ax.set_xticks(range(len(configurations)))
        ax.set_xticklabels([f"{d:g}\n({w})" for d, w in configurations])
        ax.set_xlabel("Dataset size in GB (workers)", fontsize=FONT_SIZE)
        ax.set_ylabel("Lambda cost per execution (USD)", fontsize=FONT_SIZE)
        ax.set_title(f"Billed Lambda cost per execution, {partition} MB partitions", fontsize=TITLE_SIZE)
        _legend_below(ax, ncol=min(len(arms), 3), anchor=(0.5, -0.3))
        files.append(_save(fig, output_dir, f"exp2_lambda_cost_{partition}MB", chart_format, dpi))
    ratios = pd.DataFrame([r for r in cost_ratio_by_run_group(executions) if r["series"] == "scaling"])
    files.append(partition_panels(ratios, [("cost_ratio", "cost_ratio_mean", "cost_ratio_std")],
                                  f"{subexperiment('B')} cost / {subexperiment('A')} cost",
                                  "Lambda cost of direct communication relative to S3, per run group", output_dir,
                                  "exp2_lambda_cost_ratio", chart_format, dpi))
    return files


def _variant(spec):
    label, path = spec.split("=", 1)
    return label, path


def load_rows(manifests, variants):
    rows = []
    for path in manifests:
        with open(path) as handle:
            rows += [json.loads(line) for line in handle]
    for label, path in variants:
        with open(path) as handle:
            rows += [{**json.loads(line), "variant": label} for line in handle]
    return rows


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", default=[], help="campaign manifests")
    p.add_argument("--variant", action="append", default=[], type=_variant, metavar="LABEL=MANIFEST",
                   help="a manifest whose rows are reported apart under LABEL, such as an archived earlier run")
    p.add_argument("--experiment", nargs="+", default=["exp2"], help="experiments to price")
    p.add_argument("--usage-cache", required=True,
                   help="JSON of billed usage per execution; read first, queried only for missing executions")
    p.add_argument("--executions-out", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chart-dir", help="also write the cost chart here")
    p.add_argument("--chart-format", default="png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--log-groups", nargs="+", default=[
        "/aws/lambda/cylon-armada-cosmic-ai-executor", "/aws/lambda/cylon-armada-cosmic-ai-init",
        "/aws/lambda/cylon-armada-cosmic-ai-summarize", "/aws/lambda/cylon-armada-cosmic-ai-fmi-init",
        "/aws/lambda/cylon-armada-cosmic-ai-fmi-summarize"])
    p.add_argument("--executor-function", default="cylon-armada-cosmic-ai-executor",
                   help="function whose invocation count must equal the worker count")
    p.add_argument("--gb-second-usd", type=float, default=0.0000166667, help="Lambda x86 price per GB-second")
    p.add_argument("--request-usd-per-million", type=float, default=0.20, help="Lambda price per million requests")
    p.add_argument("--window-slack-s", type=int, default=2,
                   help="seconds after an execution's stop that its REPORT records may still arrive")
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--profile", default="cylon")
    p.add_argument("--account", action="append", default=[], type=_account_profile, metavar="ACCOUNT_ID=PROFILE",
                   help="query executions whose arn is in ACCOUNT_ID with PROFILE; others use --profile")
    p.add_argument("--region", default="us-east-1")
    return p


def _account_profile(spec):
    account, profile = spec.split("=", 1)
    return account, profile


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = [r for r in load_rows(args.manifest, args.variant) if r["experiment"] in args.experiment]
    cache = {}
    if os.path.exists(args.usage_cache):
        with open(args.usage_cache) as handle:
            cache = json.load(handle)
    if any(r["status"] == "SUCCEEDED" and r["name"] not in cache for r in rows):
        import boto3
        def logs_client(profile):
            return boto3.Session(profile_name=profile).client("logs", region_name=args.region)

        logs_by_account = {account: logs_client(profile) for account, profile in args.account}
        try:
            collect_usage_by_account(rows, logs_by_account, logs_client(args.profile), args.log_groups, cache,
                                     args.window_slack_s, args.poll_s)
        finally:
            with open(args.usage_cache, "w") as handle:
                json.dump(cache, handle, indent=1)
    usage = collect_usage(rows, None, args.log_groups, cache, args.window_slack_s, args.poll_s)
    executions, summary = summarize_costs(rows, usage, args.executor_function, args.gb_second_usd,
                                          args.request_usd_per_million / 1000000)
    write_csv(executions, args.executions_out)
    write_csv(summary, args.out)
    dropped = [e["name"] for e in executions if not e["attributed"]]
    print(f"{len(executions)} executions priced, {len(dropped)} left out of the summary: {dropped}")
    write_csv(cost_ratio_by_run_group(executions), args.out.rsplit(".", 1)[0] + "_ratio.csv")
    if args.chart_dir:
        for name in render_cost_chart(executions, summary, args.chart_dir, args.chart_format, args.dpi):
            print(os.path.join(args.chart_dir, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())