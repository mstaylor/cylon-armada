"""Cosmic AI results computed the way the published AI-for-Astronomy scripts computed them.

Follows aws/stats.py and aws/collect_logs.ipynb of AI-for-Astronomy at the paper version
(commit 566e722, arXiv:2501.06249) so Subexperiments A and B sit beside the published figures:

    map_s               Map state time, MapStateEntered to MapStateExited (the paper's inference time, 28 s)
    cpu_time_s          per worker profiler CPU time, IQR filtered mean over workers (paper Fig. 8)
    throughput_sum_bps  sum over workers of bits / own CPU time (paper Fig. 9 values)
    throughput_map_bps  total bits / Map state time (the paper's stated definition, Fig. 7)
    duration_s          end to end execution time (the published total_execution_time.csv)

Subexperiment A collects results in the Summarize state after the Map, Subexperiment B gathers
inside the Map, so map_s leaves out A's collection and includes B's.

    python -m results.cosmic_paper_metrics --system A=<export>/exp2/armA \
        --system B=<export>/exp2/armB_early_exit --manifest <manifests...> \
        --published-stats result_stats.csv --published-adjusted result_stats_adjusted.csv \
        --published-cost published_table6.csv --billed-cost final_cost_executions.csv \
        --cache paper_metrics_cache.json --account 448324707516=cylon --account 881908115028=cosmicai-admin \
        --output <dir>
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.chart_cosmic_campaign import CAPSIZE, FONT_SIZE, LEGEND_SIZE, TITLE_SIZE, _save, subexperiment
from results.cosmic_lambda_cost import TERMINAL_QUERY_STATUS

CONFIG_KEYS = ["system", "series", "partition_mb", "data_gb", "batch_size"]
RUN_METRICS = ["map_s", "duration_s", "cpu_time_s", "throughput_sum_bps", "throughput_map_bps", "total_bits",
               "memory_mb", "workers"]
CPU_TIME = "total_cpu_time (seconds)"
THROUGHPUT = "throughput_bps"
MEMORY = "total_cpu_memory (MB)"
SYSTEM_STYLES = {"published": ("Published (paper)", "black", ":", "^"),
                 "A": (f"{subexperiment('A')} (S3)", "#1f77b4", "-", "o"),
                 "B": (f"{subexperiment('B')} (direct communication)", "#d62728", "--", "s")}
LIFECYCLE_QUERY = 'filter @type="START" or @type="REPORT" | fields @timestamp, @type, @log'
INSIGHTS_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
MEMORY_QUERY = 'filter @type="REPORT" | stats count(*) as n, avg(@maxMemoryUsed) as peak by @log'
BYTES_PER_GB = 1e9


class MissingMapStateError(ValueError):
    pass


def _first_event(sfn, arn, kind, reverse):
    token = None
    while True:
        kwargs = {"executionArn": arn, "reverseOrder": reverse, "maxResults": 1000}
        if token:
            kwargs["nextToken"] = token
        page = sfn.get_execution_history(**kwargs)
        for event in page["events"]:
            if event["type"] == kind:
                return event["timestamp"]
            if event["type"] in ("MapIterationSucceeded", "MapIterationStarted"):
                return None
        token = page.get("nextToken")
        if not token:
            return None


def fetch_map_seconds(sfn, arn):
    """Seconds from MapStateEntered to MapStateExited, reading from each end of the history."""
    entered = _first_event(sfn, arn, "MapStateEntered", reverse=False)
    exited = _first_event(sfn, arn, "MapStateExited", reverse=True)
    if entered is None or exited is None:
        raise MissingMapStateError(f"{arn} has no complete Map state")
    return (exited - entered).total_seconds()


def map_seconds_from_log_rows(rows, init_function, summarize_function, started):
    """Map state time from Lambda logs: the execution's Initialize REPORT to the next Summarize START, for
    executions whose Step Functions history no longer exists. `started` is the execution start in UTC."""
    stamps = {}
    for row in rows:
        fields = {f["field"]: f["value"] for f in row}
        function = fields["@log"].rsplit("/", 1)[-1]
        stamps.setdefault((function, fields["@type"]), []).append(
            datetime.strptime(fields["@timestamp"], INSIGHTS_TIME_FORMAT))
    init_end = min((t for t in stamps.get((init_function, "REPORT"), []) if t >= started), default=None)
    summarize_start = [t for t in stamps.get((summarize_function, "START"), []) if init_end and t > init_end]
    if not summarize_start:
        raise MissingMapStateError(f"logs hold no {init_function} REPORT followed by a {summarize_function} START")
    return (min(summarize_start) - init_end).total_seconds()


def _run_insights(logs, groups, start, stop, query, poll_s):
    query_id = logs.start_query(logGroupNames=groups, startTime=start, endTime=stop, queryString=query)["queryId"]
    while True:
        result = logs.get_query_results(queryId=query_id)
        if result["status"] in TERMINAL_QUERY_STATUS:
            return result
        time.sleep(poll_s)


def map_seconds_from_logs(logs, init_group, summarize_group, started, stopped, slack_s, poll_s):
    """`started` and `stopped` are the execution's ISO 8601 start and stop."""
    start, stop = datetime.fromisoformat(started), datetime.fromisoformat(stopped)
    result = _run_insights(logs, [init_group, summarize_group], int(start.timestamp()),
                           int(stop.timestamp()) + slack_s, LIFECYCLE_QUERY, poll_s)
    started_utc = start.astimezone(timezone.utc).replace(tzinfo=None)
    return map_seconds_from_log_rows(result["results"], init_group.rsplit("/", 1)[-1],
                                     summarize_group.rsplit("/", 1)[-1], started_utc)


def iqr_mean(values, multiplier=1.5):
    """Mean after dropping values beyond multiplier x IQR, as aws/stats.py remove_outliers_and_mean."""
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    low, high = q1 - multiplier * (q3 - q1), q3 + multiplier * (q3 - q1)
    return float(np.mean([v for v in values if low <= v <= high]))


def per_run_metrics(workers):
    """One row per execution from per worker records, aggregated as aws/stats.py aggregates a run."""
    bits = workers[THROUGHPUT] * workers[CPU_TIME]
    grouped = workers.assign(bits=bits).groupby("execution")
    return pd.DataFrame({"cpu_time_s": grouped[CPU_TIME].apply(lambda s: iqr_mean(list(s))),
                         "throughput_sum_bps": grouped[THROUGHPUT].sum(), "total_bits": grouped["bits"].sum(),
                         "memory_mb": grouped[MEMORY].sum(), "workers": grouped.size()}).reset_index()


def summarize_runs(runs, metrics):
    """Mean, sample standard deviation and run count per configuration."""
    keys = [k for k in CONFIG_KEYS if k in runs]
    grouped = runs.groupby(keys)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.assign(n=grouped.size()).reset_index()


def paper_cost_estimate(requests, duration_s, memory_gb, gb_second_usd):
    """The paper's Table 6 estimate: requests x duration x memory x price per GB second."""
    return requests * duration_s * memory_gb * gb_second_usd


def load_workers(export_dir, series):
    frames = [pd.read_csv(path) for s in series
              for path in glob.glob(os.path.join(export_dir, s, "*", "*", "*", "cosmic_results.csv"))]
    workers = pd.concat(frames, ignore_index=True)
    return workers[workers.phase == "measured"]


def load_manifest_arns(paths):
    arns = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("arn"):
                    arns[row["name"]] = row["arn"]
    return arns


def _execution_gone(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code") == "ExecutionDoesNotExist"


def map_seconds(row, arn, clients, cache, log_fallback):
    """Map state time from the execution history, or from the Lambda logs when the history is gone."""
    if row.execution in cache["map_s"]:
        return cache["map_s"][row.execution]
    account = arn.split(":")[4]
    try:
        seconds, source = fetch_map_seconds(clients["sfn"][account], arn), "history"
    except Exception as error:
        if not _execution_gone(error):
            raise
        seconds = map_seconds_from_logs(clients["logs"][account], *log_fallback[:2], row.start, row.stop,
                                        *log_fallback[2:])
        source = "logs"
    cache["map_s"][row.execution] = seconds
    cache["map_source"][row.execution] = source
    return seconds


def system_runs(system, workers, arns, clients, cache, log_fallback):
    runs = per_run_metrics(workers)
    context = workers.groupby("execution")[["series", "partition_mb", "data_gb", "batch_size", "run_id",
                                            "duration_s", "start", "stop"]].first().reset_index()
    runs = context.merge(runs, on="execution")
    runs["map_s"] = [map_seconds(row, arns[row.execution], clients, cache, log_fallback)
                     for row in runs.itertuples()]
    runs["map_source"] = [cache["map_source"].get(name, "history") for name in runs.execution]
    runs["throughput_map_bps"] = runs.total_bits / runs.map_s
    runs.insert(0, "system", system)
    runs["account"] = [arns[name].split(":")[4] for name in runs.execution]
    return runs


def published_runs(stats_csv, adjusted_csv):
    """The paper version per run rows: raw throughput from result_stats, Map and total time from the adjusted file."""
    keys = ["partition (MB)", "data (GB)", "run"]
    raw = pd.read_csv(stats_csv)
    adjusted = pd.read_csv(adjusted_csv)
    raw, adjusted = raw[raw.run != "mean"], adjusted[adjusted.run != "mean"]
    joined = adjusted.merge(raw[keys + [THROUGHPUT]].rename(columns={THROUGHPUT: "throughput_sum_bps"}), on=keys)
    return pd.DataFrame({"system": "published", "series": "scaling", "partition_mb": joined["partition (MB)"],
                         "data_gb": joined["data (GB)"], "batch_size": 512, "run_id": joined.run.astype(int),
                         "map_s": joined["inference_duration (s)"], "duration_s": joined["total_duration (s)"],
                         "cpu_time_s": joined[CPU_TIME], "throughput_sum_bps": joined.throughput_sum_bps,
                         "throughput_map_bps": joined[THROUGHPUT], "total_bits": joined.total_bits,
                         "memory_mb": joined[MEMORY], "workers": joined.num_worlds.astype(int)})


def _query_peak_memory(logs, log_group, start, stop, slack_s, poll_s):
    query_id = logs.start_query(logGroupNames=[log_group], startTime=start, endTime=stop + slack_s,
                                queryString=MEMORY_QUERY)["queryId"]
    while True:
        result = logs.get_query_results(queryId=query_id)
        if result["status"] in TERMINAL_QUERY_STATUS:
            break
        time.sleep(poll_s)
    fields = {f["field"]: f["value"] for f in result["results"][0]} if result["results"] else {}
    return {"invocations": int(fields.get("n", 0)), "peak_gb": float(fields.get("peak", 0)) / BYTES_PER_GB}


def peak_memory(runs, logs_by_account, log_group, cache, slack_s, poll_s):
    """Mean worker peak memory (REPORT Max Memory Used) per execution, the paper's Table 6 memory column."""
    peaks = []
    for row in runs.itertuples():
        if row.execution not in cache["peak"]:
            start = int(datetime.fromisoformat(row.start).timestamp())
            stop = int(datetime.fromisoformat(row.stop).timestamp())
            cache["peak"][row.execution] = _query_peak_memory(logs_by_account[row.account], log_group, start, stop,
                                                              slack_s, poll_s)
        usage = cache["peak"][row.execution]
        peaks.append(usage["peak_gb"] if usage["invocations"] == row.workers else np.nan)
    return runs.assign(peak_memory_gb=peaks)


def cost_table(runs, billed, published_cost, gb_second_usd):
    """Table 6 at the full 12.6 GB dataset: the paper's estimate for each system beside the billed cost."""
    full = runs[(runs.series == "scaling") & (runs.data_gb == runs.data_gb.max())]
    full = full.merge(billed[["name", "usd"]].rename(columns={"name": "execution", "usd": "billed_usd"}),
                      on="execution", how="left")
    rows = full.groupby(["system", "partition_mb"]).agg(
        requests=("workers", "first"), cpu_time_s=("cpu_time_s", "mean"), peak_memory_gb=("peak_memory_gb", "mean"),
        billed_usd=("billed_usd", "mean"), billed_usd_std=("billed_usd", "std"), n=("execution", "size")).reset_index()
    rows["paper_estimate_usd"] = paper_cost_estimate(rows.requests, rows.cpu_time_s, rows.peak_memory_gb,
                                                     gb_second_usd)
    published = published_cost.assign(system="published")
    return pd.concat([published, rows], ignore_index=True).sort_values(["partition_mb", "system"])


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.minorticks_off()


def _plot_systems(ax, rows, x, metric, scale, systems):
    for system in systems:
        part = rows[rows.system == system].sort_values(x)
        if part.empty:
            continue
        label, color, linestyle, marker = SYSTEM_STYLES[system]
        positions = list(range(len(part)))
        err = (part[f"{metric}_std"].fillna(0) / scale).tolist()
        ax.errorbar(positions, (part[f"{metric}_mean"] / scale).tolist(), yerr=err if any(err) else None,
                    color=color, linestyle=linestyle, marker=marker, lw=2, capsize=CAPSIZE, label=label)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{v:g}" for v in part[x]])


def partition_grid(summary, metric, ylabel, title, name, output_dir, chart_format, dpi, scale=1.0,
                   systems=("published", "A", "B")):
    """Paper style metric against dataset size, one panel per partition size."""
    rows = summary[summary.series == "scaling"]
    partitions = sorted(rows.partition_mb.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    for ax, partition in zip(axes.flat, partitions):
        _plot_systems(ax, rows[rows.partition_mb == partition], "data_gb", metric, scale, systems)
        _style_axes(ax, "Dataset size (GB)", ylabel, f"{int(partition)} MB partitions")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=len(labels), frameon=True,
               fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, output_dir, name, chart_format, dpi)


def batch_chart(summary, output_dir, chart_format, dpi):
    """The paper's batch size sweep: CPU time and throughput (total bits / Map time) against batch size."""
    rows = summary[summary.series == "batch"]
    first = rows.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (metric, ylabel, scale) in zip(axes, (("cpu_time_s", "Per worker CPU time (s)", 1.0),
                                                  ("throughput_map_bps", "Throughput (billion bps)", 1e9))):
        _plot_systems(ax, rows, "batch_size", metric, scale, ("A", "B"))
        _style_axes(ax, "Batch size", ylabel, ylabel.split(" (")[0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Batch size sweep, {int(first.partition_mb)} MB partitions, {first.data_gb:g} GB",
                 fontsize=TITLE_SIZE)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(labels), frameon=True,
               fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, output_dir, "paper_batch_sweep", chart_format, dpi)


def render_all(summary, output_dir, chart_format, dpi):
    return [
        partition_grid(summary, "map_s", "Map state time (s)",
                       "Inference time as the paper measured it (Step Functions Map state)", "paper_map_time",
                       output_dir, chart_format, dpi),
        partition_grid(summary, "duration_s", "End to end time (s)", "End to end execution time",
                       "paper_end_to_end_time", output_dir, chart_format, dpi),
        partition_grid(summary, "cpu_time_s", "Per worker CPU time (s)",
                       "Average model inference time per worker (profiler CPU time)", "paper_cpu_time",
                       output_dir, chart_format, dpi),
        partition_grid(summary, "throughput_sum_bps", "Throughput (billion bps)",
                       "Throughput as the paper's figure computed it (sum of per worker throughput)",
                       "paper_throughput_sum", output_dir, chart_format, dpi, scale=1e9),
        partition_grid(summary, "throughput_map_bps", "Throughput (billion bps)",
                       "Throughput as the paper defines it (total bits / Map state time)", "paper_throughput_map",
                       output_dir, chart_format, dpi, scale=1e9),
        batch_chart(summary, output_dir, chart_format, dpi),
    ]


def _system(spec):
    key, _, path = spec.partition("=")
    return key, path


def _account(spec):
    account, _, profile = spec.partition("=")
    return account, profile


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", action="append", type=_system, required=True, metavar="KEY=EXPORT_DIR")
    p.add_argument("--series", nargs="+", default=["scaling", "batch"])
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--published-stats", required=True, help="AI-for-Astronomy aws/results/result_stats.csv")
    p.add_argument("--published-adjusted", required=True, help="AI-for-Astronomy aws/results/result_stats_adjusted.csv")
    p.add_argument("--published-cost", required=True, help="the paper's Table 6 as CSV")
    p.add_argument("--billed-cost", required=True, help="cost_executions CSV from results.cosmic_lambda_cost")
    p.add_argument("--cache", required=True)
    p.add_argument("--account", action="append", type=_account, required=True, metavar="ACCOUNT_ID=PROFILE")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--map-log-groups", nargs=2, action="append", metavar=("INIT_GROUP", "SUMMARIZE_GROUP"),
                   help="per system, in --system order: log groups that time the Map when the history is gone")
    p.add_argument("--executor-log-group", default="/aws/lambda/cylon-armada-cosmic-ai-executor")
    p.add_argument("--gb-second-usd", type=float, default=0.0000166667)
    p.add_argument("--window-slack-s", type=int, default=2)
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--dpi", type=int, default=200)
    return p


def main(argv=None):
    import boto3
    args = build_parser().parse_args(argv)
    sessions = {account: boto3.Session(profile_name=profile, region_name=args.region)
                for account, profile in args.account}
    clients = {"sfn": {account: s.client("stepfunctions") for account, s in sessions.items()},
               "logs": {account: s.client("logs") for account, s in sessions.items()}}
    logs_by_account = clients["logs"]
    cache = {"map_s": {}, "map_source": {}, "peak": {}}
    if os.path.exists(args.cache):
        with open(args.cache) as handle:
            cache.update(json.load(handle))
    arns = load_manifest_arns(args.manifest)
    try:
        runs = pd.concat([system_runs(key, load_workers(path, args.series), arns, clients, cache,
                                      (*groups, args.window_slack_s, args.poll_s))
                          for (key, path), groups in zip(args.system, args.map_log_groups)], ignore_index=True)
        full = runs[(runs.series == "scaling") & (runs.data_gb == runs.data_gb.max())]
        runs = runs.merge(peak_memory(full, logs_by_account, args.executor_log_group, cache, args.window_slack_s,
                                      args.poll_s)[["execution", "peak_memory_gb"]], on="execution", how="left")
    finally:
        with open(args.cache, "w") as handle:
            json.dump(cache, handle, indent=1)
    published = published_runs(args.published_stats, args.published_adjusted)
    everything = pd.concat([published, runs], ignore_index=True)
    summary = summarize_runs(everything, [m for m in RUN_METRICS if m != "workers"] + ["workers"])
    costs = cost_table(runs, pd.read_csv(args.billed_cost), pd.read_csv(args.published_cost), args.gb_second_usd)
    os.makedirs(args.output, exist_ok=True)
    everything.to_csv(os.path.join(args.output, "paper_metrics_runs.csv"), index=False)
    summary.to_csv(os.path.join(args.output, "paper_metrics_summary.csv"), index=False)
    costs.to_csv(os.path.join(args.output, "paper_cost_table.csv"), index=False)
    for name in render_all(summary, args.output, args.format, args.dpi):
        print(os.path.join(args.output, name))


if __name__ == "__main__":
    main()
