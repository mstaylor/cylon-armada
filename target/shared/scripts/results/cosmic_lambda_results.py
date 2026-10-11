"""Aggregate Cosmic AI Lambda campaign results (Arm A and Arm B).

Reads each execution's combined_data.json and aggregate_metrics.json, rejects incomplete rank
sets, restores integer counts (Arm B gathers them as float64), excludes warm ups, and reports
mean and sample standard deviation per configuration with the true n.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime

INTEGER_FIELDS = ("num_samples", "num_batches", "batch_size")
SUMMARY_METRICS = ("duration_s", "total_s_max", "inference_s_max", "aggregate_s", "collect_s",
                   "cold_worker_fraction")


class IncompleteExecutionError(ValueError):
    pass


def _is_missing(error):
    code = getattr(error, "response", {}).get("Error", {}).get("Code")
    return isinstance(error, KeyError) or code in ("NoSuchKey", "404")


def _read_json(s3, bucket, key, started_at):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        if _is_missing(error):
            raise IncompleteExecutionError(f"{key} missing") from error
        raise
    modified = response.get("LastModified")
    if started_at and modified and modified < datetime.fromisoformat(started_at):
        raise IncompleteExecutionError(f"{key} is stale: written {modified}, execution started {started_at}")
    return json.loads(response["Body"].read())


ABORT_MARKER_SOURCES = {"campaign_driver", "worker", "operator"}


def load_execution(s3, bucket, result_path, workers, started_at=None):
    combined = _read_json(s3, bucket, f"{result_path}/combined_data.json", started_at)
    metrics = _read_json(s3, bucket, f"{result_path}/aggregate_metrics.json", started_at)
    records = [r for r in combined if r.get("source") not in ABORT_MARKER_SOURCES]
    markers = len(combined) - len(records)
    if len(records) != workers or metrics.get("ranks_aggregated") not in (workers, workers + markers):
        raise IncompleteExecutionError(
            f"{result_path}: {len(records)} records, ranks_aggregated="
            f"{metrics.get('ranks_aggregated')}, expected {workers}")
    for record in records:
        for field in INTEGER_FIELDS:
            if record.get(field) is not None:
                record[field] = int(record[field])
    return {"records": records, "metrics": metrics}


def _cold_worker_fraction(records, cold_init_threshold_s):
    flags = [r["cold_start"] for r in records if r.get("cold_start") is not None]
    if flags:
        return sum(1 for f in flags if f) / len(flags)
    if cold_init_threshold_s is None:
        return None
    inits = [r["init_s"] for r in records if r.get("init_s") is not None]
    return sum(1 for i in inits if i > cold_init_threshold_s) / len(inits) if inits else None


def _execution_metrics(row, loaded, cold_init_threshold_s):
    records = loaded["records"]

    def slowest(field):
        values = [r[field] for r in records if r.get(field) is not None]
        return max(values) if values else None

    metrics = loaded["metrics"]
    parts = [slowest("publish_s"), metrics.get("trailing_gather_s", 0.0), metrics.get("aggregate_s")]
    collect_s = None if None in parts else sum(parts)
    return {"duration_s": row["duration_s"], "total_s_max": slowest("total_s"),
            "inference_s_max": slowest("inference_s"),
            "aggregate_s": metrics.get("aggregate_s"),
            "collect_s": collect_s,
            "cold_worker_fraction": _cold_worker_fraction(records, cold_init_threshold_s)}


REPORTED_PHASE = {"warmup": "cold_start", "measured": "measured"}


def _expected_runs(measured_runs, series):
    return measured_runs if isinstance(measured_runs, int) else measured_runs[series]


def summarize(manifest_rows, loader, measured_runs, cold_init_threshold_s=None):
    """One row per experiment, arm, configuration and phase.

    measured_runs is an int, or a mapping from series to its published run count. The
    cold-start run (manifest phase warmup) gets its own row and is never averaged into
    the measured row.
    """
    cells = defaultdict(list)
    attempted = defaultdict(set)
    for row in manifest_rows:
        key = (row["experiment"], row["arm"], *row["slot"][:5], REPORTED_PHASE[row["phase"]])
        attempted[key].add(row["slot"][6])
        cells.setdefault(key, [])
        if row["status"] != "SUCCEEDED":
            continue
        try:
            cells[key].append(_execution_metrics(row, loader(row), cold_init_threshold_s))
        except IncompleteExecutionError:
            continue
    summary = []
    for key, runs in cells.items():
        phase = key[7]
        expected = _expected_runs(measured_runs, key[2]) if phase == "measured" else len(attempted[key])
        out = {"experiment": key[0], "arm": key[1], "series": key[2], "partition_mb": key[3],
               "data_gb": key[4], "batch_size": key[5], "workers": key[6], "phase": phase,
               "n": len(runs), "complete": len(runs) == expected}
        for metric in SUMMARY_METRICS:
            values = [r[metric] for r in runs if r[metric] is not None]
            out[f"{metric}_mean"] = statistics.mean(values) if values else None
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        summary.append(out)
    return summary


CONFIGURATION_FIELDS = ("series", "partition_mb", "data_gb", "batch_size", "workers")


def partition_samples(manifest_rows, loader):
    """One row per partition of every successful execution, with the samples its rank processed."""
    samples = []
    for row in manifest_rows:
        if row["status"] != "SUCCEEDED":
            continue
        try:
            records = loader(row)["records"]
        except IncompleteExecutionError:
            continue
        configuration = dict(zip(CONFIGURATION_FIELDS, row["slot"][:5]))
        for record in records:
            samples.append({"experiment": row["experiment"], "arm": row["arm"], **configuration,
                            "run": f"{row['phase']}{row['slot'][6]}", "data_path": record["data_path"],
                            "num_samples": int(record["num_samples"])})
    return samples


def compare_samples(samples, reference):
    """Each partition's sample count against the reference experiment and arm's count for it."""
    def partition_key(s):
        return tuple(s[f] for f in CONFIGURATION_FIELDS[:4]) + (s["data_path"],)

    expected = {}
    for s in samples:
        if (s["experiment"], s["arm"]) == reference:
            expected.setdefault(partition_key(s), s["num_samples"])
    others = [s for s in samples if (s["experiment"], s["arm"]) != reference]
    mismatches = [{**s, "reference_samples": expected.get(partition_key(s))} for s in others
                  if expected.get(partition_key(s)) != s["num_samples"]]
    return {"reference": list(reference), "compared": len(others), "mismatches": mismatches}


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, nargs="+",
                   help="campaign manifests to aggregate together (rows carry their experiment and arm)")
    p.add_argument("--out", required=True)
    p.add_argument("--samples-out",
                   help="also write each partition's sample count to this CSV and compare it with the "
                        "reference experiment and arm")
    p.add_argument("--coverage-out", help="JSON coverage report path (default: beside --samples-out)")
    p.add_argument("--reference", nargs=2, default=["exp1", "A"], metavar=("EXPERIMENT", "ARM"))
    p.add_argument("--bucket", default="cosmicai-data-cylon",
                   help="data bucket holding the results (cosmicai: cosmicai)")
    p.add_argument("--profile", default="cylon",
                   help="AWS CLI profile for the target account (cosmicai: cosmicai-admin)")
    p.add_argument("--measured-runs", type=int, default=3,
                   help="expected measured runs per scaling and baseline configuration")
    p.add_argument("--batch-measured-runs", type=int, default=4,
                   help="expected measured runs per batch sweep configuration")
    p.add_argument("--cold-init-threshold-s", type=float, default=10.0,
                   help="a rank whose init_s exceeds this counts as a cold start")
    p.add_argument("--region", default="us-east-1")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import boto3

    s3 = boto3.Session(profile_name=args.profile).client("s3", region_name=args.region)
    rows = []
    for manifest in args.manifest:
        with open(manifest) as handle:
            rows += [json.loads(line) for line in handle]
    expected = {"scaling": args.measured_runs, "baseline": args.measured_runs,
                "batch": args.batch_measured_runs}
    cache = {}

    def loader(r):
        if r["result_path"] not in cache:
            cache[r["result_path"]] = load_execution(s3, args.bucket, r["result_path"], r["workers"],
                                                     started_at=r["start"])
        return cache[r["result_path"]]

    summary = summarize(rows, loader, expected, cold_init_threshold_s=args.cold_init_threshold_s)
    write_csv(summary, args.out)
    incomplete = [r for r in summary if not r["complete"]]
    print(f"{len(summary)} rows, {len(incomplete)} incomplete")
    if args.samples_out:
        samples = partition_samples(rows, loader)
        write_csv(samples, args.samples_out)
        report = compare_samples(samples, tuple(args.reference))
        coverage_out = args.coverage_out or args.samples_out.rsplit(".", 1)[0] + "_coverage.json"
        with open(coverage_out, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"{report['compared']} partitions compared with {args.reference}, "
              f"{len(report['mismatches'])} mismatches -> {coverage_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
