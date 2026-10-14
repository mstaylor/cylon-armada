"""Merge accepted outlier reruns into the Cosmic AI campaign manifests.

An accepted rerun (reruns.csv, within_20pct = 1) takes its original run's slot. A whole-group rerun
(--regroup RERUN_EXPERIMENT=EXPERIMENT) replaces every measured run of its configurations and keeps
the original cold start. Replaced rows are written apart so the originals stay on record. Merged
rows keep their own arn, so the aggregator reads each from the account that ran it.

Usage:
    python -m results.cosmic_rerun_merge --manifest exp1_cylon.jsonl exp1b_cylon.jsonl exp2_cylon.jsonl \
        --rerun-manifest rerun_cosmicai.jsonl --reruns reruns.csv --regroup regroup-exp1=exp1 \
        --out merged.jsonl --replaced-out replaced.jsonl
"""

import argparse
import csv
import json

from results.cosmic_lambda_results import CONFIGURATION_FIELDS as SLOT_FIELDS, account_of


class RerunMergeError(ValueError):
    pass


def _configuration(experiment, arm, slot):
    return (experiment, arm, slot[0], int(slot[1]), float(slot[2]), int(slot[3]), int(slot[4]))


def _slot_key(row):
    return _configuration(row["experiment"], row["arm"], row["slot"]) + (row["phase"], int(row["slot"][6]))


def _accepted_key(accepted):
    slot = [accepted[f] for f in SLOT_FIELDS]
    return _configuration(accepted["experiment"], accepted["arm"], slot) + ("measured",
                                                                            int(accepted["original_run"]))


def _relabelled(rerun, experiment, run, rerun_of):
    slot = list(rerun["slot"])
    slot[6] = run
    return {**rerun, "experiment": experiment, "slot": slot, "source_experiment": rerun["experiment"],
            "rerun_of": rerun_of}


def merge_reruns(original_rows, rerun_rows, accepted, regroups):
    """The original rows with accepted reruns swapped in, and the original rows they replaced.

    accepted holds reruns.csv rows whose rerun was accepted; regroups maps a whole-group rerun
    experiment to the experiment it replaces.
    """
    reruns_by_name = {r["name"]: r for r in rerun_rows}
    originals = {_slot_key(r): r for r in original_rows}
    swaps = {}
    for a in accepted:
        key = _accepted_key(a)
        if a["rerun_execution"] not in reruns_by_name:
            raise RerunMergeError(f"accepted rerun {a['rerun_execution']} is not in the rerun manifest")
        if key not in originals:
            raise RerunMergeError(f"accepted rerun {a['rerun_execution']} has no original run at {key}")
        if key in swaps:
            raise RerunMergeError(f"two accepted reruns for {key}: {swaps[key]['name']}, {a['rerun_execution']}")
        swaps[key] = reruns_by_name[a["rerun_execution"]]
    regrouped = {}
    for r in rerun_rows:
        if r["experiment"] in regroups and r["phase"] == "measured":
            target = regroups[r["experiment"]]
            regrouped.setdefault(_configuration(target, r["arm"], r["slot"]), []).append(r)
    merged, replaced = [], []
    for row in original_rows:
        key = _slot_key(row)
        if row["phase"] == "measured" and key[:7] in regrouped:
            replaced.append(row)
        elif key in swaps:
            replaced.append(row)
            merged.append(_relabelled(swaps[key], row["experiment"], row["slot"][6], row["name"]))
        else:
            merged.append(row)
    for configuration, rows in regrouped.items():
        merged += [_relabelled(r, configuration[0], r["slot"][6], None) for r in rows]
    return merged, replaced


def _read_jsonl(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def _write_jsonl(rows, path):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _regroup(spec):
    rerun_experiment, experiment = spec.split("=", 1)
    return rerun_experiment, experiment


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True, help="original campaign manifests")
    p.add_argument("--rerun-manifest", nargs="+", required=True, help="manifests of the rerun executions")
    p.add_argument("--reruns", required=True, help="rerun mapping CSV; rows with within_20pct = 1 are accepted")
    p.add_argument("--regroup", action="append", default=[], type=_regroup, metavar="RERUN_EXPERIMENT=EXPERIMENT",
                   help="a whole-group rerun that replaces every measured run of its configurations")
    p.add_argument("--out", required=True, help="merged manifest")
    p.add_argument("--replaced-out", required=True, help="the original rows the merge replaced")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.reruns) as handle:
        accepted = [r for r in csv.DictReader(handle) if r["within_20pct"] == "1"]
    merged, replaced = merge_reruns(_read_jsonl(args.manifest), _read_jsonl(args.rerun_manifest), accepted,
                                    dict(args.regroup))
    _write_jsonl(merged, args.out)
    _write_jsonl(replaced, args.replaced_out)
    print(f"{len(merged)} rows merged, {len(replaced)} originals replaced -> {args.replaced_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())