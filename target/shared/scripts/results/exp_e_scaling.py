# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment E scaling summary over the Fargate sweep's per-rank records.

Reads <root>/<scaling>/<arm>/ws<N>/run<k>/rank<r>.json, the layout
fargate_cosmic_poc.py writes; warmup<k>/ directories and _timing.json are
never read. Every run is gated with exp_e_gate.check_arms across all arms
present, and a failing run is discarded for every arm.

Per run, each metric is the slowest rank's value, because an arm ends when its
slowest rank does. Across the measured runs the central value is the mean and
the spread the sample standard deviation, matching the established scaling
methodology.

comparable_s is the end-to-end figure the arms are compared on. Under Armada
and ray-cylon, waiting for the slowest rank happens inside each epoch's
AllGather and so inside run_s. Under ray-native there is no per-epoch barrier;
that wait moves to the end-of-run barrier, so its comparable figure is
run_s + teardown_barrier_s. ray_cluster_s, forming the Ray cluster, is reported
separately and never folded into either.
"""

import csv
import json
import os
import re
import statistics

from results.exp_e_gate import check_arms

ARMS_WITH_TEARDOWN_BARRIER = ("ray-native",)
METRICS = ("comparable_s", "run_s", "teardown_barrier_s", "establish_s", "ray_cluster_s")

_WS_DIR = re.compile(r"^ws(\d+)$")
_RUN_DIR = re.compile(r"^run(\d+)$")
_RANK_FILE = re.compile(r"^rank\d+\.json$")


def comparable_s(record):
    """The end-to-end time this rank's arm is compared on."""
    if record.get("backend") in ARMS_WITH_TEARDOWN_BARRIER:
        if "teardown_barrier_s" not in record:
            raise ValueError(
                f"{record.get('backend')} rank {record.get('rank')} has no teardown_barrier_s; "
                f"its run_s excludes the wait for peers and is not comparable on its own"
            )
        return record["run_s"] + record["teardown_barrier_s"]
    return record["run_s"]


def _subdirs(path, pattern):
    if not os.path.isdir(path):
        return []
    found = []
    for entry in sorted(os.listdir(path)):
        match = pattern.match(entry)
        if match and os.path.isdir(os.path.join(path, entry)):
            found.append((int(match.group(1)), os.path.join(path, entry)))
    return found


def load_runs(root):
    """{(scaling, world_size, run_index): {arm: [rank records]}} for measured runs."""
    runs = {}
    for scaling in sorted(os.listdir(root)):
        scaling_dir = os.path.join(root, scaling)
        if not os.path.isdir(scaling_dir):
            continue
        for arm in sorted(os.listdir(scaling_dir)):
            for world_size, ws_dir in _subdirs(os.path.join(scaling_dir, arm), _WS_DIR):
                for run_index, run_dir in _subdirs(ws_dir, _RUN_DIR):
                    records = []
                    for name in sorted(os.listdir(run_dir)):
                        if _RANK_FILE.match(name):
                            with open(os.path.join(run_dir, name)) as handle:
                                records.append(json.load(handle))
                    runs.setdefault((scaling, world_size, run_index), {})[arm] = records
    return dict(sorted(runs.items()))


def _slowest(records, metric):
    if metric == "comparable_s":
        return max(comparable_s(r) for r in records)
    values = [r[metric] for r in records if r.get(metric) is not None]
    return max(values) if values else None


def summarize(runs):
    """Rows per (scaling, arm, world_size) and the list of discarded runs."""
    per_cell = {}
    discarded = []
    for (scaling, world_size, run_index), by_arm in runs.items():
        gate = check_arms(by_arm)
        if not gate.passed:
            discarded.append({"scaling": scaling, "world_size": world_size, "run": run_index,
                              "arms": sorted(by_arm), "failures": gate.failures})
            continue
        for arm, records in by_arm.items():
            cell = per_cell.setdefault((scaling, arm, world_size), {m: [] for m in METRICS})
            for metric in METRICS:
                value = _slowest(records, metric)
                if value is not None:
                    cell[metric].append(value)

    rows = []
    for (scaling, arm, world_size), cell in sorted(per_cell.items()):
        row = {"scaling": scaling, "arm": arm, "world_size": world_size,
               "n_runs": len(cell["comparable_s"])}
        for metric in METRICS:
            values = cell[metric]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        rows.append(row)
    return rows, discarded


def write_summary(root, output_dir):
    """Write exp_e_scaling_summary.csv and exp_e_scaling_discarded.json."""
    rows, discarded = summarize(load_runs(root))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "exp_e_scaling_summary.csv")
    columns = ["scaling", "arm", "world_size", "n_runs"] + [
        f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    discarded_path = os.path.join(output_dir, "exp_e_scaling_discarded.json")
    with open(discarded_path, "w") as handle:
        json.dump(discarded, handle, indent=1)
    return csv_path, discarded_path
