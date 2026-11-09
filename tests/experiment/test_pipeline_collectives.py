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

"""Unit test for aggregate_collectives_runs (SP4 Task 4).

Writes two synthetic per-run CSVs and checks the aggregator produces mean central
values plus cross-run sample-std columns — no redis/pycylon needed.
"""

import csv
import os
import statistics
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from results.pipeline import aggregate_collectives_runs  # noqa: E402

_FIELDS = [
    "channel", "collective", "msg_size", "payload_bytes",
    "latency_p50_ms", "latency_p99_ms", "latency_mean_ms",
    "throughput_MBps", "barrier_latency_ms", "reps", "unsupported",
    "rank", "world_size", "N", "run_id",
]


def _write_run(path, run_id, p50, tput):
    row = {
        "channel": "ucc", "collective": "broadcast", "msg_size": 4096, "payload_bytes": 4096,
        "latency_p50_ms": p50, "latency_p99_ms": p50 * 2, "latency_mean_ms": p50,
        "throughput_MBps": tput, "barrier_latency_ms": 0.0, "reps": 4, "unsupported": False,
        "rank": 0, "world_size": 4, "N": 4, "run_id": run_id,
    }
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerow(row)


def test_aggregate_collectives_mean_and_std(tmp_path):
    d = str(tmp_path)
    _write_run(os.path.join(d, "run1_exp_b_collectives_results.csv"), 1, p50=0.010, tput=200.0)
    _write_run(os.path.join(d, "run2_exp_b_collectives_results.csv"), 2, p50=0.020, tput=300.0)

    assert aggregate_collectives_runs(d) is True

    with open(os.path.join(d, "exp_b_collectives_results.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]

    # Central = mean across the two runs.
    assert float(r["latency_p50_ms"]) == 0.015
    assert float(r["throughput_MBps"]) == 250.0
    # Sample std (n-1) across runs (aggregator rounds to 6 decimals).
    assert abs(float(r["latency_p50_ms_std"]) - statistics.stdev([0.010, 0.020])) < 1e-5
    assert abs(float(r["throughput_MBps_std"]) - statistics.stdev([200.0, 300.0])) < 1e-3
    # run_id is an identity column, not averaged to a fractional value.
    assert r["run_id"] in ("1", "2")


def test_aggregate_collectives_no_runs_returns_false(tmp_path):
    # Only the canonical file present (no per-run CSVs) → nothing to aggregate.
    _write_run(os.path.join(str(tmp_path), "exp_b_collectives_results.csv"), 1, 0.01, 100.0)
    assert aggregate_collectives_runs(str(tmp_path)) is False

def _write_rows(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _cell(**overrides):
    row = {
        "channel": "ray-gloo", "collective": "allgather", "msg_size": 4096, "payload_bytes": 4096,
        "latency_p50_ms": 1.0, "latency_p99_ms": 2.0, "latency_mean_ms": 1.0,
        "throughput_MBps": 4.0, "barrier_latency_ms": 0.0, "reps": 20, "unsupported": False,
        "rank": 0, "world_size": 2, "N": 2, "run_id": 1,
    }
    row.update(overrides)
    return row


def _aggregated(d):
    with open(os.path.join(d, "exp_b_collectives_results.csv")) as f:
        return list(csv.DictReader(f))


def test_old_csvs_without_payload_aggregate_as_float32(tmp_path):
    d = str(tmp_path)
    _write_run(os.path.join(d, "run1_exp_b_collectives_results.csv"), 1, p50=0.010, tput=200.0)
    _write_run(os.path.join(d, "run2_exp_b_collectives_results.csv"), 2, p50=0.020, tput=300.0)

    assert aggregate_collectives_runs(d) is True

    rows = _aggregated(d)
    assert len(rows) == 1
    assert rows[0]["payload"] == "float32"


def test_payload_kinds_aggregate_separately(tmp_path):
    d = str(tmp_path)
    fields = ["channel", "payload"] + _FIELDS[1:]
    for run_id in (1, 2):
        _write_rows(os.path.join(d, f"run{run_id}_exp_b_collectives_results.csv"), fields,
                    [_cell(payload="float32", run_id=run_id),
                     _cell(payload="kv_pairs", run_id=run_id)])

    aggregate_collectives_runs(d)

    assert sorted(r["payload"] for r in _aggregated(d)) == ["float32", "kv_pairs"]


def test_inner_p50_gets_a_cross_run_std(tmp_path):
    d = str(tmp_path)
    fields = _FIELDS + ["latency_inner_p50_ms", "latency_inner_p99_ms"]
    for run_id, inner in ((1, 1.0), (2, 3.0)):
        _write_rows(os.path.join(d, f"run{run_id}_exp_b_collectives_results.csv"), fields,
                    [_cell(run_id=run_id, latency_inner_p50_ms=inner, latency_inner_p99_ms=inner)])

    aggregate_collectives_runs(d)

    row = _aggregated(d)[0]
    assert float(row["latency_inner_p50_ms"]) == 2.0
    assert abs(float(row["latency_inner_p50_ms_std"]) - statistics.stdev([1.0, 3.0])) < 1e-5


def test_blank_inner_columns_aggregate_to_blank_with_zero_std(tmp_path):
    d = str(tmp_path)
    fields = _FIELDS + ["latency_inner_p50_ms", "latency_inner_p99_ms"]
    for run_id in (1, 2):
        _write_rows(os.path.join(d, f"run{run_id}_exp_b_collectives_results.csv"), fields,
                    [_cell(run_id=run_id, latency_inner_p50_ms="", latency_inner_p99_ms="")])

    aggregate_collectives_runs(d)

    row = _aggregated(d)[0]
    assert row["latency_inner_p50_ms"] == ""
    assert float(row["latency_inner_p50_ms_std"]) == 0.0
