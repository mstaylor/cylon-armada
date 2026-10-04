"""Experiment E scaling summary over the Fargate sweep's result records.

The comparable time differs by arm on purpose: ray-native has no per-epoch
barrier, so waiting for the slowest rank lands in its end-of-run barrier
(teardown_barrier_s) instead of inside run_s as it does under Armada's
per-epoch AllGather. Comparing run_s alone would flatter ray-native.

Run: pytest tests/test_exp_e_scaling.py -v
"""

import csv
import json
import os
import statistics
import sys

import pytest

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "shared",
                                        "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from results.exp_e_scaling import comparable_s, load_runs, summarize, write_summary


def _record(rank, world_size, backend, run_s, per_rank=5, **extra):
    record = {"rank": rank, "world_size": world_size, "backend": backend,
              "galaxies": per_rank, "shard": [rank * per_rank, (rank + 1) * per_rank],
              "records_written": per_rank, "cache_hits": 0, "records_failed": 0,
              "throttle_events": 0, "run_s": run_s, "establish_s": 0.5}
    record.update(extra)
    return record


def _write(root, scaling, arm, world_size, run_dir, records):
    path = os.path.join(root, scaling, arm, f"ws{world_size}", run_dir)
    os.makedirs(path, exist_ok=True)
    for record in records:
        with open(os.path.join(path, f"rank{record['rank']}.json"), "w") as handle:
            json.dump(record, handle)


def test_ray_native_comparable_time_adds_the_teardown_barrier():
    record = _record(0, 2, "ray-native", 3.0, teardown_barrier_s=1.5, ray_cluster_s=9.0)
    assert comparable_s(record) == 4.5


def test_armada_and_ray_cylon_comparable_time_is_run_s():
    assert comparable_s(_record(0, 2, "armada", 3.0)) == 3.0
    assert comparable_s(_record(0, 2, "ray-cylon", 3.0, ray_cluster_s=9.0)) == 3.0


def test_a_ray_native_record_missing_its_barrier_field_is_refused():
    """A ray-native record without teardown_barrier_s came from a runner that
    predates the barrier; its run_s is not comparable, so it must not be
    silently summed as if the barrier cost zero."""
    with pytest.raises(ValueError, match="teardown_barrier_s"):
        comparable_s(_record(0, 2, "ray-native", 3.0))


def test_warmups_and_timing_files_are_never_read(tmp_path):
    root = str(tmp_path)
    _write(root, "strong", "armada", 2, "warmup0", [_record(r, 2, "armada", 99.0) for r in range(2)])
    _write(root, "strong", "armada", 2, "run0", [_record(r, 2, "armada", 1.0) for r in range(2)])
    with open(os.path.join(root, "strong", "armada", "ws2", "run0", "_timing.json"), "w") as f:
        json.dump({"ranks": 2}, f)

    runs = load_runs(root)

    assert list(runs) == [("strong", 2, 0)]
    assert [r["run_s"] for r in runs[("strong", 2, 0)]["armada"]] == [1.0, 1.0]


def _sweep(root, run_s_by_run, world_size=2):
    for run_index, (armada, native, cylon) in enumerate(run_s_by_run):
        _write(root, "strong", "armada", world_size, f"run{run_index}",
               [_record(r, world_size, "armada", armada + r) for r in range(world_size)])
        _write(root, "strong", "ray-native", world_size, f"run{run_index}",
               [_record(r, world_size, "ray-native", native, teardown_barrier_s=0.5 * r,
                        ray_cluster_s=10.0 + r) for r in range(world_size)])
        _write(root, "strong", "ray-cylon", world_size, f"run{run_index}",
               [_record(r, world_size, "ray-cylon", cylon, ray_cluster_s=20.0)
                for r in range(world_size)])


def test_summary_takes_the_slowest_rank_per_run_then_mean_and_sample_std(tmp_path):
    root = str(tmp_path)
    _sweep(root, [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (3.0, 4.0, 5.0), (4.0, 5.0, 6.0)])

    rows, discarded = summarize(load_runs(root))
    by_arm = {row["arm"]: row for row in rows}

    assert discarded == []
    armada_per_run = [2.0, 3.0, 4.0, 5.0]
    assert by_arm["armada"]["n_runs"] == 4
    assert by_arm["armada"]["comparable_s_mean"] == pytest.approx(statistics.mean(armada_per_run))
    assert by_arm["armada"]["comparable_s_std"] == pytest.approx(statistics.stdev(armada_per_run))

    native_per_run = [2.5, 3.5, 4.5, 5.5]
    assert by_arm["ray-native"]["comparable_s_mean"] == pytest.approx(statistics.mean(native_per_run))
    assert by_arm["ray-native"]["run_s_mean"] == pytest.approx(3.5)
    assert by_arm["ray-native"]["teardown_barrier_s_mean"] == pytest.approx(0.5)
    assert by_arm["ray-native"]["ray_cluster_s_mean"] == pytest.approx(11.0)

    assert by_arm["ray-cylon"]["comparable_s_mean"] == pytest.approx(4.5)
    assert by_arm["ray-cylon"]["ray_cluster_s_mean"] == pytest.approx(20.0)
    assert by_arm["armada"]["ray_cluster_s_mean"] is None
    assert by_arm["ray-cylon"]["teardown_barrier_s_mean"] is None


def test_a_run_that_fails_the_gate_is_discarded_for_every_arm(tmp_path):
    """Paired-run semantics: a run is comparable only if every arm in it
    did the same work, so a failing arm takes the whole run with it rather
    than leaving the other arms averaged over a different set of runs."""
    root = str(tmp_path)
    _sweep(root, [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)])
    bad = os.path.join(root, "strong", "ray-cylon", "ws2", "run1", "rank1.json")
    with open(bad) as handle:
        record = json.load(handle)
    record["error"] = "RuntimeError: boom"
    with open(bad, "w") as handle:
        json.dump(record, handle)

    rows, discarded = summarize(load_runs(root))

    assert [(d["scaling"], d["world_size"], d["run"]) for d in discarded] == [("strong", 2, 1)]
    assert any("boom" in failure for failure in discarded[0]["failures"])
    assert all(row["n_runs"] == 1 for row in rows)
    assert all(row["comparable_s_std"] is None for row in rows)


def test_write_summary_emits_csv_and_discarded_json(tmp_path):
    root = str(tmp_path / "in")
    _sweep(root, [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)])
    out = str(tmp_path / "out")

    csv_path, discarded_path = write_summary(root, out)

    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["arm"] for row in rows} == {"armada", "ray-native", "ray-cylon"}
    assert {"comparable_s_mean", "comparable_s_std", "ray_cluster_s_mean",
            "teardown_barrier_s_mean", "establish_s_mean", "n_runs"} <= set(rows[0])
    assert json.load(open(discarded_path)) == []


def test_the_results_pipeline_runs_the_scaling_summary(tmp_path, monkeypatch):
    from results import pipeline

    root = str(tmp_path / "in")
    _sweep(root, [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)])
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["pipeline", "--experiment", "exp_e_scaling",
                                      "--local-dir", root, "--output-dir", out])

    pipeline.main()

    assert os.path.exists(os.path.join(out, "exp_e_scaling_summary.csv"))
    assert os.path.exists(os.path.join(out, "exp_e_scaling_discarded.json"))
    charts = sorted(os.listdir(os.path.join(out, "charts")))
    assert any(name.startswith("exp_e_strong_comparable_time.") for name in charts)
    assert any(name.startswith("exp_e_strong_ray_cluster_formation.") for name in charts)


def _captured_figures(monkeypatch, tmp_path):
    from results import chart_exp_e_scaling

    figures = {}

    def capture(fig, output_dir, name, chart_format, chart_dpi):
        figures[name] = fig
        return name

    monkeypatch.setattr(chart_exp_e_scaling, "_save", capture)
    root = str(tmp_path / "in")
    for world_size in (1, 2, 4):
        _sweep(root, [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)], world_size=world_size)
    csv_path, _ = write_summary(root, str(tmp_path / "out"))
    chart_exp_e_scaling.generate_exp_e_scaling_charts(csv_path, str(tmp_path / "charts"))
    return figures


def test_charts_follow_the_quals_notebook_style(monkeypatch, tmp_path):
    """Box frame, no gridlines, major ticks only, legend boxed below, error
    bars with caps (CLAUDE.md chart conventions)."""
    figures = _captured_figures(monkeypatch, tmp_path)
    assert set(figures) == {"exp_e_strong_comparable_time", "exp_e_strong_ray_cluster_formation"}

    for fig in figures.values():
        ax = fig.axes[0]
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
        assert len(ax.xaxis.get_minor_locator().tick_values(1, 64)) == 0
        assert len(ax.yaxis.get_minor_locator().tick_values(0.1, 100)) == 0
        legend = ax.get_legend()
        assert legend.get_frame_on()
        assert legend._loc == 8
        assert ax.get_title()

    line_ax = figures["exp_e_strong_comparable_time"].axes[0]
    assert [int(t) for t in line_ax.get_xticks()] == [1, 2, 4]
    labels = [text.get_text() for text in line_ax.get_legend().get_texts()]
    assert any("ray native" in label.lower() and "teardown barrier" in label for label in labels)
    caps = [c for c in line_ax.containers if hasattr(c, "lines")]
    assert caps and all(c.lines[1] for c in caps)

    bars = [p for p in figures["exp_e_strong_ray_cluster_formation"].axes[0].patches]
    assert bars and all(p.get_edgecolor()[:3] == (0.0, 0.0, 0.0) for p in bars)
    import matplotlib.pyplot as plt
    for fig in figures.values():
        plt.close(fig)
