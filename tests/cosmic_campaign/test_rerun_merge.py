"""Merging accepted outlier reruns and whole-group reruns into the campaign manifests."""

import pytest

from results.cosmic_rerun_merge import RerunMergeError, account_of, merge_reruns

CYLON_ARN = "arn:aws:states:us-east-1:448324707516:execution:wf:{}"
COSMICAI_ARN = "arn:aws:states:us-east-1:881908115028:execution:wf:{}"


def _row(experiment, phase, run, duration, series="scaling", batch=512, workers=41, arn=CYLON_ARN):
    name = f"{experiment}-{series}-b{batch}-n{workers}-{phase}{run}"
    return {"experiment": experiment, "arm": "A", "slot": [series, 25, 1.0, batch, workers, phase, run],
            "name": name, "arn": arn.format(name), "status": "SUCCEEDED", "phase": phase, "workers": workers,
            "result_path": f"{experiment}/{phase}{run}", "duration_s": duration}


def _accepted(rerun_name, run=2, series="scaling", batch=512, workers=41):
    return {"experiment": "exp2", "arm": "A", "series": series, "partition_mb": 25, "data_gb": 1.0,
            "batch_size": batch, "workers": workers, "original_run": run, "rerun_execution": rerun_name}


def _originals():
    return [_row("exp2", "warmup", 0, 30.0)] + [_row("exp2", "measured", k, d) for k, d in ((1, 21.0), (2, 16.5),
                                                                                         (3, 21.2))]


def test_an_accepted_rerun_takes_the_original_slot_and_the_original_is_set_aside():
    rerun = _row("rerun5-exp2", "measured", 1, 20.8, arn=COSMICAI_ARN)
    merged, replaced = merge_reruns(_originals(), [rerun], [_accepted(rerun["name"])], {})
    measured = sorted((r["slot"][6], r["duration_s"]) for r in merged if r["phase"] == "measured")
    assert measured == [(1, 21.0), (2, 20.8), (3, 21.2)]
    swapped = next(r for r in merged if r["duration_s"] == 20.8)
    assert swapped["experiment"] == "exp2" and swapped["rerun_of"] == "exp2-scaling-b512-n41-measured2"
    assert swapped["source_experiment"] == "rerun5-exp2"
    assert [r["duration_s"] for r in replaced] == [16.5]
    assert len(merged) == 4


def test_rerun_passes_that_were_not_accepted_never_enter_the_merge():
    passes = [_row(f"rerun{p}-exp2", "measured", 1, d, arn=COSMICAI_ARN) for p, d in ((1, 31.0), (5, 20.8))]
    merged, _ = merge_reruns(_originals(), passes, [_accepted(passes[1]["name"])], {})
    assert 31.0 not in [r["duration_s"] for r in merged]


def test_a_whole_group_rerun_replaces_every_measured_run_of_its_configuration_but_keeps_the_cold_start():
    originals = [_row("exp1", "warmup", 0, 65.0, series="batch", batch=64, workers=11)]
    originals += [_row("exp1", "measured", k, d, series="batch", batch=64, workers=11)
                  for k, d in ((1, 76.0), (2, 58.5), (3, 114.7), (4, 76.5))]
    originals += [_row("exp1", "measured", 1, 40.0, series="batch", batch=128, workers=11)]
    regroup = [_row("regroup-exp1", "warmup", 0, 62.1, series="batch", batch=64, workers=11, arn=COSMICAI_ARN)]
    regroup += [_row("regroup-exp1", "measured", k, d, series="batch", batch=64, workers=11, arn=COSMICAI_ARN)
                for k, d in ((1, 63.3), (2, 62.2), (3, 64.1), (4, 58.8))]
    merged, replaced = merge_reruns(originals, regroup, [], {"regroup-exp1": "exp1"})
    b64 = sorted((r["phase"], r["slot"][6], r["duration_s"]) for r in merged if r["slot"][3] == 64)
    assert b64 == [("measured", 1, 63.3), ("measured", 2, 62.2), ("measured", 3, 64.1), ("measured", 4, 58.8),
                   ("warmup", 0, 65.0)]
    assert all(r["experiment"] == "exp1" for r in merged)
    assert sorted(r["duration_s"] for r in replaced) == [58.5, 76.0, 76.5, 114.7]
    assert any(r["slot"][3] == 128 for r in merged)


def test_an_accepted_rerun_missing_from_the_rerun_manifest_fails():
    with pytest.raises(RerunMergeError, match="rerun-missing"):
        merge_reruns(_originals(), [], [_accepted("rerun-missing")], {})


def test_an_accepted_rerun_whose_original_slot_does_not_exist_fails():
    rerun = _row("rerun1-exp2", "measured", 1, 20.8, arn=COSMICAI_ARN)
    with pytest.raises(RerunMergeError, match="no original"):
        merge_reruns(_originals(), [rerun], [_accepted(rerun["name"], run=7)], {})


def test_two_accepted_reruns_for_one_slot_fail():
    reruns = [_row(f"rerun{p}-exp2", "measured", 1, 20.0 + p, arn=COSMICAI_ARN) for p in (1, 2)]
    with pytest.raises(RerunMergeError, match="two accepted"):
        merge_reruns(_originals(), reruns, [_accepted(r["name"]) for r in reruns], {})


def test_the_account_of_a_row_comes_from_its_execution_arn():
    assert account_of(_row("exp2", "measured", 1, 1.0, arn=COSMICAI_ARN)) == "881908115028"