"""The campaign grid reproduces the original Cosmic AI experiment exactly."""

import pytest

from cosmic_campaign.grid import (
    CampaignSettings,
    Configuration,
    RunSlot,
    baseline_configurations,
    batch_sweep,
    format_gb,
    num_workers,
    reference_input,
    result_path,
    run_slots,
    scaling_grid,
)

PARTITIONS = (25, 50, 75, 100)
DATA_SIZES = (1, 2, 4, 6, 8, 10, 12.6)

ORIGINAL_NUM_WORLDS = {
    25: [41, 82, 164, 246, 328, 410, 517],
    50: [21, 41, 82, 123, 164, 205, 259],
    75: [14, 28, 55, 82, 110, 137, 173],
    100: [11, 21, 41, 62, 82, 103, 130],
}

SETTINGS = CampaignSettings(
    bucket="cosmicai-data-cylon",
    data_bucket="cosmicai-data-cylon",
    object_name="Anomaly Detection",
    object_type="folder",
    scripts={"A": "/tmp/Anomaly Detection/Inference/inference.py",
             "B": "/tmp/Anomaly Detection/Inference/inference_FMI.py"},
    result_prefix="cylon-armada-track1/exp1",
)


@pytest.mark.parametrize("partition_mb", PARTITIONS)
def test_worker_counts_match_every_row_of_the_original_campaign(partition_mb):
    assert [num_workers(d, partition_mb) for d in DATA_SIZES] == ORIGINAL_NUM_WORLDS[partition_mb]


def test_scaling_grid_is_the_28_original_configurations():
    grid = scaling_grid(PARTITIONS, DATA_SIZES, 512)
    assert len(grid) == 28
    assert {c.series for c in grid} == {"scaling"}
    assert max(c.workers for c in grid) == 517
    assert Configuration("scaling", 100, 1, 512, 11) in grid


def test_batch_sweep_is_one_gigabyte_at_100mb_over_five_batch_sizes():
    sweep = batch_sweep(100, 1, (32, 64, 128, 256, 512))
    assert [c.batch_size for c in sweep] == [32, 64, 128, 256, 512]
    assert {(c.series, c.partition_mb, c.data_gb, c.workers) for c in sweep} == {("batch", 100, 1, 11)}


def test_baselines_give_one_and_two_workers():
    baselines = baseline_configurations(100, (1, 2), 512)
    assert [c.workers for c in baselines] == [1, 2]
    assert {c.series for c in baselines} == {"baseline"}


def test_run_slots_put_the_warmup_first_then_measured_runs_per_configuration():
    configs = baseline_configurations(100, (1, 2), 512)
    slots = run_slots(configs, warmup_runs=1, measured_runs=4)
    assert len(slots) == 10
    assert [(s.phase, s.index) for s in slots[:5]] == [
        ("warmup", 0), ("measured", 1), ("measured", 2), ("measured", 3), ("measured", 4)]
    assert all(s.configuration == configs[0] for s in slots[:5])


@pytest.mark.parametrize("data_gb,expected", [(1, "1GB"), (1.0, "1GB"), (12.6, "12.6GB"), (0.2, "0.2GB")])
def test_format_gb_never_prints_float_noise(data_gb, expected):
    assert format_gb(data_gb) == expected


def test_result_paths_are_unique_and_separate_warmups():
    configs = (scaling_grid(PARTITIONS, DATA_SIZES, 512)
               + batch_sweep(100, 1, (32, 64, 128, 256, 512))
               + baseline_configurations(100, (1, 2), 512))
    slots = run_slots(configs, 1, 4)
    paths = [result_path("p", "A", s) for s in slots]
    assert len(paths) == len(set(paths))
    slot = RunSlot(Configuration("scaling", 25, 12.6, 512, 517), "warmup", 0)
    assert result_path("p", "A", slot) == "p/A/result-partition-25MB/12.6GB/warmup0"
    slot = RunSlot(Configuration("batch", 100, 1, 64, 11), "measured", 2)
    assert result_path("p", "B", slot) == "p/B/result-partition-100MB/1GB/Batches/batch64/run2"
    slot = RunSlot(Configuration("baseline", 100, 0.1, 512, 1), "measured", 1)
    assert result_path("p", "A", slot) == "p/A/result-partition-100MB/baseline-ws1/run1"


def test_reference_input_matches_the_original_step_functions_input():
    slot = RunSlot(Configuration("scaling", 100, 1, 512, 11), "measured", 1)
    assert reference_input(slot, "A", SETTINGS) == {
        "bucket": "cosmicai-data-cylon",
        "file_limit": "11",
        "world_size": 11,
        "batch_size": 512,
        "object_type": "folder",
        "S3_object_name": "Anomaly Detection",
        "script": "/tmp/Anomaly Detection/Inference/inference.py",
        "result_path": "cylon-armada-track1/exp1/A/result-partition-100MB/1GB/run1",
        "data_bucket": "cosmicai-data-cylon",
        "data_prefix": "100MB",
    }


def test_arm_b_differs_only_in_script_and_result_path():
    slot = RunSlot(Configuration("scaling", 50, 2, 512, 41), "measured", 3)
    a, b = reference_input(slot, "A", SETTINGS), reference_input(slot, "B", SETTINGS)
    assert {k for k in a if a[k] != b[k]} == {"script", "result_path"}


def test_unknown_arm_is_rejected():
    slot = RunSlot(Configuration("scaling", 100, 1, 512, 11), "measured", 1)
    with pytest.raises(ValueError, match="arm"):
        reference_input(slot, "C", SETTINGS)
