"""Classic Cosmic AI on Vista: the file-backed driver around the Lambda inference code.

Runs the real AstroMAE model on a few samples of the AI-for-Astronomy sample file, so it
needs torch, timm and a local AI-for-Astronomy checkout; it skips without them.
Run: pytest tests/tacc/test_cosmic_classic.py -v
"""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from torch.utils.data import TensorDataset  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "target", "tacc", "scripts"))

import cosmic_classic  # noqa: E402

_ASTRO = os.environ.get("AI_FOR_ASTRONOMY", os.path.expanduser("~/AI-for-Astronomy"))
_CODE_DIR = os.path.join(_ASTRO, "code", "Anomaly Detection")
_SAMPLE = os.path.join(_CODE_DIR, "Inference", "resized_inference.pt")
_MODEL = os.path.join(_CODE_DIR, "Fine_Tune_Model", "Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt")

needs_model = pytest.mark.skipif(not (os.path.exists(_SAMPLE) and os.path.exists(_MODEL)),
                                 reason="AI-for-Astronomy sample data or weights not present")

PARTITION_ROWS = 32


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    if not os.path.exists(_SAMPLE):
        pytest.skip("AI-for-Astronomy sample data not present")
    sample = torch.load(_SAMPLE, weights_only=False)
    images, magnitudes, redshifts = sample[:][0], sample[:][1], sample[:][2]
    directory = tmp_path_factory.mktemp("25MB")
    for index, name in enumerate(["1.pt", "10.pt", "2.pt"]):
        rows = slice(index * PARTITION_ROWS, (index + 1) * PARTITION_ROWS)
        torch.save(TensorDataset(images[rows], magnitudes[rows], redshifts[rows]),
                   os.path.join(directory, name))
    (directory / "notes.txt").write_text("not a partition")
    return str(directory)


def _init(tmp_path, data_dir, world_size, file_limit=3):
    run_dir = str(tmp_path / "run")
    cosmic_classic.main(["init", "--data-dir", data_dir, "--file-limit", str(file_limit),
                         "--world-size", str(world_size), "--batch-size", "512", "--run-dir", run_dir])
    return run_dir


def _worker(run_dir, data_dir, rank):
    return cosmic_classic.run_worker(os.path.join(run_dir, "payload.json"), data_dir, _MODEL, _CODE_DIR,
                                     rank, "cpu")


def test_payload_lists_files_in_s3_order_and_maps_them_like_the_lambda_initializer(tmp_path, data_dir):
    one_each = cosmic_classic.build_payload(data_dir, 3, 3, 512, "results")
    one_rank = cosmic_classic.build_payload(data_dir, 3, 1, 512, "results")

    assert one_each["data_map"] == {"0": "1.pt", "1": "10.pt", "2": "2.pt"}
    assert one_rank["data_map"] == {"0": ["1.pt", "10.pt", "2.pt"]}
    assert (one_each["world_size"], one_each["file_limit"], one_each["batch_size"]) == (3, 3, 512)


def test_file_limit_keeps_the_first_files_in_s3_order(tmp_path, data_dir):
    payload = cosmic_classic.build_payload(data_dir, 2, 2, 512, "results")

    assert payload["data_map"] == {"0": "1.pt", "1": "10.pt"}


def test_more_ranks_than_files_reuses_files_like_assign_partitions(tmp_path, data_dir):
    payload = cosmic_classic.build_payload(data_dir, 3, 5, 512, "results")

    assert payload["data_map"] == {"0": "1.pt", "1": "10.pt", "2": "2.pt", "3": "1.pt", "4": "10.pt"}


def test_init_writes_the_payload_into_the_run_directory(tmp_path, data_dir):
    run_dir = _init(tmp_path, data_dir, 2)

    with open(os.path.join(run_dir, "payload.json")) as f:
        payload = json.load(f)
    assert payload["data_map"] == {"0": ["1.pt", "10.pt"], "1": "2.pt"}
    assert payload["result_path"] == os.path.join(run_dir, "results")


def test_partitions_default_to_the_file_count(tmp_path, data_dir):
    payload = cosmic_classic.build_payload(data_dir, 3, 2, 512, "results")

    assert payload["partitions"] == 3


def test_partitions_past_the_file_count_cycle_the_files_then_split_across_ranks(tmp_path, data_dir):
    payload = cosmic_classic.build_payload(data_dir, 3, 2, 512, "results", partitions=7)

    assert payload["data_map"] == {"0": ["1.pt", "10.pt", "2.pt", "1.pt"], "1": ["10.pt", "2.pt", "1.pt"]}
    assert (payload["partitions"], payload["file_limit"]) == (7, 3)


def test_init_command_passes_the_partition_count(tmp_path, data_dir):
    run_dir = str(tmp_path / "run")
    cosmic_classic.main(["init", "--data-dir", data_dir, "--file-limit", "2", "--world-size", "1",
                         "--batch-size", "512", "--run-dir", run_dir, "--partitions", "5"])

    with open(os.path.join(run_dir, "payload.json")) as f:
        payload = json.load(f)
    assert payload["data_map"] == {"0": ["1.pt", "10.pt", "1.pt", "10.pt", "1.pt"]}


def test_a_partition_count_below_one_is_refused(tmp_path, data_dir):
    with pytest.raises(ValueError, match="partitions"):
        cosmic_classic.build_payload(data_dir, 3, 1, 512, "results", partitions=0)


def test_an_empty_data_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no .pt partitions"):
        cosmic_classic.build_payload(str(tmp_path), 3, 1, 512, "results")


@needs_model
def test_worker_writes_the_lambda_result_fields_for_its_partition(tmp_path, data_dir):
    run_dir = _init(tmp_path, data_dir, 3)

    info = _worker(run_dir, data_dir, 1)

    with open(os.path.join(run_dir, "results", "1.json")) as f:
        written = json.load(f)
    assert written == info
    assert info["num_samples"] == PARTITION_ROWS
    assert info["data_path"] == "10.pt"
    assert info["comm_init_s"] == 0.0 and info["barrier_s"] == 0.0
    for field in ("inference_s", "total_cpu_time (seconds)", "scatter_s", "model_load_s",
                  "payload_fetch_s", "publish_s", "total_s"):
        assert info[field] >= 0.0, field
    assert info["code_fetch_s"] is None and info["cold_start"] is None


def _fake_results(run_dir, ranks):
    results = os.path.join(run_dir, "results")
    os.makedirs(results, exist_ok=True)
    for rank in ranks:
        with open(os.path.join(results, f"{rank}.json"), "w") as f:
            json.dump({"rank": rank, "num_samples": 10 + rank}, f)
    return results


def test_summary_concatenates_every_rank_in_s3_key_order_and_counts_bytes(tmp_path):
    run_dir = str(tmp_path)
    results = _fake_results(run_dir, range(11))
    expected_bytes = sum(os.path.getsize(os.path.join(results, f"{r}.json")) for r in range(11))

    metrics = cosmic_classic.summarize(run_dir, 11)

    with open(os.path.join(results, "combined_data.json")) as f:
        combined = json.load(f)
    assert [row["rank"] for row in combined] == [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
    assert metrics["ranks_aggregated"] == 11
    assert metrics["bytes_aggregated"] == expected_bytes
    assert metrics["aggregate_s"] >= 0.0
    with open(os.path.join(results, "aggregate_metrics.json")) as f:
        assert json.load(f) == metrics


def test_summary_rerun_ignores_its_own_outputs(tmp_path):
    run_dir = str(tmp_path)
    _fake_results(run_dir, range(3))
    first = cosmic_classic.summarize(run_dir, 3)

    second = cosmic_classic.summarize(run_dir, 3)

    assert (second["ranks_aggregated"], second["bytes_aggregated"]) == (
        first["ranks_aggregated"], first["bytes_aggregated"])


def test_summary_of_a_run_missing_a_rank_raises_with_the_missing_ranks(tmp_path):
    run_dir = str(tmp_path)
    _fake_results(run_dir, [0, 2])

    with pytest.raises(cosmic_classic.IncompleteRun) as raised:
        cosmic_classic.summarize(run_dir, 4)

    assert raised.value.missing_ranks == [1, 3]
    assert not os.path.exists(os.path.join(run_dir, "results", "combined_data.json"))


def test_summarize_command_exits_non_zero_on_a_missing_rank(tmp_path):
    run_dir = str(tmp_path)
    _fake_results(run_dir, [0])

    with pytest.raises(SystemExit) as raised:
        cosmic_classic.main(["summarize", "--run-dir", run_dir, "--world-size", "2"])

    assert "missing ranks [1]" in str(raised.value.code)


@needs_model
def test_a_rank_with_several_files_reads_them_all(tmp_path, data_dir):
    run_dir = _init(tmp_path, data_dir, 1)

    info = _worker(run_dir, data_dir, 0)

    assert info["num_samples"] == 3 * PARTITION_ROWS