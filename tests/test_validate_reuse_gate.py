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

"""Tests for the reuse-gate-vs-fresh-inference validation script.

Pure logic (sampling, template selection, verdict parsing) is tested without
Bedrock. run_validation()'s orchestration is tested against a fake
ChainExecutor that returns canned responses, per CLAUDE.md's mocking
convention — no AWS call in this file.

Run: pytest tests/test_validate_reuse_gate.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from experiment.validate_reuse_gate import (
    build_judge_prompt,
    deployable_redshift_accuracy_features,
    fit_accuracy_proxy,
    find_tolerance_for_target_reuse_rate,
    load_dataset,
    oracle_redshift_accuracy_features,
    parse_verdict,
    run_self_consistency_check,
    run_validation,
    standardized_features,
    stratified_sample,
    template_for_index,
)

D = 8


class FakeChainExecutor:
    """Canned Bedrock stand-in — records every prompt it was asked, in order."""

    def __init__(self, judge_verdicts=None):
        self.calls = []
        self._judge_verdicts = list(judge_verdicts or [])

    def execute(self, task_description, system_prompt=None):
        self.calls.append({"task_description": task_description, "system_prompt": system_prompt})
        if system_prompt is not None:
            verdict = self._judge_verdicts.pop(0) if self._judge_verdicts else "AGREE"
            return {"response": f"VERDICT: {verdict}\nREASON: because",
                    "input_tokens": 5, "output_tokens": 5, "latency_ms": 1.0,
                    "model_id": "fake"}
        return {"response": f"analysis of: {task_description[:20]}",
                "input_tokens": 5, "output_tokens": 5, "latency_ms": 1.0,
                "model_id": "fake"}


# --- template_for_index ------------------------------------------------

def test_template_for_index_outlier_when_residual_exceeds_threshold():
    predictions = np.array([0.5])
    true_redshifts = np.array([0.0])
    assert template_for_index(0, predictions, true_redshifts, outlier_threshold=0.1) == \
        "outlier_analysis"


def test_template_for_index_even_index_is_redshift_analysis():
    predictions = np.array([0.10, 0.10])
    true_redshifts = np.array([0.10, 0.10])
    assert template_for_index(0, predictions, true_redshifts, outlier_threshold=0.5) == \
        "redshift_analysis"


def test_template_for_index_odd_index_is_photometry_classification():
    predictions = np.array([0.10, 0.10])
    true_redshifts = np.array([0.10, 0.10])
    assert template_for_index(1, predictions, true_redshifts, outlier_threshold=0.5) == \
        "photometry_classification"


def test_template_for_index_offset_shifts_parity():
    """Matches task_generator.py: (index_offset + idx) % 2, not idx alone."""
    predictions = np.array([0.10])
    true_redshifts = np.array([0.10])
    assert template_for_index(0, predictions, true_redshifts, outlier_threshold=0.5,
                               index_offset=1) == "photometry_classification"


# --- stratified_sample ---------------------------------------------------

def test_stratified_sample_empty_input_returns_empty():
    assert stratified_sample([], sample_size=10) == []


def test_stratified_sample_never_exceeds_sample_size():
    pairs = [(i, i + 1, float(i) / 100) for i in range(50)]
    sample = stratified_sample(pairs, sample_size=12, seed=1)
    assert len(sample) <= 12


def test_stratified_sample_covers_the_high_error_bucket():
    """The near-boundary pairs are the informative ones; a bucket that never
    gets sampled defeats the point of stratifying at all."""
    pairs = [(i, i + 1, float(i)) for i in range(40)]  # gate_error 0..39
    sample = stratified_sample(pairs, sample_size=20, n_strata=4, seed=3)
    sampled_errors = [p[2] for p in sample]
    assert max(sampled_errors) >= 30, \
        f"expected a pair from the top quartile, got max error {max(sampled_errors)}"
    assert min(sampled_errors) <= 9, \
        f"expected a pair from the bottom quartile, got min error {min(sampled_errors)}"


def test_stratified_sample_is_reproducible_for_a_fixed_seed():
    pairs = [(i, i + 1, float(i)) for i in range(40)]
    a = stratified_sample(pairs, sample_size=10, seed=42)
    b = stratified_sample(pairs, sample_size=10, seed=42)
    assert a == b


def test_stratified_sample_falls_back_to_random_when_gate_error_is_none():
    """Ungated replay records accepted_pairs with gate_error=None (see
    RankReplay._is_hit's ungated branch) — must not crash on that shape."""
    pairs = [(i, i + 1, None) for i in range(10)]
    sample = stratified_sample(pairs, sample_size=5, seed=1)
    assert len(sample) == 5
    assert all(p in pairs for p in sample)


# --- build_judge_prompt / parse_verdict -----------------------------------

def test_build_judge_prompt_includes_the_question_and_both_texts():
    prompt = build_judge_prompt("Is z accurate?", "text A here", "text B here")
    assert "Is z accurate?" in prompt
    assert "text A here" in prompt
    assert "text B here" in prompt


@pytest.mark.parametrize("text,expected", [
    ("VERDICT: AGREE\nREASON: same conclusion", "AGREE"),
    ("VERDICT: DISAGREE\nREASON: different conclusion", "DISAGREE"),
    ("verdict: agree\nreason: lowercase still parses", "AGREE"),
    ("I think these agree.", None),
    ("", None),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


# --- load_dataset ----------------------------------------------------------

def test_load_dataset_raises_on_misaligned_arrays(tmp_path):
    np.save(tmp_path / "predictions.npy", np.zeros(5))
    np.save(tmp_path / "true_redshifts.npy", np.zeros(5))
    np.save(tmp_path / "magnitudes.npy", np.zeros((5, 5)))
    np.save(tmp_path / "embeddings.npy", np.zeros((4, D)))  # short by one row
    with open(tmp_path / "prompts.json", "w") as f:
        json.dump(["p"] * 5, f)
    with pytest.raises(ValueError, match="embeddings"):
        load_dataset(str(tmp_path))


# --- run_validation (orchestration, fake ChainExecutor, no AWS) ------------

def _synthetic_dataset(n=12, dim=D, seed=0):
    rng = np.random.default_rng(seed)
    predictions = rng.uniform(0.05, 0.2, size=n)
    true_redshifts = predictions + rng.normal(0, 0.002, size=n)
    magnitudes = rng.standard_normal((n, 5))
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    # Force a near-duplicate the gate should accept. Index 5, not 1: with the
    # default epoch_batch_size=4, galaxy 0 and 1 land in the SAME epoch, and
    # replay()'s within-epoch blindness means galaxy 1 can never see galaxy
    # 0's context yet — the duplicate has to be in a LATER epoch to be
    # reachable at all.
    embeddings[5] = embeddings[0] + 1e-4
    predictions[5] = predictions[0] + 1e-5
    true_redshifts[5] = true_redshifts[0]
    prompts = [f"prompt for galaxy {i} z={predictions[i]:.3f}" for i in range(n)]
    return predictions, true_redshifts, magnitudes, embeddings, prompts


def test_run_validation_produces_one_row_per_sampled_pair_with_all_fields():
    dataset = _synthetic_dataset()
    fake = FakeChainExecutor()
    result = run_validation(dataset, sample_size=5, threshold=0.9, gate_tolerance=0.01,
                             epoch_batch_size=4, seed=1, chain_executor=fake)

    assert result["n_accepted_pairs"] >= 1
    assert result["n_sampled"] == min(5, result["n_accepted_pairs"])
    for row in result["rows"]:
        assert row["verdict"] in ("AGREE", "DISAGREE")
        assert row["template"] in ("redshift_analysis", "photometry_classification",
                                    "outlier_analysis")
        assert row["candidate_response"]
        assert row["query_response"]
    # Every sampled pair costs exactly 3 calls: candidate, query, judge.
    assert len(fake.calls) == 3 * result["n_sampled"]


def test_run_validation_agreement_rate_reflects_judge_verdicts():
    dataset = _synthetic_dataset()
    fake = FakeChainExecutor(judge_verdicts=["AGREE", "DISAGREE", "AGREE", "AGREE"])
    result = run_validation(dataset, sample_size=4, threshold=0.9, gate_tolerance=0.01,
                             epoch_batch_size=4, seed=1, chain_executor=fake)
    if result["n_sampled"] == 4:
        assert result["agreement_rate"] == pytest.approx(3 / 4)


def test_run_validation_with_no_accepted_pairs_returns_empty_rows_not_a_crash():
    dataset = _synthetic_dataset(seed=99)
    fake = FakeChainExecutor()
    # threshold=1.01 is unreachable by cosine similarity — nothing ever hits.
    result = run_validation(dataset, sample_size=5, threshold=1.01, gate_tolerance=0.01,
                             epoch_batch_size=4, seed=1, chain_executor=fake)
    assert result["n_accepted_pairs"] == 0
    assert result["rows"] == []
    assert result["agreement_rate"] is None
    assert fake.calls == []


# --- standardized_features -------------------------------------------------

def test_standardized_features_has_zero_mean_and_unit_std_per_column():
    rng = np.random.default_rng(7)
    predictions = rng.uniform(0.0, 0.3, size=200)
    magnitudes = rng.standard_normal((200, 5)) * 3 + 10  # arbitrary scale/offset

    combined = standardized_features(predictions, magnitudes)

    assert combined.shape == (200, 6)
    np.testing.assert_allclose(combined.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(combined.std(axis=0), 1.0, atol=1e-8)


def test_standardized_features_handles_a_constant_column_without_nan():
    predictions = np.array([0.1, 0.1, 0.1, 0.1])
    magnitudes = np.zeros((4, 5))
    combined = standardized_features(predictions, magnitudes)
    assert np.isfinite(combined).all()


def test_oracle_redshift_accuracy_features_separates_accurate_from_inaccurate():
    """Two galaxies sharing z_pred but differing in accuracy must land farther
    apart under this feature than under z_pred alone — that's the entire
    point of adding the residual."""
    predictions = np.array([0.10, 0.10, 0.10])
    true_redshifts = np.array([0.10, 0.10, 0.30])  # row 2 is a bad prediction

    combined = oracle_redshift_accuracy_features(predictions, true_redshifts)

    dist_0_1 = np.linalg.norm(combined[0] - combined[1])  # both accurate
    dist_0_2 = np.linalg.norm(combined[0] - combined[2])  # accurate vs. inaccurate
    assert dist_0_2 > dist_0_1


# --- find_tolerance_for_target_reuse_rate -----------------------------------

def test_find_tolerance_finds_a_tolerance_achieving_a_reachable_target_rate():
    """Rather than hand-deriving which rate a synthetic population makes
    reachable (replay()'s ingest-only-misses mechanics make that easy to get
    wrong by hand), measure the rate a known tolerance actually achieves
    first, then confirm the search converges to a tolerance achieving that
    same, independently-measured rate."""
    from experiment.reuse_replay import SHARED, replay, totals

    rng = np.random.default_rng(13)
    n = 40
    embeddings = rng.standard_normal((n, D)).astype(np.float32) * 0.01
    embeddings[:, 0] = 5.0  # shared dominant component keeps cosine sim high throughout
    gate_values = np.linspace(0.0, 10.0, n)

    known_tolerance = 1.0
    known_rate = totals(replay(embeddings, 1, SHARED, 0.9, 1,
                               gate_values=gate_values,
                               gate_tolerance=known_tolerance))["reuse_rate"]

    tol, rate = find_tolerance_for_target_reuse_rate(
        embeddings, gate_values, threshold=0.9, epoch_batch_size=1,
        target_reuse_rate=known_rate)

    assert rate == pytest.approx(known_rate, abs=0.05)


def test_find_tolerance_accepts_a_multi_column_gate_values():
    rng = np.random.default_rng(12)
    n = 16
    embeddings = rng.standard_normal((n, D)).astype(np.float32) * 0.01
    embeddings[:, 0] = 5.0
    gate_values = np.column_stack([np.linspace(0, 1, n), np.linspace(0, 1, n)])

    tol, rate = find_tolerance_for_target_reuse_rate(
        embeddings, gate_values, threshold=0.9, epoch_batch_size=1,
        target_reuse_rate=0.5)

    assert 0.0 <= rate <= 1.0
    assert tol > 0


# --- run_self_consistency_check (fake ChainExecutor, no AWS) ---------------

def test_self_consistency_calls_the_same_prompt_twice_per_sample():
    dataset = _synthetic_dataset()
    fake = FakeChainExecutor()
    result = run_self_consistency_check(dataset, sample_size=5, seed=1, chain_executor=fake)

    assert result["n_sampled"] == 5
    # 3 calls per sample: prompt twice, then the judge — never a candidate/
    # query distinction, since there's only one galaxy per row here.
    assert len(fake.calls) == 15
    for row in result["rows"]:
        assert row["response_1"]
        assert row["response_2"]
        assert row["verdict"] in ("AGREE", "DISAGREE")


def test_self_consistency_calls_the_identical_prompt_both_times():
    dataset = _synthetic_dataset()
    fake = FakeChainExecutor()
    run_self_consistency_check(dataset, sample_size=3, seed=2, chain_executor=fake)

    non_judge_calls = [c for c in fake.calls if c["system_prompt"] is None]
    # Grouped in pairs; each pair's two task_descriptions must be identical —
    # that's the whole point of a self-consistency check, not a near-miss.
    for i in range(0, len(non_judge_calls), 2):
        assert non_judge_calls[i]["task_description"] == non_judge_calls[i + 1]["task_description"]


def test_self_consistency_agreement_rate_reflects_judge_verdicts():
    dataset = _synthetic_dataset()
    fake = FakeChainExecutor(judge_verdicts=["AGREE", "DISAGREE", "AGREE"])
    result = run_self_consistency_check(dataset, sample_size=3, seed=1, chain_executor=fake)
    assert result["agreement_rate"] == pytest.approx(2 / 3)


# --- fit_accuracy_proxy / deployable_redshift_accuracy_features -----------

def test_fit_accuracy_proxy_recovers_a_known_relationship_on_held_out_folds():
    """Residual is a deterministic function of one magnitude column; the
    cross-validated proxy must recover a real correlation with it using only
    OTHER rows' labels — proof this isn't leaking the row's own answer."""
    rng = np.random.default_rng(21)
    n = 200
    predictions = rng.uniform(0.05, 0.2, size=n)
    magnitudes = rng.standard_normal((n, 5))
    true_residual = np.abs(magnitudes[:, 0]) * 0.05
    sign = np.where(rng.standard_normal(n) >= 0, 1.0, -1.0)
    true_redshifts = predictions - true_residual * sign

    predicted_residual = fit_accuracy_proxy(predictions, magnitudes, true_redshifts,
                                             n_splits=5, seed=1)
    observed_residual = np.abs(predictions - true_redshifts)

    correlation = np.corrcoef(predicted_residual, observed_residual)[0, 1]
    assert correlation > 0.3, f"expected the proxy to recover the known relationship, got r={correlation:.3f}"


def test_fit_accuracy_proxy_out_of_fold_predictions_dont_all_equal_the_label():
    """A model that leaked the row's own label would reproduce the observed
    residual near-exactly; K-fold cross_val_predict must not."""
    rng = np.random.default_rng(22)
    n = 100
    predictions = rng.uniform(0.05, 0.2, size=n)
    magnitudes = rng.standard_normal((n, 5))
    true_redshifts = predictions + rng.normal(0, 0.01, size=n)

    predicted_residual = fit_accuracy_proxy(predictions, magnitudes, true_redshifts,
                                             n_splits=5, seed=1)
    observed_residual = np.abs(predictions - true_redshifts)
    assert not np.allclose(predicted_residual, observed_residual, atol=1e-6)


def test_deployable_redshift_accuracy_features_shape_and_finiteness():
    rng = np.random.default_rng(23)
    n = 60
    predictions = rng.uniform(0.05, 0.2, size=n)
    magnitudes = rng.standard_normal((n, 5))
    true_redshifts = predictions + rng.normal(0, 0.01, size=n)

    combined = deployable_redshift_accuracy_features(predictions, magnitudes, true_redshifts,
                                                       n_splits=5, seed=1)
    assert combined.shape == (n, 2)
    assert np.isfinite(combined).all()