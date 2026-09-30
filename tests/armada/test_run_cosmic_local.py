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

"""Per-template reuse_key selection (SP1 reuse-validity follow-up).

reuse_keys_for is what closes the loop between
docs/Reuse_Gate_Validity_Findings_2026-09-28.md's measurements and the
deployed gate: a scalar comparison in reuse_policy.py stays correct only if
this function feeds it the right quantity per template.

Run: pytest tests/armada/test_run_cosmic_local.py -v
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.reuse_policy import PHOTOMETRY_KEY_OFFSET, photometry_reuse_key_for
from armada.run_cosmic_local import (
    assert_photometry_band_order,
    assert_photometry_config_is_safe,
    live_services,
    photometry_features_for,
    reuse_keys_for,
    workload_hash,
)

_TYPICAL_MAGS = [0.5, -0.5, 1.0, -1.0, 0.0]


def _magnitudes(n, row=None):
    """n rows of typical (unflagged) magnitudes, or `row` at every position."""
    return np.array([row or _TYPICAL_MAGS for _ in range(n)])


def test_redshift_analysis_row_gets_the_signed_residual():
    """Even index, small residual -> redshift_analysis (task_generator.py's
    own branching). Signed, not absolute: an over-prediction and an
    under-prediction of the same magnitude must be distinguishable."""
    predictions = np.array([0.150])
    true_redshifts = np.array([0.140])  # over-predicted by 0.010
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, _magnitudes(1),
                          outlier_threshold=0.5, index_offset=0, n_local=1)

    assert keys == pytest.approx([0.010])


def test_redshift_analysis_key_preserves_sign():
    over = reuse_keys_for(np.array([0.150]), np.array([0.140]), np.array([0.010]),
                          _magnitudes(1), outlier_threshold=0.5, index_offset=0, n_local=1)
    under = reuse_keys_for(np.array([0.140]), np.array([0.150]), np.array([0.010]),
                           _magnitudes(1), outlier_threshold=0.5, index_offset=0, n_local=1)
    assert over[0] > 0
    assert under[0] < 0
    assert over[0] != under[0]


def test_outlier_analysis_row_gets_z_pred_unchanged():
    """Residual above threshold -> outlier_analysis, which keeps the
    z_pred-only key measured at 0.917-1.00 agreement."""
    predictions = np.array([0.150])
    true_redshifts = np.array([0.500])  # residual 0.35, well above threshold
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, _magnitudes(1),
                          outlier_threshold=0.1, index_offset=0, n_local=1)

    assert keys == pytest.approx([0.150])


def test_an_outlier_key_landing_in_the_photometry_range_is_dropped():
    """z_pred=1000 would otherwise equal the unflagged photometry key."""
    predictions = np.array([1000.0])
    true_redshifts = np.array([0.0])
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, _magnitudes(1),
                          outlier_threshold=0.1, index_offset=0, n_local=1)

    assert keys[0] is None


def test_photometry_classification_row_gets_an_upfront_key_from_its_own_magnitudes():
    """Odd index, small residual -> photometry_classification, key from magnitudes alone."""
    predictions = np.array([0.0, 0.150])
    true_redshifts = np.array([0.0, 0.140])
    magnitudes = _magnitudes(2)
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                          outlier_threshold=0.5, index_offset=0, n_local=2)

    assert keys[1] == photometry_reuse_key_for(magnitudes[1])
    assert keys[1] == pytest.approx(PHOTOMETRY_KEY_OFFSET)


def test_photometry_classification_row_gets_no_key_for_malformed_magnitudes():
    predictions = np.array([0.0, 0.150])
    true_redshifts = np.array([0.0, 0.140])
    magnitudes = [list(_TYPICAL_MAGS), [0.0, 0.0, 0.0]]
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                          outlier_threshold=0.5, index_offset=0, n_local=2)

    assert keys[1] is None


def test_a_mixed_shard_gets_one_key_per_row_matching_its_own_template():
    """index_offset=0: idx0 (even, small residual) -> redshift_analysis;
    idx1 (odd, small residual) -> photometry_classification; idx2 (residual
    above threshold) -> outlier_analysis, regardless of parity."""
    predictions = np.array([0.10, 0.20, 0.30])
    true_redshifts = np.array([0.09, 0.19, 0.90])
    magnitudes = _magnitudes(3)
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                          outlier_threshold=0.5, index_offset=0, n_local=3)

    assert keys[0] == pytest.approx(0.01)                      # redshift_analysis: signed residual
    assert keys[1] == pytest.approx(PHOTOMETRY_KEY_OFFSET)      # photometry_classification: upfront key
    assert keys[2] == pytest.approx(0.30)                       # outlier_analysis: z_pred


def test_index_offset_shifts_which_rows_are_photometry_vs_redshift():
    """A shard starting at a global odd offset flips local-even to
    global-odd — the same population-size independence template_for already
    guarantees, exercised through reuse_keys_for specifically."""
    predictions = np.array([0.10])
    true_redshifts = np.array([0.09])
    magnitudes = _magnitudes(1)
    residuals = np.abs(predictions - true_redshifts)

    at_offset_0 = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                                 outlier_threshold=0.5, index_offset=0, n_local=1)
    at_offset_1 = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                                 outlier_threshold=0.5, index_offset=1, n_local=1)

    assert at_offset_0[0] is not None            # redshift_analysis: signed residual
    assert at_offset_1[0] == pytest.approx(PHOTOMETRY_KEY_OFFSET)  # photometry_classification: upfront key


def test_workload_hash_accepts_none_reuse_keys_without_crashing():
    """photometry_classification rows carry a None key all the way through
    to the workload identity hash — this must not raise."""
    digest = workload_hash(0, ["prompt a", "prompt b"], [0.01, None])
    assert isinstance(digest, str)
    assert len(digest) == 16


# --- photometry_features_for -------------------------------------------------

def test_photometry_classification_row_gets_its_own_magnitudes():
    predictions = np.array([0.0, 0.20])
    true_redshifts = np.array([0.0, 0.19])
    magnitudes = np.array([[9.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]])
    residuals = np.abs(predictions - true_redshifts)

    features = photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                                       outlier_threshold=0.5, index_offset=0, n_local=2)

    assert features[1] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_non_photometry_rows_get_no_features():
    predictions = np.array([0.10, 0.30])
    true_redshifts = np.array([0.09, 0.90])  # idx1: residual 0.6, above threshold
    magnitudes = np.array([[1.0] * 5, [2.0] * 5])
    residuals = np.abs(predictions - true_redshifts)

    features = photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                                       outlier_threshold=0.5, index_offset=0, n_local=2)

    assert features[0] is None  # redshift_analysis (even index, small residual)
    assert features[1] is None  # outlier_analysis (residual above threshold)


def test_photometry_features_for_gets_no_features_for_malformed_magnitudes():
    predictions = np.array([0.0, 0.150])
    true_redshifts = np.array([0.0, 0.140])
    magnitudes = [list(_TYPICAL_MAGS), [0.0, 0.0, 0.0]]
    residuals = np.abs(predictions - true_redshifts)

    features = photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                                       outlier_threshold=0.5, index_offset=0, n_local=2)

    assert features[1] is None


def test_a_valid_photometry_row_gets_both_a_key_and_features():
    predictions = np.array([0.150])
    true_redshifts = np.array([0.140])
    magnitudes = _magnitudes(1)
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                          outlier_threshold=0.5, index_offset=1, n_local=1)
    features = photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                                       outlier_threshold=0.5, index_offset=1, n_local=1)

    assert keys[0] is not None
    assert features[0] is not None


def test_photometry_features_and_reuse_keys_agree_on_which_row_is_valid():
    predictions = np.array([0.10, 0.20, 0.30])
    true_redshifts = np.array([0.09, 0.19, 0.90])
    magnitudes = np.array([[1.0] * 5, [2.0] * 5, [3.0] * 5])
    residuals = np.abs(predictions - true_redshifts)

    keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                          outlier_threshold=0.5, index_offset=0, n_local=3)
    features = photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                                       outlier_threshold=0.5, index_offset=0, n_local=3)

    # Positive check: an "always None" regression would pass the implication below vacuously.
    assert any(feature is not None for feature in features)
    assert any(key is not None for key in keys)

    for key, feature in zip(keys, features):
        if feature is not None:
            assert key is not None


# --- assert_photometry_band_order --------------------------------------------

def test_the_default_band_order_matches_reuse_policy(monkeypatch):
    monkeypatch.delenv("COSMIC_AI_CONFIG", raising=False)
    assert_photometry_band_order()


def test_a_config_file_with_a_different_band_order_is_refused(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"bands": ["g", "u", "r", "i", "z"]}')

    with pytest.raises(ValueError, match="PHOTOMETRY_BANDS"):
        assert_photometry_band_order(config_path=str(config_path))


# --- assert_photometry_config_is_safe ----------------------------------------

def test_the_default_config_passes_both_guards(monkeypatch):
    monkeypatch.delenv("COSMIC_AI_CONFIG", raising=False)
    assert_photometry_config_is_safe()


def test_a_config_file_overriding_the_photometry_template_is_refused(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"templates": {"photometry_classification": '
        '"Custom question with a different threshold: {band_str}"}}'
    )

    with pytest.raises(ValueError, match="photometry_classification template"):
        assert_photometry_config_is_safe(config_path=str(config_path))


def test_a_config_file_overriding_only_bands_still_fails_the_combined_guard(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"bands": ["g", "u", "r", "i", "z"]}')

    with pytest.raises(ValueError, match="PHOTOMETRY_BANDS"):
        assert_photometry_config_is_safe(config_path=str(config_path))


# --- live_services honors --dimensions ----------------------------------------

def test_live_services_threads_dimensions_into_bedrock_config(monkeypatch):
    """--dimensions must reach BedrockConfig, or the operator schema and the
    real EmbeddingService can disagree on vector length and crash Arrow
    construction (reproduced live: BEDROCK_EMBEDDING_DIMENSIONS defaults to
    1024 while --dimensions defaults to 256)."""
    from unittest.mock import MagicMock, patch

    resolved = {}

    def fake_resolve(payload=None, config_path=None):
        resolved.update(payload or {})
        return MagicMock(embedding_dimensions=payload["embedding_dimensions"])

    with patch("cost.bedrock_pricing.BedrockConfig.resolve", side_effect=fake_resolve), \
         patch("context.embedding.EmbeddingService"), \
         patch("context.manager.ContextManager"), \
         patch("context.router.ContextRouter"), \
         patch("chain.executor.ChainExecutor"):
        live_services(512)

    assert resolved["embedding_dimensions"] == 512