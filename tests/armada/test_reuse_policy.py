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

"""Cosmic AI's reuse validity policy.

The runtime owns the gate; this module owns what counts as a valid reuse for
this workload. Keeping the two apart is what lets the five canonical operators
stay domain-agnostic.

Run: pytest tests/armada/test_reuse_policy.py -v
"""

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.reuse_policy import (
    DEFAULT_TOLERANCE,
    MAX_TOLERANCE,
    PHOTOMETRY_KEY_OFFSET,
    is_photometry_key,
    photometry_ground_truth_flags,
    photometry_reuse_key_for,
    redshift_validator,
    resolve_tolerance,
    synthesize_photometry_response,
    verified_photometry_reuse_key,
)


def test_a_near_redshift_is_reusable():
    assert redshift_validator(tolerance=0.01)(0.150, 0.155) is True


def test_a_far_redshift_is_not():
    """The measured failure case: near-identical prompts, opposite ends of the
    catalogue."""
    assert redshift_validator(tolerance=0.01)(0.151, 0.942) is False


def test_the_boundary_is_inclusive():
    """The comparison is <=, not <.

    Values chosen to be exactly representable in binary so the assertion tests
    the policy rather than float rounding: 0.160 - 0.150 is 0.010000000000000009,
    which is not <= 0.01 and says nothing about the intended semantics. Exact
    boundary behaviour on arbitrary decimals is inherently imprecise and is not
    a property this policy promises.
    """
    assert redshift_validator(tolerance=0.25)(0.5, 0.75) is True


def test_the_tolerance_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "0.5")
    assert redshift_validator()(0.10, 0.50) is True


def test_an_explicit_tolerance_beats_the_environment(monkeypatch):
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "0.5")
    assert redshift_validator(tolerance=0.01)(0.10, 0.50) is False


def test_the_default_applies_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("REUSE_KEY_TOLERANCE", raising=False)
    assert resolve_tolerance() == DEFAULT_TOLERANCE


def test_a_non_numeric_tolerance_fails_fast(monkeypatch):
    """A typo silently falling back to the default would change the reuse rate,
    and therefore the experiment's headline number, with nothing to show it."""
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "loose")
    with pytest.raises(ValueError, match="REUSE_KEY_TOLERANCE"):
        redshift_validator()


def test_a_missing_key_on_either_side_is_refused():
    """Absence of evidence is not permission: a context stored before this
    policy existed was never checked against it."""
    validator = redshift_validator(tolerance=0.01)
    assert validator(None, 0.15) is False
    assert validator(0.15, None) is False


def test_a_zero_tolerance_admits_only_an_exact_match():
    validator = redshift_validator(tolerance=0.0)
    assert validator(0.15, 0.15) is True
    assert validator(0.15, 0.151) is False


def test_the_validator_is_symmetric():
    """Nothing downstream guarantees which side is query and which is candidate,
    so an asymmetric policy would make reuse depend on arrival order."""
    validator = redshift_validator(tolerance=0.01)
    assert validator(0.150, 0.158) == validator(0.158, 0.150)


def test_an_infinite_tolerance_is_refused(monkeypatch):
    """The worst available outcome: inf accepts every candidate, so the run is
    ungated while still recorded as gated. Nothing fails and the numbers look
    like a gated result."""
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "inf")
    with pytest.raises(ValueError, match="infinite"):
        redshift_validator()


def test_a_nan_tolerance_is_refused(monkeypatch):
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "nan")
    with pytest.raises(ValueError, match="NaN"):
        redshift_validator()


def test_a_negative_tolerance_is_refused(monkeypatch):
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "-0.01")
    with pytest.raises(ValueError, match="negative"):
        redshift_validator()


def test_a_tolerance_above_the_maximum_is_refused(monkeypatch):
    """Large enough to risk confusing photometry_classification keys (>=1000)
    with redshift_analysis/outlier_analysis keys — never a real measurement."""
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", "50")
    with pytest.raises(ValueError, match="MAX_TOLERANCE"):
        redshift_validator()


def test_the_maximum_tolerance_itself_is_accepted(monkeypatch):
    monkeypatch.setenv("REUSE_KEY_TOLERANCE", str(MAX_TOLERANCE))
    assert resolve_tolerance() == MAX_TOLERANCE


def test_an_explicit_infinite_tolerance_is_refused_the_same_as_the_environment():
    """An earlier version validated only environment-sourced values, so a
    caller passing tolerance=float("inf") directly bypassed the check
    entirely — exactly the failure the check exists to catch, reachable from
    a different entry point."""
    with pytest.raises(ValueError, match="infinite"):
        redshift_validator(tolerance=float("inf"))


def test_an_explicit_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="negative"):
        redshift_validator(tolerance=-0.01)


def test_an_explicit_tolerance_above_the_maximum_is_refused():
    with pytest.raises(ValueError, match="MAX_TOLERANCE"):
        redshift_validator(tolerance=50.0)


# --- photometry_ground_truth_flags ------------------------------------------

def test_ground_truth_is_empty_when_every_band_is_typical():
    assert photometry_ground_truth_flags([0.5, -0.5, 1.0, -1.0, 0.0]) == frozenset()


def test_ground_truth_flags_a_band_above_the_threshold():
    assert photometry_ground_truth_flags([0.0, 0.0, 2.50, 0.0, 0.0]) == frozenset({"r"})


def test_ground_truth_flags_a_band_below_negative_threshold():
    assert photometry_ground_truth_flags([-2.50, 0.0, 0.0, 0.0, 0.0]) == frozenset({"u"})


def test_ground_truth_can_flag_more_than_one_band():
    assert photometry_ground_truth_flags([2.5, 0.0, 0.0, -3.0, 0.0]) == frozenset({"u", "i"})


def test_the_boundary_is_exclusive_not_inclusive():
    """Exactly 2.0 is not '> 2' — matches the prompt's own wording."""
    assert photometry_ground_truth_flags([2.00, 0.0, 0.0, 0.0, 0.0]) == frozenset()


def test_ground_truth_rounds_to_the_same_precision_the_prompt_displays():
    """Raw 2.001 displays as 2.00 in the prompt (_format_bands' :.2f) and
    must read as unflagged — the LLM never saw the unrounded value."""
    assert photometry_ground_truth_flags([2.001, 0.0, 0.0, 0.0, 0.0]) == frozenset()
    assert photometry_ground_truth_flags([2.006, 0.0, 0.0, 0.0, 0.0]) == frozenset({"u"})


def test_ground_truth_is_none_for_the_wrong_number_of_bands():
    assert photometry_ground_truth_flags([0.0, 0.0, 0.0, 0.0]) is None
    assert photometry_ground_truth_flags([]) is None
    assert photometry_ground_truth_flags(None) is None


def test_ground_truth_is_none_for_non_finite_values():
    assert photometry_ground_truth_flags([float("nan"), 0.0, 0.0, 0.0, 0.0]) is None
    assert photometry_ground_truth_flags([float("inf"), 0.0, 0.0, 0.0, 0.0]) is None


def test_ground_truth_is_none_for_a_value_too_large_to_convert():
    """int.__float__ raises OverflowError, not ValueError, for an integer
    too large to represent as a float — a distinct exception the original
    guard did not catch."""
    assert photometry_ground_truth_flags([10 ** 400, 0.0, 0.0, 0.0, 0.0]) is None


def test_ground_truth_is_none_for_a_scalar_instead_of_a_vector():
    """A bare scalar has no len() — must not raise."""
    assert photometry_ground_truth_flags(1.0) is None


def test_ground_truth_is_none_for_a_zero_d_numpy_array():
    """len() of a 0-d array raises TypeError — must not propagate."""
    import numpy as np
    assert photometry_ground_truth_flags(np.array(1.0)) is None


def test_ground_truth_is_none_for_a_string_even_when_it_has_five_characters():
    """"12345" has len() 5 and iterates as 5 valid floats — must still be refused."""
    assert photometry_ground_truth_flags("12345") is None


def test_ground_truth_is_none_for_a_dict():
    """Iterating a dict yields its keys, not per-band values — a dict with
    5 numeric keys would otherwise be silently misread as a magnitude
    vector."""
    assert photometry_ground_truth_flags({0: 1, 1: 2, 2: 3, 3: 4, 4: 5}) is None


# --- photometry_reuse_key_for (upfront, ground-truth-only key) --------------

def test_photometry_reuse_key_for_needs_no_response():
    assert photometry_reuse_key_for(_TYPICAL_MAGS) == PHOTOMETRY_KEY_OFFSET
    assert photometry_reuse_key_for(_ONE_FLAGGED_MAGS) is not None


def test_photometry_reuse_key_for_agrees_with_the_verified_key_for_a_correct_response():
    upfront = photometry_reuse_key_for(_ONE_FLAGGED_MAGS)
    verified = verified_photometry_reuse_key("FLAGGED_BANDS: r", _ONE_FLAGGED_MAGS)
    assert upfront == verified


def test_photometry_reuse_key_for_is_none_for_malformed_magnitudes():
    assert photometry_reuse_key_for([0.0, 0.0, 0.0]) is None
    assert photometry_reuse_key_for(None) is None


# --- synthesize_photometry_response ------------------------------------------

def test_synthesized_response_states_the_querys_own_values():
    response = synthesize_photometry_response(_ONE_FLAGGED_MAGS)
    assert "r=2.50" in response
    assert "u=0.00" in response


def test_synthesized_response_ends_with_a_valid_flagged_bands_marker():
    response = synthesize_photometry_response(_ONE_FLAGGED_MAGS)
    assert verified_photometry_reuse_key(response, _ONE_FLAGGED_MAGS) is not None


def test_synthesized_response_for_a_typical_galaxy_states_none():
    response = synthesize_photometry_response(_TYPICAL_MAGS)
    assert response.splitlines()[-1] == "FLAGGED_BANDS: NONE"


def test_synthesized_response_is_none_for_malformed_magnitudes():
    assert synthesize_photometry_response([0.0, 0.0, 0.0]) is None


# --- _parse_flagged_bands / verified_photometry_reuse_key: last line only ---

def test_a_self_correction_earlier_in_the_response_is_ignored():
    """Only the final line is the real answer."""
    response = "FLAGGED_BANDS: r\nOn reflection, none are flagged.\nFLAGGED_BANDS: NONE"
    key = verified_photometry_reuse_key(response, _TYPICAL_MAGS)
    assert key == PHOTOMETRY_KEY_OFFSET


def test_a_marker_not_on_the_final_line_is_refused():
    response = "FLAGGED_BANDS: NONE\nThanks for asking!"
    assert verified_photometry_reuse_key(response, _TYPICAL_MAGS) is None


def test_trailing_blank_lines_after_the_marker_are_tolerated():
    response = "FLAGGED_BANDS: NONE\n\n\n"
    assert verified_photometry_reuse_key(response, _TYPICAL_MAGS) is not None


# --- verified_photometry_reuse_key -------------------------------------------

_TYPICAL_MAGS = [0.5, -0.5, 1.0, -1.0, 0.0]
_ONE_FLAGGED_MAGS = [0.0, 0.0, 2.50, 0.0, 0.0]  # flags 'r'


def test_a_correct_typical_response_is_verified():
    response = "u=0.50: not flagged\nFLAGGED_BANDS: NONE"
    key = verified_photometry_reuse_key(response, _TYPICAL_MAGS)
    assert key == PHOTOMETRY_KEY_OFFSET


def test_a_correct_anomalous_response_is_verified():
    response = "r=2.50: flagged\nFLAGGED_BANDS: r"
    key = verified_photometry_reuse_key(response, _ONE_FLAGGED_MAGS)
    assert key is not None
    assert PHOTOMETRY_KEY_OFFSET < key <= PHOTOMETRY_KEY_OFFSET + 31


def test_two_different_flagged_sets_encode_to_different_keys():
    key_r = verified_photometry_reuse_key("FLAGGED_BANDS: r", _ONE_FLAGGED_MAGS)
    key_none = verified_photometry_reuse_key("FLAGGED_BANDS: NONE", _TYPICAL_MAGS)
    assert key_r != key_none


def test_a_wrong_response_is_refused_even_though_it_parses_cleanly():
    """The model claims 'typical' but the galaxy's own magnitudes say
    otherwise — this is the exact failure mode this mechanism exists to
    catch before a wrong answer ever enters the store."""
    response = "FLAGGED_BANDS: NONE"
    assert verified_photometry_reuse_key(response, _ONE_FLAGGED_MAGS) is None


def test_a_response_missing_the_marker_is_refused():
    response = "The photometry looks typical, nothing stands out."
    assert verified_photometry_reuse_key(response, _TYPICAL_MAGS) is None


def test_the_marker_is_case_insensitive_and_whitespace_tolerant():
    response = "flagged_bands:   R  "
    assert verified_photometry_reuse_key(response, _ONE_FLAGGED_MAGS) is not None


def test_a_marker_with_trailing_text_on_the_same_line_is_refused():
    """Fails closed on anything not exactly the required format, rather
    than guessing at intent from nearby prose."""
    response = "FLAGGED_BANDS: r (this seems significant)"
    assert verified_photometry_reuse_key(response, _ONE_FLAGGED_MAGS) is None


def test_is_photometry_key_identifies_the_reserved_range():
    assert is_photometry_key(PHOTOMETRY_KEY_OFFSET) is True
    assert is_photometry_key(PHOTOMETRY_KEY_OFFSET + 31) is True
    assert is_photometry_key(0.15) is False
    assert is_photometry_key(-0.15) is False
    assert is_photometry_key(None) is False


@pytest.mark.parametrize("tolerance", [0.0, 0.01, 1.0, MAX_TOLERANCE])
def test_a_photometry_key_never_collides_with_a_redshift_key_at_any_valid_tolerance(tolerance):
    photometry_key = verified_photometry_reuse_key("FLAGGED_BANDS: NONE", _TYPICAL_MAGS)
    validator = redshift_validator(tolerance=tolerance)
    assert validator(photometry_key, 0.15) is False
    assert validator(photometry_key, -0.15) is False
    assert validator(photometry_key, PHOTOMETRY_KEY_OFFSET - 0.5) is False


@pytest.mark.parametrize("tolerance", [0.0, 0.01, 1.0, MAX_TOLERANCE])
def test_adjacent_photometry_keys_never_collide_at_any_valid_tolerance(tolerance):
    """1000 vs 1001: the tightest possible adjacent bitmask pair."""
    validator = redshift_validator(tolerance=tolerance)
    assert validator(PHOTOMETRY_KEY_OFFSET, PHOTOMETRY_KEY_OFFSET + 1) is False


def test_infinity_does_not_validate_against_itself():
    validator = redshift_validator(tolerance=0.0)
    assert validator(float("inf"), float("inf")) is False


def test_an_out_of_range_continuous_value_is_not_treated_as_a_photometry_key():
    validator = redshift_validator(tolerance=0.0)
    assert is_photometry_key(2000.0) is False
    assert validator(2000.0, 2000.0) is True


@pytest.mark.parametrize("tolerance", [0.0, 0.01, 1.0, MAX_TOLERANCE])
def test_two_different_photometry_keys_never_collide_at_any_valid_tolerance(tolerance):
    """Only an exact match may ever validate two photometry keys."""
    key_none = verified_photometry_reuse_key("FLAGGED_BANDS: NONE", _TYPICAL_MAGS)
    key_r = verified_photometry_reuse_key("FLAGGED_BANDS: r", _ONE_FLAGGED_MAGS)
    validator = redshift_validator(tolerance=tolerance)
    assert validator(key_none, key_r) is False


@pytest.mark.parametrize("tolerance", [0.0, 0.01, 1.0, MAX_TOLERANCE])
def test_an_identical_photometry_key_always_validates_at_any_valid_tolerance(tolerance):
    """The exact-match rule must not become so strict it rejects a genuine
    match — the SAME flagged-band set must still validate against itself
    regardless of tolerance."""
    key = verified_photometry_reuse_key("FLAGGED_BANDS: r", _ONE_FLAGGED_MAGS)
    validator = redshift_validator(tolerance=tolerance)
    assert validator(key, key) is True
