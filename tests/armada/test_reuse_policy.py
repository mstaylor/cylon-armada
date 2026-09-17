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

from armada.reuse_policy import DEFAULT_TOLERANCE, redshift_validator, resolve_tolerance


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
