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
this workload — kept apart so the five canonical operators stay
domain-agnostic. Per-template key choice, measured agreement numbers, the
photometry redesign history, and the known storage-persistence gap are all
in docs/Reuse_Gate_Validity_Findings_2026-09-28.md; this module docstring
does not restate them.
"""

import math
import os
import re

TOLERANCE_VAR = "REUSE_KEY_TOLERANCE"
DEFAULT_TOLERANCE = 0.01
MAX_TOLERANCE = 10.0

PHOTOMETRY_BANDS = ("u", "g", "r", "i", "z")
PHOTOMETRY_FLAG_THRESHOLD = 2.0
PHOTOMETRY_KEY_OFFSET = 1000.0
PHOTOMETRY_KEY_MAX = PHOTOMETRY_KEY_OFFSET + 31

_FLAGGED_BANDS_LINE_RE = re.compile(
    r"^FLAGGED_BANDS:\s*(NONE|[ugriz](?:\s*,\s*[ugriz])*)\s*$",
    re.IGNORECASE,
)

_NOT_A_MAGNITUDE_VECTOR = (str, bytes, dict, set, frozenset)


def photometry_ground_truth_flags(magnitudes):
    """Which of the 5 bands are anomalous, or None if magnitudes is not
    exactly 5 finite numeric values."""
    if magnitudes is None or isinstance(magnitudes, _NOT_A_MAGNITUDE_VECTOR):
        return None
    try:
        length = len(magnitudes)
    except TypeError:
        return None
    if length != len(PHOTOMETRY_BANDS):
        return None
    rounded = []
    for value in magnitudes:
        try:
            as_float = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(as_float):
            return None
        rounded.append(round(as_float, 2))
    return frozenset(
        band for band, value in zip(PHOTOMETRY_BANDS, rounded)
        if abs(value) > PHOTOMETRY_FLAG_THRESHOLD
    )


def _encode_photometry_key(flags):
    bitmask = sum(1 << i for i, band in enumerate(PHOTOMETRY_BANDS) if band in flags)
    return PHOTOMETRY_KEY_OFFSET + bitmask


def photometry_reuse_key_for(magnitudes):
    """Up-front reuse_key for a photometry_classification row from its own
    magnitudes alone, or None if malformed."""
    ground_truth = photometry_ground_truth_flags(magnitudes)
    if ground_truth is None:
        return None
    return _encode_photometry_key(ground_truth)


def synthesize_photometry_response(magnitudes):
    """A photometry_classification response built from the query's own
    magnitudes, or None if malformed. Used on a reuse hit instead of the
    matched candidate's stored text."""
    ground_truth = photometry_ground_truth_flags(magnitudes)
    if ground_truth is None:
        return None
    lines = [
        f"{band}={float(value):.2f}: {'flagged' if band in ground_truth else 'not flagged'}"
        for band, value in zip(PHOTOMETRY_BANDS, magnitudes)
    ]
    flagged_str = "NONE" if not ground_truth else ",".join(sorted(ground_truth))
    lines.append(f"FLAGGED_BANDS: {flagged_str}")
    return "\n".join(lines)


def _last_nonblank_line(text):
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return None


def _parse_flagged_bands(response_text):
    """Bands the response's last non-blank line states are flagged, or
    None if that line is not an exact FLAGGED_BANDS marker."""
    if not response_text:
        return None
    last_line = _last_nonblank_line(response_text)
    if last_line is None:
        return None
    match = _FLAGGED_BANDS_LINE_RE.match(last_line.strip())
    if match is None:
        return None
    raw = match.group(1).strip()
    if raw.upper() == "NONE":
        return frozenset()
    return frozenset(band.strip().lower() for band in raw.split(","))


def verified_photometry_reuse_key(response_text, magnitudes):
    """Confirms a fresh response against this row's own ground truth,
    returning its reuse_key if it matches, else None."""
    ground_truth = photometry_ground_truth_flags(magnitudes)
    if ground_truth is None:
        return None
    stated = _parse_flagged_bands(response_text)
    if stated is None or stated != ground_truth:
        return None
    return _encode_photometry_key(ground_truth)


def resolve_tolerance(tolerance=None):
    """Tolerance from the argument, else the environment, else the
    default. Raises on an unparseable or out-of-range value."""
    if tolerance is not None:
        return _validated("tolerance", tolerance, float(tolerance))

    raw = os.environ.get(TOLERANCE_VAR)
    if raw is None:
        return DEFAULT_TOLERANCE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{TOLERANCE_VAR}={raw!r} is not a number; it is a redshift "
            f"tolerance such as 0.01"
        ) from None
    return _validated(TOLERANCE_VAR, raw, value)


def _validated(name, raw, value):
    if value != value:
        raise ValueError(f"{name}={raw!r} is NaN; it would reject every reuse")
    if value == float("inf"):
        raise ValueError(
            f"{name}={raw!r} is infinite; it would accept every candidate and "
            f"leave the run ungated while still reporting as gated"
        )
    if value < 0:
        raise ValueError(f"{name}={raw!r} is negative; a tolerance is a distance")
    if value > MAX_TOLERANCE:
        raise ValueError(f"{name}={raw!r} exceeds MAX_TOLERANCE={MAX_TOLERANCE}")
    return value


def is_photometry_key(key):
    """True if key is in [PHOTOMETRY_KEY_OFFSET, PHOTOMETRY_KEY_MAX]."""
    return key is not None and PHOTOMETRY_KEY_OFFSET <= key <= PHOTOMETRY_KEY_MAX


def redshift_validator(tolerance=None):
    """Predicate accepting a reuse whose reuse_key is near enough.

    A photometry_classification key (is_photometry_key) is categorical, so
    it only matches another key with the exact same value, and never
    matches a non-photometry key. redshift_analysis and outlier_analysis
    keys are continuous and compare within tolerance — see
    docs/Reuse_Gate_Validity_Findings_2026-09-28.md for why, and for the
    known gap that those two templates share no discriminator from each
    other.
    """
    limit = resolve_tolerance(tolerance)

    def validate(query_key, candidate_key):
        if query_key is None or candidate_key is None:
            return False
        query_photo = is_photometry_key(query_key)
        candidate_photo = is_photometry_key(candidate_key)
        if query_photo != candidate_photo:
            return False
        if query_photo:
            return float(query_key) == float(candidate_key)
        return abs(float(query_key) - float(candidate_key)) <= limit

    return validate