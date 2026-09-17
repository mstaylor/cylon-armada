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
this workload. Keeping them apart is what lets the five canonical operators stay
domain-agnostic — a redshift tolerance inside Retrieve would put astronomy in a
general-purpose agentic runtime.

Why a policy is needed at all: cosine similarity over these prompts says two
galaxies read alike, not that one's analysis answers the other's question. The
prompts are template-generated, so most of their characters are identical
boilerplate and the numbers that distinguish them carry little of the embedding.
Measured on real AstroMAE inference over 1253 SDSS galaxies, 10.1% of reuses
returned an analysis about a galaxy whose predicted redshift differed by more
than the model's own median residual.

**The tolerance is an exploratory parameter, not a validated equivalence bound.**
It bounds how far apart two predictions may be; it does not establish that the
cached analysis is correct for the query. A population-level prediction residual
measures disagreement between predictions and truth, not how far an input can
move before the right answer changes — two close predictions can still warrant
opposite anomaly judgments. Report reuse under this policy as *gate acceptance*,
never as validated reuse, and choose production tolerances against answer-level
error on held-out data rather than against model error.
"""

import os

TOLERANCE_VAR = "REUSE_KEY_TOLERANCE"
DEFAULT_TOLERANCE = 0.01


def resolve_tolerance(tolerance=None):
    """Tolerance from the argument, else the environment, else the default.

    Raises rather than guessing on an unparseable value: a typo silently
    resolving to the default would change the reuse rate, and therefore the
    experiment's headline number, with nothing to show it happened.
    """
    if tolerance is not None:
        return float(tolerance)

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
    """Reject values that parse but silently destroy the measurement.

    `inf` accepts every candidate, so the run is ungated while still recorded
    as gated — the worst outcome available, because nothing fails and the
    numbers look like a gated result. `nan` compares false against everything
    and rejects all reuse. A negative tolerance is not a distance.
    """
    if value != value:
        raise ValueError(f"{name}={raw!r} is NaN; it would reject every reuse")
    if value == float("inf"):
        raise ValueError(
            f"{name}={raw!r} is infinite; it would accept every candidate and "
            f"leave the run ungated while still reporting as gated"
        )
    if value < 0:
        raise ValueError(f"{name}={raw!r} is negative; a tolerance is a distance")
    return value


def redshift_validator(tolerance=None):
    """A predicate accepting a reuse whose predicted redshift is near enough.

    Returns ``(query_key, candidate_key) -> bool`` for injection as
    ``build_retrieve_operator(..., reuse_validator=...)``. Either key being None
    means unverifiable, which is refused: absence of evidence is not permission,
    and a context stored before this policy existed was never checked against it.
    """
    limit = resolve_tolerance(tolerance)

    def validate(query_key, candidate_key):
        if query_key is None or candidate_key is None:
            return False
        return abs(float(query_key) - float(candidate_key)) <= limit

    return validate