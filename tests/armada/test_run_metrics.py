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

"""Per-arm run metrics for the Experiment E comparison.

The reuse rate is reported per arm because the arms may not achieve the same
one: Arm A's AllGather makes a new context visible to every rank at a barrier,
while Arm B's Redis visibility is eventual. If h diverges, the cost comparison
is comparing different cache hit rates rather than different transports, and
that has to be visible in the results rather than averaged away.

Run: pytest tests/armada/test_run_metrics.py -v
"""

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.run_metrics import RunMetrics, is_throttle_error


def test_reuse_rate_is_hits_over_retrievals():
    m = RunMetrics()
    for hit in (True, True, False, True):
        m.record_retrieval(hit)

    summary = m.summary()

    assert summary["retrievals"] == 4
    assert summary["cache_hits"] == 3
    assert summary["reuse_rate"] == pytest.approx(0.75)


def test_reuse_rate_is_zero_with_no_retrievals_not_a_division_error():
    """A rank with an empty shard performs no retrievals."""
    assert RunMetrics().summary()["reuse_rate"] == 0.0


def test_llm_latency_is_summarised():
    m = RunMetrics()
    m.record_llm_call(100.0)
    m.record_llm_call(200.0)

    summary = m.summary()

    assert summary["llm_calls"] == 2
    assert summary["llm_latency_ms_total"] == pytest.approx(300.0)
    assert summary["llm_latency_ms_mean"] == pytest.approx(150.0)


def test_records_written_counts_stores_this_rank_made():
    """The one record count that means the same thing on both arms — an
    AllGather'd ack table holds every rank's acks, a LangChain rank's only its
    own, so ack counts cannot be compared across arms but store counts can."""
    m = RunMetrics()
    m.record_store()
    m.record_store()
    m.record_store()

    assert m.summary()["records_written"] == 3


def test_throttle_events_are_counted_not_swallowed():
    m = RunMetrics()
    m.record_throttle()
    m.record_throttle()

    assert m.summary()["throttle_events"] == 2


def test_bedrock_throttling_is_recognised():
    class ClientError(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "ThrottlingException"}}

    assert is_throttle_error(ClientError()) is True


def test_other_errors_are_not_throttling():
    class ClientError(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "ValidationException"}}

    assert is_throttle_error(ClientError()) is False
    assert is_throttle_error(ValueError("boom")) is False


def test_a_fresh_summary_has_every_key_the_gate_reads():
    """The correctness gate reads records_written and throttle_events off every
    rank's record; a rank that did nothing must still emit them."""
    summary = RunMetrics().summary()
    for key in ("records_written", "throttle_events", "reuse_rate", "llm_calls"):
        assert key in summary


def test_records_failed_counts_stores_this_rank_could_not_make():
    """records_failed has to mean the same thing on both arms, exactly like
    records_written — an ack-derived count would be post-AllGather on Armada
    and own-only on LangChain."""
    m = RunMetrics()
    m.record_store()
    m.record_store_failure()

    summary = m.summary()

    assert summary["records_written"] == 1
    assert summary["records_failed"] == 1


def test_a_call_without_a_reported_latency_does_not_deflate_the_mean():
    """Recording a missing latency as zero would quietly pull the mean down —
    and the constant-inference check reads that mean."""
    m = RunMetrics()
    m.record_llm_call(200.0)
    m.record_llm_call(None)

    summary = m.summary()

    assert summary["llm_calls"] == 2
    assert summary["llm_calls_without_latency"] == 1
    assert summary["llm_latency_ms_mean"] == pytest.approx(200.0)


def test_ingested_contexts_are_counted_apart_from_originated_ones():
    """After the AllGather every rank stores every rank's contexts. Counting
    those together would make records_written world_size times larger on Arm A
    than Arm B for identical work."""
    m = RunMetrics()
    m.record_store()
    m.record_ingest()
    m.record_ingest()

    summary = m.summary()

    assert summary["records_written"] == 1
    assert summary["contexts_ingested"] == 2


def test_gate_rejections_are_counted_apart_from_misses():
    """Acceptance is not correctness. The rejection count is the denominator a
    write-up needs to say what the policy refused, rather than implying the
    accepted reuses were verified."""
    m = RunMetrics()
    m.record_retrieval(False)
    m.record_gate_rejection()

    summary = m.summary()

    assert summary["gate_rejections"] == 1
    assert summary["cache_hits"] == 0
