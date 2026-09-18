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

"""Per-rank metrics for one Experiment E run.

Reported per arm rather than aggregated across them: the two arms share
context by different mechanisms and may not reach the same reuse rate, which
would confound a cost comparison unless it is visible.

records_written counts the contexts this rank itself originated. It is the same
quantity on both arms, unlike the ack table an executor hands back — after an
AllGather that table holds every rank's acks, while LangChain's holds only
this rank's, so ack counts are not comparable across arms.

contexts_ingested counts the contexts this rank stored on another rank's
behalf. It is necessarily arm-dependent — Arm A's AllGather replicates every
rank's contexts to every rank, Arm B's ranks each write only their own into a
shared Redis — which is why it is a separate number rather than folded into
records_written. Folded together the same work would report world_size times
more records on Arm A than on Arm B.
"""

_THROTTLE_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "ServiceQuotaExceededException",
})


def is_throttle_error(exc):
    """True when this exception is Bedrock refusing for rate reasons.

    Throttling is the most likely way this experiment produces a wrong answer
    that looks right: a throttled rank is indistinguishable from a slow rank in
    timing data unless it is recorded separately.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    return response.get("Error", {}).get("Code") in _THROTTLE_CODES


class RunMetrics:
    def __init__(self):
        self.retrievals = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.llm_calls_without_latency = 0
        self.llm_latency_ms_total = 0.0
        self.records_written = 0
        self.records_failed = 0
        self.contexts_ingested = 0
        self.gate_rejections = 0
        self.throttle_events = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.cost_usd_total = 0.0
        self.unpriced_calls = 0

    def record_retrieval(self, hit):
        self.retrievals += 1
        if hit:
            self.cache_hits += 1

    def record_llm_call(self, latency_ms, input_tokens=0, output_tokens=0, cost_usd=0.0):
        """Count a call; fold its latency into the mean only if one was reported.

        A call that reports no latency is counted separately rather than as
        zero, because a zero would silently deflate llm_latency_ms_mean — the
        constant-inference check reads that number.

        Tokens and cost are accumulated only for calls that actually happened.
        A reused row never reaches here, so cost_usd_total is what the run
        spent rather than what it would have spent without reuse. The avoided
        amount is deliberately not accumulated: it is an estimate, and the
        estimator belongs in the analysis where it can be stated, not buried
        in instrumentation. cost_usd_mean plus cache_hits is enough to compute
        it downstream under whatever assumption is being defended.
        """
        self.llm_calls += 1
        if latency_ms is None:
            self.llm_calls_without_latency += 1
        else:
            self.llm_latency_ms_total += float(latency_ms)
        self.input_tokens_total += int(input_tokens or 0)
        self.output_tokens_total += int(output_tokens or 0)
        self.cost_usd_total += float(cost_usd or 0.0)

    def record_store(self):
        """Count a context this rank originated."""
        self.records_written += 1

    def record_ingest(self):
        """Count a context another rank originated that this rank stored.

        Kept apart from record_store so records_written keeps meaning the same
        thing on both arms once the collective replicates contexts.
        """
        self.contexts_ingested += 1

    def record_gate_rejection(self):
        """Count a row whose cosine-eligible candidates the validity policy all refused.

        Kept apart from an ordinary miss: a miss means nothing matched, a
        rejection means something matched and the policy said no. Conflating
        them would inflate the rejection rate with cold-start misses and make
        the reported denominator meaningless. Acceptance is not correctness, so
        this count is what lets results say what the policy refused rather than
        imply the accepted reuses were verified.
        """
        self.gate_rejections += 1

    def record_store_failure(self):
        self.records_failed += 1

    def record_unpriced_call(self):
        """Count a call whose model had no price entry.

        Its cost came from the most expensive registered model, so any run with
        a non-zero count here is reporting an upper bound rather than a cost.
        """
        self.unpriced_calls += 1

    def record_throttle(self):
        self.throttle_events += 1

    def summary(self):
        return {
            "retrievals": self.retrievals,
            "cache_hits": self.cache_hits,
            "reuse_rate": (self.cache_hits / self.retrievals) if self.retrievals else 0.0,
            "llm_calls": self.llm_calls,
            "llm_calls_without_latency": self.llm_calls_without_latency,
            "llm_latency_ms_total": round(self.llm_latency_ms_total, 3),
            "llm_latency_ms_mean": (
                round(self.llm_latency_ms_total / timed, 3)
                if (timed := self.llm_calls - self.llm_calls_without_latency) else 0.0),
            "records_written": self.records_written,
            "records_failed": self.records_failed,
            "contexts_ingested": self.contexts_ingested,
            "gate_rejections": self.gate_rejections,
            "throttle_events": self.throttle_events,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "cost_usd_total": round(self.cost_usd_total, 8),
            "cost_usd_mean": (round(self.cost_usd_total / self.llm_calls, 8)
                              if self.llm_calls else 0.0),
            "unpriced_calls": self.unpriced_calls,
        }
