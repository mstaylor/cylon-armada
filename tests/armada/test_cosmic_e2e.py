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

"""Cosmic AI workflow end to end (SP1 Task 6).

Drives the whole authored pipeline — Preprocess | Embed | Retrieve | Reason |
Bind | MemoryUpsert — through ArmadaExecutor, from a fixture of AstroMAE
inference outputs (redshift predictions + magnitudes) turned into real analysis
prompts by cosmic_ai.task_generator. Bedrock, Redis and DynamoDB are all mocked;
nothing here touches AWS.

Two things are proved:
  1. every rank turns its own shard into stored analysis records, and the
     collectives fired in the order the compiled plan assigns (scatter ->
     allgather). Under the row-distributed layout Embed and Retrieve are local,
     so the only cross-rank movement is MemoryUpsert's allgather, which is what
     makes a context published on one rank visible to the others;
  2. at world_size == 1 the executor's result is identical to LangChain's own
     seq.invoke() on the same input — the swap-equivalence control for E-SP2,
     which is what lets a measured S(N) be attributed to the data plane rather
     than to two pipelines that merely resemble each other.

Run: pytest tests/armada/test_cosmic_e2e.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np
import pyarrow as pa
import pytest

from armada.cosmic_workflow import build_cosmic_workflow
from armada.executor import ArmadaExecutor
from cosmic_ai.task_generator import generate_tasks_from_results

D = 8
WORKFLOW_ID = "wf-cosmic-e2e"


# ---------------------------------------------------------------------------
# Fixture — AstroMAE inference outputs (real model wiring is an SP5 item)
# ---------------------------------------------------------------------------

def _astromae_fixture(n_galaxies=4):
    """n galaxies: predicted vs true redshift, plus SDSS ugriz magnitudes.

    Deterministic stand-in for a real AstroMAE inference pass — one prediction
    per galaxy plus its five-band photometry, which is all task_generator needs
    to build genuine analysis prompts. Wiring the real checkpoint is an SP5 item.
    """
    rng = np.random.default_rng(42)
    predictions = np.round(rng.uniform(0.15, 0.95, n_galaxies), 3)
    residuals = np.round(rng.normal(0.0, 0.06, n_galaxies), 3)
    true_redshifts = np.clip(predictions - residuals, 0.01, None)
    base = rng.uniform(17.0, 20.5, n_galaxies)
    # ugriz decreases monotonically for a typical red-sequence galaxy.
    magnitudes = np.round(base[:, None] - np.array([0.0, 0.7, 1.2, 1.5, 1.7]), 2)
    return predictions, true_redshifts, magnitudes


def _analysis_prompts(n_tasks=4):
    predictions, true_redshifts, magnitudes = _astromae_fixture(n_tasks)
    tasks = generate_tasks_from_results(
        predictions, true_redshifts, magnitudes, max_tasks=n_tasks, seed=42,
    )
    return tasks[:n_tasks]


def _mock_services():
    embedding_service = MagicMock()
    embedding_service.embed.side_effect = lambda text: (
        np.full(D, float(len(text) % 7 + 1), dtype=np.float32), {"token_count": len(text)}
    )

    context_router = MagicMock()
    context_router.find_similar.return_value = []

    context_manager = MagicMock()
    context_manager.store_context.return_value = "ctx-id"

    chain_executor = MagicMock()
    chain_executor.execute.side_effect = lambda prompt: {
        "response": f"analysis::{prompt[:40]}",
        "input_tokens": 11, "output_tokens": 7,
        "latency_ms": 3.5, "model_id": "mock-model",
    }
    return embedding_service, context_router, context_manager, chain_executor


def _workflow(services):
    embedding_service, context_router, context_manager, chain_executor = services
    return build_cosmic_workflow(
        embedding_service, context_router, context_manager, chain_executor,
        workflow_id=WORKFLOW_ID, dimensions=D,
    )


class SpyBridge:
    """Records each collective and returns what a real one would hand back.

    scatter delivers this rank's own shard; allgather returns every rank's
    contribution, which under the row-distributed layout is how a context
    published on one rank becomes visible to the others. The values moving
    through the pipeline are the real ones, produced by the operator bodies the
    executor runs — only the transport is simulated.

    gather and broadcast are deliberately absent. Under this layout no operator
    should reach for them, so a pattern that regressed to one fails loudly with
    an AttributeError naming the collective rather than being quietly recorded.
    """

    def __init__(self, world_size, my_shard):
        self.world_size = world_size
        self.available = world_size > 1
        self.calls = []
        self._my_shard = my_shard

    def scatter(self, tables, root=0):
        self.calls.append("scatter")
        return self._my_shard

    def allgather(self, table):
        self.calls.append("allgather")
        return [table] * self.world_size


# ---------------------------------------------------------------------------
# Distributed end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [4, 8])
def test_every_galaxy_produces_an_analysis_record_over_the_collectives(world_size):
    """One rank's view of the pipeline: it reasons over its own shard only, and
    stores every rank's contexts.

    The per-rank LLM count is the assertion that matters. Under the previous
    mapping Embed's gather funnelled every row to root, so one rank made every
    LLM call while the others idled — which would have flattened S(N) in both
    arms of Experiment E for reasons unrelated to the data plane.

    Reasoning stays sharded while storage is replicated, and the two counts
    diverging by exactly world_size is what all-to-all visibility looks like
    from inside one rank: one LLM call for its own galaxy, world_size stores
    because the collective handed it everyone's contexts.
    """
    prompts = _analysis_prompts(world_size)
    assert len(prompts) == world_size

    services = _mock_services()
    seq = _workflow(services)
    preprocess_op, embed_op = seq.operators[0], seq.operators[1]

    def raw_table(texts):
        return pa.table({"raw_text": list(texts)}, schema=preprocess_op.schema_in)

    bridge = SpyBridge(world_size=world_size, my_shard=raw_table([prompts[0]]))
    executor = ArmadaExecutor(bridge)

    result = executor.run(seq, input_tables=[raw_table([p]) for p in prompts],
                          ctx=None, root=0)

    assert bridge.calls == ["scatter", "allgather"]

    acks = result.column("ack").to_pylist()
    assert len(acks) == world_size
    assert all(acks)

    _, _, context_manager, chain_executor = services
    assert chain_executor.execute.call_count == 1
    assert context_manager.store_context.call_count == world_size


def test_stored_records_carry_the_real_prompt_and_embedding():
    """Provenance has to survive to MemoryUpsert or the reuse mechanism (H4)
    stores contexts that similarity search can never find again."""
    prompts = _analysis_prompts(4)
    services = _mock_services()
    seq = _workflow(services)
    preprocess_op, embed_op = seq.operators[0], seq.operators[1]

    def raw_table(texts):
        return pa.table({"raw_text": list(texts)}, schema=preprocess_op.schema_in)

    bridge = SpyBridge(4, raw_table([prompts[0]]))
    ArmadaExecutor(bridge).run(seq, input_tables=[raw_table([p]) for p in prompts],
                               ctx=None, root=0)

    _, _, context_manager, _ = services
    stored = [c.kwargs for c in context_manager.store_context.call_args_list]
    assert {s["task_description"] for s in stored} == {prompts[0]}

    for s in stored:
        assert s["workflow_id"] == WORKFLOW_ID
        assert s["response"].startswith("analysis::")
        assert not np.allclose(s["embedding"], np.zeros(D, dtype=np.float32))
        assert s["cost_metadata"]["model_id"] == "mock-model"


# ---------------------------------------------------------------------------
# Swap equivalence (the E-SP2 control)
# ---------------------------------------------------------------------------

def test_world_size_one_matches_langchain_native_invoke():
    prompts = _analysis_prompts(4)

    executor_services = _mock_services()
    executor_seq = _workflow(executor_services)
    schema_in = executor_seq.operators[0].schema_in
    table = pa.table({"raw_text": list(prompts)}, schema=schema_in)

    single_rank = SpyBridge(world_size=1, my_shard=None)
    via_executor = ArmadaExecutor(single_rank).run(executor_seq, input_tables=table,
                                                   ctx=None, root=0)

    # A separate service set, so the two paths cannot share mock state.
    invoke_services = _mock_services()
    via_invoke = _workflow(invoke_services).invoke(
        pa.table({"raw_text": list(prompts)}, schema=schema_in))

    assert single_rank.calls == []
    assert via_executor.column("ack").to_pylist() == via_invoke.column("ack").to_pylist()
    assert via_executor.num_rows == len(prompts)

    executor_stored = [c.kwargs for c in executor_services[2].store_context.call_args_list]
    invoke_stored = [c.kwargs for c in invoke_services[2].store_context.call_args_list]
    assert [s["task_description"] for s in executor_stored] == \
           [s["task_description"] for s in invoke_stored]
    assert [s["response"] for s in executor_stored] == [s["response"] for s in invoke_stored]


# ---------------------------------------------------------------------------
# photometry_classification write-time verification, full chain
# ---------------------------------------------------------------------------

def test_photometry_features_survive_the_full_chain_and_produce_a_verified_reuse_key():
    """photometry_features has to ride unchanged through Preprocess -> Embed
    -> Retrieve -> Reason -> Bind for Bind's write-time verification to have
    anything to check. Each of those four operators got its own passthrough
    line added individually — this is what proves the wiring holds together
    as a chain, not just in each operator's own isolated unit test."""
    from cosmic_ai.task_generator import template_for
    from armada.reuse_policy import PHOTOMETRY_BANDS, PHOTOMETRY_KEY_OFFSET, photometry_ground_truth_flags

    predictions, true_redshifts, magnitudes = _astromae_fixture(4)
    residuals = np.abs(predictions - true_redshifts)
    outlier_threshold = float(np.percentile(residuals, 90))
    photo_idx = next(i for i in range(4)
                     if template_for(i, float(residuals[i]), outlier_threshold, 0)
                     == "photometry_classification")

    prompts = generate_tasks_from_results(predictions, true_redshifts, magnitudes,
                                          max_tasks=4, seed=42,
                                          outlier_threshold=outlier_threshold)
    mags = magnitudes[photo_idx]
    ground_truth = photometry_ground_truth_flags(mags)
    flagged_str = "NONE" if not ground_truth else ",".join(sorted(ground_truth))

    services = _mock_services()
    _, _, context_manager, chain_executor = services
    chain_executor.execute.side_effect = lambda p: {
        "response": f"analysis\nFLAGGED_BANDS: {flagged_str}",
        "input_tokens": 1, "output_tokens": 1, "latency_ms": 1.0, "model_id": "mock",
    }

    table = pa.table({
        "raw_text": pa.array([prompts[photo_idx]], type=pa.large_utf8()),
        "reuse_key": pa.array([None], type=pa.float64()),
        "photometry_features": pa.array([[float(v) for v in mags]],
                                        type=pa.list_(pa.float64(), 5)),
    })

    bridge = SpyBridge(world_size=1, my_shard=None)
    ArmadaExecutor(bridge).run(_workflow(services), input_tables=table, ctx=None, root=0)

    stored = context_manager.store_context.call_args_list
    assert len(stored) == 1
    expected_bitmask = sum(1 << i for i, b in enumerate(PHOTOMETRY_BANDS) if b in ground_truth)
    assert stored[0].kwargs["reuse_key"] == pytest.approx(PHOTOMETRY_KEY_OFFSET + expected_bitmask)


def test_a_wrong_photometry_response_stores_with_no_reuse_key_over_the_full_chain():
    from cosmic_ai.task_generator import template_for
    from armada.reuse_policy import photometry_ground_truth_flags

    predictions, true_redshifts, magnitudes = _astromae_fixture(4)
    residuals = np.abs(predictions - true_redshifts)
    outlier_threshold = float(np.percentile(residuals, 90))
    photo_idx = next(i for i in range(4)
                     if template_for(i, float(residuals[i]), outlier_threshold, 0)
                     == "photometry_classification")

    prompts = generate_tasks_from_results(predictions, true_redshifts, magnitudes,
                                          max_tasks=4, seed=42,
                                          outlier_threshold=outlier_threshold)
    mags = magnitudes[photo_idx]
    ground_truth = photometry_ground_truth_flags(mags)
    # Deliberately wrong: claim the opposite of the truth.
    wrong_str = "NONE" if ground_truth else "u"

    services = _mock_services()
    _, _, context_manager, chain_executor = services
    chain_executor.execute.side_effect = lambda p: {
        "response": f"analysis\nFLAGGED_BANDS: {wrong_str}",
        "input_tokens": 1, "output_tokens": 1, "latency_ms": 1.0, "model_id": "mock",
    }

    table = pa.table({
        "raw_text": pa.array([prompts[photo_idx]], type=pa.large_utf8()),
        "reuse_key": pa.array([None], type=pa.float64()),
        "photometry_features": pa.array([[float(v) for v in mags]],
                                        type=pa.list_(pa.float64(), 5)),
    })

    bridge = SpyBridge(world_size=1, my_shard=None)
    ArmadaExecutor(bridge).run(_workflow(services), input_tables=table, ctx=None, root=0)

    stored = context_manager.store_context.call_args_list
    assert len(stored) == 1
    assert stored[0].kwargs["reuse_key"] is None


def test_photometry_reuse_actually_fires_end_to_end_on_a_matching_query():
    """A photometry query whose upfront key matches an existing candidate
    must actually reuse it through the real gated Retrieve/Reason, and the
    returned response must come from the query's own values, not the
    candidate's borrowed text."""
    from armada.cosmic_workflow import (
        build_embed_operator,
        build_preprocess_operator,
        build_reason_operator,
        build_retrieve_operator,
    )
    from armada.reuse_policy import photometry_reuse_key_for, redshift_validator

    candidate_mags = [0.0, 0.0, 2.50, 0.0, 0.0]      # flags 'r'
    query_mags = [0.0, 0.0, 3.00, 0.0, 0.0]           # also flags only 'r' — same key
    candidate_key = photometry_reuse_key_for(candidate_mags)
    query_key = photometry_reuse_key_for(query_mags)
    assert candidate_key == query_key                # precondition: same flagged set

    borrowed_text = "some other galaxy's own numbers\nFLAGGED_BANDS: r"
    embedding_service, context_router, context_manager, chain_executor = _mock_services()
    context_router.find_similar.return_value = [{"context_id": "candidate-1", "similarity": 0.9}]
    context_router.context_manager.get_context.return_value = {
        "response": borrowed_text,
        "reuse_key": candidate_key,
    }

    # Preprocess | Embed | Retrieve | Reason only — Bind/MemoryUpsert are not
    # needed to observe the response, and a reused row is never bound for
    # storage anyway (nothing would reach store_context to inspect there).
    partial_workflow = (
        build_preprocess_operator(dimensions=D)
        | build_embed_operator(embedding_service, dimensions=D)
        | build_retrieve_operator(context_router, workflow_id=WORKFLOW_ID, dimensions=D,
                                  reuse_validator=redshift_validator(tolerance=0.0))
        | build_reason_operator(chain_executor, dimensions=D)
    )

    table = pa.table({
        "raw_text": pa.array(["query prompt"], type=pa.large_utf8()),
        "reuse_key": pa.array([query_key], type=pa.float64()),
        "photometry_features": pa.array([[float(v) for v in query_mags]],
                                        type=pa.list_(pa.float64(), 5)),
    })

    bridge = SpyBridge(world_size=1, my_shard=None)
    result = ArmadaExecutor(bridge).run(partial_workflow, input_tables=table, ctx=None, root=0)

    chain_executor.execute.assert_not_called()

    response = result.column("response").to_pylist()[0]
    assert result.column("reused").to_pylist() == [True]
    assert response != borrowed_text
    assert "r=3.00" in response       # the query's own value
    assert "2.50" not in response    # the candidate's borrowed value, absent
    assert "FLAGGED_BANDS: r" in response


def test_photometry_reuse_stores_nothing_end_to_end_on_a_matching_query():
    """The storage-side half of the same reproduction: a reused row must
    never be bound, so context_manager.store_context is never called for it.
    Split from the response-content test above so a failure names exactly
    which contract broke."""
    from armada.reuse_policy import photometry_reuse_key_for, redshift_validator

    candidate_mags = [0.0, 0.0, 2.50, 0.0, 0.0]
    query_mags = [0.0, 0.0, 3.00, 0.0, 0.0]
    candidate_key = photometry_reuse_key_for(candidate_mags)
    query_key = photometry_reuse_key_for(query_mags)

    embedding_service, context_router, context_manager, chain_executor = _mock_services()
    context_router.find_similar.return_value = [{"context_id": "candidate-1", "similarity": 0.9}]
    context_router.context_manager.get_context.return_value = {
        "response": "some other galaxy's own numbers\nFLAGGED_BANDS: r",
        "reuse_key": candidate_key,
    }

    workflow = build_cosmic_workflow(
        embedding_service, context_router, context_manager, chain_executor,
        workflow_id=WORKFLOW_ID, dimensions=D, reuse_validator=redshift_validator(tolerance=0.0),
    )

    table = pa.table({
        "raw_text": pa.array(["query prompt"], type=pa.large_utf8()),
        "reuse_key": pa.array([query_key], type=pa.float64()),
        "photometry_features": pa.array([[float(v) for v in query_mags]],
                                        type=pa.list_(pa.float64(), 5)),
    })

    bridge = SpyBridge(world_size=1, my_shard=None)
    ArmadaExecutor(bridge).run(workflow, input_tables=table, ctx=None, root=0)

    chain_executor.execute.assert_not_called()
    context_manager.store_context.assert_not_called()