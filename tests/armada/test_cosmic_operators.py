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

"""Unit tests for the five Cosmic AI ArmadaOperators (SP1 Task 5).

Every service (EmbeddingService, ContextRouter, ContextManager, ChainExecutor)
is a MagicMock — no AWS, no Redis, no DynamoDB. Each test checks that the
operator's fn transforms schema_in -> schema_out correctly in isolation, not
through the executor/bridge (that's Task 4's and Task 6's job).

Run: pytest tests/armada/test_cosmic_operators.py -v
"""

import base64
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

from armada.cosmic_workflow import (
    build_bind_operator,
    build_embed_operator,
    build_memory_upsert_operator,
    build_preprocess_operator,
    build_reason_operator,
    build_retrieve_operator,
)
from cylon_armada.dag_compiler import CollectivePattern

D = 8


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def test_preprocess_truncates_to_max_chars():
    op = build_preprocess_operator(max_chars=5, dimensions=D)
    assert op.pattern == CollectivePattern.Scatter

    table = pa.table({"raw_text": ["hello world", "hi"]}, schema=op.schema_in)
    result = op.fn(table)

    # chunked_text (canonical) is column 0, matching op.schema_out's field
    # exactly; raw_text is an additive passthrough (task_description for
    # Bind/MemoryUpsert downstream), not part of the declared A2 contract.
    assert result.schema.field(0) == op.schema_out.field(0)
    assert result.column("chunked_text").to_pylist() == ["hello", "hi"]
    assert result.column("raw_text").to_pylist() == ["hello world", "hi"]


def test_preprocess_noop_when_max_chars_none():
    op = build_preprocess_operator(max_chars=None, dimensions=D)
    table = pa.table({"raw_text": ["hello world", "hi"]}, schema=op.schema_in)
    result = op.fn(table)
    assert result.column("chunked_text").to_pylist() == ["hello world", "hi"]


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


def test_embed_produces_fixed_size_list_float32():
    embedding_service = MagicMock()
    vec1 = np.arange(D, dtype=np.float32)
    vec2 = np.arange(D, dtype=np.float32) * 2
    embedding_service.embed.side_effect = [
        (vec1, {"token_count": 3}),
        (vec2, {"token_count": 4}),
    ]

    op = build_embed_operator(embedding_service, dimensions=D)
    assert op.pattern == CollectivePattern.PointToPoint

    table = pa.table({"chunked_text": ["a", "b"]}, schema=op.schema_in)
    result = op.fn(table)

    assert result.schema == op.schema_out
    list_type = result.schema.field(0).type
    assert pa.types.is_fixed_size_list(list_type)
    assert list_type.list_size == D
    assert list_type.value_type == pa.float32()

    out = result.column("embedding").to_pylist()
    assert np.allclose(out[0], vec1)
    assert np.allclose(out[1], vec2)
    assert embedding_service.embed.call_args_list[0].args == ("a",)
    assert embedding_service.embed.call_args_list[1].args == ("b",)


def test_embed_forwards_raw_text_passthrough_when_present():
    embedding_service = MagicMock()
    embedding_service.embed.return_value = (np.zeros(D, dtype=np.float32), {})
    op = build_embed_operator(embedding_service, dimensions=D)

    table = pa.table({"chunked_text": ["a"], "raw_text": ["original task text"]})
    result = op.fn(table)

    assert result.column("raw_text").to_pylist() == ["original task text"]


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


def test_retrieve_returns_best_match_per_row():
    context_router = MagicMock()
    context_router.find_similar.return_value = [
        {"context_id": "c1", "similarity": 0.9},
        {"context_id": "c2", "similarity": 0.7},
    ]
    context_router.context_manager.get_context.side_effect = lambda cid: {
        "context_id": cid, "response": f"the {cid} analysis"}

    op = build_retrieve_operator(context_router, workflow_id="wf-1", dimensions=D)
    assert op.pattern == CollectivePattern.PointToPoint

    list_type = op.schema_in.field(0).type
    query = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(D, dtype=np.float32)), D
    ).cast(list_type)
    table = pa.table({"query_embedding": query}, schema=op.schema_in)
    result = op.fn(table)

    # ranked_docs (canonical) is column 0; query_embedding is self-forwarded
    # (Retrieve's own canonical output doesn't carry the embedding it just
    # searched with, but Bind needs it downstream).
    assert result.schema.field(0) == op.schema_out.field(0)
    row = result.column("ranked_docs").to_pylist()[0]
    assert row == {"doc": "the c1 analysis", "score": pytest.approx(0.9)}
    assert result.column("query_embedding").to_pylist()[0] == query.to_pylist()[0]
    _, kwargs = context_router.find_similar.call_args
    assert kwargs["workflow_id"] == "wf-1"


def test_retrieve_concatenates_a_list_of_gathered_tables():
    """Retrieve still accepts a gather-shaped list, not only a single table.

    The row-distributed pipeline hands it a single table, since Embed is now
    PointToPoint. This pins the list form so a gather-shaped caller — which is
    what a corpus-distributed layout would use — keeps working.
    """
    context_router = MagicMock()
    context_router.find_similar.return_value = [{"context_id": "c1", "similarity": 0.5}]
    context_router.context_manager.get_context.return_value = {"context_id": "c1",
                                                               "response": "an analysis"}

    op = build_retrieve_operator(context_router, workflow_id="wf-1", dimensions=D)
    list_type = op.schema_in.field(0).type

    def one_row(v):
        arr = pa.FixedSizeListArray.from_arrays(pa.array(np.full(D, v, dtype=np.float32)), D).cast(list_type)
        return pa.table({"query_embedding": arr}, schema=op.schema_in)

    result = op.fn([one_row(1.0), one_row(2.0), one_row(3.0)])

    assert result.num_rows == 3
    assert context_router.find_similar.call_count == 3


def test_retrieve_returns_empty_match_when_no_hits():
    context_router = MagicMock()
    context_router.find_similar.return_value = []

    op = build_retrieve_operator(context_router, workflow_id="wf-1", dimensions=D)
    list_type = op.schema_in.field(0).type
    query = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(D, dtype=np.float32)), D
    ).cast(list_type)
    table = pa.table({"query_embedding": query}, schema=op.schema_in)
    result = op.fn(table)

    row = result.column("ranked_docs").to_pylist()[0]
    assert row == {"doc": "", "score": 0.0}


# ---------------------------------------------------------------------------
# Reason
# ---------------------------------------------------------------------------


def test_reason_puts_its_canonical_response_column_first():
    """Reason's declared A2 output must stay column 0, whichever branch ran.

    This test previously also asserted that the LLM was called with the
    retrieved doc text. That assertion encoded the defect the prompt-fidelity
    gate caught: the doc was a context id, never a prompt. The branch
    behaviours are asserted in the reuse-or-call tests below.
    """
    chain_executor = MagicMock()
    chain_executor.execute.return_value = {"response": "the answer"}

    op = build_reason_operator(chain_executor, dimensions=D)
    assert op.pattern == CollectivePattern.PointToPoint

    struct_type = op.schema_in.field(0).type
    ctx = pa.array([{"doc": "", "score": 0.0}], type=struct_type)
    table = pa.table({"context": ctx}, schema=op.schema_in)
    result = op.fn(table)

    assert result.schema.field(0) == op.schema_out.field(0)
    assert result.column("response").to_pylist() == ["the answer"]


def test_reason_carries_cost_metadata_forward_as_passthrough():
    chain_executor = MagicMock()
    chain_executor.execute.return_value = {
        "response": "the answer", "input_tokens": 12, "output_tokens": 34,
        "latency_ms": 5.5, "model_id": "mock-model",
    }
    op = build_reason_operator(chain_executor, dimensions=D)
    struct_type = op.schema_in.field(0).type
    ctx = pa.array([{"doc": "", "score": 0.0}], type=struct_type)
    table = pa.table({"context": ctx}, schema=op.schema_in)
    result = op.fn(table)

    cost = json.loads(result.column("cost_metadata_json").to_pylist()[0])
    assert cost == {"input_tokens": 12, "output_tokens": 34, "latency_ms": 5.5, "model_id": "mock-model"}


# ---------------------------------------------------------------------------
# MemoryUpsert
# ---------------------------------------------------------------------------

def _kv_envelope(context_id, embedding):
    envelope = {
        "workflow_id": "wf-1",
        "task_description": "analyze galaxy",
        "response": "analysis text",
        "cost_metadata": {"cost_usd": 0.001},
        "embedding_b64": base64.b64encode(embedding.astype(np.float32).tobytes()).decode(),
    }
    return {"k": context_id, "v": json.dumps(envelope)}


def test_memory_upsert_acks_and_calls_store_context():
    context_manager = MagicMock()
    embedding = np.arange(D, dtype=np.float32)

    op = build_memory_upsert_operator(context_manager, dimensions=D)
    assert op.pattern == CollectivePattern.AllGather

    struct_type = op.schema_in.field(0).type
    kv = pa.array([_kv_envelope("ctx-1", embedding)], type=struct_type)
    table = pa.table({"kv_pairs": kv}, schema=op.schema_in)
    result = op.fn(table)

    assert result.schema == op.schema_out
    assert result.column("ack").to_pylist() == [True]

    _, kwargs = context_manager.store_context.call_args
    assert kwargs["workflow_id"] == "wf-1"
    assert kwargs["task_description"] == "analyze galaxy"
    assert kwargs["response"] == "analysis text"
    assert kwargs["cost_metadata"] == {"cost_usd": 0.001}
    assert kwargs["context_id"] == "ctx-1"
    assert np.allclose(kwargs["embedding"], embedding)


def test_memory_upsert_acks_false_on_bad_envelope():
    context_manager = MagicMock()
    op = build_memory_upsert_operator(context_manager, dimensions=D)

    struct_type = op.schema_in.field(0).type
    kv = pa.array([{"k": "ctx-1", "v": "not json"}], type=struct_type)
    table = pa.table({"kv_pairs": kv}, schema=op.schema_in)
    result = op.fn(table)

    assert result.column("ack").to_pylist() == [False]
    context_manager.store_context.assert_not_called()


def test_memory_upsert_falls_back_to_placeholder_embedding_when_envelope_omits_it():
    """Bind's glued envelope only has workflow_id + response — task_description
    and embedding_b64 are genuinely absent, not malformed. MemoryUpsert must
    still ack (a real store_context call, with a placeholder embedding), not
    treat a missing optional field as a bad envelope."""
    context_manager = MagicMock()
    op = build_memory_upsert_operator(context_manager, dimensions=D)

    struct_type = op.schema_in.field(0).type
    minimal_envelope = json.dumps({"workflow_id": "wf-1", "response": "analysis text"})
    kv = pa.array([{"k": "ctx-1", "v": minimal_envelope}], type=struct_type)
    table = pa.table({"kv_pairs": kv}, schema=op.schema_in)
    result = op.fn(table)

    assert result.column("ack").to_pylist() == [True]
    _, kwargs = context_manager.store_context.call_args
    assert kwargs["task_description"] == ""
    assert np.allclose(kwargs["embedding"], np.zeros(D, dtype=np.float32))


# ---------------------------------------------------------------------------
# Bind (non-canonical glue: Reason's response -> MemoryUpsert's kv_pairs)
# ---------------------------------------------------------------------------


def test_bind_wraps_response_into_kv_pairs_with_unique_keys():
    op = build_bind_operator(workflow_id="wf-1")
    assert op.pattern == CollectivePattern.PointToPoint

    table = pa.table({"response": ["analysis A", "analysis B"]}, schema=op.schema_in)
    result = op.fn(table)

    assert result.schema == op.schema_out
    rows = result.column("kv_pairs").to_pylist()
    assert len(rows) == 2
    assert rows[0]["k"] != rows[1]["k"]

    envelope = json.loads(rows[0]["v"])
    assert envelope == {"workflow_id": "wf-1", "response": "analysis A"}


def test_bind_output_feeds_memory_upsert_end_to_end():
    context_manager = MagicMock()
    bind = build_bind_operator(workflow_id="wf-1")
    upsert = build_memory_upsert_operator(context_manager, dimensions=D)
    assert bind.schema_out == upsert.schema_in

    table = pa.table({"response": ["analysis A"]}, schema=bind.schema_in)
    result = upsert.fn(bind.fn(table))

    assert result.column("ack").to_pylist() == [True]
    context_manager.store_context.assert_called_once()


def test_cosmic_uses_row_distributed_patterns_not_the_canonical_ones():
    """Rows are distributed and the corpus is replicated, so Embed and Retrieve
    are local and MemoryUpsert is the one genuine collective. Table III's
    mapping describes a corpus-distributed layout and is deliberately not used
    here (spec 3.3)."""
    from armada.cosmic_workflow import COSMIC_PATTERNS
    from cylon_armada.dag_compiler import CollectivePattern

    assert COSMIC_PATTERNS == {
        "Preprocess": CollectivePattern.Scatter,
        "Embed": CollectivePattern.PointToPoint,
        "Retrieve": CollectivePattern.PointToPoint,
        "Reason": CollectivePattern.PointToPoint,
        "Bind": CollectivePattern.PointToPoint,
        "MemoryUpsert": CollectivePattern.AllGather,
    }


def test_operators_still_carry_the_canonical_a2_schemas():
    """Patterns diverge from Table III; schemas must not. The A2 contract is a
    schema contract and this workflow stays under it."""
    from unittest.mock import MagicMock

    from armada.cosmic_workflow import build_embed_operator, build_retrieve_operator
    from experiment.exp_a2_schema import canonical_operators

    canon = {o.name: o for o in canonical_operators(8)}
    embed = build_embed_operator(MagicMock(), dimensions=8)
    retrieve = build_retrieve_operator(MagicMock(), workflow_id="wf", dimensions=8)

    assert embed.schema_in == canon["Embed"].schema_in
    assert embed.schema_out == canon["Embed"].schema_out
    assert retrieve.schema_in == canon["Retrieve"].schema_in
    assert retrieve.schema_out == canon["Retrieve"].schema_out


def test_the_full_chain_still_compiles():
    """compile_workflow enforces schema compatibility on every edge; changing
    patterns must not break the chain."""
    from unittest.mock import MagicMock

    from armada.cosmic_workflow import build_cosmic_workflow
    from armada.executor import lower

    seq = build_cosmic_workflow(MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                workflow_id="wf", dimensions=8)
    plan = lower(seq)

    assert plan.assignments["MemoryUpsert"].name == "AllGather"
    assert plan.assignments["Embed"].name == "PointToPoint"


def test_memory_upsert_records_a_store_failure_and_acks_false():
    """The failure has to be counted at the store site, not read back off the
    ack table — an AllGather'd ack table holds every rank's acks, so an
    ack-derived failure count would mean something different per arm."""
    from armada.cosmic_workflow import build_memory_upsert_operator
    from armada.run_metrics import RunMetrics

    context_manager = MagicMock()
    context_manager.store_context.side_effect = [None, RuntimeError("redis down")]
    metrics = RunMetrics()
    op = build_memory_upsert_operator(context_manager, dimensions=D, metrics=metrics)

    envelope = json.dumps({"workflow_id": "wf", "response": "r"})
    table = pa.table({"kv_pairs": pa.array(
        [{"k": "a", "v": envelope}, {"k": "b", "v": envelope}],
        type=op.schema_in.field(0).type)}, schema=op.schema_in)

    result = op.fn(table)

    assert result.column("ack").to_pylist() == [True, False]
    assert metrics.summary()["records_written"] == 1
    assert metrics.summary()["records_failed"] == 1


def test_memory_upsert_counts_only_this_ranks_contexts_as_written():
    """A rank stores everyone's contexts but originates only its own."""
    import json

    from armada.cosmic_workflow import build_memory_upsert_operator
    from armada.run_metrics import RunMetrics

    context_manager = MagicMock()
    metrics = RunMetrics()
    op = build_memory_upsert_operator(context_manager, dimensions=D, metrics=metrics, rank=0)

    mine = json.dumps({"workflow_id": "wf", "response": "r", "rank": 0})
    theirs = json.dumps({"workflow_id": "wf", "response": "r", "rank": 1})
    table = pa.table({"kv_pairs": pa.array(
        [{"k": "a", "v": mine}, {"k": "b", "v": theirs}, {"k": "c", "v": theirs}],
        type=op.schema_in.field(0).type)}, schema=op.schema_in)

    op.fn(table)

    assert context_manager.store_context.call_count == 3
    assert metrics.summary()["records_written"] == 1
    assert metrics.summary()["contexts_ingested"] == 2


def test_bind_stamps_the_originating_rank_into_the_envelope():
    import json

    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf", rank=3)
    table = pa.table({"response": ["an analysis"]}, schema=op.schema_in)

    envelope = json.loads(op.fn(table).column("kv_pairs").to_pylist()[0]["v"])

    assert envelope["rank"] == 3


def test_an_untagged_envelope_counts_as_originated():
    """Single-rank runs and every existing caller pass no rank, so the envelope
    carries no rank key. Those must stay originated, or records_written would
    drop to zero everywhere the tag is absent."""
    import json

    from armada.cosmic_workflow import build_memory_upsert_operator
    from armada.run_metrics import RunMetrics

    context_manager = MagicMock()
    metrics = RunMetrics()
    op = build_memory_upsert_operator(context_manager, dimensions=D, metrics=metrics)

    envelope = json.dumps({"workflow_id": "wf", "response": "r"})
    table = pa.table({"kv_pairs": pa.array(
        [{"k": "a", "v": envelope}], type=op.schema_in.field(0).type)},
        schema=op.schema_in)

    op.fn(table)

    assert metrics.summary()["records_written"] == 1
    assert metrics.summary()["contexts_ingested"] == 0


def test_retrieve_resolves_a_match_to_its_stored_response_text():
    """ranked_docs.doc feeds Reason. A context id is not a document: Reason
    cannot reason over a UUID, and sending one to an LLM is what the
    prompt-fidelity gate caught."""
    from armada.cosmic_workflow import build_retrieve_operator

    context_manager = MagicMock()
    context_manager.get_context.return_value = {"context_id": "c-1",
                                                "response": "an earlier analysis"}
    context_router = MagicMock()
    context_router.context_manager = context_manager
    context_router.find_similar.return_value = [{"context_id": "c-1", "similarity": 0.97}]

    op = build_retrieve_operator(context_router, workflow_id="wf", dimensions=D)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=op.schema_in.field(0).type)},
        schema=op.schema_in)

    row = op.fn(table).column("ranked_docs").to_pylist()[0]

    assert row["doc"] == "an earlier analysis"
    assert row["score"] == pytest.approx(0.97)


def test_retrieve_on_a_miss_emits_an_empty_doc_and_zero_score():
    """The miss shape is the hit/miss discriminator Reason reads; score 0.0
    must mean miss unambiguously."""
    from armada.cosmic_workflow import build_retrieve_operator

    context_router = MagicMock()
    context_router.context_manager = MagicMock()
    context_router.find_similar.return_value = []

    op = build_retrieve_operator(context_router, workflow_id="wf", dimensions=D)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=op.schema_in.field(0).type)},
        schema=op.schema_in)

    row = op.fn(table).column("ranked_docs").to_pylist()[0]

    assert row["doc"] == ""
    assert row["score"] == 0.0
    context_router.context_manager.get_context.assert_not_called()


def test_retrieve_treats_a_vanished_context_as_a_miss():
    """A context id that no longer resolves — evicted by TTL, or another rank's
    id not yet ingested — must degrade to a miss rather than produce a row whose
    doc is None and whose score claims a hit."""
    from armada.cosmic_workflow import build_retrieve_operator

    context_manager = MagicMock()
    context_manager.get_context.return_value = None
    context_router = MagicMock()
    context_router.context_manager = context_manager
    context_router.find_similar.return_value = [{"context_id": "gone", "similarity": 0.97}]

    op = build_retrieve_operator(context_router, workflow_id="wf", dimensions=D)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=op.schema_in.field(0).type)},
        schema=op.schema_in)

    row = op.fn(table).column("ranked_docs").to_pylist()[0]

    assert row["doc"] == ""
    assert row["score"] == 0.0


def test_a_resolved_hit_is_what_counts_as_reuse():
    """reuse_rate must count reuse that actually happened. A similarity match
    whose context cannot be fetched avoids no LLM call, so counting it would
    report a reuse rate the cost numbers cannot support."""
    from armada.cosmic_workflow import build_retrieve_operator
    from armada.run_metrics import RunMetrics

    context_manager = MagicMock()
    context_manager.get_context.return_value = None
    context_router = MagicMock()
    context_router.context_manager = context_manager
    context_router.find_similar.return_value = [{"context_id": "gone", "similarity": 0.97}]

    metrics = RunMetrics()
    op = build_retrieve_operator(context_router, workflow_id="wf", dimensions=D,
                                 metrics=metrics)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=op.schema_in.field(0).type)},
        schema=op.schema_in)

    op.fn(table)

    assert metrics.summary()["retrievals"] == 1
    assert metrics.summary()["cache_hits"] == 0


def _reason_table(op, docs, scores, raw):
    ctx_type = op.schema_in.field(0).type
    return pa.table({
        "context": pa.array([{"doc": d, "score": s} for d, s in zip(docs, scores)],
                            type=ctx_type),
        "raw_text": pa.array(raw, type=pa.large_utf8()),
    })


def test_reason_calls_the_llm_with_raw_text_on_a_miss():
    from armada.cosmic_workflow import build_reason_operator
    from armada.run_metrics import RunMetrics

    chain_executor = MagicMock()
    chain_executor.execute.return_value = {"response": "fresh", "input_tokens": 1,
                                           "output_tokens": 2, "latency_ms": 5.0,
                                           "model_id": "m"}
    metrics = RunMetrics()
    op = build_reason_operator(chain_executor, dimensions=D, metrics=metrics)

    out = op.fn(_reason_table(op, [""], [0.0], ["the real analysis prompt"]))

    chain_executor.execute.assert_called_once_with("the real analysis prompt")
    assert out.column("response").to_pylist() == ["fresh"]
    assert out.column("reused").to_pylist() == [False]
    assert metrics.summary()["llm_calls"] == 1


def test_reason_reuses_the_retrieved_response_without_calling_the_llm():
    from armada.cosmic_workflow import build_reason_operator
    from armada.run_metrics import RunMetrics

    chain_executor = MagicMock()
    metrics = RunMetrics()
    op = build_reason_operator(chain_executor, dimensions=D, metrics=metrics)

    out = op.fn(_reason_table(op, ["an earlier analysis"], [0.97], ["prompt"]))

    chain_executor.execute.assert_not_called()
    assert out.column("response").to_pylist() == ["an earlier analysis"]
    assert out.column("reused").to_pylist() == [True]
    assert metrics.summary()["llm_calls"] == 0


def test_reason_mixes_hits_and_misses_in_one_batch():
    """Rows are decided independently. A batch that called once for every row
    because one row missed would put H4's cost saving back to zero."""
    from armada.cosmic_workflow import build_reason_operator
    from armada.run_metrics import RunMetrics

    chain_executor = MagicMock()
    chain_executor.execute.return_value = {"response": "fresh", "input_tokens": 1,
                                           "output_tokens": 2, "latency_ms": 5.0,
                                           "model_id": "m"}
    metrics = RunMetrics()
    op = build_reason_operator(chain_executor, dimensions=D, metrics=metrics)

    out = op.fn(_reason_table(op, ["reused one", "", "reused two"], [0.9, 0.0, 0.95],
                              ["p0", "p1", "p2"]))

    chain_executor.execute.assert_called_once_with("p1")
    assert out.column("response").to_pylist() == ["reused one", "fresh", "reused two"]
    assert out.column("reused").to_pylist() == [True, False, True]
    assert metrics.summary()["llm_calls"] == 1


def test_a_reused_row_reports_no_tokens():
    """Cost accounting must not bill a call that never happened."""
    import json as _json

    from armada.cosmic_workflow import build_reason_operator

    op = build_reason_operator(MagicMock(), dimensions=D)

    out = op.fn(_reason_table(op, ["an earlier analysis"], [0.97], ["prompt"]))
    cost = _json.loads(out.column("cost_metadata_json").to_pylist()[0])

    assert cost["input_tokens"] == 0
    assert cost["output_tokens"] == 0


def test_bind_emits_no_envelope_for_a_reused_row():
    """A hit stores nothing: duplicating the context it just reused would grow
    the store with copies and slow every later similarity search."""
    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf", rank=0)
    table = pa.table({
        "response": pa.array(["reused", "fresh"], type=pa.large_utf8()),
        "reused": pa.array([True, False], type=pa.bool_()),
    })

    out = op.fn(table)

    assert out.num_rows == 1
    assert json.loads(out.column("kv_pairs").to_pylist()[0]["v"])["response"] == "fresh"


def test_bind_without_a_reused_column_binds_every_row():
    """A bare single-column caller (a standalone unit test, or a caller
    predating the reuse path) must keep working unchanged."""
    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf")
    table = pa.table({"response": ["a", "b"]}, schema=op.schema_in)

    assert op.fn(table).num_rows == 2


def test_retrieve_and_reason_agree_on_what_a_hit_is_at_similarity_zero():
    """Both operators must key the hit on the same field.

    A resolved match carrying similarity 0.0 is the case where a score-based
    discriminator in Reason disagrees with Retrieve's response-based one: the
    galaxy is counted in cache_hits AND in records_written, written+hits
    exceeds the shard, and the gate fails a run that was actually fine.
    """
    from armada.cosmic_workflow import build_reason_operator, build_retrieve_operator
    from armada.run_metrics import RunMetrics

    context_manager = MagicMock()
    context_manager.get_context.return_value = {"context_id": "c-1",
                                                "response": "an earlier analysis"}
    context_router = MagicMock()
    context_router.context_manager = context_manager
    context_router.find_similar.return_value = [{"context_id": "c-1", "similarity": 0.0}]

    metrics = RunMetrics()
    retrieve = build_retrieve_operator(context_router, workflow_id="wf", dimensions=D,
                                       metrics=metrics)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=retrieve.schema_in.field(0).type)},
        schema=retrieve.schema_in)

    retrieved = retrieve.fn(table)
    assert metrics.summary()["cache_hits"] == 1

    chain_executor = MagicMock()
    chain_executor.execute.return_value = {"response": "a fresh call", "input_tokens": 1,
                                           "output_tokens": 2, "latency_ms": 1.0,
                                           "model_id": "m"}
    reason = build_reason_operator(chain_executor, dimensions=D, metrics=metrics)
    out = reason.fn(pa.table({
        "context": retrieved.column("ranked_docs"),
        "raw_text": pa.array(["a prompt"], type=pa.large_utf8()),
    }))

    chain_executor.execute.assert_not_called()
    assert out.column("response").to_pylist() == ["an earlier analysis"]
    assert out.column("reused").to_pylist() == [True]
    assert metrics.summary()["llm_calls"] == 0


def test_bind_stamps_a_reuse_key_per_row():
    """The key is per galaxy, not per batch — a batch-level key would make every
    row in an epoch mutually reusable regardless of its own value."""
    import json

    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf")
    table = pa.table({"response": pa.array(["a", "b"], type=pa.large_utf8()),
                      "reuse_key": pa.array([0.12, 0.44], type=pa.float64())})

    envelopes = [json.loads(kv["v"]) for kv in op.fn(table).column("kv_pairs").to_pylist()]

    assert [e["reuse_key"] for e in envelopes] == [0.12, 0.44]


def test_bind_without_reuse_keys_omits_the_field():
    """Every existing caller passes none; the envelope must be unchanged for them."""
    import json

    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf")
    table = pa.table({"response": ["a"]}, schema=op.schema_in)

    assert "reuse_key" not in json.loads(op.fn(table).column("kv_pairs").to_pylist()[0]["v"])


def test_bind_indexes_reuse_keys_by_input_row_even_when_some_are_reused():
    """A reused row contributes no envelope, so the key for a bound row must
    still be its OWN key — indexing by output position would shift every key
    after the first reuse onto the wrong galaxy."""
    import json

    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf")
    table = pa.table({
        "response": pa.array(["skip", "keep", "keep2"], type=pa.large_utf8()),
        "reused": pa.array([True, False, False], type=pa.bool_()),
        "reuse_key": pa.array([0.10, 0.20, 0.30], type=pa.float64()),
    })

    envelopes = [json.loads(kv["v"]) for kv in op.fn(table).column("kv_pairs").to_pylist()]

    assert [e["reuse_key"] for e in envelopes] == [0.20, 0.30]


def test_memory_upsert_persists_the_reuse_key():
    import json

    from armada.cosmic_workflow import build_memory_upsert_operator

    context_manager = MagicMock()
    op = build_memory_upsert_operator(context_manager, dimensions=D)
    envelope = json.dumps({"workflow_id": "wf", "response": "r", "reuse_key": 0.31})
    table = pa.table({"kv_pairs": pa.array(
        [{"k": "a", "v": envelope}], type=op.schema_in.field(0).type)}, schema=op.schema_in)

    op.fn(table)

    assert context_manager.store_context.call_args.kwargs["reuse_key"] == 0.31


def _gated_router(candidates):
    """Router whose find_similar returns `candidates` and whose manager resolves
    each to a stored context carrying its reuse_key."""
    store = {cid: {"context_id": cid, "response": resp, "reuse_key": key}
             for cid, _, resp, key in candidates}
    context_manager = MagicMock()
    context_manager.get_context.side_effect = lambda cid: store.get(cid)
    router = MagicMock()
    router.context_manager = context_manager
    router.find_similar.return_value = [
        {"context_id": cid, "similarity": sim} for cid, sim, _, _ in candidates]
    return router


def _query_table(op, reuse_key=None):
    cols = {"query_embedding": pa.array([np.ones(D, dtype=np.float32)],
                                        type=op.schema_in.field(0).type)}
    if reuse_key is not None:
        cols["reuse_key"] = pa.array([reuse_key], type=pa.float64())
    return pa.table(cols)


def test_retrieve_rejects_a_candidate_the_validator_refuses():
    """The defect the gate exists for: a near-identical embedding whose domain
    value is far away. Cosine accepts it; the validator must not."""
    from armada.run_metrics import RunMetrics

    router = _gated_router([("c-1", 0.99, "an earlier analysis", 0.94)])
    metrics = RunMetrics()
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D, metrics=metrics,
                                 reuse_validator=lambda q, c: abs(q - c) <= 0.01)

    row = op.fn(_query_table(op, 0.15)).column("ranked_docs").to_pylist()[0]

    assert row["doc"] == ""
    assert metrics.summary()["gate_rejections"] == 1
    assert metrics.summary()["cache_hits"] == 0


def test_retrieve_accepts_a_lower_ranked_candidate_the_validator_allows():
    """The validator filters the candidate list, not only the best match: a
    slightly worse cosine match that is valid must still be reused."""
    router = _gated_router([("c-far", 0.99, "wrong", 0.94),
                            ("c-near", 0.97, "right", 0.151)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D,
                                 reuse_validator=lambda q, c: abs(q - c) <= 0.01)

    assert op.fn(_query_table(op, 0.15)).column("ranked_docs").to_pylist()[0]["doc"] == "right"


def test_no_validator_reproduces_todays_behaviour():
    """The gate is opt-in. Without a validator the operator must behave exactly
    as before, or every measurement taken so far becomes incomparable."""
    router = _gated_router([("c-1", 0.99, "reused", 0.94)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D)

    assert op.fn(_query_table(op)).column("ranked_docs").to_pylist()[0]["doc"] == "reused"


def test_ungated_retrieval_never_looks_past_the_best_match():
    """The regression the single-candidate gate above cannot see.

    Ungated, only matches[0] was ever consulted, so a best match that failed to
    resolve was a miss. A candidate walk that runs unconditionally would quietly
    find a later match instead — improving the BASELINE and destroying
    comparability with every measurement taken before the gate existed.
    """
    router = _gated_router([("c-empty", 0.99, "", None),
                            ("c-real", 0.97, "a later analysis", None)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D)

    assert op.fn(_query_table(op)).column("ranked_docs").to_pylist()[0]["doc"] == ""


def test_a_gated_run_does_look_past_an_unresolvable_best_match():
    """The converse, so the restriction above is scoped to the ungated path and
    does not quietly disable the candidate walk the gate depends on."""
    router = _gated_router([("c-empty", 0.99, "", 0.15),
                            ("c-real", 0.97, "a later analysis", 0.15)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D,
                                 reuse_validator=lambda q, c: abs(q - c) <= 0.01)

    row = op.fn(_query_table(op, 0.15)).column("ranked_docs").to_pylist()[0]
    assert row["doc"] == "a later analysis"


def test_a_missing_query_key_is_a_miss_not_an_unchecked_hit():
    """Fail closed. A row with no key cannot be validated, and treating it as a
    hit would bypass the gate for exactly the rows that lack metadata."""
    router = _gated_router([("c-1", 0.99, "x", 0.15)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D,
                                 reuse_validator=lambda q, c: abs(q - c) <= 0.01)

    assert op.fn(_query_table(op)).column("ranked_docs").to_pylist()[0]["doc"] == ""


def test_a_candidate_with_no_stored_key_is_refused_when_gating():
    """A context stored before the key existed must not be reused under a policy
    it was never checked against."""
    router = _gated_router([("c-1", 0.99, "x", None)])
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D,
                                 reuse_validator=lambda q, c: True)

    assert op.fn(_query_table(op, 0.15)).column("ranked_docs").to_pylist()[0]["doc"] == ""


def test_a_row_with_no_candidates_is_a_miss_not_a_gate_rejection():
    """gate_rejections must mean 'the policy refused an eligible match', not
    'nothing matched'. Conflating them would inflate the rejection rate with
    ordinary cold-start misses and make the reported denominator meaningless."""
    from armada.run_metrics import RunMetrics

    router = _gated_router([])
    metrics = RunMetrics()
    op = build_retrieve_operator(router, workflow_id="wf", dimensions=D, metrics=metrics,
                                 reuse_validator=lambda q, c: True)

    op.fn(_query_table(op, 0.15))

    assert metrics.summary()["gate_rejections"] == 0
    assert metrics.summary()["retrievals"] == 1


def test_bind_keys_follow_the_rows_across_epochs():
    """The bug the live/offline reconciliation caught.

    This fn is invoked once per EPOCH. A positional key list would restamp the
    first epoch's keys onto every later epoch, so all but the first few contexts
    would be stored under another galaxy's redshift and the gate would compare
    correct query keys against wrong candidate keys. Keys ride with the rows
    precisely so a later epoch cannot inherit an earlier one's.
    """
    import json

    from armada.cosmic_workflow import build_bind_operator

    op = build_bind_operator(workflow_id="wf")

    def epoch(responses, keys):
        table = pa.table({"response": pa.array(responses, type=pa.large_utf8()),
                          "reuse_key": pa.array(keys, type=pa.float64())})
        return [json.loads(kv["v"])["reuse_key"]
                for kv in op.fn(table).column("kv_pairs").to_pylist()]

    assert epoch(["g0", "g1"], [0.10, 0.11]) == [0.10, 0.11]
    assert epoch(["g2", "g3"], [0.80, 0.81]) == [0.80, 0.81]


def test_the_reuse_key_survives_retrieve_and_reason():
    """The key has to reach Bind from the input table, four operators away."""
    from unittest.mock import MagicMock

    from armada.cosmic_workflow import build_reason_operator

    router = _gated_router([])
    retrieve = build_retrieve_operator(router, workflow_id="wf", dimensions=D)
    table = pa.table({"query_embedding": pa.array(
        [np.ones(D, dtype=np.float32)], type=retrieve.schema_in.field(0).type),
        "raw_text": pa.array(["a galaxy"], type=pa.large_utf8()),
        "reuse_key": pa.array([0.42], type=pa.float64())})

    after_retrieve = retrieve.fn(table)
    assert after_retrieve.column("reuse_key").to_pylist() == [0.42]

    chain = MagicMock()
    chain.execute.return_value = {"response": "r", "input_tokens": 1, "output_tokens": 1,
                                  "latency_ms": 1.0, "model_id": "m"}
    after_reason = build_reason_operator(chain, dimensions=D).fn(after_retrieve)

    assert after_reason.column("reuse_key").to_pylist() == [0.42]
