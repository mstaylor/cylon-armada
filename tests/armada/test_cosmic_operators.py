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
    assert op.pattern == CollectivePattern.ScatterGather

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

    op = build_retrieve_operator(context_router, workflow_id="wf-1", dimensions=D)
    assert op.pattern == CollectivePattern.Reduce

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
    assert row == {"doc": "c1", "score": pytest.approx(0.9)}
    assert result.column("query_embedding").to_pylist()[0] == query.to_pylist()[0]
    _, kwargs = context_router.find_similar.call_args
    assert kwargs["workflow_id"] == "wf-1"


def test_retrieve_concatenates_a_list_of_gathered_tables():
    """After Embed's real ScatterGather, Retrieve's input is bridge.gather()'s
    return: a list of N single-row per-rank tables, not one table."""
    context_router = MagicMock()
    context_router.find_similar.return_value = [{"context_id": "c1", "similarity": 0.5}]

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

def test_reason_calls_chain_executor_with_top_doc_text():
    chain_executor = MagicMock()
    chain_executor.execute.return_value = {"response": "the answer"}

    op = build_reason_operator(chain_executor, dimensions=D)
    assert op.pattern == CollectivePattern.PointToPoint

    struct_type = op.schema_in.field(0).type
    ctx = pa.array([{"doc": "some retrieved context", "score": 0.9}], type=struct_type)
    table = pa.table({"context": ctx}, schema=op.schema_in)
    result = op.fn(table)

    # response (canonical) is column 0; cost_metadata_json is an additive
    # passthrough (LLM token usage ChainExecutor.execute() returns, which
    # response alone would otherwise discard).
    assert result.schema.field(0) == op.schema_out.field(0)
    assert result.column("response").to_pylist() == ["the answer"]
    chain_executor.execute.assert_called_once_with("some retrieved context")


def test_reason_carries_cost_metadata_forward_as_passthrough():
    chain_executor = MagicMock()
    chain_executor.execute.return_value = {
        "response": "the answer", "input_tokens": 12, "output_tokens": 34,
        "latency_ms": 5.5, "model_id": "mock-model",
    }
    op = build_reason_operator(chain_executor, dimensions=D)
    struct_type = op.schema_in.field(0).type
    ctx = pa.array([{"doc": "x", "score": 0.9}], type=struct_type)
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
    assert op.pattern == CollectivePattern.Broadcast

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