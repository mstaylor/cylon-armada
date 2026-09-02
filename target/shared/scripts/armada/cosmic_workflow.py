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

"""The five Cosmic AI ArmadaOperators (SP1 Task 5).

Each builder wires an injected service (EmbeddingService, ContextRouter,
ContextManager, ChainExecutor) into an ArmadaOperator whose pattern and
schema_in/schema_out come from experiment/exp_a2_schema.py's
canonical_operators() — the same contracts Experiment A2 validates — so this
workflow runs under the identical schema-compatibility check, not a
parallel/duplicated one.

Every operator's fn takes and returns a single-column pyarrow.Table matching
its schema. "context" and "kv_pairs" are single struct<doc,score> /
struct<k,v> per row (not lists) per the A2 contract, so Reason and
MemoryUpsert each act on exactly one upstream result per row.

MemoryUpsert's kv_pairs.v carries a JSON envelope
{workflow_id, task_description, response, cost_metadata, embedding_b64}
(embedding_b64 base64-encoded float32 bytes, the same convention
EmbeddingService's Redis cache already uses) — kv_pairs stays the generic
{k, v} string pair the A2 schema declares, while v's content is what
ContextManager.store_context() actually needs.
"""

import base64
import json
from typing import Optional

import numpy as np
import pyarrow as pa

from armada.operator import ArmadaOperator, ArmadaSequence
from experiment.exp_a2_schema import canonical_operators


def _canonical(dimensions: int) -> dict:
    return {op.name: op for op in canonical_operators(dimensions)}


def build_preprocess_operator(max_chars: Optional[int] = None, dimensions: int = 1024) -> ArmadaOperator:
    """Scatter: raw_text -> chunked_text. Truncates each row to max_chars (no-op if None).

    Single table in, single table out, like every other operator in this
    module: ArmadaExecutor now moves data (scatter) *before* calling fn for a
    Scatter pattern, so by the time fn runs, this rank already has its own
    single shard — the same uniform shape Runnable.invoke() expects, which is
    what makes world_size==1 execution equal plain seq.invoke() chaining.
    """
    canon = _canonical(dimensions)["Preprocess"]

    def fn(table: pa.Table) -> pa.Table:
        texts = table.column("raw_text").to_pylist()
        if max_chars is not None:
            texts = [t[:max_chars] for t in texts]
        return pa.table({"chunked_text": texts}, schema=canon.schema_out)

    return ArmadaOperator("Preprocess", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_embed_operator(embedding_service, dimensions: int = 1024) -> ArmadaOperator:
    """ScatterGather: chunked_text -> embedding, via EmbeddingService.embed()."""
    canon = _canonical(dimensions)["Embed"]
    list_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        texts = table.column("chunked_text").to_pylist()
        vectors = [np.asarray(embedding_service.embed(t)[0], dtype=np.float32) for t in texts]
        flat = np.concatenate(vectors) if vectors else np.array([], dtype=np.float32)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dimensions).cast(list_type)
        return pa.table({"embedding": arr}, schema=canon.schema_out)

    return ArmadaOperator("Embed", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_retrieve_operator(context_router, workflow_id: str, dimensions: int = 1024) -> ArmadaOperator:
    """Reduce: query_embedding -> ranked_docs (the single best match per row), via ContextRouter.find_similar().

    Retrieve is the operator right after Embed's ScatterGather, so its real
    input is bridge.gather()'s return — a list of N single-row per-rank
    tables, not one table — while a standalone unit test still hands it a
    single table directly. Accepts either.
    """
    canon = _canonical(dimensions)["Retrieve"]
    struct_type = canon.schema_out.field(0).type

    def fn(table_or_tables) -> pa.Table:
        table = pa.concat_tables(table_or_tables) if isinstance(table_or_tables, list) else table_or_tables
        embeddings = table.column("query_embedding")
        rows = []
        for i in range(len(embeddings)):
            vec = np.asarray(embeddings[i].values.to_numpy(zero_copy_only=False), dtype=np.float32)
            matches = context_router.find_similar(vec, workflow_id=workflow_id)
            best = matches[0] if matches else {"context_id": "", "similarity": 0.0}
            rows.append({"doc": best["context_id"], "score": float(best["similarity"])})
        arr = pa.array(rows, type=struct_type)
        return pa.table({"ranked_docs": arr}, schema=canon.schema_out)

    return ArmadaOperator("Retrieve", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_reason_operator(chain_executor, dimensions: int = 1024) -> ArmadaOperator:
    """PointToPoint: context (best retrieved doc) -> response, via the shared ChainExecutor/Bedrock Runnable."""
    canon = _canonical(dimensions)["Reason"]

    def fn(table: pa.Table) -> pa.Table:
        contexts = table.column("context").to_pylist()
        responses = [chain_executor.execute(ctx.get("doc") or "")["response"] for ctx in contexts]
        return pa.table({"response": responses}, schema=canon.schema_out)

    return ArmadaOperator("Reason", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_memory_upsert_operator(context_manager, dimensions: int = 1024) -> ArmadaOperator:
    """Broadcast: kv_pairs -> ack, via ContextManager.store_context(). See module docstring for the v envelope."""
    canon = _canonical(dimensions)["MemoryUpsert"]

    def fn(table: pa.Table) -> pa.Table:
        kv_pairs = table.column("kv_pairs").to_pylist()
        acks = []
        for kv in kv_pairs:
            try:
                envelope = json.loads(kv["v"])
                embedding = np.frombuffer(base64.b64decode(envelope["embedding_b64"]), dtype=np.float32)
                context_manager.store_context(
                    workflow_id=envelope["workflow_id"],
                    task_description=envelope["task_description"],
                    embedding=embedding,
                    response=envelope["response"],
                    cost_metadata=envelope.get("cost_metadata", {}),
                    context_id=kv["k"] or None,
                )
                acks.append(True)
            except Exception:
                acks.append(False)
        return pa.table({"ack": acks}, schema=canon.schema_out)

    return ArmadaOperator("MemoryUpsert", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_cosmic_workflow(
    embedding_service,
    context_router,
    context_manager,
    chain_executor,
    workflow_id: str,
    dimensions: int = 1024,
    max_chars: Optional[int] = None,
) -> ArmadaSequence:
    """Preprocess | Embed | Retrieve | Reason | MemoryUpsert, wired to the injected services.

    Deviation from the plan's `build_cosmic_workflow(config) -> ArmadaSequence` sketch:
    takes the four live service instances plus workflow_id directly (dependency
    injection, CLAUDE.md's Dependency Inversion rule) rather than constructing
    real Bedrock/Redis/DynamoDB clients internally from a config object — a
    plain BedrockConfig alone can't supply live service instances, and Task 6's
    own e2e test requires mocked Bedrock.
    """
    return (
        build_preprocess_operator(max_chars=max_chars, dimensions=dimensions)
        | build_embed_operator(embedding_service, dimensions=dimensions)
        | build_retrieve_operator(context_router, workflow_id=workflow_id, dimensions=dimensions)
        | build_reason_operator(chain_executor, dimensions=dimensions)
        | build_memory_upsert_operator(context_manager, dimensions=dimensions)
    )