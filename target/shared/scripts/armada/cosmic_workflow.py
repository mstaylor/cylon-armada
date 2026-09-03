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
ContextManager.store_context() actually needs. task_description/embedding_b64/
cost_metadata are all optional in the envelope: they're populated for real
(not placeholders) when Bind's input table carries the passthrough columns
below, and omitted when it doesn't (e.g. a standalone unit test).

Two things beyond the five canonical operators' own schema_in/schema_out:

1. **Column access is by position (table.column(0)), not by the consumer's
   declared field name.** compile_workflow()'s A2 check compares field
   *types* between schema_out(producer) and schema_in(consumer), not field
   *names* — e.g. Embed's schema_out names its column "embedding" while
   Retrieve's schema_in names the same-typed column "query_embedding"; A2
   accepts that edge as compatible (it is, by type), but a runtime table
   only ever has the producer's column names. Reading by declared name
   crashes the moment two operators are actually chained for real (confirmed
   empirically — this bug existed, unexercised, before this fix). Reading by
   position matches what A2 itself checks.
2. **Passthrough columns carry task_description, the real embedding, and LLM
   cost metadata forward** through Retrieve and Reason — data those
   operators' own canonical single-field schemas don't declare, but which
   downstream Bind needs to build a real (non-placeholder) MemoryUpsert
   envelope: `raw_text` (task_description, from Preprocess), `query_embedding`
   (Embed's own output, self-forwarded through Retrieve since Retrieve's
   canonical output doesn't include it), `cost_metadata_json` (LLM token
   usage, computed fresh by Reason from ChainExecutor.execute's return, which
   Reason's canonical schema_out — just `response` — would otherwise
   discard). Each operator forwards whichever of these it received, so a
   context stored via Bind's envelope really is retrievable — this is the
   fix for the interim limitation flagged when Bind was first added; see
   docs/superpowers/specs/2026-08-07-expE-cosmic-ai-e2e-design.md and this
   plan's Task 6 section.

Every extra column is additive and optional at every stage (checked via
`"col" in table.column_names`), so every operator's fn still works standalone
in its own unit test against a bare single-column table.

build_bind_operator is a sixth, non-canonical operator: Reason's schema_out
(response: large_utf8) and MemoryUpsert's schema_in (kv_pairs: struct<k,v>)
are genuinely incompatible Arrow types — compile_workflow() rejects chaining
them directly with a real SchemaMismatchError, confirmed empirically, not
assumed. The five canonical A2 operators (exp_a2_schema.py) don't include
anything that builds a keyed envelope from a bare response, so Bind supplies
that connective step.
"""

import base64
import json
import uuid
from typing import Optional

import numpy as np
import pyarrow as pa

from armada.operator import ArmadaOperator, ArmadaSequence
from cylon_armada.dag_compiler import CollectivePattern
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
    chunked_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        texts = table.column(0).to_pylist()
        chunked = texts if max_chars is None else [t[:max_chars] for t in texts]
        return pa.table({
            "chunked_text": pa.array(chunked, type=chunked_type),
            "raw_text": pa.array(texts, type=pa.large_utf8()),
        })

    return ArmadaOperator("Preprocess", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_embed_operator(embedding_service, dimensions: int = 1024) -> ArmadaOperator:
    """ScatterGather: chunked_text -> embedding, via EmbeddingService.embed()."""
    canon = _canonical(dimensions)["Embed"]
    list_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        texts = table.column(0).to_pylist()
        vectors = [np.asarray(embedding_service.embed(t)[0], dtype=np.float32) for t in texts]
        flat = np.concatenate(vectors) if vectors else np.array([], dtype=np.float32)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dimensions).cast(list_type)
        out = {"embedding": arr}
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        return pa.table(out)

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
        embeddings = table.column(0)
        rows = []
        for i in range(len(embeddings)):
            vec = np.asarray(embeddings[i].values.to_numpy(zero_copy_only=False), dtype=np.float32)
            matches = context_router.find_similar(vec, workflow_id=workflow_id)
            best = matches[0] if matches else {"context_id": "", "similarity": 0.0}
            rows.append({"doc": best["context_id"], "score": float(best["similarity"])})
        arr = pa.array(rows, type=struct_type)
        out = {"ranked_docs": arr}
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        # Retrieve's own canonical output (ranked_docs) doesn't carry the
        # embedding forward — self-forward it under a fixed name so Bind can
        # still recover it, regardless of what the producer (Embed) named it.
        out["query_embedding"] = embeddings
        return pa.table(out)

    return ArmadaOperator("Retrieve", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_reason_operator(chain_executor, dimensions: int = 1024) -> ArmadaOperator:
    """PointToPoint: context (best retrieved doc) -> response, via the shared ChainExecutor/Bedrock Runnable."""
    canon = _canonical(dimensions)["Reason"]
    response_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        contexts = table.column(0).to_pylist()
        results = [chain_executor.execute(ctx.get("doc") or "") for ctx in contexts]
        out = {"response": pa.array([r["response"] for r in results], type=response_type)}
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        if "query_embedding" in table.column_names:
            out["query_embedding"] = table.column("query_embedding")
        # Reason's own canonical output (response) doesn't carry LLM token
        # usage forward — ChainExecutor.execute() returns it, so capture it
        # here (as JSON text, Arrow has no generic dict type) or it's lost.
        out["cost_metadata_json"] = pa.array([
            json.dumps({
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "latency_ms": r.get("latency_ms", 0.0),
                "model_id": r.get("model_id", ""),
            }) for r in results
        ], type=pa.large_utf8())
        return pa.table(out)

    return ArmadaOperator("Reason", canon.pattern, canon.schema_in, canon.schema_out, fn=fn)


def build_memory_upsert_operator(context_manager, dimensions: int = 1024) -> ArmadaOperator:
    """Broadcast: kv_pairs -> ack, via ContextManager.store_context(). See module docstring for the v envelope.

    workflow_id and response are required in the envelope; task_description
    and embedding_b64 are optional (default "" and an all-zero vector) since
    build_bind_operator's glued envelope can't supply them — see its
    docstring and the module docstring for why.
    """
    canon = _canonical(dimensions)["MemoryUpsert"]

    def fn(table: pa.Table) -> pa.Table:
        kv_pairs = table.column("kv_pairs").to_pylist()
        acks = []
        for kv in kv_pairs:
            try:
                envelope = json.loads(kv["v"])
                embedding_b64 = envelope.get("embedding_b64")
                embedding = (
                    np.frombuffer(base64.b64decode(embedding_b64), dtype=np.float32)
                    if embedding_b64 else np.zeros(dimensions, dtype=np.float32)
                )
                context_manager.store_context(
                    workflow_id=envelope["workflow_id"],
                    task_description=envelope.get("task_description", ""),
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


def build_bind_operator(workflow_id: str) -> ArmadaOperator:
    """PointToPoint glue: response -> kv_pairs. NOT one of the five canonical A2 operators.

    Reason's schema_out (response: large_utf8) and MemoryUpsert's schema_in
    (kv_pairs: struct<k,v>) are genuinely incompatible Arrow types — nothing
    in the five canonical operators builds a keyed envelope from a bare
    response, so chaining Reason directly into MemoryUpsert raises a real
    SchemaMismatchError at lower() time (confirmed empirically). Bind
    supplies that missing connective step: generates a context_id and packs
    the envelope.

    Declared schema_in is still just `response` (the A2-compatible contract
    with Reason), but fn also reads Reason's passthrough columns —
    `raw_text` (task_description), `query_embedding` (the real embedding),
    `cost_metadata_json` — when present, so the envelope going into
    MemoryUpsert is real, not a placeholder. Each is independently optional:
    a bare single-column `response` table (e.g. a standalone unit test)
    still produces a valid, minimal envelope.
    """
    schema_in = pa.schema([pa.field("response", pa.large_utf8())])
    kv_type = pa.struct([pa.field("k", pa.large_utf8()), pa.field("v", pa.large_utf8())])
    schema_out = pa.schema([pa.field("kv_pairs", kv_type)])

    def fn(table: pa.Table) -> pa.Table:
        responses = table.column(0).to_pylist()
        n = len(responses)

        task_descriptions = table.column("raw_text").to_pylist() if "raw_text" in table.column_names else None

        embeddings_b64 = None
        if "query_embedding" in table.column_names:
            emb_col = table.column("query_embedding")
            embeddings_b64 = [
                base64.b64encode(
                    np.asarray(emb_col[i].values.to_numpy(zero_copy_only=False), dtype=np.float32).tobytes()
                ).decode()
                for i in range(n)
            ]

        cost_list = None
        if "cost_metadata_json" in table.column_names:
            cost_list = [json.loads(c) for c in table.column("cost_metadata_json").to_pylist()]

        rows = []
        for i in range(n):
            envelope = {"workflow_id": workflow_id, "response": responses[i]}
            if task_descriptions is not None:
                envelope["task_description"] = task_descriptions[i]
            if embeddings_b64 is not None:
                envelope["embedding_b64"] = embeddings_b64[i]
            if cost_list is not None:
                envelope["cost_metadata"] = cost_list[i]
            rows.append({"k": str(uuid.uuid4()), "v": json.dumps(envelope)})

        arr = pa.array(rows, type=kv_type)
        return pa.table({"kv_pairs": arr}, schema=schema_out)

    return ArmadaOperator("Bind", CollectivePattern.PointToPoint, schema_in, schema_out, fn=fn)


def build_cosmic_workflow(
    embedding_service,
    context_router,
    context_manager,
    chain_executor,
    workflow_id: str,
    dimensions: int = 1024,
    max_chars: Optional[int] = None,
) -> ArmadaSequence:
    """Preprocess | Embed | Retrieve | Reason | Bind | MemoryUpsert, wired to the injected services.

    Deviations from the plan's `build_cosmic_workflow(config) -> ArmadaSequence` sketch:
    (1) takes the four live service instances plus workflow_id directly (dependency
    injection, CLAUDE.md's Dependency Inversion rule) rather than constructing
    real Bedrock/Redis/DynamoDB clients internally from a config object — a
    plain BedrockConfig alone can't supply live service instances, and Task 6's
    own e2e test requires mocked Bedrock. (2) Bind is inserted between Reason
    and MemoryUpsert — see build_bind_operator's docstring for why the five
    canonical operators alone don't chain.
    """
    return (
        build_preprocess_operator(max_chars=max_chars, dimensions=dimensions)
        | build_embed_operator(embedding_service, dimensions=dimensions)
        | build_retrieve_operator(context_router, workflow_id=workflow_id, dimensions=dimensions)
        | build_reason_operator(chain_executor, dimensions=dimensions)
        | build_bind_operator(workflow_id)
        | build_memory_upsert_operator(context_manager, dimensions=dimensions)
    )