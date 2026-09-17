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
{workflow_id, task_description, response, cost_metadata, embedding_b64, rank}
(embedding_b64 base64-encoded float32 bytes, the same convention
EmbeddingService's Redis cache already uses) — kv_pairs stays the generic
{k, v} string pair the A2 schema declares, while v's content is what
ContextManager.store_context() actually needs. task_description/embedding_b64/
cost_metadata are all optional in the envelope: they're populated for real
(not placeholders) when Bind's input table carries the passthrough columns
below, and omitted when it doesn't (e.g. a standalone unit test).

`rank` is the rank that originated the context, stamped by Bind and read by
MemoryUpsert. It is likewise optional, and omitted means originated: after the
AllGather every rank stores every rank's contexts, so without it a rank could
not tell which of the contexts it holds are its own, and records_written would
stop meaning the same thing on the two arms of Experiment E.

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

Patterns come from COSMIC_PATTERNS rather than from canonical_operators(),
while the schemas still come from the canonical operators. Cosmic AI is
row-distributed with a replicated corpus: each rank owns its galaxies end to
end, so Embed and Retrieve are local and the only cross-rank requirement is
that every rank sees every rank's new contexts, which is an AllGather.
Table III of the proposal maps Embed to scatter-gather and Retrieve to reduce,
which describes a corpus-distributed layout where Retrieve's reduce merges
partial top-k lists. SELECTCOLLECTIVE is evaluated at plan-compilation time,
so the collective a typed operator maps to may depend on layout; the operator
abstraction is what stays fixed. See
docs/superpowers/specs/2026-09-10-expE-langchain-comparison-design.md 3.2-3.3.
"""

import base64
import json
import uuid
from typing import Optional

import numpy as np
import pyarrow as pa

from armada.operator import ArmadaOperator, ArmadaSequence
from armada.run_metrics import is_throttle_error
from cylon_armada.dag_compiler import CollectivePattern
from experiment.exp_a2_schema import canonical_operators


COSMIC_PATTERNS = {
    "Preprocess": CollectivePattern.Scatter,
    "Embed": CollectivePattern.PointToPoint,
    "Retrieve": CollectivePattern.PointToPoint,
    "Reason": CollectivePattern.PointToPoint,
    "Bind": CollectivePattern.PointToPoint,
    "MemoryUpsert": CollectivePattern.AllGather,
}


def _canonical(dimensions: int) -> dict:
    return {op.name: op for op in canonical_operators(dimensions)}


def _bedrock_call(call, metrics):
    """Run one Bedrock call, recording a throttle before letting it propagate.

    A throttled rank must still fail the run — a 429 invalidates that
    configuration's run and it is repeated — but the count has to survive the
    failure so the rank's record says why it failed rather than looking merely
    slow.
    """
    try:
        return call()
    except Exception as exc:
        if metrics is not None and is_throttle_error(exc):
            metrics.record_throttle()
        raise


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
        out = {
            "chunked_text": pa.array(chunked, type=chunked_type),
            "raw_text": pa.array(texts, type=pa.large_utf8()),
        }
        if "reuse_key" in table.column_names:
            out["reuse_key"] = table.column("reuse_key")
        return pa.table(out)

    return ArmadaOperator("Preprocess", COSMIC_PATTERNS["Preprocess"],
                          canon.schema_in, canon.schema_out, fn=fn)


def build_embed_operator(embedding_service, dimensions: int = 1024, metrics=None) -> ArmadaOperator:
    """PointToPoint: chunked_text -> embedding, via EmbeddingService.embed().

    Local under the row-distributed layout: each rank embeds its own rows and
    there is nothing to gather.
    """
    canon = _canonical(dimensions)["Embed"]
    list_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        texts = table.column(0).to_pylist()
        vectors = [np.asarray(_bedrock_call(lambda: embedding_service.embed(t), metrics)[0],
                              dtype=np.float32)
                   for t in texts]
        flat = np.concatenate(vectors) if vectors else np.array([], dtype=np.float32)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dimensions).cast(list_type)
        out = {"embedding": arr}
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        if "reuse_key" in table.column_names:
            out["reuse_key"] = table.column("reuse_key")
        return pa.table(out)

    return ArmadaOperator("Embed", COSMIC_PATTERNS["Embed"],
                          canon.schema_in, canon.schema_out, fn=fn)


def build_retrieve_operator(context_router, workflow_id: str, dimensions: int = 1024,
                            metrics=None, reuse_validator=None) -> ArmadaOperator:
    """PointToPoint: query_embedding -> ranked_docs (the single best match per row), via ContextRouter.find_similar().

    Local under the row-distributed layout: the lookup runs against the shared
    context store rather than reducing partial rankings across ranks.

    ranked_docs.doc carries the matched context's RESPONSE TEXT, empty on a
    miss, and is the hit discriminator Reason reads — the same field this
    operator keys its own reuse counter on, so the two cannot disagree.
    score stays informational. find_similar
    returns only {context_id, similarity}, so the id is resolved here: a
    context id is not a document, and passing one downstream is how Reason came
    to send Bedrock a bare UUID as its prompt.

    A match whose context cannot be fetched — evicted by TTL, or another rank's
    id not yet ingested — degrades to a miss, and only a resolved hit is
    recorded as reuse. Counting an unresolvable match would report a reuse rate
    that avoided no LLM call and that the cost numbers cannot support.

    The manager is reached through the router because it is the same instance
    ContextRouter.route() already uses, so there is no second source of truth
    for what a context id resolves to.

    `reuse_validator(query_key, candidate_key) -> bool` is an optional
    application-supplied validity policy. Cosine similarity says two prompts
    read alike; it does not say the cached answer is correct for this query.
    The runtime never interprets the key — the policy belongs to the workload,
    which is what keeps the five canonical operators domain-agnostic.

    When a policy is supplied, candidates are walked in cosine order and the
    first it admits is reused, so a slightly worse match that is valid beats a
    better one that is not. The policy fails closed: an unverifiable row — no
    query key, or a candidate stored before the policy existed — is a miss,
    never an unchecked hit. A row whose eligible candidates were all refused is
    counted as a gate rejection, distinct from a row that simply matched nothing.

    Without a policy only the single best match is considered, which is the
    ungated behaviour exactly. Walking further would be a silent improvement to
    the baseline — a first match that fails to resolve became a miss before and
    must still, or an ungated control run stops being comparable with every
    measurement taken before the gate existed.

    Under the row-distributed layout Embed is PointToPoint, so the pipeline
    hands this a single table: the rank's own rows. The list form is still
    accepted for a gather-shaped caller — that was the real input when Embed
    was a ScatterGather, and a rank handed nothing must still produce an empty
    result of the right shape rather than exit early, since every rank has to
    reach MemoryUpsert's collective. The empty branch therefore mirrors the
    populated branch column for column.
    """
    canon = _canonical(dimensions)["Retrieve"]
    struct_type = canon.schema_out.field(0).type

    def fn(table_or_tables) -> pa.Table:
        if isinstance(table_or_tables, list):
            if not table_or_tables:
                return pa.table({
                    "ranked_docs": pa.array([], type=struct_type),
                    "raw_text": pa.array([], type=pa.large_utf8()),
                    "query_embedding": pa.array([], type=canon.schema_in.field(0).type),
                })
            table = pa.concat_tables(table_or_tables)
        else:
            table = table_or_tables
        embeddings = table.column(0)
        query_keys = (table.column("reuse_key").to_pylist()
                      if "reuse_key" in table.column_names else [None] * len(embeddings))
        rows = []
        for i in range(len(embeddings)):
            vec = np.asarray(embeddings[i].values.to_numpy(zero_copy_only=False), dtype=np.float32)
            matches = context_router.find_similar(vec, workflow_id=workflow_id)
            candidates = matches if reuse_validator is not None else matches[:1]
            accepted, refused = None, False
            for match in candidates:
                stored = context_router.context_manager.get_context(match["context_id"])
                response = stored.get("response") if stored else None
                if not response:
                    continue
                if reuse_validator is not None:
                    candidate_key = stored.get("reuse_key")
                    if query_keys[i] is None or candidate_key is None:
                        refused = True
                        continue
                    if not reuse_validator(query_keys[i], candidate_key):
                        refused = True
                        continue
                accepted = {"doc": response, "score": float(match["similarity"])}
                break

            if metrics is not None:
                metrics.record_retrieval(accepted is not None)
                if accepted is None and refused:
                    metrics.record_gate_rejection()
            rows.append(accepted if accepted else {"doc": "", "score": 0.0})
        arr = pa.array(rows, type=struct_type)
        out = {"ranked_docs": arr}
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        if "reuse_key" in table.column_names:
            out["reuse_key"] = table.column("reuse_key")
        # Retrieve's own canonical output (ranked_docs) doesn't carry the
        # embedding forward — self-forward it under a fixed name so Bind can
        # still recover it, regardless of what the producer (Embed) named it.
        out["query_embedding"] = embeddings
        return pa.table(out)

    return ArmadaOperator("Retrieve", COSMIC_PATTERNS["Retrieve"],
                          canon.schema_in, canon.schema_out, fn=fn)


def build_reason_operator(chain_executor, dimensions: int = 1024, metrics=None) -> ArmadaOperator:
    """PointToPoint: context (best retrieved doc) -> response, via the shared ChainExecutor/Bedrock Runnable.

    Reuse-or-call, per row. A non-empty `doc` means Retrieve resolved a
    reusable context, so its response is returned and Bedrock is not called;
    otherwise the row's `raw_text` is the prompt. This is the mechanism H4's
    cost claim rests on: while every row called Bedrock unconditionally,
    llm_calls equalled the galaxy count at every reuse rate and the claim could
    not be measured.

    The discriminator is `doc`, the same field Retrieve keys its own hit
    counter on, and deliberately not `score`. Keying the two on different
    fields lets them disagree when a resolved match carries similarity 0.0:
    Retrieve would count reuse while Reason called and stored anyway, so one
    galaxy would land in both cache_hits and records_written and the gate's
    written-plus-hits total would exceed the shard.

    A reused row reports zero tokens and no latency, so cost accounting does
    not bill a call that never happened, and `reused` is emitted so Bind can
    leave hits out of storage.

    A row with neither a resolved hit nor `raw_text` falls back to the empty
    prompt, which keeps a bare single-column caller working.
    """
    canon = _canonical(dimensions)["Reason"]
    response_type = canon.schema_out.field(0).type

    def fn(table: pa.Table) -> pa.Table:
        contexts = table.column(0).to_pylist()
        prompts = (table.column("raw_text").to_pylist()
                   if "raw_text" in table.column_names else [""] * len(contexts))
        results, reused = [], []
        for ctx, prompt in zip(contexts, prompts):
            if ctx and ctx.get("doc"):
                results.append({"response": ctx["doc"], "input_tokens": 0, "output_tokens": 0,
                                "latency_ms": 0.0, "model_id": ""})
                reused.append(True)
                continue
            result = _bedrock_call(lambda: chain_executor.execute(prompt), metrics)
            if metrics is not None:
                metrics.record_llm_call(result.get("latency_ms"))
            results.append(result)
            reused.append(False)
        out = {"response": pa.array([r["response"] for r in results], type=response_type)}
        out["reused"] = pa.array(reused, type=pa.bool_())
        if "raw_text" in table.column_names:
            out["raw_text"] = table.column("raw_text")
        if "query_embedding" in table.column_names:
            out["query_embedding"] = table.column("query_embedding")
        if "reuse_key" in table.column_names:
            out["reuse_key"] = table.column("reuse_key")
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

    return ArmadaOperator("Reason", COSMIC_PATTERNS["Reason"],
                          canon.schema_in, canon.schema_out, fn=fn)


def build_memory_upsert_operator(context_manager, dimensions: int = 1024,
                                 metrics=None, rank=None) -> ArmadaOperator:
    """AllGather: kv_pairs -> ack, via ContextManager.store_context(). See module docstring for the v envelope.

    Every rank publishes contexts for its own galaxies, and reuse requires each
    rank to see all of them — so this is all-to-all, not root-to-all. The
    executor moves before computing, so the table handed to fn already holds
    every rank's kv_pairs and this rank stores all of them.

    `rank` is this rank's identity, used only to split the store count: an
    envelope stamped with a different rank is counted as ingested rather than
    written, so records_written stays "contexts this rank originated" and keeps
    the same meaning on an arm that replicates and an arm that does not. An
    envelope carrying no rank key counts as originated, which is the
    single-rank case and every caller that passes no rank.

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
                    reuse_key=envelope.get("reuse_key"),
                )
                if metrics is not None:
                    origin = envelope.get("rank")
                    if origin is None or origin == rank:
                        metrics.record_store()
                    else:
                        metrics.record_ingest()
                acks.append(True)
            except Exception:
                if metrics is not None:
                    metrics.record_store_failure()
                acks.append(False)
        return pa.table({"ack": acks}, schema=canon.schema_out)

    return ArmadaOperator("MemoryUpsert", COSMIC_PATTERNS["MemoryUpsert"],
                          canon.schema_in, canon.schema_out, fn=fn)


def build_bind_operator(workflow_id: str, rank=None) -> ArmadaOperator:
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

    A row Reason marked `reused` is deliberately NOT bound for storage: it
    reused an existing context, and storing a duplicate would grow the store
    with copies and slow every later similarity search. A table without the
    `reused` column binds every row, which keeps callers predating the reuse
    path working.

    The application's per-row validity key rides in as a `reuse_key` passthrough
    column and is written into each envelope. It travels with the rows rather
    than arriving as a positional list because this fn is invoked once per
    EPOCH: a list indexed by position would restamp the first epoch's keys onto
    every later epoch, storing all but the first few contexts under the wrong
    key. Absent the column the field is omitted and reuse stays ungated.

    `rank` stamps the originating rank into the envelope so MemoryUpsert can
    tell, after the collective has replicated every rank's contexts to every
    rank, which ones this rank actually created. Left None the key is omitted
    and the context counts as originated wherever it is stored, which is the
    single-rank case.
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

        reused = (table.column("reused").to_pylist()
                  if "reused" in table.column_names else [False] * n)
        row_keys = (table.column("reuse_key").to_pylist()
                    if "reuse_key" in table.column_names else [None] * n)

        rows = []
        for i in range(n):
            if reused[i]:
                continue
            envelope = {"workflow_id": workflow_id, "response": responses[i]}
            if rank is not None:
                envelope["rank"] = rank
            if row_keys[i] is not None:
                envelope["reuse_key"] = row_keys[i]
            if task_descriptions is not None:
                envelope["task_description"] = task_descriptions[i]
            if embeddings_b64 is not None:
                envelope["embedding_b64"] = embeddings_b64[i]
            if cost_list is not None:
                envelope["cost_metadata"] = cost_list[i]
            rows.append({"k": str(uuid.uuid4()), "v": json.dumps(envelope)})

        arr = pa.array(rows, type=kv_type)
        return pa.table({"kv_pairs": arr}, schema=schema_out)

    return ArmadaOperator("Bind", COSMIC_PATTERNS["Bind"], schema_in, schema_out, fn=fn)


def build_cosmic_workflow(
    embedding_service,
    context_router,
    context_manager,
    chain_executor,
    workflow_id: str,
    dimensions: int = 1024,
    max_chars: Optional[int] = None,
    metrics=None,
    rank=None,
    reuse_validator=None,
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
        | build_embed_operator(embedding_service, dimensions=dimensions, metrics=metrics)
        | build_retrieve_operator(context_router, workflow_id=workflow_id, dimensions=dimensions,
                                  metrics=metrics, reuse_validator=reuse_validator)
        | build_reason_operator(chain_executor, dimensions=dimensions, metrics=metrics)
        | build_bind_operator(workflow_id, rank=rank)
        | build_memory_upsert_operator(context_manager, dimensions=dimensions, metrics=metrics,
                                       rank=rank)
    )