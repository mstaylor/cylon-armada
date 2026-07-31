"""Experiment A2 driver: validate Algorithm 1 (AgentDAGCompiler) schema enforcement.

Runs the proposal's three A2 cases through the compiled cylon_armada.dag_compiler:
  1. compatible edges -> zero-copy (the per-edge matrix),
  2. incompatible edge -> SchemaMismatchError at compile time,
  3. appended nullable column -> tolerated.

Fails loudly if the compiled compiler is unavailable; it never falls back to a
string comparison, since validating *real* schema enforcement is the point.
"""

import logging

import pyarrow as pa

logger = logging.getLogger(__name__)

try:
    from cylon_armada.dag_compiler import (
        AgentOperator, WorkflowDAG, CollectivePattern,
        compile_workflow, check_edges, SchemaMismatchError, TransferMode,
    )
    A2_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    A2_AVAILABLE = False
    _IMPORT_ERROR = _e


def _require():
    if not A2_AVAILABLE:
        raise RuntimeError(
            "cylon_armada.dag_compiler is not importable (%r). Experiment A2 "
            "validates the compiled AgentDAGCompiler and will not fall back to a "
            "string comparison. Build the extension / rebuild the image." % (_IMPORT_ERROR,)
        )


def _embedding(d, name):
    return pa.schema([pa.field(name, pa.list_(pa.field("item", pa.float32()), d))])


def _text(name):
    return pa.schema([pa.field(name, pa.large_utf8())])


def _struct(name, fields):
    return pa.schema([pa.field(name, pa.struct(fields))])


def _bool(name):
    return pa.schema([pa.field(name, pa.bool_())])


def canonical_operators(d=1024):
    """The five agentic operators as real pyarrow schemas (proposal contracts)."""
    _require()
    docs = pa.struct([pa.field("doc", pa.large_utf8()), pa.field("score", pa.float32())])
    ctx = pa.struct([pa.field("doc", pa.large_utf8()), pa.field("score", pa.float32())])
    kv = pa.struct([pa.field("k", pa.large_utf8()), pa.field("v", pa.large_utf8())])
    return [
        AgentOperator("Preprocess", CollectivePattern.Scatter,
                      _text("raw_text"), _text("chunked_text")),
        AgentOperator("Embed", CollectivePattern.ScatterGather,
                      _text("chunked_text"), _embedding(d, "embedding")),
        AgentOperator("Retrieve", CollectivePattern.Reduce,
                      _embedding(d, "query_embedding"), _struct("ranked_docs", list(docs))),
        AgentOperator("Reason", CollectivePattern.PointToPoint,
                      _struct("context", list(ctx)), _text("response")),
        AgentOperator("MemoryUpsert", CollectivePattern.Broadcast,
                      _struct("kv_pairs", list(kv)), _bool("ack")),
    ]


def pipeline_edges():
    return [("Preprocess", "Embed"), ("Embed", "Retrieve"),
            ("Retrieve", "Reason"), ("Reason", "MemoryUpsert")]


def _payload_class(schema):
    t = schema.field(0).type
    if pa.types.is_fixed_size_list(t):
        return "dense"
    if pa.types.is_large_string(t) or pa.types.is_string(t):
        return "variable"
    return "nested"


def run_a2(d=1024):
    """Return A2 CSV rows across the three cases."""
    _require()
    ops = {op.name: op for op in canonical_operators(d)}
    rows = []

    # Case 1: per-edge compatibility matrix over the canonical pipeline.
    dag = WorkflowDAG(list(ops.values()), pipeline_edges())
    for ep in check_edges(dag):
        prod = ops[ep.producer]
        rows.append({
            "edge": f"{ep.producer}->{ep.consumer}",
            "schema_out": str(prod.schema_out.field(0).type),
            "schema_in": str(ops[ep.consumer].schema_in.field(0).type),
            "arrow_compatible": ep.mode == TransferMode.ZeroCopy,
            "zero_copy_eligible": ep.mode == TransferMode.ZeroCopy
                                  and _payload_class(prod.schema_out) == "dense",
            "payload_class": _payload_class(prod.schema_out),
            "case": "compatible",
            "result": "zero_copy" if ep.mode == TransferMode.ZeroCopy else "convert",
        })

    # Case 2: an incompatible edge must be rejected at compile time.
    a = AgentOperator("A", CollectivePattern.PointToPoint,
                      _text("x"), _text("response"))
    b = AgentOperator("B", CollectivePattern.Broadcast,
                      _embedding(d, "needs_embedding"), _bool("ack"))
    incompat = WorkflowDAG([a, b], [("A", "B")])
    try:
        compile_workflow(incompat)
        result = "NOT_REJECTED"
    except SchemaMismatchError:
        result = "rejected"
    rows.append({
        "edge": "A->B", "schema_out": "large_utf8",
        "schema_in": f"fixed_size_list<float>[{d}]",
        "arrow_compatible": False, "zero_copy_eligible": False,
        "payload_class": "variable", "case": "incompatible", "result": result,
    })

    # Case 3: an appended nullable column must be tolerated.
    prod_out = pa.schema([
        pa.field("embedding", pa.list_(pa.field("item", pa.float32()), d)),
        pa.field("provenance", pa.large_utf8(), nullable=True),
    ])
    e1 = AgentOperator("E1", CollectivePattern.ScatterGather, _embedding(d, "x"), prod_out)
    e2 = AgentOperator("E2", CollectivePattern.Reduce,
                       _embedding(d, "embedding"), _text("y"))
    evo = WorkflowDAG([e1, e2], [("E1", "E2")])
    try:
        compile_workflow(evo)
        result = "tolerated"
    except SchemaMismatchError:
        result = "REJECTED"
    rows.append({
        "edge": "E1->E2 (+nullable)", "schema_out": f"fixed_size_list<float>[{d}]+nullable",
        "schema_in": f"fixed_size_list<float>[{d}]",
        "arrow_compatible": True, "zero_copy_eligible": True,
        "payload_class": "dense", "case": "evolution", "result": result,
    })

    logger.info("A2: %d rows across cases %s", len(rows), {r["case"] for r in rows})
    return rows
