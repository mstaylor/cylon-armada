import pyarrow as pa
import pytest

dag_mod = pytest.importorskip("cylon_armada.dag_compiler")
from cylon_armada.dag_compiler import (
    CollectivePattern, TransferMode, AgentOperator, WorkflowDAG,
    SchemaMismatchError, compile_workflow, check_edges,
)


def _emb_schema(d, name="embedding"):
    return pa.schema([pa.field(name, pa.list_(pa.field("item", pa.float32()), d))])


def test_skeleton_imports_and_empty_dag_compiles():
    dag = WorkflowDAG(operators=[], edges=[])
    plan = compile_workflow(dag)
    assert list(plan.edges) == []
    assert dict(plan.assignments) == {}
    assert check_edges(dag) == []


def test_operator_and_dag_construct():
    op = AgentOperator("Embed", CollectivePattern.ScatterGather,
                       _emb_schema(8, "chunked"), _emb_schema(8, "embedding"))
    assert op.name == "Embed"
    assert op.pattern == CollectivePattern.ScatterGather


def _op(name, pattern, sin, sout):
    return AgentOperator(name, pattern, sin, sout)


def test_agent_operator_rejects_non_schema_type():
    with pytest.raises(TypeError):
        AgentOperator("Bad", CollectivePattern.Broadcast,
                      "not_a_schema", _emb_schema(8, "y"))


def test_check_edges_compatible_names_differ_is_zerocopy():
    # Embed outputs "embedding"; Retrieve consumes "query_embedding": same type, different name.
    embed = _op("Embed", CollectivePattern.ScatterGather,
                _emb_schema(8, "chunked"), _emb_schema(8, "embedding"))
    retrieve = _op("Retrieve", CollectivePattern.Reduce,
                   _emb_schema(8, "query_embedding"),
                   pa.schema([pa.field("ranked", pa.float32())]))
    dag = WorkflowDAG([embed, retrieve], [("Embed", "Retrieve")])
    edges = check_edges(dag)
    assert len(edges) == 1
    assert edges[0].mode == TransferMode.ZeroCopy


def test_check_edges_type_mismatch_is_convert_with_reason():
    a = _op("A", CollectivePattern.PointToPoint,
            pa.schema([pa.field("x", pa.int32())]),
            pa.schema([pa.field("out", pa.large_utf8())]))     # produces large_utf8
    b = _op("B", CollectivePattern.Broadcast,
            _emb_schema(8, "needs_embedding"),                 # requires FixedSizeList
            pa.schema([pa.field("ack", pa.bool_())]))
    dag = WorkflowDAG([a, b], [("A", "B")])
    edges = check_edges(dag)
    assert edges[0].mode == TransferMode.Convert
    assert "type mismatch" in edges[0].note


def test_compile_rejects_type_mismatch():
    a = _op("A", CollectivePattern.PointToPoint,
            pa.schema([pa.field("x", pa.int32())]),
            pa.schema([pa.field("out", pa.large_utf8())]))
    b = _op("B", CollectivePattern.Broadcast,
            _emb_schema(8, "needs_embedding"),
            pa.schema([pa.field("ack", pa.bool_())]))
    dag = WorkflowDAG([a, b], [("A", "B")])
    with pytest.raises(SchemaMismatchError) as exc:
        compile_workflow(dag)
    assert "A -> B" in str(exc.value)


def test_compile_tolerates_appended_nullable_column():
    # Producer appends a nullable column the consumer does not require -> still compiles.
    prod_out = pa.schema([
        pa.field("embedding", pa.list_(pa.field("item", pa.float32()), 8)),
        pa.field("provenance", pa.large_utf8(), nullable=True),   # appended nullable
    ])
    cons_in = _emb_schema(8, "embedding")
    a = _op("A", CollectivePattern.ScatterGather, _emb_schema(8, "x"), prod_out)
    b = _op("B", CollectivePattern.Reduce, cons_in,
            pa.schema([pa.field("y", pa.float32())]))
    dag = WorkflowDAG([a, b], [("A", "B")])
    plan = compile_workflow(dag)              # must not raise
    assert plan.edges[0].mode == TransferMode.ZeroCopy


def test_compile_rejects_appended_non_nullable_column():
    prod_out = pa.schema([
        pa.field("embedding", pa.list_(pa.field("item", pa.float32()), 8)),
        pa.field("required_extra", pa.int32(), nullable=False),   # appended non-nullable
    ])
    a = _op("A", CollectivePattern.ScatterGather, _emb_schema(8, "x"), prod_out)
    b = _op("B", CollectivePattern.Reduce, _emb_schema(8, "embedding"),
            pa.schema([pa.field("y", pa.float32())]))
    dag = WorkflowDAG([a, b], [("A", "B")])
    with pytest.raises(SchemaMismatchError):
        compile_workflow(dag)


def test_plan_records_collective_pattern_per_operator():
    a = _op("Embed", CollectivePattern.ScatterGather, _emb_schema(8, "x"),
            _emb_schema(8, "embedding"))
    b = _op("Retrieve", CollectivePattern.Reduce, _emb_schema(8, "embedding"),
            pa.schema([pa.field("y", pa.float32())]))
    dag = WorkflowDAG([a, b], [("Embed", "Retrieve")])
    plan = compile_workflow(dag)
    assert plan.assignments["Embed"] == CollectivePattern.ScatterGather
    assert plan.assignments["Retrieve"] == CollectivePattern.Reduce


def test_a2_driver_runs_three_cases():
    import importlib, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "target", "shared", "scripts"))
    a2 = importlib.import_module("experiment.exp_a2_schema")
    rows = a2.run_a2(d=256)
    cases = {r["case"] for r in rows}
    assert {"compatible", "incompatible", "evolution"} <= cases
    incompat = [r for r in rows if r["case"] == "incompatible"][0]
    assert incompat["result"] == "rejected"
    evo = [r for r in rows if r["case"] == "evolution"][0]
    assert evo["result"] == "tolerated"


def test_missing_required_field_rejected():
    # Producer supplies zero fields; consumer requires one.
    a = _op("A", CollectivePattern.Scatter,
            pa.schema([pa.field("x", pa.int32())]), pa.schema([]))
    b = _op("B", CollectivePattern.Reduce,
            _emb_schema(8, "needs"), pa.schema([pa.field("y", pa.float32())]))
    dag = WorkflowDAG([a, b], [("A", "B")])
    with pytest.raises(SchemaMismatchError):
        compile_workflow(dag)


def test_single_operator_no_edges_compiles():
    a = _op("Solo", CollectivePattern.Broadcast,
            _emb_schema(8, "x"), _emb_schema(8, "y"))
    dag = WorkflowDAG([a], [])
    plan = compile_workflow(dag)
    assert plan.assignments["Solo"] == CollectivePattern.Broadcast
    assert list(plan.edges) == []


def test_check_edges_reports_all_without_throwing():
    a = _op("A", CollectivePattern.PointToPoint,
            pa.schema([pa.field("x", pa.int32())]), _text_out())
    b = _op("B", CollectivePattern.Broadcast,
            _emb_schema(8, "needs"), _bool_out())
    c = _op("C", CollectivePattern.Reduce,
            _emb_schema(8, "needs"), pa.schema([pa.field("z", pa.float32())]))
    dag = WorkflowDAG([a, b, c], [("A", "B"), ("B", "C")])
    edges = check_edges(dag)             # must not raise even though A->B mismatches
    assert len(edges) == 2
    assert edges[0].mode == TransferMode.Convert


def _text_out():
    return pa.schema([pa.field("out", pa.large_utf8())])


def _bool_out():
    return pa.schema([pa.field("ack", pa.bool_())])