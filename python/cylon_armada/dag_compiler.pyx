# distutils: language = c++
from enum import IntEnum

import pyarrow as pa

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr

from pyarrow.lib cimport CSchema, pyarrow_unwrap_schema

from cylon_armada.dag_compiler cimport (
    _CollectivePattern as CPat, _TransferMode as CMode,
    CAgentOperator, CWorkflowDAG, CEdgePlan, CExecutionPlan,
    CCompileResult, CAgentDAGCompiler,
    Scatter, ScatterGather, Reduce, PointToPoint, Broadcast,
    ZeroCopy, Convert,
)


class CollectivePattern(IntEnum):
    Scatter = 0
    ScatterGather = 1
    Reduce = 2
    PointToPoint = 3
    Broadcast = 4


class TransferMode(IntEnum):
    ZeroCopy = 0
    Convert = 1


class SchemaMismatchError(Exception):
    """Raised when the AgentDAGCompiler rejects an edge whose producer output
    is not schema-compatible with the consumer input (proposal A2 case 2)."""


cdef CPat _to_cpat(int p) except *:
    if p == 0: return Scatter
    if p == 1: return ScatterGather
    if p == 2: return Reduce
    if p == 3: return PointToPoint
    if p == 4: return Broadcast
    raise ValueError(f"unknown collective pattern {p}")


cdef int _from_cpat(CPat p):
    if p == Scatter: return 0
    if p == ScatterGather: return 1
    if p == Reduce: return 2
    if p == PointToPoint: return 3
    return 4  # Broadcast


class AgentOperator:
    def __init__(self, name, pattern, schema_in, schema_out):
        if not isinstance(schema_in, pa.Schema):
            raise TypeError(
                f"schema_in must be a pyarrow.Schema, got {type(schema_in).__name__}"
            )
        if not isinstance(schema_out, pa.Schema):
            raise TypeError(
                f"schema_out must be a pyarrow.Schema, got {type(schema_out).__name__}"
            )
        self.name = name
        self.pattern = CollectivePattern(int(pattern))
        self.schema_in = schema_in    # pyarrow.Schema
        self.schema_out = schema_out  # pyarrow.Schema


class WorkflowDAG:
    def __init__(self, operators, edges):
        self.operators = list(operators)
        self.edges = [(str(a), str(b)) for (a, b) in edges]


class EdgePlanView:
    def __init__(self, producer, consumer, mode, note):
        self.producer = producer
        self.consumer = consumer
        self.mode = mode
        self.note = note


class ExecutionPlan:
    def __init__(self, edges, assignments):
        self.edges = edges              # list[EdgePlanView]
        self.assignments = assignments  # dict[str, CollectivePattern]


cdef CWorkflowDAG _build_cdag(object dag) except *:
    cdef CWorkflowDAG cdag
    cdef CAgentOperator cop
    cdef shared_ptr[CSchema] sin
    cdef shared_ptr[CSchema] sout
    for op in dag.operators:
        cop = CAgentOperator()
        cop.name = op.name.encode("utf-8")
        cop.pattern = _to_cpat(int(op.pattern))
        sin = pyarrow_unwrap_schema(op.schema_in)
        sout = pyarrow_unwrap_schema(op.schema_out)
        cop.schema_in = sin
        cop.schema_out = sout
        cdag.operators.push_back(cop)
    for (a, b) in dag.edges:
        cdag.edges.push_back(pair[string, string](a.encode("utf-8"), b.encode("utf-8")))
    return cdag


def check_edges(dag):
    cdef CAgentDAGCompiler comp
    cdef CWorkflowDAG cdag = _build_cdag(dag)
    cdef vector[CEdgePlan] res = comp.CheckEdges(cdag)
    out = []
    cdef size_t i
    for i in range(res.size()):
        out.append(EdgePlanView(
            res[i].producer.decode("utf-8"),
            res[i].consumer.decode("utf-8"),
            TransferMode(0 if res[i].mode == ZeroCopy else 1),
            res[i].note.decode("utf-8"),
        ))
    return out


def compile_workflow(dag):
    cdef CAgentDAGCompiler comp
    cdef CWorkflowDAG cdag = _build_cdag(dag)
    cdef CCompileResult r = comp.Compile(cdag)
    if not r.ok:
        raise SchemaMismatchError(r.error_message.decode("utf-8"))
    edges = []
    cdef size_t i
    for i in range(r.plan.edges.size()):
        edges.append(EdgePlanView(
            r.plan.edges[i].producer.decode("utf-8"),
            r.plan.edges[i].consumer.decode("utf-8"),
            TransferMode(0 if r.plan.edges[i].mode == ZeroCopy else 1),
            r.plan.edges[i].note.decode("utf-8"),
        ))
    assignments = {}
    cdef object key
    for item in r.plan.assignments:
        key = item.first.decode("utf-8")
        assignments[key] = CollectivePattern(_from_cpat(item.second))
    return ExecutionPlan(edges, assignments)
