# distutils: language = c++
from libcpp cimport bool as cbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr

from pyarrow.lib cimport CSchema

cdef extern from "cylon_armada/compiler/dag_compiler.hpp" namespace "cylon_armada":
    # NOTE: Cython-side names are prefixed with `_` (rather than the bare
    # `CollectivePattern` / `TransferMode` the brief originally specified)
    # because this .pxd and dag_compiler.pyx share the same module name.
    # Cython implicitly merges a same-named .pxd's extern declarations into
    # the .pyx's own namespace, so a bare `CollectivePattern` here collides
    # with the `class CollectivePattern(IntEnum)` defined in the .pyx and
    # Cython reports "Calling non-function type 'CollectivePattern'". The
    # underlying C++ symbol (the quoted string) is unchanged.
    cdef enum _CollectivePattern "cylon_armada::CollectivePattern":
        Scatter "cylon_armada::CollectivePattern::Scatter"
        ScatterGather "cylon_armada::CollectivePattern::ScatterGather"
        Reduce "cylon_armada::CollectivePattern::Reduce"
        PointToPoint "cylon_armada::CollectivePattern::PointToPoint"
        Broadcast "cylon_armada::CollectivePattern::Broadcast"

    cdef enum _TransferMode "cylon_armada::TransferMode":
        ZeroCopy "cylon_armada::TransferMode::ZeroCopy"
        Convert "cylon_armada::TransferMode::Convert"

    cdef cppclass CAgentOperator "cylon_armada::AgentOperator":
        CAgentOperator()
        string name
        _CollectivePattern pattern
        shared_ptr[CSchema] schema_in
        shared_ptr[CSchema] schema_out

    cdef cppclass CWorkflowDAG "cylon_armada::WorkflowDAG":
        CWorkflowDAG()
        vector[CAgentOperator] operators
        vector[pair[string, string]] edges

    cdef cppclass CEdgePlan "cylon_armada::EdgePlan":
        string producer
        string consumer
        _TransferMode mode
        string note

    cdef cppclass CExecutionPlan "cylon_armada::ExecutionPlan":
        vector[CEdgePlan] edges
        cmap[string, _CollectivePattern] assignments

    cdef cppclass CCompileResult "cylon_armada::CompileResult":
        cbool ok
        string error_message
        CExecutionPlan plan

    cdef cppclass CAgentDAGCompiler "cylon_armada::AgentDAGCompiler":
        CAgentDAGCompiler()
        vector[CEdgePlan] CheckEdges(const CWorkflowDAG&)
        CCompileResult Compile(const CWorkflowDAG&)
