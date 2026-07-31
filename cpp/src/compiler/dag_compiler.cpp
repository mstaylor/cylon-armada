#include "dag_compiler.hpp"

namespace cylon_armada {

const AgentOperator* AgentDAGCompiler::FindOperator(const WorkflowDAG& dag,
                                                    const std::string& name) const {
  for (const auto& op : dag.operators) {
    if (op.name == name) return &op;
  }
  return nullptr;
}

bool AgentDAGCompiler::EdgeCompatible(const AgentOperator& producer,
                                      const AgentOperator& consumer,
                                      std::string* reason) const {
  const std::shared_ptr<arrow::Schema>& P = producer.schema_out;
  const std::shared_ptr<arrow::Schema>& C = consumer.schema_in;
  if (P->num_fields() < C->num_fields()) {
    *reason = "producer supplies " + std::to_string(P->num_fields()) +
              " fields but consumer requires " + std::to_string(C->num_fields());
    return false;
  }
  for (int i = 0; i < C->num_fields(); ++i) {
    if (!P->field(i)->type()->Equals(*C->field(i)->type())) {
      *reason = "type mismatch at field " + std::to_string(i) + ": producer " +
                P->field(i)->type()->ToString() + " vs consumer " +
                C->field(i)->type()->ToString();
      return false;
    }
  }
  for (int j = C->num_fields(); j < P->num_fields(); ++j) {
    if (!P->field(j)->nullable()) {
      *reason = "extra non-nullable producer field: " + P->field(j)->name();
      return false;
    }
  }
  return true;
}

std::vector<EdgePlan> AgentDAGCompiler::CheckEdges(const WorkflowDAG& dag) const {
  std::vector<EdgePlan> out;
  for (const auto& e : dag.edges) {
    EdgePlan ep;
    ep.producer = e.first;
    ep.consumer = e.second;
    const AgentOperator* prod = FindOperator(dag, e.first);
    const AgentOperator* cons = FindOperator(dag, e.second);
    if (prod == nullptr || cons == nullptr) {
      ep.mode = TransferMode::Convert;
      ep.note = "edge references unknown operator";
    } else {
      std::string reason;
      if (EdgeCompatible(*prod, *cons, &reason)) {
        ep.mode = TransferMode::ZeroCopy;
        ep.note = "";
      } else {
        ep.mode = TransferMode::Convert;
        ep.note = reason;
      }
    }
    out.push_back(ep);
  }
  return out;
}

CompileResult AgentDAGCompiler::Compile(const WorkflowDAG& dag) const {
  CompileResult r;
  r.ok = true;
  for (const auto& e : dag.edges) {
    const AgentOperator* prod = FindOperator(dag, e.first);
    const AgentOperator* cons = FindOperator(dag, e.second);
    if (prod == nullptr || cons == nullptr) {
      r.ok = false;
      r.error_message = "edge " + e.first + " -> " + e.second +
                        " references unknown operator";
      return r;
    }
    std::string reason;
    if (!EdgeCompatible(*prod, *cons, &reason)) {
      r.ok = false;
      r.error_message = "schema_mismatch on edge " + e.first + " -> " + e.second +
                        ": " + reason;
      return r;
    }
    EdgePlan ep;
    ep.producer = e.first;
    ep.consumer = e.second;
    ep.mode = TransferMode::ZeroCopy;
    ep.note = "";
    r.plan.edges.push_back(ep);
  }
  for (const auto& op : dag.operators) {
    r.plan.assignments[op.name] = op.pattern;
  }
  return r;
}

}  // namespace cylon_armada
