#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/type.h>

namespace cylon_armada {

enum class CollectivePattern { Scatter, ScatterGather, Reduce, PointToPoint, Broadcast };
enum class TransferMode { ZeroCopy, Convert };

struct AgentOperator {
  std::string name;
  CollectivePattern pattern;
  std::shared_ptr<arrow::Schema> schema_in;
  std::shared_ptr<arrow::Schema> schema_out;
};

struct WorkflowDAG {
  std::vector<AgentOperator> operators;
  std::vector<std::pair<std::string, std::string>> edges;  // (producer_name, consumer_name)
};

struct EdgePlan {
  std::string producer;
  std::string consumer;
  TransferMode mode;
  std::string note;
};

struct ExecutionPlan {
  std::vector<EdgePlan> edges;
  std::map<std::string, CollectivePattern> assignments;
};

// Plain result (no cylon/arrow Status) so the Cython binding needs only pyarrow.
struct CompileResult {
  bool ok;
  std::string error_message;  // set when !ok (schema mismatch)
  ExecutionPlan plan;
};

class AgentDAGCompiler {
 public:
  // Non-throwing diagnostic: per-edge compatibility report for the whole DAG.
  std::vector<EdgePlan> CheckEdges(const WorkflowDAG& dag) const;

  // Enforcement: ok=false + error_message on the first incompatible edge,
  // before any plan is emitted. On success, ok=true and plan is populated.
  CompileResult Compile(const WorkflowDAG& dag) const;

 private:
  bool EdgeCompatible(const AgentOperator& producer, const AgentOperator& consumer,
                      std::string* reason) const;
  const AgentOperator* FindOperator(const WorkflowDAG& dag, const std::string& name) const;
};

}  // namespace cylon_armada
