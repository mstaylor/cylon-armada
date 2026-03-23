/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_CONTEXT_TABLE_HPP
#define CYLON_CONTEXT_TABLE_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/ipc/api.h>

#include <cylon/simd/simd_ops.hpp>
#include <cylon/status.hpp>

namespace cylon {
class CylonContext;  // forward declaration
namespace context {

/// Metadata associated with a context entry.
struct ContextMetadata {
  std::string workflow_id;
  std::string response;
  std::string model_id;
  int64_t input_tokens = 0;
  int64_t output_tokens = 0;
  double cost_usd = 0.0;
};

/// Arrow builders for all columns. Supports O(1) append.
struct ContextBuilders {
  std::unique_ptr<arrow::StringBuilder> context_id;
  std::unique_ptr<arrow::StringBuilder> workflow_id;
  std::unique_ptr<arrow::FixedSizeListBuilder> embedding;
  std::unique_ptr<arrow::LargeStringBuilder> response;
  std::unique_ptr<arrow::StringBuilder> model_id;
  std::unique_ptr<arrow::Int64Builder> input_tokens;
  std::unique_ptr<arrow::Int64Builder> output_tokens;
  std::unique_ptr<arrow::DoubleBuilder> cost_usd;
  std::unique_ptr<arrow::TimestampBuilder> created_at;
  std::unique_ptr<arrow::Int64Builder> reuse_count;

  static std::unique_ptr<ContextBuilders> Create(int embedding_dim);

  void Append(const std::string& context_id,
              const float* embedding, int dim,
              const ContextMetadata& metadata);

  std::shared_ptr<arrow::RecordBatch> Finish(
      const std::shared_ptr<arrow::Schema>& schema);

  int64_t length = 0;
};

/// Arrow-native key-value store optimized for embedding storage and
/// SIMD similarity search.
///
/// Uses Arrow builders for O(1) inserts. Data is materialized into a
/// RecordBatch lazily before reads (search, get, serialize). Deleted rows
/// are tracked in a tombstone set. Call Compact() to reclaim space.
class ContextTable {
 public:
  static arrow::Result<std::shared_ptr<ContextTable>> Create(int embedding_dim);

  static Status MakeEmpty(int embedding_dim, std::shared_ptr<ContextTable>* out);

  static arrow::Result<std::shared_ptr<ContextTable>> FromRecordBatch(
      const std::shared_ptr<arrow::RecordBatch>& batch);

  static Status MakeFromIpc(const uint8_t* data, int64_t size,
                            std::shared_ptr<ContextTable>* out);

  /// Insert or update a context entry. O(1).
  Status Put(const std::string& context_id,
             const float* embedding, int dim,
             const ContextMetadata& metadata);

  /// Retrieve a single row by context_id. O(1).
  std::optional<std::shared_ptr<arrow::RecordBatch>> Get(
      const std::string& context_id);

  /// Status-based Get for Cython bindings. O(1).
  Status GetRow(const std::string& context_id,
                std::shared_ptr<arrow::RecordBatch>* out);

  /// Mark a row as deleted. O(1).
  Status Remove(const std::string& context_id);

  std::shared_ptr<arrow::RecordBatch> GetWorkflow(
      const std::string& workflow_id);

  std::vector<simd::SearchResult> Search(
      const float* query, int dim,
      float threshold, int top_k = 5,
      const std::string& workflow_id = "");

  int64_t Size() const;
  int64_t TotalRows() const;
  std::shared_ptr<arrow::RecordBatch> Batch();
  std::shared_ptr<arrow::Schema> GetSchema() const;
  int EmbeddingDim() const { return embedding_dim_; }

  Status Compact();
  Status ToIpc(std::vector<uint8_t>* data);

  static arrow::Result<std::shared_ptr<ContextTable>> FromIpc(
      const uint8_t* data, int64_t size);

  Status Broadcast(const std::shared_ptr<CylonContext>& ctx, int root = 0);
  Status AllGather(const std::shared_ptr<CylonContext>& ctx);

 private:
  ContextTable() = default;

  static std::shared_ptr<arrow::Schema> MakeSchema(int embedding_dim);
  void RebuildIndex();

  /// Flush builders into the main batch if dirty.
  void MaterializeIfDirty();

  std::shared_ptr<arrow::RecordBatch> batch_;
  std::unique_ptr<ContextBuilders> builders_;
  bool dirty_ = false;
  int64_t builder_count_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::unordered_map<std::string, int64_t> index_;
  std::unordered_set<int64_t> deleted_;
  int embedding_dim_ = 0;
};

}  // namespace context
}  // namespace cylon

#endif  // CYLON_CONTEXT_TABLE_HPP