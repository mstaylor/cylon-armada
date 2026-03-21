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

/// Arrow-native key-value store optimized for embedding storage and
/// SIMD similarity search. Backed by an Arrow RecordBatch with a
/// FixedSizeList<Float32> embedding column and a hash index on context_id.
///
/// Put/Get/Remove are O(1) via hash index. Deleted rows are tracked in a
/// tombstone set and skipped during search. Call Compact() to reclaim space.
class ContextTable {
 public:
  /// Create an empty ContextTable with the given embedding dimension.
  /// @param embedding_dim The fixed size of each embedding vector (required).
  static arrow::Result<std::shared_ptr<ContextTable>> Create(int embedding_dim);

  /// Status-based factory (for Cython bindings).
  static Status MakeEmpty(int embedding_dim, std::shared_ptr<ContextTable>* out);

  /// Reconstruct a ContextTable from a RecordBatch (e.g., after IPC
  /// deserialization). Infers embedding_dim from the FixedSizeList column.
  static arrow::Result<std::shared_ptr<ContextTable>> FromRecordBatch(
      const std::shared_ptr<arrow::RecordBatch>& batch);

  /// Status-based factory from IPC (for Cython bindings).
  static Status MakeFromIpc(const uint8_t* data, int64_t size,
                            std::shared_ptr<ContextTable>* out);

  /// Insert or update a context entry. O(1) amortized.
  Status Put(const std::string& context_id,
             const float* embedding, int dim,
             const ContextMetadata& metadata);

  /// Retrieve a single row by context_id. O(1). Returns nullopt if not found.
  std::optional<std::shared_ptr<arrow::RecordBatch>> Get(
      const std::string& context_id);

  /// Status-based Get for Cython bindings. O(1).
  /// Sets *out to the row batch, or nullptr if not found.
  Status GetRow(const std::string& context_id,
                std::shared_ptr<arrow::RecordBatch>* out);

  /// Mark a row as deleted. O(1). Call Compact() to reclaim space.
  Status Remove(const std::string& context_id);

  /// Retrieve all rows for a given workflow_id.
  std::shared_ptr<arrow::RecordBatch> GetWorkflow(
      const std::string& workflow_id);

  /// SIMD cosine similarity search across all (or workflow-filtered) embeddings.
  /// Returns up to @p top_k results with similarity >= @p threshold,
  /// sorted by descending similarity. Skips tombstoned rows.
  std::vector<simd::SearchResult> Search(
      const float* query, int dim,
      float threshold, int top_k = 5,
      const std::string& workflow_id = "");

  /// Number of active (non-deleted) rows.
  int64_t Size() const;

  /// Total rows including deleted (tombstoned).
  int64_t TotalRows() const;

  /// The underlying RecordBatch (includes tombstoned rows).
  std::shared_ptr<arrow::RecordBatch> Batch() const;

  /// The Arrow schema.
  std::shared_ptr<arrow::Schema> GetSchema() const;

  /// Embedding dimension.
  int EmbeddingDim() const { return embedding_dim_; }

  /// Rebuild the RecordBatch, removing tombstoned rows and compacting memory.
  Status Compact();

  /// Serialize to Arrow IPC stream format.
  Status ToIpc(std::vector<uint8_t>* data) const;

  /// Deserialize from Arrow IPC stream format.
  static arrow::Result<std::shared_ptr<ContextTable>> FromIpc(
      const uint8_t* data, int64_t size);

  /// Broadcast this ContextTable from root rank to all workers.
  /// On non-root ranks, replaces the current contents with the broadcast data.
  /// Compacts before broadcasting to avoid sending tombstoned rows.
  /// @param ctx CylonContext with distributed communicator.
  /// @param root Rank of the broadcasting process.
  Status Broadcast(const std::shared_ptr<CylonContext>& ctx, int root = 0);

  /// AllGather: each worker contributes its ContextTable, and all workers
  /// receive the merged result. Compacts before gathering.
  /// @param ctx CylonContext with distributed communicator.
  Status AllGather(const std::shared_ptr<CylonContext>& ctx);

 private:
  ContextTable() = default;

  /// Build the schema for the given embedding dimension.
  static std::shared_ptr<arrow::Schema> MakeSchema(int embedding_dim);

  /// Rebuild the hash index from the current batch (used after Compact/FromIpc).
  void RebuildIndex();

  /// Append a single row to the batch. Returns the new row index.
  arrow::Result<int64_t> AppendRow(
      const std::string& context_id,
      const float* embedding, int dim,
      const ContextMetadata& metadata);

  std::shared_ptr<arrow::RecordBatch> batch_;
  std::unordered_map<std::string, int64_t> index_;    // context_id → row
  std::unordered_set<int64_t> deleted_;                // tombstoned row indices
  int embedding_dim_ = 0;
};

}  // namespace context
}  // namespace cylon

#endif  // CYLON_CONTEXT_TABLE_HPP