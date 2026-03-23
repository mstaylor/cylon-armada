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

#include "context_table.hpp"

#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <chrono>
#include <cstring>

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/table.hpp>
#include <cylon/util/macros.hpp>

namespace cylon {
namespace context {

// Column indices
static constexpr int kContextIdCol = 0;
static constexpr int kWorkflowIdCol = 1;
static constexpr int kEmbeddingCol = 2;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

std::shared_ptr<arrow::Schema> ContextTable::MakeSchema(int embedding_dim) {
  return arrow::schema({
      arrow::field("context_id", arrow::utf8()),
      arrow::field("workflow_id", arrow::utf8()),
      arrow::field("embedding",
                   arrow::fixed_size_list(arrow::field("item", arrow::float32()),
                                          embedding_dim)),
      arrow::field("response", arrow::large_utf8()),
      arrow::field("model_id", arrow::utf8()),
      arrow::field("input_tokens", arrow::int64()),
      arrow::field("output_tokens", arrow::int64()),
      arrow::field("cost_usd", arrow::float64()),
      arrow::field("created_at", arrow::timestamp(arrow::TimeUnit::MILLI, "UTC")),
      arrow::field("reuse_count", arrow::int64()),
  });
}

// ---------------------------------------------------------------------------
// ContextBuilders
// ---------------------------------------------------------------------------

std::unique_ptr<ContextBuilders> ContextBuilders::Create(int embedding_dim) {
  auto b = std::make_unique<ContextBuilders>();
  auto pool = arrow::default_memory_pool();
  b->context_id = std::make_unique<arrow::StringBuilder>(pool);
  b->workflow_id = std::make_unique<arrow::StringBuilder>(pool);
  auto values_builder = std::make_shared<arrow::FloatBuilder>(pool);
  b->embedding = std::make_unique<arrow::FixedSizeListBuilder>(
      pool, values_builder, embedding_dim);
  b->response = std::make_unique<arrow::LargeStringBuilder>(pool);
  b->model_id = std::make_unique<arrow::StringBuilder>(pool);
  b->input_tokens = std::make_unique<arrow::Int64Builder>(pool);
  b->output_tokens = std::make_unique<arrow::Int64Builder>(pool);
  b->cost_usd = std::make_unique<arrow::DoubleBuilder>(pool);
  b->created_at = std::make_unique<arrow::TimestampBuilder>(
      arrow::timestamp(arrow::TimeUnit::MILLI, "UTC"), pool);
  b->reuse_count = std::make_unique<arrow::Int64Builder>(pool);
  b->length = 0;
  return b;
}

void ContextBuilders::Append(const std::string& ctx_id,
                              const float* emb, int dim,
                              const ContextMetadata& meta) {
  (void)context_id->Append(ctx_id);
  (void)workflow_id->Append(meta.workflow_id);
  (void)embedding->Append();
  auto* vals = static_cast<arrow::FloatBuilder*>(embedding->value_builder());
  (void)vals->AppendValues(emb, dim);
  (void)response->Append(meta.response);
  (void)model_id->Append(meta.model_id);
  (void)input_tokens->Append(meta.input_tokens);
  (void)output_tokens->Append(meta.output_tokens);
  (void)cost_usd->Append(meta.cost_usd);
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count();
  (void)created_at->Append(millis);
  (void)reuse_count->Append(0);
  ++length;
}

std::shared_ptr<arrow::RecordBatch> ContextBuilders::Finish(
    const std::shared_ptr<arrow::Schema>& schema) {
  std::shared_ptr<arrow::Array> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;
  (void)context_id->Finish(&a0);
  (void)workflow_id->Finish(&a1);
  (void)embedding->Finish(&a2);
  (void)response->Finish(&a3);
  (void)model_id->Finish(&a4);
  (void)input_tokens->Finish(&a5);
  (void)output_tokens->Finish(&a6);
  (void)cost_usd->Finish(&a7);
  (void)created_at->Finish(&a8);
  (void)reuse_count->Finish(&a9);
  auto rows = length;
  length = 0;
  return arrow::RecordBatch::Make(schema, rows,
      {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9});
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

arrow::Result<std::shared_ptr<ContextTable>> ContextTable::Create(int embedding_dim) {
  if (embedding_dim <= 0) {
    return arrow::Status::Invalid("embedding_dim must be positive");
  }
  auto schema = MakeSchema(embedding_dim);

  // Create empty arrays for the empty batch
  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (int i = 0; i < schema->num_fields(); ++i) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    ARROW_RETURN_NOT_OK(arrow::MakeBuilder(arrow::default_memory_pool(),
                                           schema->field(i)->type(), &builder));
    std::shared_ptr<arrow::Array> empty_array;
    ARROW_RETURN_NOT_OK(builder->Finish(&empty_array));
    columns.push_back(std::move(empty_array));
  }

  auto table = std::shared_ptr<ContextTable>(new ContextTable());
  table->embedding_dim_ = embedding_dim;
  table->schema_ = schema;
  table->batch_ = arrow::RecordBatch::Make(schema, 0, std::move(columns));
  table->builders_ = ContextBuilders::Create(embedding_dim);
  return table;
}

Status ContextTable::MakeEmpty(int embedding_dim,
                               std::shared_ptr<ContextTable>* out) {
  auto result = Create(embedding_dim);
  if (!result.ok()) {
    return {Code::Invalid, result.status().ToString()};
  }
  *out = std::move(*result);
  return Status::OK();
}

arrow::Result<std::shared_ptr<ContextTable>> ContextTable::FromRecordBatch(
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!batch) {
    return arrow::Status::Invalid("batch is null");
  }
  auto embedding_field = batch->schema()->field(kEmbeddingCol);
  auto fsl_type = std::dynamic_pointer_cast<arrow::FixedSizeListType>(
      embedding_field->type());
  if (!fsl_type) {
    return arrow::Status::Invalid("embedding column is not FixedSizeList");
  }
  if (fsl_type->value_type()->id() != arrow::Type::FLOAT) {
    return arrow::Status::Invalid("embedding inner type must be Float32");
  }

  auto table = std::shared_ptr<ContextTable>(new ContextTable());
  table->embedding_dim_ = fsl_type->list_size();
  table->schema_ = batch->schema();
  table->batch_ = batch;
  table->builders_ = ContextBuilders::Create(table->embedding_dim_);
  table->RebuildIndex();
  return table;
}

Status ContextTable::MakeFromIpc(const uint8_t* data, int64_t size,
                                 std::shared_ptr<ContextTable>* out) {
  auto result = FromIpc(data, size);
  if (!result.ok()) {
    return {Code::ExecutionError, result.status().ToString()};
  }
  *out = std::move(*result);
  return Status::OK();
}

// ---------------------------------------------------------------------------
// Key-Value Operations — O(1)
// ---------------------------------------------------------------------------

Status ContextTable::Put(const std::string& context_id,
                         const float* embedding, int dim,
                         const ContextMetadata& metadata) {
  if (dim != embedding_dim_) {
    return {Code::Invalid, "embedding dim mismatch: expected "
                           + std::to_string(embedding_dim_) + ", got "
                           + std::to_string(dim)};
  }

  // Tombstone old row if key exists — O(1)
  auto it = index_.find(context_id);
  if (it != index_.end()) {
    deleted_.insert(it->second);
    index_.erase(it);
  }

  // Append to builders — O(1)
  builders_->Append(context_id, embedding, dim, metadata);
  auto new_idx = batch_->num_rows() + builder_count_;
  index_[context_id] = new_idx;
  ++builder_count_;
  dirty_ = true;
  return Status::OK();
}

std::optional<std::shared_ptr<arrow::RecordBatch>> ContextTable::Get(
    const std::string& context_id) {
  auto it = index_.find(context_id);
  if (it == index_.end()) {
    return std::nullopt;
  }
  MaterializeIfDirty();
  return batch_->Slice(it->second, 1);
}

Status ContextTable::GetRow(const std::string& context_id,
                            std::shared_ptr<arrow::RecordBatch>* out) {
  auto it = index_.find(context_id);
  if (it == index_.end()) {
    *out = nullptr;
    return Status::OK();
  }
  MaterializeIfDirty();
  *out = batch_->Slice(it->second, 1);
  return Status::OK();
}

Status ContextTable::Remove(const std::string& context_id) {
  auto it = index_.find(context_id);
  if (it == index_.end()) {
    return {Code::KeyError, "context_id not found: " + context_id};
  }
  deleted_.insert(it->second);
  index_.erase(it);
  return Status::OK();
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

std::shared_ptr<arrow::RecordBatch> ContextTable::GetWorkflow(
    const std::string& workflow_id) {
  MaterializeIfDirty();

  auto wf_array = std::static_pointer_cast<arrow::StringArray>(
      batch_->column(kWorkflowIdCol));

  std::vector<int64_t> indices;
  for (int64_t i = 0; i < wf_array->length(); ++i) {
    if (deleted_.count(i)) continue;
    if (!wf_array->IsNull(i) && wf_array->GetView(i) == workflow_id) {
      indices.push_back(i);
    }
  }

  if (indices.empty()) {
    std::vector<std::shared_ptr<arrow::Array>> empty_cols;
    for (int c = 0; c < schema_->num_fields(); ++c) {
      std::unique_ptr<arrow::ArrayBuilder> builder;
      (void)arrow::MakeBuilder(arrow::default_memory_pool(),
                                schema_->field(c)->type(), &builder);
      std::shared_ptr<arrow::Array> arr;
      (void)builder->Finish(&arr);
      empty_cols.push_back(std::move(arr));
    }
    return arrow::RecordBatch::Make(schema_, 0, std::move(empty_cols));
  }

  arrow::Int64Builder idx_builder;
  (void)idx_builder.AppendValues(indices);
  std::shared_ptr<arrow::Array> idx_array;
  (void)idx_builder.Finish(&idx_array);

  auto datum = arrow::compute::Take(batch_, idx_array);
  if (datum.ok()) {
    return datum->record_batch();
  }
  return nullptr;
}

std::vector<simd::SearchResult> ContextTable::Search(
    const float* query, int dim,
    float threshold, int top_k,
    const std::string& workflow_id) {
  MaterializeIfDirty();

  if (!batch_ || batch_->num_rows() == 0 || dim != embedding_dim_) {
    return {};
  }

  auto embedding_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
      batch_->column(kEmbeddingCol));

  // Fast path: no deletions, no workflow filter
  if (deleted_.empty() && workflow_id.empty()) {
    return simd::batch_cosine_search_arrow(query, dim, embedding_col,
                                            threshold, top_k);
  }

  auto values = std::static_pointer_cast<arrow::FloatArray>(
      embedding_col->values());
  const float* data = values->raw_values();

  std::shared_ptr<arrow::StringArray> wf_array;
  if (!workflow_id.empty()) {
    wf_array = std::static_pointer_cast<arrow::StringArray>(
        batch_->column(kWorkflowIdCol));
  }

  using Pair = std::pair<float, int64_t>;
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

  for (int64_t i = 0; i < batch_->num_rows(); ++i) {
    if (deleted_.count(i)) continue;
    if (wf_array && (wf_array->IsNull(i) || wf_array->GetView(i) != workflow_id)) {
      continue;
    }
    const float* row = data + i * dim;
    float sim = simd::cosine_similarity_f32(query, row, dim);
    if (sim >= threshold) {
      if (static_cast<int>(heap.size()) < top_k) {
        heap.emplace(sim, i);
      } else if (sim > heap.top().first) {
        heap.pop();
        heap.emplace(sim, i);
      }
    }
  }

  std::vector<simd::SearchResult> results;
  results.reserve(heap.size());
  while (!heap.empty()) {
    auto [sim, idx] = heap.top();
    heap.pop();
    results.push_back({idx, sim});
  }
  std::reverse(results.begin(), results.end());
  return results;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int64_t ContextTable::Size() const {
  return static_cast<int64_t>(index_.size());
}

int64_t ContextTable::TotalRows() const {
  return (batch_ ? batch_->num_rows() : 0) + builder_count_;
}

std::shared_ptr<arrow::RecordBatch> ContextTable::Batch() {
  MaterializeIfDirty();
  return batch_;
}

std::shared_ptr<arrow::Schema> ContextTable::GetSchema() const {
  return schema_;
}

// ---------------------------------------------------------------------------
// Materialize — flush builders into batch
// ---------------------------------------------------------------------------

void ContextTable::MaterializeIfDirty() {
  if (!dirty_ || builder_count_ == 0) {
    return;
  }

  auto new_batch = builders_->Finish(schema_);
  if (batch_->num_rows() == 0) {
    batch_ = std::move(new_batch);
  } else {
    std::vector<std::shared_ptr<arrow::Array>> merged;
    for (int c = 0; c < schema_->num_fields(); ++c) {
      auto concat_result = arrow::Concatenate(
          {batch_->column(c), new_batch->column(c)});
      merged.push_back(std::move(*concat_result));
    }
    batch_ = arrow::RecordBatch::Make(
        schema_, batch_->num_rows() + new_batch->num_rows(), std::move(merged));
  }

  builders_ = ContextBuilders::Create(embedding_dim_);
  builder_count_ = 0;
  dirty_ = false;
}

// ---------------------------------------------------------------------------
// Compact
// ---------------------------------------------------------------------------

Status ContextTable::Compact() {
  MaterializeIfDirty();

  if (!batch_ || batch_->num_rows() == 0 || deleted_.empty()) {
    return Status::OK();
  }

  std::vector<int64_t> keep;
  keep.reserve(batch_->num_rows() - deleted_.size());
  for (int64_t i = 0; i < batch_->num_rows(); ++i) {
    if (!deleted_.count(i)) {
      keep.push_back(i);
    }
  }

  arrow::Int64Builder idx_builder;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.AppendValues(keep));
  std::shared_ptr<arrow::Array> idx_array;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(idx_builder.Finish(&idx_array));

  auto take_result = arrow::compute::Take(batch_, idx_array);
  if (!take_result.ok()) {
    return {Code::ExecutionError, take_result.status().ToString()};
  }
  batch_ = take_result->record_batch();
  RebuildIndex();
  return Status::OK();
}

void ContextTable::RebuildIndex() {
  index_.clear();
  deleted_.clear();
  if (!batch_ || batch_->num_rows() == 0) {
    return;
  }
  auto id_array = std::static_pointer_cast<arrow::StringArray>(
      batch_->column(kContextIdCol));
  for (int64_t i = 0; i < id_array->length(); ++i) {
    if (!id_array->IsNull(i)) {
      auto key = id_array->GetString(i);
      auto it = index_.find(key);
      if (it != index_.end()) {
        deleted_.insert(it->second);
      }
      index_[key] = i;
    }
  }
}

// ---------------------------------------------------------------------------
// Arrow IPC Serialization
// ---------------------------------------------------------------------------

Status ContextTable::ToIpc(std::vector<uint8_t>* data) {
  MaterializeIfDirty();

  // Filter tombstoned rows
  std::shared_ptr<arrow::RecordBatch> batch_to_write;
  if (!deleted_.empty()) {
    std::vector<int64_t> keep;
    keep.reserve(batch_->num_rows() - deleted_.size());
    for (int64_t i = 0; i < batch_->num_rows(); ++i) {
      if (!deleted_.count(i)) {
        keep.push_back(i);
      }
    }
    arrow::Int64Builder idx_builder;
    (void)idx_builder.AppendValues(keep);
    std::shared_ptr<arrow::Array> idx_array;
    (void)idx_builder.Finish(&idx_array);
    auto take_result = arrow::compute::Take(batch_, idx_array);
    if (!take_result.ok()) {
      return {Code::ExecutionError, take_result.status().ToString()};
    }
    batch_to_write = take_result->record_batch();
  } else {
    batch_to_write = batch_;
  }

  auto stream_result = arrow::io::BufferOutputStream::Create();
  if (!stream_result.ok()) {
    return {Code::ExecutionError, stream_result.status().ToString()};
  }
  auto stream = std::move(*stream_result);

  auto writer_result = arrow::ipc::MakeStreamWriter(stream, batch_to_write->schema());
  if (!writer_result.ok()) {
    return {Code::ExecutionError, writer_result.status().ToString()};
  }
  auto writer = std::move(*writer_result);

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->WriteRecordBatch(*batch_to_write));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->Close());

  auto buffer_result = stream->Finish();
  if (!buffer_result.ok()) {
    return {Code::ExecutionError, buffer_result.status().ToString()};
  }
  auto buffer = std::move(*buffer_result);
  data->resize(buffer->size());
  std::memcpy(data->data(), buffer->data(), buffer->size());
  return Status::OK();
}

arrow::Result<std::shared_ptr<ContextTable>> ContextTable::FromIpc(
    const uint8_t* data, int64_t size) {
  auto buffer = std::make_shared<arrow::Buffer>(data, size);
  auto input = std::make_shared<arrow::io::BufferReader>(buffer);

  auto reader_result = arrow::ipc::RecordBatchStreamReader::Open(input);
  ARROW_RETURN_NOT_OK(reader_result.status());
  auto reader = std::move(*reader_result);

  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
  if (!batch) {
    return arrow::Status::Invalid("Empty IPC stream");
  }
  return FromRecordBatch(batch);
}

// ---------------------------------------------------------------------------
// Distributed Operations
// ---------------------------------------------------------------------------

Status ContextTable::Broadcast(const std::shared_ptr<CylonContext>& ctx,
                               int root) {
  if (!ctx->IsDistributed()) {
    return Status::OK();
  }
  auto comm = ctx->GetCommunicator();

  MaterializeIfDirty();
  auto status = Compact();
  if (!status.is_ok()) {
    return status;
  }

  auto arrow_table = arrow::Table::FromRecordBatches({batch_});
  if (!arrow_table.ok()) {
    return {Code::ExecutionError, arrow_table.status().ToString()};
  }

  std::shared_ptr<cylon::Table> cylon_table;
  status = cylon::Table::FromArrowTable(ctx, std::move(*arrow_table), cylon_table);
  if (!status.is_ok()) {
    return status;
  }

  status = comm->Bcast(&cylon_table, root, ctx);
  if (!status.is_ok()) {
    return status;
  }

  std::shared_ptr<arrow::Table> result_arrow;
  status = cylon_table->ToArrowTable(result_arrow);
  if (!status.is_ok()) {
    return status;
  }

  auto combined = result_arrow->CombineChunksToBatch();
  if (!combined.ok()) {
    return {Code::ExecutionError, combined.status().ToString()};
  }
  batch_ = std::move(*combined);
  RebuildIndex();
  return Status::OK();
}

Status ContextTable::AllGather(const std::shared_ptr<CylonContext>& ctx) {
  if (!ctx->IsDistributed()) {
    return Status::OK();
  }
  auto comm = ctx->GetCommunicator();

  MaterializeIfDirty();
  auto status = Compact();
  if (!status.is_ok()) {
    return status;
  }

  auto arrow_table = arrow::Table::FromRecordBatches({batch_});
  if (!arrow_table.ok()) {
    return {Code::ExecutionError, arrow_table.status().ToString()};
  }

  std::shared_ptr<cylon::Table> cylon_table;
  status = cylon::Table::FromArrowTable(ctx, std::move(*arrow_table), cylon_table);
  if (!status.is_ok()) {
    return status;
  }

  std::vector<std::shared_ptr<cylon::Table>> gathered;
  status = comm->AllGather(cylon_table, &gathered);
  if (!status.is_ok()) {
    return status;
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches;
  for (auto& t : gathered) {
    std::shared_ptr<arrow::Table> at;
    status = t->ToArrowTable(at);
    if (!status.is_ok()) {
      return status;
    }
    auto batch_result = at->CombineChunksToBatch();
    if (!batch_result.ok()) {
      return {Code::ExecutionError, batch_result.status().ToString()};
    }
    all_batches.push_back(std::move(*batch_result));
  }

  if (all_batches.empty()) {
    return Status::OK();
  }

  std::vector<std::shared_ptr<arrow::Array>> merged_columns;
  for (int c = 0; c < schema_->num_fields(); ++c) {
    std::vector<std::shared_ptr<arrow::Array>> col_arrays;
    for (auto& b : all_batches) {
      col_arrays.push_back(b->column(c));
    }
    auto concat_result = arrow::Concatenate(col_arrays);
    if (!concat_result.ok()) {
      return {Code::ExecutionError, concat_result.status().ToString()};
    }
    merged_columns.push_back(std::move(*concat_result));
  }

  int64_t total_rows = 0;
  for (auto& b : all_batches) {
    total_rows += b->num_rows();
  }
  batch_ = arrow::RecordBatch::Make(schema_, total_rows, std::move(merged_columns));
  RebuildIndex();
  return Status::OK();
}

}  // namespace context
}  // namespace cylon