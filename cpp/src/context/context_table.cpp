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

// Column indices in the schema
static constexpr int kContextIdCol = 0;
static constexpr int kWorkflowIdCol = 1;
static constexpr int kEmbeddingCol = 2;
static constexpr int kResponseCol = 3;
static constexpr int kModelIdCol = 4;
static constexpr int kInputTokensCol = 5;
static constexpr int kOutputTokensCol = 6;
static constexpr int kCostUsdCol = 7;
static constexpr int kCreatedAtCol = 8;
static constexpr int kReuseCountCol = 9;

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
// Construction
// ---------------------------------------------------------------------------

arrow::Result<std::shared_ptr<ContextTable>> ContextTable::Create(int embedding_dim) {
  if (embedding_dim <= 0) {
    return arrow::Status::Invalid("embedding_dim must be positive");
  }
  auto schema = MakeSchema(embedding_dim);

  // Create an empty RecordBatch with 0 rows
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
  table->batch_ = arrow::RecordBatch::Make(schema, 0, std::move(columns));
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

Status ContextTable::MakeFromIpc(const uint8_t* data, int64_t size,
                                 std::shared_ptr<ContextTable>* out) {
  auto result = FromIpc(data, size);
  if (!result.ok()) {
    return {Code::ExecutionError, result.status().ToString()};
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

  auto table = std::shared_ptr<ContextTable>(new ContextTable());
  table->embedding_dim_ = fsl_type->list_size();
  table->batch_ = batch;
  table->RebuildIndex();
  return table;
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
        // Duplicate key — tombstone the earlier row
        deleted_.insert(it->second);
      }
      index_[key] = i;
    }
  }
}

// ---------------------------------------------------------------------------
// Key-Value Operations — all O(1) amortized
// ---------------------------------------------------------------------------

arrow::Result<int64_t> ContextTable::AppendRow(
    const std::string& context_id,
    const float* embedding, int dim,
    const ContextMetadata& metadata) {
  auto schema = batch_->schema();
  auto pool = arrow::default_memory_pool();

  arrow::StringBuilder context_id_builder(pool);
  arrow::StringBuilder workflow_id_builder(pool);
  arrow::StringBuilder model_id_builder(pool);
  arrow::LargeStringBuilder response_builder(pool);
  arrow::Int64Builder input_tokens_builder(pool);
  arrow::Int64Builder output_tokens_builder(pool);
  arrow::DoubleBuilder cost_usd_builder(pool);
  arrow::TimestampBuilder created_at_builder(
      arrow::timestamp(arrow::TimeUnit::MILLI, "UTC"), pool);
  arrow::Int64Builder reuse_count_builder(pool);

  auto values_builder = std::make_shared<arrow::FloatBuilder>(pool);
  arrow::FixedSizeListBuilder embedding_builder(pool, values_builder,
                                                 embedding_dim_);

  ARROW_RETURN_NOT_OK(context_id_builder.Append(context_id));
  ARROW_RETURN_NOT_OK(workflow_id_builder.Append(metadata.workflow_id));
  ARROW_RETURN_NOT_OK(embedding_builder.Append());
  ARROW_RETURN_NOT_OK(values_builder->AppendValues(embedding, dim));
  ARROW_RETURN_NOT_OK(response_builder.Append(metadata.response));
  ARROW_RETURN_NOT_OK(model_id_builder.Append(metadata.model_id));
  ARROW_RETURN_NOT_OK(input_tokens_builder.Append(metadata.input_tokens));
  ARROW_RETURN_NOT_OK(output_tokens_builder.Append(metadata.output_tokens));
  ARROW_RETURN_NOT_OK(cost_usd_builder.Append(metadata.cost_usd));

  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count();
  ARROW_RETURN_NOT_OK(created_at_builder.Append(millis));
  ARROW_RETURN_NOT_OK(reuse_count_builder.Append(0));

  // Finish arrays
  std::shared_ptr<arrow::Array> ctx_arr, wf_arr, emb_arr, resp_arr, model_arr,
      in_tok_arr, out_tok_arr, cost_arr, ts_arr, reuse_arr;

  ARROW_RETURN_NOT_OK(context_id_builder.Finish(&ctx_arr));
  ARROW_RETURN_NOT_OK(workflow_id_builder.Finish(&wf_arr));
  ARROW_RETURN_NOT_OK(embedding_builder.Finish(&emb_arr));
  ARROW_RETURN_NOT_OK(response_builder.Finish(&resp_arr));
  ARROW_RETURN_NOT_OK(model_id_builder.Finish(&model_arr));
  ARROW_RETURN_NOT_OK(input_tokens_builder.Finish(&in_tok_arr));
  ARROW_RETURN_NOT_OK(output_tokens_builder.Finish(&out_tok_arr));
  ARROW_RETURN_NOT_OK(cost_usd_builder.Finish(&cost_arr));
  ARROW_RETURN_NOT_OK(created_at_builder.Finish(&ts_arr));
  ARROW_RETURN_NOT_OK(reuse_count_builder.Finish(&reuse_arr));

  auto new_row = arrow::RecordBatch::Make(schema, 1, {
      ctx_arr, wf_arr, emb_arr, resp_arr, model_arr,
      in_tok_arr, out_tok_arr, cost_arr, ts_arr, reuse_arr
  });

  // Concatenate with existing batch
  if (batch_->num_rows() == 0) {
    batch_ = std::move(new_row);
  } else {
    std::vector<std::shared_ptr<arrow::Array>> merged;
    merged.reserve(schema->num_fields());
    for (int c = 0; c < schema->num_fields(); ++c) {
      ARROW_ASSIGN_OR_RAISE(auto concat,
          arrow::Concatenate({batch_->column(c), new_row->column(c)}));
      merged.push_back(std::move(concat));
    }
    batch_ = arrow::RecordBatch::Make(
        schema, batch_->num_rows() + 1, std::move(merged));
  }

  return batch_->num_rows() - 1;
}

Status ContextTable::Put(const std::string& context_id,
                         const float* embedding, int dim,
                         const ContextMetadata& metadata) {
  if (dim != embedding_dim_) {
    return {Code::Invalid, "embedding dim mismatch: expected "
                           + std::to_string(embedding_dim_) + ", got "
                           + std::to_string(dim)};
  }

  // If key exists, tombstone the old row — O(1)
  auto it = index_.find(context_id);
  if (it != index_.end()) {
    deleted_.insert(it->second);
    index_.erase(it);
  }

  // Append new row — O(1) amortized (Arrow concatenate)
  auto row_result = AppendRow(context_id, embedding, dim, metadata);
  if (!row_result.ok()) {
    return {Code::ExecutionError, row_result.status().ToString()};
  }
  index_[context_id] = *row_result;
  return Status::OK();
}

std::optional<std::shared_ptr<arrow::RecordBatch>> ContextTable::Get(
    const std::string& context_id) {
  auto it = index_.find(context_id);
  if (it == index_.end()) {
    return std::nullopt;
  }
  return batch_->Slice(it->second, 1);
}

Status ContextTable::GetRow(const std::string& context_id,
                            std::shared_ptr<arrow::RecordBatch>* out) {
  auto it = index_.find(context_id);
  if (it == index_.end()) {
    *out = nullptr;
    return Status::OK();
  }
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
  if (!batch_ || batch_->num_rows() == 0) {
    return batch_;
  }

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
    auto schema = batch_->schema();
    std::vector<std::shared_ptr<arrow::Array>> empty_cols;
    for (int c = 0; c < schema->num_fields(); ++c) {
      std::unique_ptr<arrow::ArrayBuilder> builder;
      (void)arrow::MakeBuilder(arrow::default_memory_pool(),
                                schema->field(c)->type(), &builder);
      std::shared_ptr<arrow::Array> arr;
      (void)builder->Finish(&arr);
      empty_cols.push_back(std::move(arr));
    }
    return arrow::RecordBatch::Make(schema, 0, std::move(empty_cols));
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

// ---------------------------------------------------------------------------
// SIMD Search
// ---------------------------------------------------------------------------

std::vector<simd::SearchResult> ContextTable::Search(
    const float* query, int dim,
    float threshold, int top_k,
    const std::string& workflow_id) {
  if (!batch_ || batch_->num_rows() == 0 || dim != embedding_dim_) {
    return {};
  }

  // Fast path: no deletions, no workflow filter — pure zero-copy SIMD
  if (deleted_.empty() && workflow_id.empty()) {
    auto embedding_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
        batch_->column(kEmbeddingCol));
    return simd::batch_cosine_search_arrow(query, dim, embedding_col,
                                            threshold, top_k);
  }

  // Filtered path: skip tombstoned rows and optionally filter by workflow
  auto embedding_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
      batch_->column(kEmbeddingCol));
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
  return batch_ ? batch_->num_rows() : 0;
}

std::shared_ptr<arrow::RecordBatch> ContextTable::Batch() const {
  return batch_;
}

std::shared_ptr<arrow::Schema> ContextTable::GetSchema() const {
  return batch_ ? batch_->schema() : nullptr;
}

// ---------------------------------------------------------------------------
// Compact — removes tombstoned rows, rebuilds batch and index
// ---------------------------------------------------------------------------

Status ContextTable::Compact() {
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

// ---------------------------------------------------------------------------
// Arrow IPC Serialization
// ---------------------------------------------------------------------------

Status ContextTable::ToIpc(std::vector<uint8_t>* data) const {
  // Filter out tombstoned rows before serializing
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

  auto status = Compact();
  if (!status.is_ok()) {
    return status;
  }

  // RecordBatch → arrow::Table → cylon::Table (O(1) pointer wraps)
  auto arrow_table = arrow::Table::FromRecordBatches({batch_});
  if (!arrow_table.ok()) {
    return {Code::ExecutionError, arrow_table.status().ToString()};
  }

  std::shared_ptr<cylon::Table> cylon_table;
  status = cylon::Table::FromArrowTable(ctx, std::move(*arrow_table), cylon_table);
  if (!status.is_ok()) {
    return status;
  }

  // Delegate to existing communicator Bcast
  status = comm->Bcast(&cylon_table, root, ctx);
  if (!status.is_ok()) {
    return status;
  }

  // cylon::Table → RecordBatch → rebuild index
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

  auto status = Compact();
  if (!status.is_ok()) {
    return status;
  }

  // RecordBatch → arrow::Table → cylon::Table (O(1) pointer wraps)
  auto arrow_table = arrow::Table::FromRecordBatches({batch_});
  if (!arrow_table.ok()) {
    return {Code::ExecutionError, arrow_table.status().ToString()};
  }

  std::shared_ptr<cylon::Table> cylon_table;
  status = cylon::Table::FromArrowTable(ctx, std::move(*arrow_table), cylon_table);
  if (!status.is_ok()) {
    return status;
  }

  // Delegate to existing communicator AllGather
  std::vector<std::shared_ptr<cylon::Table>> gathered;
  status = comm->AllGather(cylon_table, &gathered);
  if (!status.is_ok()) {
    return status;
  }

  // Merge gathered tables into a single RecordBatch
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

  // Concatenate column-by-column
  auto schema = all_batches[0]->schema();
  std::vector<std::shared_ptr<arrow::Array>> merged_columns;
  for (int c = 0; c < schema->num_fields(); ++c) {
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
  batch_ = arrow::RecordBatch::Make(schema, total_rows, std::move(merged_columns));
  RebuildIndex();
  return Status::OK();
}

}  // namespace context
}  // namespace cylon