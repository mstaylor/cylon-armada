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

#include "common/test_header.hpp"

#include <cylon/simd/simd_ops.hpp>
#include <cylon/context/context_table.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace cylon;

namespace {

// Generate a normalized random embedding
std::vector<float> make_embedding(int dim, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(dim);
  float norm = 0.0f;
  for (auto& x : v) {
    x = dist(gen);
    norm += x * x;
  }
  norm = std::sqrt(norm);
  for (auto& x : v) {
    x /= norm;
  }
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// SIMD Tests
// ---------------------------------------------------------------------------

TEST_CASE("SIMD cosine_similarity_f32 - identical vectors", "[simd]") {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  float sim = simd::cosine_similarity_f32(a.data(), a.data(), 4);
  REQUIRE(std::abs(sim - 1.0f) < 1e-5f);
}

TEST_CASE("SIMD cosine_similarity_f32 - orthogonal vectors", "[simd]") {
  std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
  float sim = simd::cosine_similarity_f32(a.data(), b.data(), 4);
  REQUIRE(std::abs(sim) < 1e-5f);
}

TEST_CASE("SIMD cosine_similarity_f32 - opposite vectors", "[simd]") {
  std::vector<float> a = {1.0f, 2.0f, 3.0f};
  std::vector<float> b = {-1.0f, -2.0f, -3.0f};
  float sim = simd::cosine_similarity_f32(a.data(), b.data(), 3);
  REQUIRE(std::abs(sim + 1.0f) < 1e-5f);
}

TEST_CASE("SIMD cosine_similarity_f32 - zero vector", "[simd]") {
  std::vector<float> a = {1.0f, 2.0f, 3.0f};
  std::vector<float> b = {0.0f, 0.0f, 0.0f};
  float sim = simd::cosine_similarity_f32(a.data(), b.data(), 3);
  REQUIRE(sim == 0.0f);
}

TEST_CASE("SIMD cosine_similarity_f32 - large dimension (1024)", "[simd]") {
  auto a = make_embedding(1024, 42);
  auto b = make_embedding(1024, 42);
  float sim = simd::cosine_similarity_f32(a.data(), b.data(), 1024);
  REQUIRE(std::abs(sim - 1.0f) < 1e-4f);
}

TEST_CASE("SIMD batch_cosine_search - basic", "[simd]") {
  const int dim = 4;
  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  // Row 0: identical to query, row 1: orthogonal, row 2: similar
  std::vector<float> embeddings = {
      1.0f, 0.0f, 0.0f, 0.0f,   // sim = 1.0
      0.0f, 1.0f, 0.0f, 0.0f,   // sim = 0.0
      0.9f, 0.1f, 0.0f, 0.0f,   // sim ~ 0.994
  };

  auto results = simd::batch_cosine_search(
      query.data(), dim, embeddings.data(), 3, 0.5f, 10);

  REQUIRE(results.size() == 2);
  REQUIRE(results[0].index == 0);
  REQUIRE(std::abs(results[0].similarity - 1.0f) < 1e-5f);
  REQUIRE(results[1].index == 2);
  REQUIRE(results[1].similarity > 0.99f);
}

TEST_CASE("SIMD batch_cosine_search - top_k limit", "[simd]") {
  const int dim = 4;
  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> embeddings = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.9f, 0.1f, 0.0f, 0.0f,
      0.8f, 0.2f, 0.0f, 0.0f,
  };

  auto results = simd::batch_cosine_search(
      query.data(), dim, embeddings.data(), 3, 0.0f, 1);

  REQUIRE(results.size() == 1);
  REQUIRE(results[0].index == 0);
}

// ---------------------------------------------------------------------------
// ContextTable Tests
// ---------------------------------------------------------------------------

TEST_CASE("ContextTable - create empty", "[context]") {
  auto result = context::ContextTable::Create(256);
  REQUIRE(result.ok());
  auto table = *result;
  REQUIRE(table->Size() == 0);
  REQUIRE(table->TotalRows() == 0);
  REQUIRE(table->EmbeddingDim() == 256);
  REQUIRE(table->GetSchema()->num_fields() == 10);
}

TEST_CASE("ContextTable - create with invalid dim", "[context]") {
  auto result = context::ContextTable::Create(0);
  REQUIRE(!result.ok());

  result = context::ContextTable::Create(-1);
  REQUIRE(!result.ok());
}

TEST_CASE("ContextTable - put and get", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  auto emb = make_embedding(dim, 1);
  context::ContextMetadata meta;
  meta.workflow_id = "wf-1";
  meta.response = "Hello world";
  meta.model_id = "claude-3-haiku";
  meta.input_tokens = 100;
  meta.output_tokens = 50;
  meta.cost_usd = 0.003;

  auto status = table->Put("ctx-1", emb.data(), dim, meta);
  REQUIRE(status.is_ok());
  REQUIRE(table->Size() == 1);

  auto row = table->Get("ctx-1");
  REQUIRE(row.has_value());
  auto batch = *row;
  REQUIRE(batch->num_rows() == 1);

  // Verify context_id
  auto id_arr = std::static_pointer_cast<arrow::StringArray>(batch->column(0));
  REQUIRE(id_arr->GetString(0) == "ctx-1");

  // Verify workflow_id
  auto wf_arr = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
  REQUIRE(wf_arr->GetString(0) == "wf-1");

  // Missing key returns nullopt
  REQUIRE(!table->Get("nonexistent").has_value());
}

TEST_CASE("ContextTable - put upsert replaces old row", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  auto emb1 = make_embedding(dim, 1);
  context::ContextMetadata meta1;
  meta1.workflow_id = "wf-1";
  meta1.response = "first";

  table->Put("ctx-1", emb1.data(), dim, meta1);
  REQUIRE(table->Size() == 1);

  auto emb2 = make_embedding(dim, 2);
  context::ContextMetadata meta2;
  meta2.workflow_id = "wf-1";
  meta2.response = "second";

  table->Put("ctx-1", emb2.data(), dim, meta2);
  REQUIRE(table->Size() == 1);        // Still 1 active
  REQUIRE(table->TotalRows() == 2);   // Old row tombstoned

  auto row = table->Get("ctx-1");
  REQUIRE(row.has_value());
  auto resp = std::static_pointer_cast<arrow::LargeStringArray>(
      (*row)->column(3));
  REQUIRE(resp->GetString(0) == "second");
}

TEST_CASE("ContextTable - remove", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  auto emb = make_embedding(dim, 1);
  context::ContextMetadata meta;
  meta.workflow_id = "wf-1";

  table->Put("ctx-1", emb.data(), dim, meta);
  REQUIRE(table->Size() == 1);

  auto status = table->Remove("ctx-1");
  REQUIRE(status.is_ok());
  REQUIRE(table->Size() == 0);
  REQUIRE(table->TotalRows() == 1);  // Tombstoned, not removed
  REQUIRE(!table->Get("ctx-1").has_value());

  // Remove nonexistent
  status = table->Remove("ctx-1");
  REQUIRE(!status.is_ok());
}

TEST_CASE("ContextTable - compact removes tombstones", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  for (int i = 0; i < 5; ++i) {
    auto emb = make_embedding(dim, i);
    context::ContextMetadata meta;
    meta.workflow_id = "wf-1";
    table->Put("ctx-" + std::to_string(i), emb.data(), dim, meta);
  }
  REQUIRE(table->Size() == 5);
  REQUIRE(table->TotalRows() == 5);

  table->Remove("ctx-1");
  table->Remove("ctx-3");
  REQUIRE(table->Size() == 3);
  REQUIRE(table->TotalRows() == 5);

  auto status = table->Compact();
  REQUIRE(status.is_ok());
  REQUIRE(table->Size() == 3);
  REQUIRE(table->TotalRows() == 3);  // Tombstones removed

  REQUIRE(table->Get("ctx-0").has_value());
  REQUIRE(table->Get("ctx-2").has_value());
  REQUIRE(table->Get("ctx-4").has_value());
  REQUIRE(!table->Get("ctx-1").has_value());
  REQUIRE(!table->Get("ctx-3").has_value());
}

TEST_CASE("ContextTable - search", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  // Insert three entries with known embeddings
  std::vector<float> e1 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> e2 = {0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> e3 = {0.9f, 0.1f, 0.0f, 0.0f};

  context::ContextMetadata meta;
  meta.workflow_id = "wf-1";
  table->Put("ctx-1", e1.data(), dim, meta);
  table->Put("ctx-2", e2.data(), dim, meta);
  table->Put("ctx-3", e3.data(), dim, meta);

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  auto results = table->Search(query.data(), dim, 0.5f, 10);

  REQUIRE(results.size() == 2);  // ctx-1 and ctx-3
  REQUIRE(results[0].similarity > results[1].similarity);
}

TEST_CASE("ContextTable - search skips tombstoned rows", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  std::vector<float> e1 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> e2 = {0.9f, 0.1f, 0.0f, 0.0f};

  context::ContextMetadata meta;
  meta.workflow_id = "wf-1";
  table->Put("ctx-1", e1.data(), dim, meta);
  table->Put("ctx-2", e2.data(), dim, meta);

  table->Remove("ctx-1");

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  auto results = table->Search(query.data(), dim, 0.5f, 10);

  REQUIRE(results.size() == 1);
  // The remaining result should be ctx-2
}

TEST_CASE("ContextTable - search with workflow filter", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  std::vector<float> e1 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> e2 = {0.9f, 0.1f, 0.0f, 0.0f};

  context::ContextMetadata meta1;
  meta1.workflow_id = "wf-1";
  context::ContextMetadata meta2;
  meta2.workflow_id = "wf-2";

  table->Put("ctx-1", e1.data(), dim, meta1);
  table->Put("ctx-2", e2.data(), dim, meta2);

  std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
  auto results = table->Search(query.data(), dim, 0.0f, 10, "wf-1");

  REQUIRE(results.size() == 1);
}

TEST_CASE("ContextTable - get workflow", "[context]") {
  const int dim = 4;
  auto table = *context::ContextTable::Create(dim);

  auto emb = make_embedding(dim, 1);
  context::ContextMetadata meta1;
  meta1.workflow_id = "wf-1";
  context::ContextMetadata meta2;
  meta2.workflow_id = "wf-2";

  table->Put("ctx-1", emb.data(), dim, meta1);
  table->Put("ctx-2", emb.data(), dim, meta2);
  table->Put("ctx-3", emb.data(), dim, meta1);

  auto wf1_batch = table->GetWorkflow("wf-1");
  REQUIRE(wf1_batch->num_rows() == 2);

  auto wf2_batch = table->GetWorkflow("wf-2");
  REQUIRE(wf2_batch->num_rows() == 1);

  auto wf3_batch = table->GetWorkflow("wf-nonexistent");
  REQUIRE(wf3_batch->num_rows() == 0);
}

TEST_CASE("ContextTable - IPC round-trip", "[context]") {
  const int dim = 8;
  auto table = *context::ContextTable::Create(dim);

  for (int i = 0; i < 3; ++i) {
    auto emb = make_embedding(dim, i);
    context::ContextMetadata meta;
    meta.workflow_id = "wf-1";
    meta.response = "response-" + std::to_string(i);
    meta.model_id = "test-model";
    meta.input_tokens = 100 + i;
    meta.output_tokens = 50 + i;
    meta.cost_usd = 0.001 * (i + 1);
    table->Put("ctx-" + std::to_string(i), emb.data(), dim, meta);
  }

  // Serialize
  std::vector<uint8_t> ipc_data;
  auto status = table->ToIpc(&ipc_data);
  REQUIRE(status.is_ok());
  REQUIRE(ipc_data.size() > 0);

  // Deserialize
  auto result = context::ContextTable::FromIpc(ipc_data.data(), ipc_data.size());
  REQUIRE(result.ok());
  auto restored = *result;

  REQUIRE(restored->Size() == 3);
  REQUIRE(restored->EmbeddingDim() == dim);

  // Verify data survived
  for (int i = 0; i < 3; ++i) {
    auto key = "ctx-" + std::to_string(i);
    auto row = restored->Get(key);
    REQUIRE(row.has_value());
    auto resp = std::static_pointer_cast<arrow::LargeStringArray>(
        (*row)->column(3));
    REQUIRE(resp->GetString(0) == "response-" + std::to_string(i));
  }
}

TEST_CASE("ContextTable - dim mismatch rejected", "[context]") {
  auto table = *context::ContextTable::Create(4);
  std::vector<float> emb(8, 1.0f);  // Wrong dimension
  context::ContextMetadata meta;
  auto status = table->Put("ctx-1", emb.data(), 8, meta);
  REQUIRE(!status.is_ok());
}