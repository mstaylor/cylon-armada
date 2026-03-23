// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Arrow-native key-value store for embeddings, LLM responses, and metadata.
//!
//! `ContextTable` provides O(1) put/get/remove via a hash index backed by
//! Arrow builders for O(1) amortized inserts at any scale.
//!
//! Internally, data lives in Arrow builders during inserts. When a read
//! operation needs the data (search, get, serialize), the builders are
//! materialized into a RecordBatch. This avoids O(n) copies per insert.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, FixedSizeListBuilder, Float32Array, Float32Builder,
    Float64Builder, Int64Builder, LargeStringBuilder, StringBuilder, StringArray,
    TimestampMillisecondBuilder,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;

use cylon::error::{CylonError, CylonResult};
use cylon::simd::{batch_cosine_search, cosine_similarity_f32, SearchResult};

// Column indices
const CONTEXT_ID_COL: usize = 0;
const WORKFLOW_ID_COL: usize = 1;
const EMBEDDING_COL: usize = 2;

/// Metadata associated with a context entry.
#[derive(Debug, Clone, Default)]
pub struct ContextMetadata {
    pub workflow_id: String,
    pub response: String,
    pub model_id: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cost_usd: f64,
}

/// Arrow builders for all columns. Supports O(1) append.
struct Builders {
    context_id: StringBuilder,
    workflow_id: StringBuilder,
    embedding: FixedSizeListBuilder<Float32Builder>,
    response: LargeStringBuilder,
    model_id: StringBuilder,
    input_tokens: Int64Builder,
    output_tokens: Int64Builder,
    cost_usd: Float64Builder,
    created_at: TimestampMillisecondBuilder,
    reuse_count: Int64Builder,
}

impl Builders {
    fn new(embedding_dim: usize) -> Self {
        Self {
            context_id: StringBuilder::new(),
            workflow_id: StringBuilder::new(),
            embedding: FixedSizeListBuilder::new(Float32Builder::new(), embedding_dim as i32),
            response: LargeStringBuilder::new(),
            model_id: StringBuilder::new(),
            input_tokens: Int64Builder::new(),
            output_tokens: Int64Builder::new(),
            cost_usd: Float64Builder::new(),
            created_at: TimestampMillisecondBuilder::new().with_timezone("UTC"),
            reuse_count: Int64Builder::new(),
        }
    }

    fn append(
        &mut self,
        context_id: &str,
        embedding: &[f32],
        metadata: &ContextMetadata,
    ) {
        self.context_id.append_value(context_id);
        self.workflow_id.append_value(&metadata.workflow_id);
        for &v in embedding {
            self.embedding.values().append_value(v);
        }
        self.embedding.append(true);
        self.response.append_value(&metadata.response);
        self.model_id.append_value(&metadata.model_id);
        self.input_tokens.append_value(metadata.input_tokens);
        self.output_tokens.append_value(metadata.output_tokens);
        self.cost_usd.append_value(metadata.cost_usd);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        self.created_at.append_value(now);
        self.reuse_count.append_value(0);
    }

    fn finish(&mut self, schema: &Arc<Schema>) -> CylonResult<RecordBatch> {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.context_id.finish()),
            Arc::new(self.workflow_id.finish()),
            Arc::new(self.embedding.finish()),
            Arc::new(self.response.finish()),
            Arc::new(self.model_id.finish()),
            Arc::new(self.input_tokens.finish()),
            Arc::new(self.output_tokens.finish()),
            Arc::new(self.cost_usd.finish()),
            Arc::new(self.created_at.finish()),
            Arc::new(self.reuse_count.finish()),
        ];
        RecordBatch::try_new(schema.clone(), columns).map_err(CylonError::Arrow)
    }

}

/// Arrow-native key-value store for embeddings and metadata.
///
/// Uses Arrow builders for O(1) inserts. Data is materialized into a
/// RecordBatch lazily before reads (search, get, serialize). Deleted rows
/// are tracked in a tombstone set. Call [`compact`] to reclaim space.
pub struct ContextTable {
    /// Materialized batch — may be stale if `dirty` is true.
    batch: RecordBatch,
    /// Pending inserts not yet in `batch`.
    builders: Builders,
    /// True if builders have data not yet merged into batch.
    dirty: bool,
    /// context_id → row index in the materialized batch.
    index: HashMap<String, usize>,
    /// Tombstoned row indices.
    deleted: HashSet<usize>,
    /// Number of rows in builders (tracked for index calculation).
    builder_count: usize,
    embedding_dim: usize,
    schema: Arc<Schema>,
}

fn make_schema(embedding_dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("context_id", DataType::Utf8, true),
        Field::new("workflow_id", DataType::Utf8, true),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            true,
        ),
        Field::new("response", DataType::LargeUtf8, true),
        Field::new("model_id", DataType::Utf8, true),
        Field::new("input_tokens", DataType::Int64, true),
        Field::new("output_tokens", DataType::Int64, true),
        Field::new("cost_usd", DataType::Float64, true),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            true,
        ),
        Field::new("reuse_count", DataType::Int64, true),
    ]))
}

impl ContextTable {
    /// Create an empty ContextTable with the given embedding dimension.
    pub fn new(embedding_dim: usize) -> CylonResult<Self> {
        if embedding_dim == 0 {
            return Err(CylonError::Invalid(
                "embedding_dim must be positive".into(),
            ));
        }
        let schema = make_schema(embedding_dim);
        let batch = RecordBatch::new_empty(schema.clone());
        Ok(Self {
            batch,
            builders: Builders::new(embedding_dim),
            dirty: false,
            index: HashMap::new(),
            deleted: HashSet::new(),
            builder_count: 0,
            embedding_dim,
            schema,
        })
    }

    /// Reconstruct from a RecordBatch.
    pub fn from_record_batch(batch: RecordBatch) -> CylonResult<Self> {
        let expected_fields = make_schema(1).fields().len();
        if batch.num_columns() != expected_fields {
            return Err(CylonError::Invalid(format!(
                "expected {} columns, got {}",
                expected_fields,
                batch.num_columns()
            )));
        }
        let dim = match batch.schema().field(EMBEDDING_COL).data_type() {
            DataType::FixedSizeList(inner, size) => {
                if inner.data_type() != &DataType::Float32 {
                    return Err(CylonError::Invalid(format!(
                        "embedding inner type must be Float32, got {:?}",
                        inner.data_type()
                    )));
                }
                *size as usize
            }
            _ => {
                return Err(CylonError::Invalid(
                    "embedding column is not FixedSizeList".into(),
                ))
            }
        };
        let schema = batch.schema();
        let mut table = Self {
            batch,
            builders: Builders::new(dim),
            dirty: false,
            index: HashMap::new(),
            deleted: HashSet::new(),
            builder_count: 0,
            embedding_dim: dim,
            schema,
        };
        table.rebuild_index();
        Ok(table)
    }

    /// Insert or update a context entry. O(1).
    pub fn put(
        &mut self,
        context_id: &str,
        embedding: &[f32],
        metadata: ContextMetadata,
    ) -> CylonResult<()> {
        if embedding.len() != self.embedding_dim {
            return Err(CylonError::Invalid(format!(
                "embedding dim mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        // Tombstone old row if key exists
        if let Some(&old_idx) = self.index.get(context_id) {
            self.deleted.insert(old_idx);
            self.index.remove(context_id);
        }

        // Append to builders — O(1)
        self.builders.append(context_id, embedding, &metadata);
        let new_idx = self.batch.num_rows() + self.builder_count;
        self.index.insert(context_id.to_string(), new_idx);
        self.builder_count += 1;
        self.dirty = true;
        Ok(())
    }

    /// Retrieve a single row by context_id. O(1).
    pub fn get(&mut self, context_id: &str) -> Option<RecordBatch> {
        if !self.index.contains_key(context_id) {
            return None;
        }
        if self.materialize_if_dirty().is_err() {
            return None;
        }
        self.index
            .get(context_id)
            .map(|&idx| self.batch.slice(idx, 1))
    }

    /// Mark a row as deleted. O(1).
    pub fn remove(&mut self, context_id: &str) -> CylonResult<()> {
        match self.index.remove(context_id) {
            Some(idx) => {
                self.deleted.insert(idx);
                Ok(())
            }
            None => Err(CylonError::Generic {
                code: cylon::error::Code::KeyError,
                message: format!("context_id not found: {}", context_id),
            }),
        }
    }

    /// Cosine similarity search.
    pub fn search(
        &mut self,
        query: &[f32],
        threshold: f32,
        top_k: usize,
        workflow_id: Option<&str>,
    ) -> Vec<SearchResult> {
        if self.materialize_if_dirty().is_err() {
            return vec![];
        }

        if self.batch.num_rows() == 0 || query.len() != self.embedding_dim {
            return vec![];
        }

        let embedding_col = self
            .batch
            .column(EMBEDDING_COL)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let values = embedding_col
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let data = values.values();

        if self.deleted.is_empty() && workflow_id.is_none() {
            return batch_cosine_search(query, data, self.embedding_dim, threshold, top_k);
        }

        let wf_col = workflow_id.map(|_| {
            self.batch
                .column(WORKFLOW_ID_COL)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        });

        let dim = self.embedding_dim;
        let mut results: Vec<SearchResult> = Vec::new();
        for i in 0..self.batch.num_rows() {
            if self.deleted.contains(&i) {
                continue;
            }
            if let Some(wf_array) = wf_col {
                if let Some(wf_filter) = workflow_id {
                    if wf_array.is_null(i) || wf_array.value(i) != wf_filter {
                        continue;
                    }
                }
            }
            let row = &data[i * dim..(i + 1) * dim];
            let sim = cosine_similarity_f32(query, row);
            if sim >= threshold {
                results.push(SearchResult {
                    index: i,
                    similarity: sim,
                });
            }
        }
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(top_k);
        results
    }

    /// Retrieve all rows for a given workflow_id.
    pub fn get_workflow(&mut self, workflow_id: &str) -> CylonResult<RecordBatch> {
        self.materialize_if_dirty()?;

        let wf_col = self
            .batch
            .column(WORKFLOW_ID_COL)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let indices: Vec<u64> = (0..self.batch.num_rows())
            .filter(|&i| {
                !self.deleted.contains(&i)
                    && !wf_col.is_null(i)
                    && wf_col.value(i) == workflow_id
            })
            .map(|i| i as u64)
            .collect();
        let idx_array = arrow::array::UInt64Array::from(indices);
        let columns: Vec<ArrayRef> = (0..self.batch.num_columns())
            .map(|c| compute::take(self.batch.column(c), &idx_array, None).unwrap())
            .collect();
        RecordBatch::try_new(self.batch.schema(), columns).map_err(CylonError::Arrow)
    }

    /// Number of active (non-deleted) rows.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total rows including tombstoned.
    pub fn total_rows(&self) -> usize {
        self.batch.num_rows() + self.builder_count
    }

    /// The underlying RecordBatch (materializes if dirty).
    pub fn batch(&mut self) -> &RecordBatch {
        let _ = self.materialize_if_dirty();
        &self.batch
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Remove tombstoned rows and rebuild.
    pub fn compact(&mut self) -> CylonResult<()> {
        self.materialize_if_dirty()?;

        if self.deleted.is_empty() {
            return Ok(());
        }
        let keep: Vec<u64> = (0..self.batch.num_rows() as u64)
            .filter(|i| !self.deleted.contains(&(*i as usize)))
            .collect();
        let idx_array = arrow::array::UInt64Array::from(keep);
        let columns: Vec<ArrayRef> = (0..self.batch.num_columns())
            .map(|c| compute::take(self.batch.column(c), &idx_array, None).unwrap())
            .collect();
        self.batch =
            RecordBatch::try_new(self.batch.schema(), columns).map_err(CylonError::Arrow)?;
        self.rebuild_index();
        Ok(())
    }

    /// Serialize to Arrow IPC. Filters tombstoned rows.
    pub fn to_ipc(&mut self) -> CylonResult<Vec<u8>> {
        self.materialize_if_dirty()?;

        let batch_to_write = if self.deleted.is_empty() {
            self.batch.clone()
        } else {
            let keep: Vec<u64> = (0..self.batch.num_rows() as u64)
                .filter(|i| !self.deleted.contains(&(*i as usize)))
                .collect();
            let idx_array = arrow::array::UInt64Array::from(keep);
            let columns: Vec<ArrayRef> = (0..self.batch.num_columns())
                .map(|c| compute::take(self.batch.column(c), &idx_array, None).unwrap())
                .collect();
            RecordBatch::try_new(self.batch.schema(), columns).map_err(CylonError::Arrow)?
        };

        let mut buf = Vec::new();
        {
            let mut writer =
                StreamWriter::try_new(&mut buf, &batch_to_write.schema())
                    .map_err(CylonError::Arrow)?;
            writer
                .write(&batch_to_write)
                .map_err(CylonError::Arrow)?;
            writer.finish().map_err(CylonError::Arrow)?;
        }
        Ok(buf)
    }

    /// Deserialize from Arrow IPC.
    pub fn from_ipc(data: &[u8]) -> CylonResult<Self> {
        let cursor = Cursor::new(data);
        let mut reader = StreamReader::try_new(cursor, None).map_err(CylonError::Arrow)?;
        let batch = reader
            .next()
            .ok_or_else(|| CylonError::Invalid("empty IPC stream".into()))?
            .map_err(CylonError::Arrow)?;
        Self::from_record_batch(batch)
    }

    /// Broadcast from root rank to all workers.
    pub fn broadcast(
        &mut self,
        ctx: &Arc<cylon::CylonContext>,
        root: i32,
    ) -> CylonResult<()> {
        let comm = match ctx.get_communicator() {
            Some(c) => c,
            None => return Ok(()),
        };

        self.materialize_if_dirty()?;
        self.compact()?;

        let cylon_table =
            cylon::Table::from_record_batch(ctx.clone(), self.batch.clone())?;
        let mut opt_table = Some(cylon_table);
        comm.bcast(&mut opt_table, root, ctx.clone())?;

        if let Some(t) = opt_table {
            if let Some(b) = t.batch(0) {
                self.batch = b.clone();
            }
        }
        self.rebuild_index();
        Ok(())
    }

    /// AllGather: each worker contributes, all receive merged result.
    pub fn all_gather(&mut self, ctx: &Arc<cylon::CylonContext>) -> CylonResult<()> {
        let comm = match ctx.get_communicator() {
            Some(c) => c,
            None => return Ok(()),
        };

        self.materialize_if_dirty()?;
        self.compact()?;

        let cylon_table =
            cylon::Table::from_record_batch(ctx.clone(), self.batch.clone())?;
        let gathered = comm.all_gather(&cylon_table, ctx.clone())?;

        let mut all_batches = Vec::new();
        for t in &gathered {
            for i in 0..t.num_batches() {
                if let Some(b) = t.batch(i) {
                    if b.num_rows() > 0 {
                        all_batches.push(b.clone());
                    }
                }
            }
        }

        if all_batches.is_empty() {
            return Ok(());
        }

        let schema = all_batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &all_batches)
            .map_err(CylonError::Arrow)?;
        self.batch = merged;
        self.rebuild_index();
        Ok(())
    }

    // --- Internal ---

    /// Flush builders into the main batch. Called before any read.
    fn materialize_if_dirty(&mut self) -> CylonResult<()> {
        if !self.dirty || self.builder_count == 0 {
            return Ok(());
        }

        let new_batch = self.builders.finish(&self.schema)?;
        if self.batch.num_rows() == 0 {
            self.batch = new_batch;
        } else {
            self.batch = arrow::compute::concat_batches(
                &self.schema,
                &[self.batch.clone(), new_batch],
            )
            .map_err(CylonError::Arrow)?;
        }

        self.builders = Builders::new(self.embedding_dim);
        self.builder_count = 0;
        self.dirty = false;
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.deleted.clear();
        let id_col = self
            .batch
            .column(CONTEXT_ID_COL)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..id_col.len() {
            if !id_col.is_null(i) {
                let key = id_col.value(i).to_string();
                if let Some(old_idx) = self.index.insert(key, i) {
                    self.deleted.insert(old_idx);
                }
            }
        }
    }
}