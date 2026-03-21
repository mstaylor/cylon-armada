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
//! `ContextTable` provides O(1) put/get/remove via a hash index over an Arrow
//! RecordBatch, with cosine similarity search on FixedSizeList<Float32>
//! embedding columns.
//!
//! Uses `cylon::simd` for SIMD-accelerated cosine similarity primitives.

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
use cylon::simd::{cosine_similarity_f32, batch_cosine_search, SearchResult};

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

/// Arrow-native key-value store for embeddings and metadata.
///
/// Uses a hash index for O(1) put/get/remove. Deleted rows are tracked in a
/// tombstone set and skipped during search. Call [`compact`] to reclaim space.
pub struct ContextTable {
    batch: RecordBatch,
    index: HashMap<String, usize>,
    deleted: HashSet<usize>,
    embedding_dim: usize,
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
    pub fn new(embedding_dim: usize) -> CylonResult<Self> {
        if embedding_dim == 0 {
            return Err(CylonError::Invalid("embedding_dim must be positive".into()));
        }
        let schema = make_schema(embedding_dim);
        let batch = RecordBatch::new_empty(schema);
        Ok(Self {
            batch,
            index: HashMap::new(),
            deleted: HashSet::new(),
            embedding_dim,
        })
    }

    pub fn from_record_batch(batch: RecordBatch) -> CylonResult<Self> {
        let dim = match batch.schema().field(EMBEDDING_COL).data_type() {
            DataType::FixedSizeList(_, size) => *size as usize,
            _ => return Err(CylonError::Invalid("embedding column is not FixedSizeList".into())),
        };
        let mut table = Self {
            batch,
            index: HashMap::new(),
            deleted: HashSet::new(),
            embedding_dim: dim,
        };
        table.rebuild_index();
        Ok(table)
    }

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

        if let Some(&old_idx) = self.index.get(context_id) {
            self.deleted.insert(old_idx);
            self.index.remove(context_id);
        }

        let new_row = self.build_row(context_id, embedding, &metadata)?;
        if self.batch.num_rows() == 0 {
            self.batch = new_row;
        } else {
            self.batch = arrow::compute::concat_batches(
                &self.batch.schema(),
                &[self.batch.clone(), new_row],
            ).map_err(CylonError::Arrow)?;
        }
        self.index.insert(context_id.to_string(), self.batch.num_rows() - 1);
        Ok(())
    }

    pub fn get(&self, context_id: &str) -> Option<RecordBatch> {
        self.index.get(context_id).map(|&idx| self.batch.slice(idx, 1))
    }

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

    pub fn search(
        &self,
        query: &[f32],
        threshold: f32,
        top_k: usize,
        workflow_id: Option<&str>,
    ) -> Vec<SearchResult> {
        if self.batch.num_rows() == 0 || query.len() != self.embedding_dim {
            return vec![];
        }

        let embedding_col = self.batch.column(EMBEDDING_COL)
            .as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let values = embedding_col.values()
            .as_any().downcast_ref::<Float32Array>().unwrap();
        let data = values.values();

        if self.deleted.is_empty() && workflow_id.is_none() {
            return batch_cosine_search(query, data, self.embedding_dim, threshold, top_k);
        }

        let wf_col = workflow_id.map(|_| {
            self.batch.column(WORKFLOW_ID_COL)
                .as_any().downcast_ref::<StringArray>().unwrap()
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
                results.push(SearchResult { index: i, similarity: sim });
            }
        }
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn get_workflow(&self, workflow_id: &str) -> CylonResult<RecordBatch> {
        let wf_col = self.batch.column(WORKFLOW_ID_COL)
            .as_any().downcast_ref::<StringArray>().unwrap();
        let indices: Vec<u64> = (0..self.batch.num_rows())
            .filter(|&i| {
                !self.deleted.contains(&i) && !wf_col.is_null(i) && wf_col.value(i) == workflow_id
            })
            .map(|i| i as u64)
            .collect();
        let idx_array = arrow::array::UInt64Array::from(indices);
        let columns: Vec<ArrayRef> = (0..self.batch.num_columns())
            .map(|c| compute::take(self.batch.column(c), &idx_array, None).unwrap())
            .collect();
        RecordBatch::try_new(self.batch.schema(), columns).map_err(CylonError::Arrow)
    }

    pub fn len(&self) -> usize { self.index.len() }
    pub fn is_empty(&self) -> bool { self.index.is_empty() }
    pub fn total_rows(&self) -> usize { self.batch.num_rows() }
    pub fn batch(&self) -> &RecordBatch { &self.batch }
    pub fn embedding_dim(&self) -> usize { self.embedding_dim }

    pub fn compact(&mut self) -> CylonResult<()> {
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
        self.batch = RecordBatch::try_new(self.batch.schema(), columns)
            .map_err(CylonError::Arrow)?;
        self.rebuild_index();
        Ok(())
    }

    pub fn to_ipc(&self) -> CylonResult<Vec<u8>> {
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
            RecordBatch::try_new(self.batch.schema(), columns)
                .map_err(CylonError::Arrow)?
        };

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &batch_to_write.schema())
                .map_err(CylonError::Arrow)?;
            writer.write(&batch_to_write).map_err(CylonError::Arrow)?;
            writer.finish().map_err(CylonError::Arrow)?;
        }
        Ok(buf)
    }

    pub fn from_ipc(data: &[u8]) -> CylonResult<Self> {
        let cursor = Cursor::new(data);
        let mut reader = StreamReader::try_new(cursor, None).map_err(CylonError::Arrow)?;
        let batch = reader
            .next()
            .ok_or_else(|| CylonError::Invalid("empty IPC stream".into()))?
            .map_err(CylonError::Arrow)?;
        Self::from_record_batch(batch)
    }

    /// Broadcast this ContextTable from root rank to all workers.
    pub fn broadcast(&mut self, ctx: &Arc<cylon::CylonContext>, root: i32) -> CylonResult<()> {
        let comm = match ctx.get_communicator() {
            Some(c) => c,
            None => return Ok(()),
        };

        self.compact()?;

        let cylon_table = cylon::Table::from_record_batch(ctx.clone(), self.batch.clone())?;
        let mut opt_table = Some(cylon_table);
        comm.bcast(&mut opt_table, root, ctx.clone())?;

        if let Some(t) = opt_table {
            if let Some(batch) = t.batch(0) {
                self.batch = batch.clone();
            }
        }
        self.rebuild_index();
        Ok(())
    }

    /// AllGather: each worker contributes its ContextTable, all receive merged result.
    pub fn all_gather(&mut self, ctx: &Arc<cylon::CylonContext>) -> CylonResult<()> {
        let comm = match ctx.get_communicator() {
            Some(c) => c,
            None => return Ok(()),
        };

        self.compact()?;

        let cylon_table = cylon::Table::from_record_batch(ctx.clone(), self.batch.clone())?;
        let gathered = comm.all_gather(&cylon_table, ctx.clone())?;

        let mut all_batches = Vec::new();
        for t in &gathered {
            for i in 0..t.num_batches() {
                if let Some(batch) = t.batch(i) {
                    if batch.num_rows() > 0 {
                        all_batches.push(batch.clone());
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

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.deleted.clear();
        let id_col = self.batch.column(CONTEXT_ID_COL)
            .as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..id_col.len() {
            if !id_col.is_null(i) {
                let key = id_col.value(i).to_string();
                if let Some(old_idx) = self.index.insert(key, i) {
                    self.deleted.insert(old_idx);
                }
            }
        }
    }

    fn build_row(
        &self,
        context_id: &str,
        embedding: &[f32],
        metadata: &ContextMetadata,
    ) -> CylonResult<RecordBatch> {
        let schema = self.batch.schema();

        let mut ctx_id = StringBuilder::new();
        ctx_id.append_value(context_id);

        let mut wf = StringBuilder::new();
        wf.append_value(&metadata.workflow_id);

        let values_builder = Float32Builder::new();
        let mut emb = FixedSizeListBuilder::new(values_builder, self.embedding_dim as i32);
        for &v in embedding {
            emb.values().append_value(v);
        }
        emb.append(true);

        let mut resp = LargeStringBuilder::new();
        resp.append_value(&metadata.response);

        let mut model = StringBuilder::new();
        model.append_value(&metadata.model_id);

        let mut in_tok = Int64Builder::new();
        in_tok.append_value(metadata.input_tokens);

        let mut out_tok = Int64Builder::new();
        out_tok.append_value(metadata.output_tokens);

        let mut cost = Float64Builder::new();
        cost.append_value(metadata.cost_usd);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let mut ts = TimestampMillisecondBuilder::new().with_timezone("UTC");
        ts.append_value(now);

        let mut reuse = Int64Builder::new();
        reuse.append_value(0);

        let columns: Vec<ArrayRef> = vec![
            Arc::new(ctx_id.finish()),
            Arc::new(wf.finish()),
            Arc::new(emb.finish()),
            Arc::new(resp.finish()),
            Arc::new(model.finish()),
            Arc::new(in_tok.finish()),
            Arc::new(out_tok.finish()),
            Arc::new(cost.finish()),
            Arc::new(ts.finish()),
            Arc::new(reuse.finish()),
        ];

        RecordBatch::try_new(schema, columns).map_err(CylonError::Arrow)
    }
}