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

//! WASM bindings for the Arrow-native ContextTable.
//!
//! Exposes `WasmContextTable` to JavaScript/Node.js via wasm-bindgen.
//! Metadata is passed as JSON strings for JS compatibility.
//!
//! Distributed operations (broadcast/all_gather) require host imports
//! from cylon-wasm and should be composed at the application level.

use wasm_bindgen::prelude::*;

use cylon_armada::context::{ContextMetadata, ContextTable};

#[wasm_bindgen]
pub struct WasmContextTable {
    inner: ContextTable,
}

#[wasm_bindgen]
impl WasmContextTable {
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dim: usize) -> Result<WasmContextTable, JsValue> {
        let inner = ContextTable::new(embedding_dim)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        Ok(Self { inner })
    }

    pub fn put(
        &mut self,
        context_id: &str,
        embedding: &[f32],
        metadata_json: &str,
    ) -> Result<(), JsValue> {
        let metadata = parse_metadata(metadata_json)?;
        self.inner
            .put(context_id, embedding, metadata)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    pub fn get(&mut self, context_id: &str) -> Result<JsValue, JsValue> {
        match self.inner.get(context_id) {
            Some(batch) => {
                let id_col = batch.column(0).as_any()
                    .downcast_ref::<arrow::array::StringArray>().unwrap();
                let wf_col = batch.column(1).as_any()
                    .downcast_ref::<arrow::array::StringArray>().unwrap();
                let resp_col = batch.column(3).as_any()
                    .downcast_ref::<arrow::array::LargeStringArray>().unwrap();
                let model_col = batch.column(4).as_any()
                    .downcast_ref::<arrow::array::StringArray>().unwrap();
                let in_tok = batch.column(5).as_any()
                    .downcast_ref::<arrow::array::Int64Array>().unwrap();
                let out_tok = batch.column(6).as_any()
                    .downcast_ref::<arrow::array::Int64Array>().unwrap();
                let cost = batch.column(7).as_any()
                    .downcast_ref::<arrow::array::Float64Array>().unwrap();

                let row = serde_json::json!({
                    "context_id": id_col.value(0),
                    "workflow_id": wf_col.value(0),
                    "response": resp_col.value(0),
                    "model_id": model_col.value(0),
                    "input_tokens": in_tok.value(0),
                    "output_tokens": out_tok.value(0),
                    "cost_usd": cost.value(0),
                });
                let json = serde_json::to_string(&row)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                Ok(JsValue::from_str(&json))
            }
            None => Ok(JsValue::NULL),
        }
    }

    pub fn remove(&mut self, context_id: &str) -> Result<(), JsValue> {
        self.inner
            .remove(context_id)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    pub fn search(
        &mut self,
        query: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let results = self.inner.search(query, threshold, top_k, None);
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"index": r.index, "similarity": r.similarity}))
            .collect();
        serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn search_workflow(
        &mut self,
        query: &[f32],
        workflow_id: &str,
        threshold: f32,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let results = self.inner.search(query, threshold, top_k, Some(workflow_id));
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"index": r.index, "similarity": r.similarity}))
            .collect();
        serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn to_ipc(&mut self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .to_ipc()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    pub fn from_ipc(data: &[u8]) -> Result<WasmContextTable, JsValue> {
        let inner = ContextTable::from_ipc(data)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        Ok(Self { inner })
    }

    pub fn compact(&mut self) -> Result<(), JsValue> {
        self.inner
            .compact()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
}

fn parse_metadata(json: &str) -> Result<ContextMetadata, JsValue> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("Invalid metadata JSON: {}", e)))?;
    Ok(ContextMetadata {
        workflow_id: v["workflow_id"].as_str().unwrap_or("").to_string(),
        response: v["response"].as_str().unwrap_or("").to_string(),
        model_id: v["model_id"].as_str().unwrap_or("").to_string(),
        input_tokens: v["input_tokens"].as_i64().unwrap_or(0),
        output_tokens: v["output_tokens"].as_i64().unwrap_or(0),
        cost_usd: v["cost_usd"].as_f64().unwrap_or(0.0),
    })
}