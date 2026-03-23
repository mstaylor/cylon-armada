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

use cylon::simd::{batch_cosine_search, cosine_similarity_f32};
use cylon_armada::context::{ContextMetadata, ContextTable};

fn make_embedding(dim: usize, seed: u32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 + seed as f32) * 0.1).sin())
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Cosine similarity tests
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    assert!((cosine_similarity_f32(&a, &a) - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_orthogonal() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    assert!(cosine_similarity_f32(&a, &b).abs() < 1e-5);
}

#[test]
fn test_cosine_opposite() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    assert!((cosine_similarity_f32(&a, &b) + 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_zero_vector() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_similarity_f32(&a, &b), 0.0);
}

#[test]
fn test_cosine_large_dim() {
    let a = make_embedding(1024, 42);
    assert!((cosine_similarity_f32(&a, &a) - 1.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Batch search tests
// ---------------------------------------------------------------------------

#[test]
fn test_batch_search_basic() {
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let embeddings = vec![
        1.0, 0.0, 0.0, 0.0, // sim = 1.0
        0.0, 1.0, 0.0, 0.0, // sim = 0.0
        0.9, 0.1, 0.0, 0.0, // sim ~ 0.994
    ];
    let results = batch_cosine_search(&query, &embeddings, 4, 0.5, 10);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].index, 0);
    assert!((results[0].similarity - 1.0).abs() < 1e-5);
    assert_eq!(results[1].index, 2);
}

#[test]
fn test_batch_search_top_k() {
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let embeddings = vec![
        1.0, 0.0, 0.0, 0.0,
        0.9, 0.1, 0.0, 0.0,
        0.8, 0.2, 0.0, 0.0,
    ];
    let results = batch_cosine_search(&query, &embeddings, 4, 0.0, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
}

// ---------------------------------------------------------------------------
// ContextTable tests
// ---------------------------------------------------------------------------

#[test]
fn test_create_empty() {
    let table = ContextTable::new(256).unwrap();
    assert_eq!(table.len(), 0);
    assert_eq!(table.total_rows(), 0);
    assert_eq!(table.embedding_dim(), 256);
    assert!(table.is_empty());
}

#[test]
fn test_create_invalid_dim() {
    assert!(ContextTable::new(0).is_err());
}

#[test]
fn test_put_and_get() {
    let mut table = ContextTable::new(4).unwrap();
    let emb = make_embedding(4, 1);
    table
        .put(
            "ctx-1",
            &emb,
            ContextMetadata {
                workflow_id: "wf-1".into(),
                response: "Hello".into(),
                model_id: "test".into(),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(table.len(), 1);
    let row = table.get("ctx-1");
    assert!(row.is_some());
    assert_eq!(row.unwrap().num_rows(), 1);
    assert!(table.get("nonexistent").is_none());
}

#[test]
fn test_upsert() {
    let mut table = ContextTable::new(4).unwrap();
    table
        .put(
            "ctx-1",
            &make_embedding(4, 1),
            ContextMetadata {
                response: "first".into(),
                ..Default::default()
            },
        )
        .unwrap();
    table
        .put(
            "ctx-1",
            &make_embedding(4, 2),
            ContextMetadata {
                response: "second".into(),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(table.len(), 1);
    assert_eq!(table.total_rows(), 2);
}

#[test]
fn test_remove() {
    let mut table = ContextTable::new(4).unwrap();
    table
        .put("ctx-1", &make_embedding(4, 1), ContextMetadata::default())
        .unwrap();

    table.remove("ctx-1").unwrap();
    assert_eq!(table.len(), 0);
    assert_eq!(table.total_rows(), 1);
    assert!(table.get("ctx-1").is_none());
    assert!(table.remove("ctx-1").is_err());
}

#[test]
fn test_compact() {
    let mut table = ContextTable::new(4).unwrap();
    for i in 0..5u32 {
        table
            .put(
                &format!("ctx-{}", i),
                &make_embedding(4, i),
                ContextMetadata::default(),
            )
            .unwrap();
    }
    table.remove("ctx-1").unwrap();
    table.remove("ctx-3").unwrap();

    assert_eq!(table.len(), 3);
    assert_eq!(table.total_rows(), 5);

    table.compact().unwrap();
    assert_eq!(table.len(), 3);
    assert_eq!(table.total_rows(), 3);

    assert!(table.get("ctx-0").is_some());
    assert!(table.get("ctx-2").is_some());
    assert!(table.get("ctx-4").is_some());
    assert!(table.get("ctx-1").is_none());
    assert!(table.get("ctx-3").is_none());
}

#[test]
fn test_search() {
    let mut table = ContextTable::new(4).unwrap();
    table
        .put("ctx-1", &[1.0, 0.0, 0.0, 0.0], ContextMetadata::default())
        .unwrap();
    table
        .put("ctx-2", &[0.0, 1.0, 0.0, 0.0], ContextMetadata::default())
        .unwrap();
    table
        .put("ctx-3", &[0.9, 0.1, 0.0, 0.0], ContextMetadata::default())
        .unwrap();

    let results = table.search(&[1.0, 0.0, 0.0, 0.0], 0.5, 10, None);
    assert_eq!(results.len(), 2);
    assert!(results[0].similarity > results[1].similarity);
}

#[test]
fn test_search_skips_tombstoned() {
    let mut table = ContextTable::new(4).unwrap();
    table
        .put("ctx-1", &[1.0, 0.0, 0.0, 0.0], ContextMetadata::default())
        .unwrap();
    table
        .put("ctx-2", &[0.9, 0.1, 0.0, 0.0], ContextMetadata::default())
        .unwrap();
    table.remove("ctx-1").unwrap();

    let results = table.search(&[1.0, 0.0, 0.0, 0.0], 0.5, 10, None);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_search_workflow_filter() {
    let mut table = ContextTable::new(4).unwrap();
    table
        .put(
            "ctx-1",
            &[1.0, 0.0, 0.0, 0.0],
            ContextMetadata {
                workflow_id: "wf-1".into(),
                ..Default::default()
            },
        )
        .unwrap();
    table
        .put(
            "ctx-2",
            &[0.9, 0.1, 0.0, 0.0],
            ContextMetadata {
                workflow_id: "wf-2".into(),
                ..Default::default()
            },
        )
        .unwrap();

    let results = table.search(&[1.0, 0.0, 0.0, 0.0], 0.0, 10, Some("wf-1"));
    assert_eq!(results.len(), 1);
}

#[test]
fn test_get_workflow() {
    let mut table = ContextTable::new(4).unwrap();
    let emb = make_embedding(4, 1);
    table
        .put(
            "ctx-1",
            &emb,
            ContextMetadata {
                workflow_id: "wf-1".into(),
                ..Default::default()
            },
        )
        .unwrap();
    table
        .put(
            "ctx-2",
            &emb,
            ContextMetadata {
                workflow_id: "wf-2".into(),
                ..Default::default()
            },
        )
        .unwrap();
    table
        .put(
            "ctx-3",
            &emb,
            ContextMetadata {
                workflow_id: "wf-1".into(),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(table.get_workflow("wf-1").unwrap().num_rows(), 2);
    assert_eq!(table.get_workflow("wf-2").unwrap().num_rows(), 1);
    assert_eq!(table.get_workflow("wf-nonexistent").unwrap().num_rows(), 0);
}

#[test]
fn test_ipc_round_trip() {
    let mut table = ContextTable::new(8).unwrap();
    for i in 0..3u32 {
        table
            .put(
                &format!("ctx-{}", i),
                &make_embedding(8, i),
                ContextMetadata {
                    workflow_id: "wf-1".into(),
                    response: format!("response-{}", i),
                    model_id: "test".into(),
                    input_tokens: 100 + i as i64,
                    output_tokens: 50 + i as i64,
                    cost_usd: 0.001 * (i + 1) as f64,
                },
            )
            .unwrap();
    }

    let data = table.to_ipc().unwrap();
    assert!(!data.is_empty());

    let mut restored = ContextTable::from_ipc(&data).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored.embedding_dim(), 8);
    for i in 0..3 {
        assert!(restored.get(&format!("ctx-{}", i)).is_some());
    }
}

#[test]
fn test_dim_mismatch() {
    let mut table = ContextTable::new(4).unwrap();
    assert!(table.put("ctx-1", &[1.0; 8], ContextMetadata::default()).is_err());
}