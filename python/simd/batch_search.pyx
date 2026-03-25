# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Cython batch similarity search — Path A2 optimization.

Pushes the entire similarity search loop into C/C++, eliminating
per-embedding Python→C++ boundary crossing overhead.

For N stored embeddings:
- Path A1 makes N Python→C++ calls (via pycylon)
- Path A2 (this module) makes 1 call and iterates entirely in C/C++

Uses Cylon's native C++ SIMD cosine similarity via cylon/simd/simd_ops.hpp.
Cylon is a required dependency — CYLON_PREFIX must be set at build time.
"""

import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float32_t FLOAT32

# Cylon C++ SIMD integration
cdef extern from "cylon/simd/simd_ops.hpp" namespace "cylon::simd":
    float cosine_similarity_f32(const float* a, const float* b, int n) nogil

cdef float _cosine_similarity(
    FLOAT32* a,
    FLOAT32* b,
    int dim,
) noexcept nogil:
    """Delegate to Cylon's native C++ SIMD cosine similarity."""
    return cosine_similarity_f32(a, b, dim)


def batch_cosine_search(
    np.ndarray[FLOAT32, ndim=1] query,
    np.ndarray[FLOAT32, ndim=2] embeddings,
    float threshold,
    int top_k=5,
):
    """Search all embeddings against query in a single C-level loop.

    Args:
        query: float32 array of shape (dim,)
        embeddings: float32 array of shape (N, dim) — must be C-contiguous
        threshold: minimum cosine similarity to include
        top_k: max results to return

    Returns:
        List of (index, similarity) tuples, sorted descending by similarity.
        Single Python→C boundary crossing for the entire search.
    """
    cdef int n = embeddings.shape[0]
    cdef int dim = embeddings.shape[1]
    cdef int query_dim = query.shape[0]

    if query_dim != dim:
        raise ValueError(
            f"Query dimension {query_dim} does not match "
            f"embedding dimension {dim}"
        )

    # Allocate result arrays in C
    cdef np.ndarray[FLOAT32, ndim=1] scores = np.empty(n, dtype=np.float32)
    cdef FLOAT32* query_ptr = <FLOAT32*> query.data
    cdef FLOAT32* emb_ptr
    cdef int i
    cdef float sim

    # Core loop — entirely in C, no Python overhead per iteration
    with nogil:
        for i in range(n):
            emb_ptr = <FLOAT32*> &embeddings[i, 0]
            scores[i] = _cosine_similarity(query_ptr, emb_ptr, dim)

    # Filter and sort in Python (small result set)
    results = []
    for i in range(n):
        if scores[i] >= threshold:
            results.append((i, float(scores[i])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def batch_cosine_all(
    np.ndarray[FLOAT32, ndim=1] query,
    np.ndarray[FLOAT32, ndim=2] embeddings,
):
    """Compute cosine similarity between query and all embeddings.

    Returns the full scores array without filtering — useful for
    analysis and threshold tuning.

    Args:
        query: float32 array of shape (dim,)
        embeddings: float32 array of shape (N, dim)

    Returns:
        float32 array of shape (N,) with similarity scores.
    """
    cdef int n = embeddings.shape[0]
    cdef int dim = embeddings.shape[1]

    if query.shape[0] != dim:
        raise ValueError(
            f"Query dimension {query.shape[0]} does not match "
            f"embedding dimension {dim}"
        )

    cdef np.ndarray[FLOAT32, ndim=1] scores = np.empty(n, dtype=np.float32)
    cdef FLOAT32* query_ptr = <FLOAT32*> query.data
    cdef FLOAT32* emb_ptr
    cdef int i

    with nogil:
        for i in range(n):
            emb_ptr = <FLOAT32*> &embeddings[i, 0]
            scores[i] = _cosine_similarity(query_ptr, emb_ptr, dim)

    return scores