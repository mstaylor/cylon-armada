"""Zero-copy embedding read path (ContextManager.get_embedding_matrix).

find_similar runs once per task, and before this path existed it rebuilt the
whole embedding set from Python objects on every call: each Arrow
FixedSizeList row became a Python list, then a numpy array, then the lot was
vstacked. That is O(N*D) Python object construction per query against a store
that grows with the agent population, so the cost rose with N — the opposite
of what the zero-copy claim asserts, and enough to bend S(N) downwards for
reasons that have nothing to do with the data plane.

What these tests pin down:
  1. the matrix really is a view over the ContextTable's own buffer, asserted
     by address rather than by np.shares_memory — combine_chunks() allocates a
     fresh buffer even for a single-chunk column, so a looser check would pass
     while still paying a full copy of the embedding block per query;
  2. the new path returns exactly what the old one did, for both backends;
  3. each SIMD backend keeps its algorithmic character — A1 still makes one
     call per row, A2 still makes one batched call. Removing the shared data
     preparation must not quietly vectorize A1, which is the comparison
     Experiment A2 exists to make.

Run: pytest tests/test_embedding_matrix.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

D = 8


def _arrow_context_table(n, workflow_id="wf-1", dims=D, seed=0):
    """Arrow table shaped like the Cylon ContextTable's to_arrow() output."""
    rng = np.random.default_rng(seed)
    flat = rng.random(n * dims, dtype=np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dims)
    return pa.table({
        "context_id": pa.array([f"ctx-{i}" for i in range(n)], type=pa.large_utf8()),
        "embedding": embeddings,
        "workflow_id": pa.array([workflow_id] * n, type=pa.large_utf8()),
    })


def _cylon_manager(table):
    """ContextManager wired to a stub ContextTable returning `table`."""
    from context.manager import ContextManager

    stub = MagicMock()
    stub.to_arrow.return_value = table
    with patch('context.manager._create_cylon_backend') as make:
        make.return_value = {"table": stub, "redis_addr": None, "redis_ttl": 3600,
                             "has_redis": False, "save_fn": None, "load_fn": None}
        cm = ContextManager(backend="cylon", embedding_dim=D)
    return cm


# ---------------------------------------------------------------------------
# Zero-copy guarantee
# ---------------------------------------------------------------------------

def test_matrix_points_at_the_context_tables_own_buffer():
    """The address has to match the *original* column's data buffer.

    np.shares_memory against a re-derived view is not enough: combine_chunks()
    allocates a fresh buffer even for a single-chunk column, so a matrix that
    merely views some copy would still pass a looser check while paying a full
    copy of the embedding block on every query.
    """
    table = _arrow_context_table(64)
    cm = _cylon_manager(table)

    _, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    original = table.column("embedding").chunk(0).values.buffers()[1]
    assert matrix.ctypes.data == original.address, (
        "matrix does not point at the ContextTable's own embedding buffer"
    )


def test_matrix_has_the_right_shape_and_dtype():
    table = _arrow_context_table(37)
    cm = _cylon_manager(table)

    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    assert matrix.shape == (37, D)
    assert matrix.dtype == np.float32
    assert len(ids) == 37


def test_matrix_values_match_the_stored_embeddings():
    table = _arrow_context_table(16)
    cm = _cylon_manager(table)

    _, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    expected = np.asarray(table.column("embedding").to_pylist(), dtype=np.float32)
    assert np.array_equal(matrix, expected)


# ---------------------------------------------------------------------------
# Equivalence with the path it replaces
# ---------------------------------------------------------------------------

def test_get_all_embeddings_still_returns_the_same_pairs():
    """Its three external callers must not notice the reimplementation."""
    table = _arrow_context_table(24)
    cm = _cylon_manager(table)

    pairs = cm.get_all_embeddings(workflow_id="wf-1")
    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    assert [cid for cid, _ in pairs] == [
        (i.as_py() if hasattr(i, "as_py") else i) for i in ids
    ]
    for row, (_, emb) in enumerate(pairs):
        assert np.array_equal(emb, matrix[row])
        assert emb.dtype == np.float32


# ---------------------------------------------------------------------------
# workflow_id filtering and empty cases
# ---------------------------------------------------------------------------

def test_filters_to_the_requested_workflow():
    mine = _arrow_context_table(5, workflow_id="wf-mine", seed=1)
    theirs = _arrow_context_table(7, workflow_id="wf-theirs", seed=2)
    cm = _cylon_manager(pa.concat_tables([mine, theirs]))

    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-mine")

    assert matrix.shape == (5, D)
    assert np.array_equal(matrix, np.asarray(mine.column("embedding").to_pylist(),
                                             dtype=np.float32))


def test_no_workflow_filter_returns_every_row():
    cm = _cylon_manager(pa.concat_tables([
        _arrow_context_table(3, workflow_id="a", seed=1),
        _arrow_context_table(4, workflow_id="b", seed=2),
    ]))

    _, matrix = cm.get_embedding_matrix()

    assert matrix.shape == (7, D)


def test_unmatched_workflow_returns_empty_not_none():
    cm = _cylon_manager(_arrow_context_table(5, workflow_id="wf-mine"))

    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-absent")

    assert len(ids) == 0
    assert matrix.shape[0] == 0


def test_empty_store_returns_empty():
    cm = _cylon_manager(_arrow_context_table(0))

    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    assert len(ids) == 0
    assert matrix.shape[0] == 0


# ---------------------------------------------------------------------------
# Redis backend keeps the same contract
# ---------------------------------------------------------------------------

@patch('context.manager._create_redis_backend')
def test_redis_backend_returns_a_matrix_too(mock_create_redis):
    from context.manager import ContextManager

    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    redis = MagicMock()
    redis.smembers.return_value = [b"ctx-1", b"ctx-2"]
    pipe = MagicMock()
    pipe.execute.return_value = [a.tobytes(), b.tobytes()]
    redis.pipeline.return_value = pipe
    mock_create_redis.return_value = {"client": redis}

    cm = ContextManager(backend="redis")
    ids, matrix = cm.get_embedding_matrix(workflow_id="wf-1")

    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    assert sorted(i.as_py() if hasattr(i, "as_py") else i for i in ids) == ["ctx-1", "ctx-2"]


# ---------------------------------------------------------------------------
# Backends keep their algorithmic character (Experiment A2's comparison)
# ---------------------------------------------------------------------------

def _router_with(backend_name, table, threshold=-1.0):
    """Router over a stub store. The threshold is set low so every row is a hit,
    which is what makes the per-row call count observable."""
    from types import SimpleNamespace

    from context.router import ContextRouter, SIMDBackend

    cm = _cylon_manager(table)
    router = ContextRouter(
        cm,
        config=SimpleNamespace(similarity_threshold=threshold),
        top_k=5,
    )
    # Set after construction: __init__ downgrades an unavailable backend to
    # NUMPY, and neither the Cython extension nor pycylon is built in this
    # environment. These tests exercise dispatch, not availability detection.
    router.backend = SIMDBackend[backend_name]
    return router


def test_a1_still_makes_one_similarity_call_per_row():
    """PYCYLON is Path A1 — N boundary crossings, deliberately. If the fix
    vectorized it, Experiment A2's A1-vs-A2 comparison would silently vanish."""
    table = _arrow_context_table(12)
    router = _router_with("PYCYLON", table)

    calls = []
    router._pycylon_cosine = lambda a, b: (calls.append(1), 0.5)[1]

    router.find_similar(np.ones(D, dtype=np.float32), workflow_id="wf-1")

    assert len(calls) == 12


def test_a2_still_makes_exactly_one_batched_call():
    """CYTHON_BATCH is Path A2 — one crossing, and the matrix must arrive
    without having been rebuilt row by row first."""
    table = _arrow_context_table(12)
    router = _router_with("CYTHON_BATCH", table)

    seen = []

    def fake_batch(query, matrix, threshold, top_k):
        seen.append(matrix)
        return [(0, 0.9)]

    router._cython_batch = fake_batch
    results = router.find_similar(np.ones(D, dtype=np.float32), workflow_id="wf-1")

    assert len(seen) == 1
    assert seen[0].shape == (12, D)
    assert results and results[0]["context_id"] == "ctx-0"


def test_find_similar_agrees_across_backends():
    """Same store, same query — the backends differ in how they compute, never
    in what they conclude."""
    from context.router import SIMDBackend

    table = _arrow_context_table(20, seed=7)
    query = np.asarray(table.column("embedding").to_pylist(), dtype=np.float32)[3]

    answers = {}
    for name in ("NUMPY", "PYCYLON"):
        router = _router_with(name, table)
        if name == "PYCYLON":
            router._pycylon_cosine = None
            router.backend = SIMDBackend.PYCYLON
        hits = router.find_similar(query, workflow_id="wf-1")
        answers[name] = [h["context_id"] for h in hits]

    assert answers["NUMPY"][0] == "ctx-3"
    assert answers["NUMPY"] == answers["PYCYLON"]