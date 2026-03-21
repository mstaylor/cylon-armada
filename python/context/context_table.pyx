##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

import numpy as np
import pyarrow as pa

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int64_t, uint8_t

from pycylon.common.status cimport CStatus
from pycylon.ctx.context cimport CCylonContext, CylonContext
from pycylon.context.context_table cimport (
    CContextMetadata,
    CContextTable,
    CSearchResult,
)
from pycylon.api.lib cimport pycylon_unwrap_context
from pyarrow.lib cimport CRecordBatch, pyarrow_wrap_batch

IF CYTHON_REDIS:
    from pycylon.context.context_table cimport CSaveToRedis, CLoadFromRedis

IF CYTHON_FMI:
    from pycylon.context.context_table cimport CSaveToS3, CLoadFromS3


cdef class ContextTable:
    """Arrow-native key-value store for embeddings and metadata.

    Optimized for SIMD cosine similarity search on FixedSizeList<Float32>
    embedding columns. All key-value operations (put/get/remove) are O(1).

    Example:
        >>> table = ContextTable(embedding_dim=1024)
        >>> table.put("ctx-1", embedding=np.random.randn(1024).astype(np.float32),
        ...           workflow_id="wf-1", response="Hello", model_id="claude")
        >>> results = table.search(query_embedding, threshold=0.85, top_k=5)
        >>> row = table.get("ctx-1")
    """

    def __cinit__(self, int embedding_dim):
        cdef CStatus status = CContextTable.MakeEmpty(
            embedding_dim, &self.table_ptr)
        if not status.is_ok():
            raise ValueError(
                f"Failed to create ContextTable: {status.get_msg().decode()}")

    def put(self, str context_id not None,
            object embedding not None,
            str workflow_id="",
            str response="",
            str model_id="",
            int64_t input_tokens=0,
            int64_t output_tokens=0,
            double cost_usd=0.0):
        """Insert or update a context entry. O(1) amortized.

        Args:
            context_id: Unique identifier for this context.
            embedding: Embedding vector (numpy float32 array, must match embedding_dim).
            workflow_id: Workflow identifier for filtering.
            response: LLM response text or payload.
            model_id: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost_usd: Cost in USD.
        """
        cdef CContextMetadata meta
        meta.workflow_id = workflow_id.encode()
        meta.response = response.encode()
        meta.model_id = model_id.encode()
        meta.input_tokens = input_tokens
        meta.output_tokens = output_tokens
        meta.cost_usd = cost_usd

        embedding = np.ascontiguousarray(embedding, dtype=np.float32).ravel()
        if embedding.ndim != 1:
            raise ValueError("embedding must be a 1-D array")
        cdef int dim = len(embedding)
        cdef const float* emb_ptr = <const float*> (<unsigned long long> embedding.ctypes.data)
        cdef CStatus status = self.table_ptr.get().Put(
            context_id.encode(),
            emb_ptr, dim,
            meta)
        if not status.is_ok():
            raise Exception(f"Put failed: {status.get_msg().decode()}")

    def get(self, str context_id not None):
        """Retrieve a single row by context_id. O(1) via hash index.

        Args:
            context_id: The context ID to look up.

        Returns:
            A pyarrow.RecordBatch with 1 row, or None if not found.
        """
        cdef shared_ptr[CRecordBatch] out
        cdef CStatus status = self.table_ptr.get().GetRow(
            context_id.encode(), &out)
        if not status.is_ok():
            raise Exception(f"Get failed: {status.get_msg().decode()}")
        if not out.get():
            return None
        return pyarrow_wrap_batch(out)

    def remove(self, str context_id not None):
        """Mark a row as deleted. O(1). Call compact() to reclaim space.

        Args:
            context_id: The context ID to remove.
        """
        cdef CStatus status = self.table_ptr.get().Remove(context_id.encode())
        if not status.is_ok():
            raise KeyError(f"Remove failed: {status.get_msg().decode()}")

    def search(self, object query not None,
               float threshold=0.85,
               int top_k=5,
               str workflow_id=""):
        """SIMD cosine similarity search across embeddings.

        Args:
            query: Query vector (numpy float32 array, must match embedding_dim).
            threshold: Minimum cosine similarity to include.
            top_k: Maximum number of results.
            workflow_id: Optional workflow filter (empty = search all).

        Returns:
            List of dicts with 'index' (int) and 'similarity' (float),
            sorted by descending similarity.
        """
        query = np.ascontiguousarray(query, dtype=np.float32).ravel()
        if query.ndim != 1:
            raise ValueError("query must be a 1-D array")
        cdef int dim = len(query)
        cdef const float* q_ptr = <const float*> (<unsigned long long> query.ctypes.data)
        cdef string wf = workflow_id.encode()

        cdef vector[CSearchResult] results = self.table_ptr.get().Search(
            q_ptr, dim,
            threshold, top_k, wf)

        return [{"index": r.index, "similarity": r.similarity} for r in results]

    def get_workflow(self, str workflow_id not None):
        """Retrieve all rows for a given workflow_id.

        Args:
            workflow_id: The workflow ID to filter by.

        Returns:
            A pyarrow.RecordBatch with matching rows.
        """
        cdef shared_ptr[CRecordBatch] batch = self.table_ptr.get().GetWorkflow(
            workflow_id.encode())
        if not batch.get():
            return None
        return pyarrow_wrap_batch(batch)

    def compact(self):
        """Remove tombstoned rows and rebuild the batch. O(n)."""
        cdef CStatus status = self.table_ptr.get().Compact()
        if not status.is_ok():
            raise Exception(f"Compact failed: {status.get_msg().decode()}")

    def to_ipc(self):
        """Serialize to Arrow IPC bytes.

        Returns:
            bytes object containing the Arrow IPC stream.
        """
        cdef vector[uint8_t] data
        cdef CStatus status = self.table_ptr.get().ToIpc(&data)
        if not status.is_ok():
            raise Exception(f"ToIpc failed: {status.get_msg().decode()}")
        return bytes(data)

    @staticmethod
    def from_ipc(bytes data not None):
        """Deserialize from Arrow IPC bytes.

        Args:
            data: bytes object from to_ipc().

        Returns:
            A new ContextTable.
        """
        cdef const uint8_t* ptr = <const uint8_t*> data
        cdef int64_t size = len(data)
        cdef shared_ptr[CContextTable] c_table
        cdef CStatus status = CContextTable.MakeFromIpc(ptr, size, &c_table)
        if not status.is_ok():
            raise Exception(f"FromIpc failed: {status.get_msg().decode()}")

        cdef ContextTable table = ContextTable.__new__(
            ContextTable, c_table.get().EmbeddingDim())
        table.table_ptr = c_table
        return table

    def to_arrow(self):
        """Get the underlying RecordBatch as a pyarrow.RecordBatch.

        Returns:
            pyarrow.RecordBatch (includes tombstoned rows).
        """
        cdef shared_ptr[CRecordBatch] batch = self.table_ptr.get().Batch()
        if not batch.get():
            return None
        return pyarrow_wrap_batch(batch)

    @property
    def size(self):
        """Number of active (non-deleted) rows."""
        return self.table_ptr.get().Size()

    @property
    def total_rows(self):
        """Total rows including tombstoned."""
        return self.table_ptr.get().TotalRows()

    @property
    def embedding_dim(self):
        """Embedding dimension."""
        return self.table_ptr.get().EmbeddingDim()

    def broadcast(self, CylonContext ctx not None, int root=0):
        """Broadcast this ContextTable from root rank to all workers.

        On non-root ranks, replaces current contents with the broadcast data.
        Compacts before broadcasting. No-op for local (non-distributed) contexts.

        Args:
            ctx: CylonContext with distributed communicator.
            root: Rank of the broadcasting process (default 0).
        """
        cdef shared_ptr[CCylonContext] ctx_ptr = pycylon_unwrap_context(ctx)
        cdef CStatus status = self.table_ptr.get().Broadcast(ctx_ptr, root)
        if not status.is_ok():
            raise Exception(f"Broadcast failed: {status.get_msg().decode()}")

    def all_gather(self, CylonContext ctx not None):
        """AllGather: each worker contributes its ContextTable, all receive merged result.

        Compacts before gathering. No-op for local (non-distributed) contexts.

        Args:
            ctx: CylonContext with distributed communicator.
        """
        cdef shared_ptr[CCylonContext] ctx_ptr = pycylon_unwrap_context(ctx)
        cdef CStatus status = self.table_ptr.get().AllGather(ctx_ptr)
        if not status.is_ok():
            raise Exception(f"AllGather failed: {status.get_msg().decode()}")


# ---------------------------------------------------------------------------
# Persistence — conditional on compile-time flags
# ---------------------------------------------------------------------------

IF CYTHON_REDIS:
    def save_context_to_redis(ContextTable table not None,
                              str key not None,
                              str redis_addr="tcp://localhost:6379",
                              int ttl_seconds=3600):
        """Save a ContextTable to Redis as Arrow IPC bytes.

        Requires CYLON_SESSION_ID environment variable for key namespacing.

        Args:
            table: The ContextTable to persist.
            key: Application-defined key (e.g., workflow_id).
            redis_addr: Redis connection string.
            ttl_seconds: TTL for the Redis key (default 3600).
        """
        cdef CStatus status = CSaveToRedis(
            table.table_ptr, key.encode(), redis_addr.encode(), ttl_seconds)
        if not status.is_ok():
            raise Exception(f"SaveToRedis failed: {status.get_msg().decode()}")

    def load_context_from_redis(str key not None,
                                str redis_addr="tcp://localhost:6379"):
        """Load a ContextTable from Redis.

        Args:
            key: Application-defined key (same as used in save_context_to_redis).
            redis_addr: Redis connection string.

        Returns:
            A new ContextTable, or None if key not found.
        """
        cdef shared_ptr[CContextTable] c_table
        cdef CStatus status = CLoadFromRedis(
            key.encode(), redis_addr.encode(), &c_table)
        if not status.is_ok():
            raise Exception(f"LoadFromRedis failed: {status.get_msg().decode()}")
        if not c_table.get():
            return None
        cdef ContextTable table = ContextTable.__new__(
            ContextTable, c_table.get().EmbeddingDim())
        table.table_ptr = c_table
        return table

IF CYTHON_FMI:
    def save_context_to_s3(ContextTable table not None,
                           str bucket not None,
                           str key not None,
                           str region="us-east-1"):
        """Save a ContextTable to S3 as Arrow IPC bytes.

        Args:
            table: The ContextTable to persist.
            bucket: S3 bucket name.
            key: S3 object key.
            region: AWS region (default "us-east-1").
        """
        cdef CStatus status = CSaveToS3(
            table.table_ptr, bucket.encode(), key.encode(), region.encode())
        if not status.is_ok():
            raise Exception(f"SaveToS3 failed: {status.get_msg().decode()}")

    def load_context_from_s3(str bucket not None,
                             str key not None,
                             str region="us-east-1"):
        """Load a ContextTable from S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key.
            region: AWS region.

        Returns:
            A new ContextTable.
        """
        cdef shared_ptr[CContextTable] c_table
        cdef CStatus status = CLoadFromS3(
            bucket.encode(), key.encode(), region.encode(), &c_table)
        if not status.is_ok():
            raise Exception(f"LoadFromS3 failed: {status.get_msg().decode()}")
        cdef ContextTable table = ContextTable.__new__(
            ContextTable, c_table.get().EmbeddingDim())
        table.table_ptr = c_table
        return table