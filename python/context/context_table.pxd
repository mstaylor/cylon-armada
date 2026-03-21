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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int64_t, uint8_t

from pycylon.common.status cimport CStatus
from pycylon.ctx.context cimport CCylonContext

from pyarrow.lib cimport CRecordBatch, CSchema


cdef extern from "cylon/simd/simd_ops.hpp" namespace "cylon::simd":
    cdef struct CSearchResult "cylon::simd::SearchResult":
        int64_t index
        float similarity


cdef extern from "cylon_armada/context/context_table.hpp" namespace "cylon::context":
    cdef cppclass CContextMetadata "cylon::context::ContextMetadata":
        string workflow_id
        string response
        string model_id
        int64_t input_tokens
        int64_t output_tokens
        double cost_usd

    cdef cppclass CContextTable "cylon::context::ContextTable":
        @staticmethod
        CStatus MakeEmpty(int embedding_dim, shared_ptr[CContextTable]* out)

        @staticmethod
        CStatus MakeFromIpc(const uint8_t* data, int64_t size,
                            shared_ptr[CContextTable]* out)

        CStatus Put(const string& context_id,
                    const float* embedding, int dim,
                    const CContextMetadata& metadata)

        CStatus GetRow(const string& context_id,
                       shared_ptr[CRecordBatch]* out)

        int64_t Size() const
        int64_t TotalRows() const
        int EmbeddingDim() const

        CStatus Remove(const string& context_id)
        CStatus Compact()

        shared_ptr[CRecordBatch] Batch() const
        shared_ptr[CSchema] GetSchema() const

        vector[CSearchResult] Search(
            const float* query, int dim,
            float threshold, int top_k,
            const string& workflow_id)

        shared_ptr[CRecordBatch] GetWorkflow(const string& workflow_id)

        CStatus ToIpc(vector[uint8_t]* data) const

        CStatus Broadcast(const shared_ptr[CCylonContext]& ctx, int root)
        CStatus AllGather(const shared_ptr[CCylonContext]& ctx)


cdef extern from "cylon_armada/context/context_serialize.hpp" namespace "cylon::context":
    CStatus CSaveToRedis "cylon::context::SaveToRedis" (
        const shared_ptr[CContextTable]& table,
        const string& key,
        const string& redis_addr,
        int ttl_seconds)

    CStatus CLoadFromRedis "cylon::context::LoadFromRedis" (
        const string& key,
        const string& redis_addr,
        shared_ptr[CContextTable]* out)

    CStatus CSaveToS3 "cylon::context::SaveToS3" (
        const shared_ptr[CContextTable]& table,
        const string& bucket,
        const string& key,
        const string& region)

    CStatus CLoadFromS3 "cylon::context::LoadFromS3" (
        const string& bucket,
        const string& key,
        const string& region,
        shared_ptr[CContextTable]* out)


cdef class ContextTable:
    cdef:
        shared_ptr[CContextTable] table_ptr