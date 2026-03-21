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

#ifndef CYLON_CONTEXT_SERIALIZE_HPP
#define CYLON_CONTEXT_SERIALIZE_HPP

#include <memory>
#include <string>

#include "context_table.hpp"
#include <cylon/status.hpp>

namespace cylon {
namespace context {

#ifdef BUILD_CYLON_REDIS

/// Save a ContextTable to Redis as Arrow IPC bytes.
/// Key format: {session_id}:context_table:{key}
/// Requires CYLON_SESSION_ID environment variable.
/// @param table The ContextTable to persist.
/// @param key Application-defined key (e.g., workflow_id).
/// @param redis_addr Redis connection string (e.g., "tcp://localhost:6379").
/// @param ttl_seconds TTL for the Redis key (default 3600).
Status SaveToRedis(const std::shared_ptr<ContextTable>& table,
                   const std::string& key,
                   const std::string& redis_addr = "tcp://localhost:6379",
                   int ttl_seconds = 3600);

/// Load a ContextTable from Redis.
/// @param key Application-defined key (same as used in SaveToRedis).
/// @param redis_addr Redis connection string.
/// @param out Output ContextTable (nullptr if key not found).
Status LoadFromRedis(const std::string& key,
                     const std::string& redis_addr,
                     std::shared_ptr<ContextTable>* out);

#endif  // BUILD_CYLON_REDIS

#ifdef BUILD_CYLON_FMI

/// Save a ContextTable to S3 as Arrow IPC bytes.
/// @param table The ContextTable to persist.
/// @param bucket S3 bucket name.
/// @param key S3 object key.
/// @param region AWS region (default "us-east-1").
Status SaveToS3(const std::shared_ptr<ContextTable>& table,
                const std::string& bucket,
                const std::string& key,
                const std::string& region = "us-east-1");

/// Load a ContextTable from S3.
/// @param bucket S3 bucket name.
/// @param key S3 object key.
/// @param region AWS region.
/// @param out Output ContextTable.
Status LoadFromS3(const std::string& bucket,
                  const std::string& key,
                  const std::string& region,
                  std::shared_ptr<ContextTable>* out);

#endif  // BUILD_CYLON_FMI

}  // namespace context
}  // namespace cylon

#endif  // CYLON_CONTEXT_SERIALIZE_HPP