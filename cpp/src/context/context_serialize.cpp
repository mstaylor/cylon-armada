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

#include "context_serialize.hpp"

#include <cstdlib>
#include <cstring>

#ifdef BUILD_CYLON_REDIS
#include "sw/redis++/redis++.h"
#endif

#ifdef BUILD_CYLON_FMI
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <sstream>
#endif

namespace cylon {
namespace context {

// ---------------------------------------------------------------------------
// Redis persistence
// ---------------------------------------------------------------------------

#ifdef BUILD_CYLON_REDIS

static std::string get_redis_key(const std::string& key) {
  const char* session_id = std::getenv("CYLON_SESSION_ID");
  if (!session_id || std::strlen(session_id) == 0) {
    throw std::runtime_error("CYLON_SESSION_ID environment variable is not set");
  }
  return std::string(session_id) + ":context_table:" + key;
}

Status SaveToRedis(const std::shared_ptr<ContextTable>& table,
                   const std::string& key,
                   const std::string& redis_addr,
                   int ttl_seconds) {
  // Serialize to IPC bytes
  std::vector<uint8_t> ipc_data;
  auto status = table->ToIpc(&ipc_data);
  if (!status.is_ok()) {
    return status;
  }

  try {
    auto redis = sw::redis::Redis(redis_addr);
    auto redis_key = get_redis_key(key);

    // Store as binary string
    redis.set(redis_key,
              std::string(reinterpret_cast<const char*>(ipc_data.data()),
                          ipc_data.size()));

    if (ttl_seconds > 0) {
      redis.expire(redis_key, std::chrono::seconds(ttl_seconds));
    }
  } catch (const std::exception& e) {
    return {Code::IOError, std::string("Redis SaveToRedis failed: ") + e.what()};
  }

  return Status::OK();
}

Status LoadFromRedis(const std::string& key,
                     const std::string& redis_addr,
                     std::shared_ptr<ContextTable>* out) {
  try {
    auto redis = sw::redis::Redis(redis_addr);
    auto redis_key = get_redis_key(key);

    auto val = redis.get(redis_key);
    if (!val) {
      *out = nullptr;
      return Status::OK();
    }

    auto result = ContextTable::FromIpc(
        reinterpret_cast<const uint8_t*>(val->data()),
        static_cast<int64_t>(val->size()));
    if (!result.ok()) {
      return {Code::ExecutionError, result.status().ToString()};
    }
    *out = std::move(*result);
  } catch (const std::exception& e) {
    return {Code::IOError, std::string("Redis LoadFromRedis failed: ") + e.what()};
  }

  return Status::OK();
}

#endif  // BUILD_CYLON_REDIS

// ---------------------------------------------------------------------------
// S3 persistence
// ---------------------------------------------------------------------------

#ifdef BUILD_CYLON_FMI

// Track AWS SDK lifecycle
static int s3_instances = 0;
static Aws::SDKOptions s3_options;

static void ensure_aws_init() {
  if (s3_instances == 0) {
    Aws::InitAPI(s3_options);
  }
  ++s3_instances;
}

static void release_aws() {
  --s3_instances;
  if (s3_instances == 0) {
    Aws::ShutdownAPI(s3_options);
  }
}

Status SaveToS3(const std::shared_ptr<ContextTable>& table,
                const std::string& bucket,
                const std::string& key,
                const std::string& region) {
  // Serialize to IPC bytes
  std::vector<uint8_t> ipc_data;
  auto status = table->ToIpc(&ipc_data);
  if (!status.is_ok()) {
    return status;
  }

  ensure_aws_init();

  Aws::Client::ClientConfiguration config;
  config.region = region;
  Aws::S3::S3Client client(config);

  Aws::S3::Model::PutObjectRequest request;
  request.WithBucket(bucket).WithKey(key);

  auto body = std::make_shared<std::stringstream>(
      std::string(reinterpret_cast<const char*>(ipc_data.data()),
                  ipc_data.size()));
  request.SetBody(body);

  auto outcome = client.PutObject(request);
  release_aws();

  if (!outcome.IsSuccess()) {
    return {Code::IOError,
            "S3 SaveToS3 failed: " + outcome.GetError().GetMessage()};
  }

  return Status::OK();
}

Status LoadFromS3(const std::string& bucket,
                  const std::string& key,
                  const std::string& region,
                  std::shared_ptr<ContextTable>* out) {
  ensure_aws_init();

  Aws::Client::ClientConfiguration config;
  config.region = region;
  Aws::S3::S3Client client(config);

  Aws::S3::Model::GetObjectRequest request;
  request.WithBucket(bucket).WithKey(key);

  auto outcome = client.GetObject(request);
  release_aws();

  if (!outcome.IsSuccess()) {
    return {Code::IOError,
            "S3 LoadFromS3 failed: " + outcome.GetError().GetMessage()};
  }

  // Read body into buffer
  auto& stream = outcome.GetResult().GetBody();
  std::ostringstream ss;
  ss << stream.rdbuf();
  auto body = ss.str();

  auto result = ContextTable::FromIpc(
      reinterpret_cast<const uint8_t*>(body.data()),
      static_cast<int64_t>(body.size()));
  if (!result.ok()) {
    return {Code::ExecutionError, result.status().ToString()};
  }
  *out = std::move(*result);

  return Status::OK();
}

#endif  // BUILD_CYLON_FMI

}  // namespace context
}  // namespace cylon