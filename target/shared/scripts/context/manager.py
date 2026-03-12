"""Context Manager — stores and retrieves contexts with embeddings.

Uses DynamoDB for persistent storage and Redis for hot cache.
Both Python (Path A) and Node.js (Path B) read/write the same
DynamoDB table and Redis keys.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
import numpy as np
import redis

logger = logging.getLogger(__name__)


class ContextManager:
    """Store and retrieve contexts with embeddings from DynamoDB and Redis."""

    def __init__(
        self,
        dynamo_table: str = "context-store",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl: int = 3600,
        dynamo_endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
    ):
        dynamo_kwargs = {"region_name": region}
        if dynamo_endpoint_url:
            dynamo_kwargs["endpoint_url"] = dynamo_endpoint_url
        self._dynamo = boto3.resource("dynamodb", **dynamo_kwargs)
        self._table = self._dynamo.Table(dynamo_table)

        self._redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self._redis_ttl = redis_ttl

    def store_context(
        self,
        workflow_id: str,
        task_description: str,
        embedding: np.ndarray,
        response: str,
        cost_metadata: dict,
        context_id: Optional[str] = None,
    ) -> str:
        """Store context in DynamoDB and cache in Redis.

        Returns the context_id (generated if not provided).
        """
        if context_id is None:
            context_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc).isoformat()
        embedding_bytes = embedding.tobytes()

        # DynamoDB item
        item = {
            "context_id": context_id,
            "workflow_id": workflow_id,
            "task_description": task_description,
            "embedding": embedding_bytes,
            "embedding_dim": int(embedding.shape[0]),
            "response": response,
            "model_id": cost_metadata.get("model_id", ""),
            "cost_input_tokens": cost_metadata.get("input_tokens", 0),
            "cost_output_tokens": cost_metadata.get("output_tokens", 0),
            "cost_usd": str(cost_metadata.get("cost_usd", 0)),
            "created_at": now,
            "reuse_count": 0,
        }

        self._table.put_item(Item=item)

        # Redis cache
        self._cache_context(context_id, workflow_id, embedding_bytes, response, cost_metadata)

        logger.debug("Stored context %s (workflow %s, %d-dim)", context_id, workflow_id, embedding.shape[0])
        return context_id

    def _cache_context(
        self,
        context_id: str,
        workflow_id: str,
        embedding_bytes: bytes,
        response: str,
        metadata: dict,
    ):
        """Write context to Redis cache."""
        pipe = self._redis.pipeline()
        pipe.setex(f"embedding:{context_id}", self._redis_ttl, embedding_bytes)
        pipe.setex(
            f"context:{context_id}",
            self._redis_ttl,
            json.dumps({"response": response, "metadata": metadata}),
        )
        pipe.sadd(f"workflow:{workflow_id}", context_id)
        pipe.expire(f"workflow:{workflow_id}", self._redis_ttl * 2)
        pipe.execute()

    def get_context(self, context_id: str) -> Optional[dict]:
        """Retrieve context — Redis first, DynamoDB fallback."""
        # Try Redis
        cached = self._redis.get(f"context:{context_id}")
        if cached:
            data = json.loads(cached)
            embedding_bytes = self._redis.get(f"embedding:{context_id}")
            if embedding_bytes:
                data["embedding"] = np.frombuffer(embedding_bytes, dtype=np.float32)
            data["context_id"] = context_id
            data["source"] = "cache"
            return data

        # Fallback to DynamoDB — need workflow_id for the sort key,
        # so we scan by context_id. For known workflow_id, use get_item directly.
        try:
            response = self._table.scan(
                FilterExpression="context_id = :cid",
                ExpressionAttributeValues={":cid": context_id},
                Limit=1,
            )
            items = response.get("Items", [])
            if not items:
                return None

            item = items[0]
            embedding_bytes = item.get("embedding")
            if isinstance(embedding_bytes, bytes):
                item["embedding"] = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
            item["source"] = "dynamodb"
            return item
        except Exception as e:
            logger.error("Failed to get context %s from DynamoDB: %s", context_id, e)
            return None

    def get_all_embeddings(
        self,
        workflow_id: Optional[str] = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Return all (context_id, embedding) pairs for similarity search.

        Tries Redis first for all known context IDs in a workflow.
        Falls back to DynamoDB scan.
        """
        results = []

        if workflow_id:
            # Try Redis workflow index
            context_ids = self._redis.smembers(f"workflow:{workflow_id}")
            if context_ids:
                return self._load_embeddings_from_redis(context_ids)

            # Fallback: DynamoDB GSI query
            response = self._table.query(
                IndexName="workflow_id-created_at-index",
                KeyConditionExpression="workflow_id = :wid",
                ExpressionAttributeValues={":wid": workflow_id},
            )
        else:
            response = self._table.scan()

        for item in response.get("Items", []):
            ctx_id = item["context_id"]
            embedding_bytes = item.get("embedding")
            if isinstance(embedding_bytes, bytes):
                emb = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
                results.append((ctx_id, emb))

        return results

    def _load_embeddings_from_redis(
        self,
        context_ids: set,
    ) -> list[tuple[str, np.ndarray]]:
        """Load embeddings from Redis for a set of context IDs."""
        results = []
        pipe = self._redis.pipeline()
        id_list = []
        for cid in context_ids:
            cid_str = cid.decode() if isinstance(cid, bytes) else cid
            id_list.append(cid_str)
            pipe.get(f"embedding:{cid_str}")

        values = pipe.execute()
        for cid_str, emb_bytes in zip(id_list, values):
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                results.append((cid_str, emb))

        return results

    def increment_reuse_count(self, context_id: str, workflow_id: str) -> None:
        """Increment reuse_count in DynamoDB."""
        try:
            self._table.update_item(
                Key={"context_id": context_id, "workflow_id": workflow_id},
                UpdateExpression="SET reuse_count = reuse_count + :inc",
                ExpressionAttributeValues={":inc": 1},
            )
        except Exception as e:
            logger.error("Failed to increment reuse count for %s: %s", context_id, e)

    def get_workflow_contexts(self, workflow_id: str) -> list[dict]:
        """Query GSI for all contexts in a workflow."""
        response = self._table.query(
            IndexName="workflow_id-created_at-index",
            KeyConditionExpression="workflow_id = :wid",
            ExpressionAttributeValues={":wid": workflow_id},
        )
        return response.get("Items", [])

    def clear_workflow(self, workflow_id: str) -> int:
        """Delete all contexts for a workflow. Returns count deleted."""
        items = self.get_workflow_contexts(workflow_id)
        count = 0
        for item in items:
            ctx_id = item["context_id"]
            self._table.delete_item(
                Key={"context_id": ctx_id, "workflow_id": workflow_id},
            )
            self._redis.delete(f"embedding:{ctx_id}", f"context:{ctx_id}")
            count += 1
        self._redis.delete(f"workflow:{workflow_id}")
        return count