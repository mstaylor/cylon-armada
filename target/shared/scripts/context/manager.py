"""Context Manager — stores and retrieves contexts with embeddings.

Search backend (required, one of):
  - "cylon" (default): Cylon ContextTable — Arrow-native, C++ SIMD search,
    zero-copy embeddings. Raises ImportError if pycylon is not installed.
  - "redis": Raw numpy+Redis — pure-Python mode for environments without pycylon.

Persistence layers:
  - Redis metadata (default ON): context JSON + workflow sets written to Redis
    alongside search embeddings. Controlled by ``persist_to_redis`` (default True).
    This is the primary persistence layer — enables cross-invocation context reuse
    within the Redis TTL window.
  - DynamoDB (default OFF): durable item store for long-term history and analytics.
    Controlled by ``dynamo_table`` (default None). Pass a table name to enable.

If only ``persist_to_redis=False`` and no ``dynamo_table``, contexts live in-memory
only for the duration of the process.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
import numpy as np

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("cylon", "redis")


def _create_cylon_backend(embedding_dim, redis_addr, redis_ttl):
    """Initialize Cylon ContextTable backend."""
    from cylon_armada.context_table import ContextTable as CylonContextTable

    try:
        from cylon_armada.context_table import (
            save_context_to_redis,
            load_context_from_redis,
        )
        has_redis = True
    except ImportError:
        has_redis = False

    return {
        "table": CylonContextTable(embedding_dim=embedding_dim),
        "redis_addr": redis_addr,
        "redis_ttl": redis_ttl,
        "has_redis": has_redis,
        "save_fn": save_context_to_redis if has_redis else None,
        "load_fn": load_context_from_redis if has_redis else None,
    }


def _create_redis_backend(redis_host, redis_port):
    """Initialize raw Redis backend."""
    import redis
    return {
        "client": redis.Redis(host=redis_host, port=redis_port, decode_responses=False),
    }


class ContextManager:
    """Store and retrieve contexts with embeddings.

    Parameters
    ----------
    backend:
        Search backend — "cylon" (Arrow SIMD) or "redis" (numpy dot product).
    persist_to_redis:
        When True (default), write full context metadata JSON and workflow sets
        to Redis. This is the primary persistence layer enabling cross-invocation
        context reuse within the Redis TTL window. When False, metadata is
        in-memory only for the lifetime of the process.
    dynamo_table:
        DynamoDB table name for durable long-term storage and analytics.
        ``None`` (default) disables DynamoDB.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl: int = 3600,
        persist_to_redis: bool = True,
        dynamo_table: Optional[str] = None,
        dynamo_endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        embedding_dim: int = 1024,
        backend: str = "cylon",
    ):
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid context_backend '{backend}'. Must be one of: {_VALID_BACKENDS}"
            )

        self._backend_name = backend
        self._embedding_dim = embedding_dim
        self._redis_ttl = redis_ttl
        self._persist_to_redis = persist_to_redis

        # DynamoDB — optional, disabled by default
        if dynamo_table is not None:
            dynamo_kwargs = {"region_name": region}
            if dynamo_endpoint_url:
                dynamo_kwargs["endpoint_url"] = dynamo_endpoint_url
            self._dynamo = boto3.resource("dynamodb", **dynamo_kwargs)
            self._table = self._dynamo.Table(dynamo_table)
            logger.info("ContextManager: DynamoDB enabled, table=%s", dynamo_table)
        else:
            self._dynamo = None
            self._table = None

        # In-memory metadata store (fallback when both persistence layers disabled)
        self._memory: dict[str, dict] = {}

        # Backend-specific initialization
        if backend == "cylon":
            redis_addr = f"tcp://{redis_host}:{redis_port}"
            self._cylon = _create_cylon_backend(embedding_dim, redis_addr, redis_ttl)
            self._redis = None
            logger.info(
                "ContextManager: backend=cylon, dim=%d, persist_to_redis=%s, dynamo=%s",
                embedding_dim, persist_to_redis, dynamo_table,
            )
        else:
            self._cylon = None
            self._redis = _create_redis_backend(redis_host, redis_port)
            logger.info(
                "ContextManager: backend=redis, dim=%d, redis=%s:%d, "
                "persist_to_redis=%s, dynamo=%s",
                embedding_dim, redis_host, redis_port, persist_to_redis, dynamo_table,
            )

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create ContextManager from BedrockConfig."""
        return cls(
            embedding_dim=config.embedding_dimensions,
            backend=config.context_backend,
            region=config.region,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_context(
        self,
        workflow_id: str,
        task_description: str,
        embedding: np.ndarray,
        response: str,
        cost_metadata: dict,
        context_id: Optional[str] = None,
    ) -> str:
        """Store context. Returns context_id."""
        if context_id is None:
            context_id = str(uuid.uuid4())

        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        now = datetime.now(timezone.utc).isoformat()

        # Search backend store
        if self._cylon is not None:
            self._store_cylon(context_id, workflow_id, embedding, response, cost_metadata)
        else:
            self._store_redis_embeddings(context_id, workflow_id, embedding)

        # Redis metadata persistence (primary persistence layer)
        if self._persist_to_redis and self._redis is not None:
            self._persist_redis_metadata(
                context_id, workflow_id, task_description,
                response, cost_metadata, now,
            )

        # DynamoDB persistence (optional durable store)
        if self._table is not None:
            self._persist_dynamo(
                context_id, workflow_id, task_description,
                embedding, response, cost_metadata, now,
            )

        # In-memory fallback when no external persistence configured
        if not self._persist_to_redis and self._table is None:
            self._memory[context_id] = {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "task_description": task_description,
                "response": response,
                "reuse_count": 0,
                "created_at": now,
                **cost_metadata,
            }

        logger.debug("Stored context %s (workflow %s, %d-dim)",
                     context_id, workflow_id, embedding.shape[0])
        return context_id

    def _store_cylon(self, context_id, workflow_id, embedding, response, cost_metadata):
        table = self._cylon["table"]
        table.put(
            context_id,
            embedding=embedding,
            workflow_id=workflow_id,
            response=response,
            model_id=cost_metadata.get("model_id", ""),
            input_tokens=cost_metadata.get("input_tokens", 0),
            output_tokens=cost_metadata.get("output_tokens", 0),
            cost_usd=float(cost_metadata.get("cost_usd", 0)),
        )
        # Arrow IPC snapshot to Redis (only when persist_to_redis enabled)
        if self._persist_to_redis and self._cylon["save_fn"]:
            try:
                self._cylon["save_fn"](
                    table,
                    key=f"context_table:{workflow_id}",
                    redis_addr=self._cylon["redis_addr"],
                    ttl_seconds=self._cylon["redis_ttl"],
                )
            except Exception as e:
                logger.warning("Failed to persist ContextTable to Redis: %s", e)

    def _store_redis_embeddings(self, context_id, workflow_id, embedding):
        """Write embedding key and workflow membership (minimum for search)."""
        client = self._redis["client"]
        pipe = client.pipeline()
        pipe.setex(f"embedding:{context_id}", self._redis_ttl, embedding.tobytes())
        pipe.sadd(f"workflow:{workflow_id}", context_id)
        pipe.expire(f"workflow:{workflow_id}", self._redis_ttl * 2)
        pipe.execute()

    def _persist_redis_metadata(
        self, context_id, workflow_id, task_description,
        response, cost_metadata, now,
    ):
        """Write full context metadata JSON to Redis."""
        client = self._redis["client"]
        client.setex(
            f"context:{context_id}",
            self._redis_ttl,
            json.dumps({
                "response": response,
                "task_description": task_description,
                "workflow_id": workflow_id,
                "created_at": now,
                "reuse_count": 0,
                "metadata": cost_metadata,
            }),
        )

    def _persist_dynamo(
        self, context_id, workflow_id, task_description,
        embedding, response, cost_metadata, now,
    ):
        """Write full context record to DynamoDB."""
        try:
            self._table.put_item(Item={
                "context_id": context_id,
                "workflow_id": workflow_id,
                "task_description": task_description,
                "embedding": embedding.tobytes(),
                "embedding_dim": int(embedding.shape[0]),
                "response": response,
                "model_id": cost_metadata.get("model_id", ""),
                "cost_input_tokens": cost_metadata.get("input_tokens", 0),
                "cost_output_tokens": cost_metadata.get("output_tokens", 0),
                "cost_usd": str(cost_metadata.get("cost_usd", 0)),
                "created_at": now,
                "reuse_count": 0,
            })
        except Exception as e:
            logger.error("Failed to persist context %s to DynamoDB: %s", context_id, e)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_context(self, context_id: str) -> Optional[dict]:
        """Retrieve context metadata by ID.

        Lookup order:
          1. Cylon Arrow table (if cylon backend)
          2. Redis metadata JSON (if persist_to_redis enabled)
          3. DynamoDB (if configured)
          4. In-memory store (fallback)
        """
        # Cylon: metadata lives in the Arrow table
        if self._cylon is not None:
            row = self._cylon["table"].get(context_id)
            if row is not None:
                return {
                    "context_id": context_id,
                    "response": row.column("response")[0].as_py(),
                    "model_id": row.column("model_id")[0].as_py(),
                    "input_tokens": row.column("input_tokens")[0].as_py(),
                    "output_tokens": row.column("output_tokens")[0].as_py(),
                    "cost_usd": row.column("cost_usd")[0].as_py(),
                    "source": "context_table",
                }

        # Redis metadata JSON (primary persistence)
        if self._persist_to_redis and self._redis is not None:
            cached = self._redis["client"].get(f"context:{context_id}")
            if cached:
                data = json.loads(cached)
                data["context_id"] = context_id
                data["source"] = "redis"
                return data

        # DynamoDB fallback
        if self._table is not None:
            try:
                response = self._table.scan(
                    FilterExpression="context_id = :cid",
                    ExpressionAttributeValues={":cid": context_id},
                    Limit=1,
                )
                items = response.get("Items", [])
                if items:
                    item = items[0]
                    item["source"] = "dynamodb"
                    return item
            except Exception as e:
                logger.error("Failed to get context %s from DynamoDB: %s", context_id, e)

        # In-memory fallback
        if context_id in self._memory:
            return dict(self._memory[context_id], source="memory")

        return None

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.85,
        top_k: int = 5,
        workflow_id: Optional[str] = None,
    ) -> list[dict]:
        """Cosine similarity search.

        Cylon backend: C++ SIMD on Arrow memory (zero-copy).
        Redis backend: numpy dot product over stored embeddings.

        Returns list of dicts with 'context_id' and 'similarity'.
        """
        if self._cylon is not None:
            return self._search_cylon(query_embedding, threshold, top_k, workflow_id)
        else:
            return self._search_redis(query_embedding, threshold, top_k, workflow_id)

    def _search_cylon(self, query, threshold, top_k, workflow_id):
        table = self._cylon["table"]
        results = table.search(
            query,
            threshold=threshold,
            top_k=top_k,
            workflow_id=workflow_id or "",
        )
        batch = table.to_arrow()
        if batch is not None:
            for r in results:
                r["context_id"] = batch.column("context_id")[r["index"]].as_py()
        return results

    def _search_redis(self, query, threshold, top_k, workflow_id):
        embeddings = self.get_all_embeddings(workflow_id)
        results = []
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return results

        for ctx_id, emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            sim = float(np.dot(query, emb) / (query_norm * emb_norm))
            if sim >= threshold:
                results.append({"context_id": ctx_id, "similarity": sim})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_all_embeddings(
        self,
        workflow_id: Optional[str] = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Return all (context_id, embedding) pairs."""
        if self._cylon is not None:
            return self._get_embeddings_cylon(workflow_id)
        else:
            return self._get_embeddings_redis(workflow_id)

    def _get_embeddings_cylon(self, workflow_id):
        batch = self._cylon["table"].to_arrow()
        # In-memory table is empty on every new invocation (Lambda/ECS cold start).
        # Attempt to restore from the Redis Arrow IPC snapshot so cross-invocation
        # context reuse works without requiring a stateful process.
        if (batch is None or batch.num_rows == 0) and workflow_id:
            self.load_from_redis(workflow_id)
            batch = self._cylon["table"].to_arrow()
        if batch is None or batch.num_rows == 0:
            return []
        results = []
        ctx_ids = batch.column("context_id")
        embeddings = batch.column("embedding")
        wf_ids = batch.column("workflow_id") if workflow_id else None
        for i in range(batch.num_rows):
            if workflow_id and wf_ids[i].as_py() != workflow_id:
                continue
            emb = np.array(embeddings[i].as_py(), dtype=np.float32)
            results.append((ctx_ids[i].as_py(), emb))
        return results

    def _get_embeddings_redis(self, workflow_id):
        client = self._redis["client"]
        results = []

        if workflow_id:
            context_ids = client.smembers(f"workflow:{workflow_id}")
            if context_ids:
                pipe = client.pipeline()
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

        # DynamoDB fallback for global scan (if configured)
        if self._table is not None:
            try:
                response = self._table.scan()
                for item in response.get("Items", []):
                    ctx_id = item["context_id"]
                    embedding_bytes = item.get("embedding")
                    if isinstance(embedding_bytes, bytes):
                        emb = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
                        results.append((ctx_id, emb))
            except Exception as e:
                logger.warning("Failed to scan embeddings from DynamoDB: %s", e)

        return results

    # ------------------------------------------------------------------
    # Cylon ContextTable access
    # ------------------------------------------------------------------

    def get_context_table(self):
        """Get the underlying Cylon ContextTable (None if redis backend)."""
        return self._cylon["table"] if self._cylon else None

    def load_from_redis(self, workflow_id: str) -> bool:
        """Load ContextTable from Redis (Arrow IPC). Cylon backend only."""
        if self._cylon is None or not self._cylon["load_fn"]:
            return False
        try:
            table = self._cylon["load_fn"](
                key=f"context_table:{workflow_id}",
                redis_addr=self._cylon["redis_addr"],
            )
            if table is not None:
                self._cylon["table"] = table
                logger.info("Loaded ContextTable from Redis: %d entries", table.size)
                return True
        except Exception as e:
            logger.warning("Failed to load ContextTable from Redis: %s", e)
        return False

    def load_from_ipc(self, ipc_data: bytes) -> None:
        """Load ContextTable from Arrow IPC bytes (e.g., from FMI broadcast)."""
        if self._cylon is None:
            raise RuntimeError("load_from_ipc requires context_backend='cylon'")
        from cylon_armada.context_table import ContextTable as CylonContextTable
        self._cylon["table"] = CylonContextTable.from_ipc(ipc_data)
        logger.info("Loaded ContextTable from IPC: %d entries", self._cylon["table"].size)

    def to_ipc(self) -> Optional[bytes]:
        """Serialize ContextTable to Arrow IPC bytes."""
        if self._cylon is None:
            return None
        return self._cylon["table"].to_ipc()

    # ------------------------------------------------------------------
    # FMI broadcast support
    # ------------------------------------------------------------------

    def cache_embedding(
        self,
        context_id: str,
        embedding: np.ndarray,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Pre-populate from FMI broadcast data."""
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        if self._cylon is not None:
            self._cylon["table"].put(
                context_id,
                embedding=embedding,
                workflow_id=workflow_id or "",
            )
        else:
            client = self._redis["client"]
            client.setex(
                f"embedding:{context_id}",
                self._redis_ttl,
                embedding.tobytes(),
            )
            if workflow_id:
                client.sadd(f"workflow:{workflow_id}", context_id)

    # ------------------------------------------------------------------
    # Persistence operations
    # ------------------------------------------------------------------

    def increment_reuse_count(self, context_id: str, workflow_id: str) -> None:
        """Increment reuse_count. Updates Redis JSON or DynamoDB if configured."""
        # Redis metadata
        if self._persist_to_redis and self._redis is not None:
            try:
                raw = self._redis["client"].get(f"context:{context_id}")
                if raw:
                    data = json.loads(raw)
                    data["reuse_count"] = data.get("reuse_count", 0) + 1
                    self._redis["client"].setex(
                        f"context:{context_id}",
                        self._redis_ttl,
                        json.dumps(data),
                    )
            except Exception as e:
                logger.error("Failed to increment reuse count in Redis for %s: %s", context_id, e)

        # DynamoDB
        if self._table is not None:
            try:
                self._table.update_item(
                    Key={"context_id": context_id, "workflow_id": workflow_id},
                    UpdateExpression="SET reuse_count = reuse_count + :inc",
                    ExpressionAttributeValues={":inc": 1},
                )
            except Exception as e:
                logger.error("Failed to increment reuse count in DynamoDB for %s: %s", context_id, e)

        # In-memory fallback
        if context_id in self._memory:
            self._memory[context_id]["reuse_count"] = (
                self._memory[context_id].get("reuse_count", 0) + 1
            )

    def get_workflow_contexts(self, workflow_id: str) -> list[dict]:
        """Return all context metadata records for a workflow.

        Sources checked in order: DynamoDB → Redis metadata → in-memory.
        Returns empty list if no persistence is configured.
        """
        # DynamoDB (authoritative when configured)
        if self._table is not None:
            try:
                response = self._table.query(
                    IndexName="workflow_id-created_at-index",
                    KeyConditionExpression="workflow_id = :wid",
                    ExpressionAttributeValues={":wid": workflow_id},
                )
                return response.get("Items", [])
            except Exception as e:
                logger.error("Failed to query workflow %s from DynamoDB: %s", workflow_id, e)

        # Redis metadata (primary persistence)
        if self._persist_to_redis and self._redis is not None:
            client = self._redis["client"]
            context_ids = client.smembers(f"workflow:{workflow_id}")
            results = []
            for cid in context_ids:
                cid_str = cid.decode() if isinstance(cid, bytes) else cid
                raw = client.get(f"context:{cid_str}")
                if raw:
                    data = json.loads(raw)
                    data["context_id"] = cid_str
                    results.append(data)
            return results

        # In-memory fallback
        return [
            v for v in self._memory.values()
            if v.get("workflow_id") == workflow_id
        ]

    def clear_workflow(self, workflow_id: str) -> int:
        """Delete all contexts for a workflow. Returns count deleted."""
        count = 0

        # DynamoDB cleanup
        if self._table is not None:
            try:
                items = self.get_workflow_contexts(workflow_id)
                for item in items:
                    ctx_id = item["context_id"]
                    self._table.delete_item(
                        Key={"context_id": ctx_id, "workflow_id": workflow_id},
                    )
                    count += 1
            except Exception as e:
                logger.error("Failed to clear workflow %s from DynamoDB: %s", workflow_id, e)

        # Search backend cleanup
        if self._cylon is not None:
            batch = self._cylon["table"].to_arrow()
            if batch is not None:
                for i in range(batch.num_rows):
                    if batch.column("workflow_id")[i].as_py() == workflow_id:
                        try:
                            self._cylon["table"].remove(
                                batch.column("context_id")[i].as_py())
                            count += 1
                        except KeyError:
                            pass
                self._cylon["table"].compact()
        elif self._redis is not None:
            client = self._redis["client"]
            context_ids = client.smembers(f"workflow:{workflow_id}")
            if context_ids:
                pipe = client.pipeline()
                for cid in context_ids:
                    cid_str = cid.decode() if isinstance(cid, bytes) else cid
                    pipe.delete(f"embedding:{cid_str}")
                    if self._persist_to_redis:
                        pipe.delete(f"context:{cid_str}")
                    count += 1
                pipe.delete(f"workflow:{workflow_id}")
                pipe.execute()

        # In-memory cleanup
        to_delete = [
            k for k, v in self._memory.items()
            if v.get("workflow_id") == workflow_id
        ]
        for k in to_delete:
            del self._memory[k]

        return count