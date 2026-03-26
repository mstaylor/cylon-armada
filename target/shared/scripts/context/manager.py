"""Context Manager — stores and retrieves contexts with embeddings.

Backend is configured via BedrockConfig.context_backend:
  - "cylon" (default): Cylon ContextTable — Arrow-native, C++ SIMD search,
    zero-copy embeddings, Redis persistence via Arrow IPC
  - "redis": Raw numpy+Redis — legacy mode for environments without pycylon

The backend is an explicit configuration choice, not a silent fallback.
If "cylon" is configured and pycylon is not available, it raises an error.
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

    # Check if Redis persistence is available (compiled with CYLON_USE_REDIS)
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

    Backend is determined by config.context_backend:
      - "cylon": Cylon ContextTable with Arrow SIMD search (default)
      - "redis": Raw numpy arrays in Redis

    DynamoDB is the durable persistence layer for both backends.
    """

    def __init__(
        self,
        dynamo_table: str = "context-store",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl: int = 3600,
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

        # DynamoDB (durable store — used by both backends)
        dynamo_kwargs = {"region_name": region}
        if dynamo_endpoint_url:
            dynamo_kwargs["endpoint_url"] = dynamo_endpoint_url
        self._dynamo = boto3.resource("dynamodb", **dynamo_kwargs)
        self._table = self._dynamo.Table(dynamo_table)

        # Backend-specific initialization
        if backend == "cylon":
            redis_addr = f"tcp://{redis_host}:{redis_port}"
            self._cylon = _create_cylon_backend(embedding_dim, redis_addr, redis_ttl)
            self._redis = None
            logger.info("ContextManager: backend=cylon, dim=%d", embedding_dim)
        else:
            self._cylon = None
            self._redis = _create_redis_backend(redis_host, redis_port)
            logger.info("ContextManager: backend=redis, dim=%d, redis=%s:%d", embedding_dim, redis_host, redis_port)

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

        if self._cylon is not None:
            self._store_cylon(context_id, workflow_id, embedding, response, cost_metadata)
        else:
            self._store_redis(context_id, workflow_id, embedding, response, cost_metadata)

        # DynamoDB (durable)
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
        # Persist to Redis via Arrow IPC
        if self._cylon["save_fn"]:
            try:
                self._cylon["save_fn"](
                    table,
                    key=f"context_table:{workflow_id}",
                    redis_addr=self._cylon["redis_addr"],
                    ttl_seconds=self._cylon["redis_ttl"],
                )
            except Exception as e:
                logger.warning("Failed to persist ContextTable to Redis: %s", e)

    def _store_redis(self, context_id, workflow_id, embedding, response, cost_metadata):
        client = self._redis["client"]
        pipe = client.pipeline()
        pipe.setex(f"embedding:{context_id}", self._redis_ttl, embedding.tobytes())
        pipe.setex(
            f"context:{context_id}",
            self._redis_ttl,
            json.dumps({"response": response, "metadata": cost_metadata}),
        )
        pipe.sadd(f"workflow:{workflow_id}", context_id)
        pipe.expire(f"workflow:{workflow_id}", self._redis_ttl * 2)
        pipe.execute()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_context(self, context_id: str) -> Optional[dict]:
        """Retrieve context by ID."""
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
        else:
            cached = self._redis["client"].get(f"context:{context_id}")
            if cached:
                data = json.loads(cached)
                data["context_id"] = context_id
                data["source"] = "cache"
                return data

        # DynamoDB fallback (durable store)
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
            item["source"] = "dynamodb"
            return item
        except Exception as e:
            logger.error("Failed to get context %s from DynamoDB: %s", context_id, e)
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
        """SIMD cosine similarity search.

        Cylon backend: delegates to C++ SIMD on Arrow memory (zero-copy).
        Redis backend: numpy cosine similarity.

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
        # Enrich with context_id from the Arrow batch
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
    # DynamoDB operations
    # ------------------------------------------------------------------

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
            count += 1

        if self._cylon is not None:
            batch = self._cylon["table"].to_arrow()
            if batch is not None:
                for i in range(batch.num_rows):
                    if batch.column("workflow_id")[i].as_py() == workflow_id:
                        try:
                            self._cylon["table"].remove(
                                batch.column("context_id")[i].as_py())
                        except KeyError:
                            pass
                self._cylon["table"].compact()
        else:
            self._redis["client"].delete(f"workflow:{workflow_id}")

        return count