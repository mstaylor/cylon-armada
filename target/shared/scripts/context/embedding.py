"""Embedding Service — generates embeddings via Amazon Titan Text Embeddings V2.

Supports configurable dimensions (256, 512, 1024) for experiment sweeps.
Cost tracking is delegated to BedrockCostTracker — this service only
reports token counts.

Redis embedding cache: embed() checks Redis before calling Bedrock. Cache key
is sha256(text + model_id + str(dimensions)) so different configs don't collide.
TTL is 7 days — long enough for multi-day sweeps, short enough to auto-expire.
Set REDIS_HOST="" to disable the cache (falls through to Bedrock every time).
"""

import base64
import hashlib
import json
import logging
import os
import time
from typing import Optional

import boto3
import numpy as np
from botocore.config import Config

from cost.bedrock_pricing import BedrockConfig

logger = logging.getLogger(__name__)

VALID_DIMENSIONS = (256, 512, 1024)
_CACHE_TTL = 7 * 24 * 3600  # 7 days


class EmbeddingService:
    """Generate embeddings via Bedrock with configurable model and dimensions."""

    def __init__(
        self,
        config: Optional[BedrockConfig] = None,
        endpoint_url: Optional[str] = None,
    ):
        if config is None:
            config = BedrockConfig.resolve()
        self.config = config

        if config.embedding_dimensions not in VALID_DIMENSIONS:
            raise ValueError(
                f"embedding_dimensions must be one of {VALID_DIMENSIONS}, "
                f"got {config.embedding_dimensions}"
            )

        client_kwargs = {"region_name": config.region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        # Adaptive retry mode backs off automatically on ThrottlingException.
        # 10 attempts gives ~90s of cumulative wait at max backoff — enough for
        # bursts when many Lambda workers embed simultaneously.
        client_kwargs["config"] = Config(retries={"max_attempts": 10, "mode": "adaptive"})
        self._client = boto3.client("bedrock-runtime", **client_kwargs)

        # Optional Redis embedding cache — avoids re-calling Bedrock for
        # identical task descriptions across sweep runs.
        self._redis = None
        redis_host = os.environ.get("REDIS_HOST", "")
        if redis_host:
            try:
                import redis as _redis
                redis_port = int(os.environ.get("REDIS_PORT", 6379))
                self._redis = _redis.Redis(
                    host=redis_host, port=redis_port,
                    decode_responses=False, socket_connect_timeout=2,
                )
                self._redis.ping()
                logger.debug("EmbeddingService: Redis cache enabled at %s:%d", redis_host, redis_port)
            except Exception as e:
                logger.warning("EmbeddingService: Redis cache unavailable, using Bedrock only: %s", e)
                self._redis = None

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(
            f"{text}|{self.model_id}|{self.dimensions}".encode()
        ).hexdigest()
        return f"emb_cache:{digest}"

    @property
    def model_id(self) -> str:
        return self.config.embedding_model_id

    @property
    def dimensions(self) -> int:
        return self.config.embedding_dimensions

    def embed(self, text: str) -> tuple[np.ndarray, dict]:
        """Generate embedding for a single text.

        Checks Redis cache first (keyed by sha256 of text+model+dims).
        Falls back to Bedrock on cache miss and stores the result.

        Returns:
            (embedding, metadata) where embedding is float32 ndarray
            and metadata contains token_count and latency_ms.
            Cost is NOT calculated here — pass token_count to BedrockCostTracker.
        """
        # --- Cache lookup ---------------------------------------------------
        if self._redis is not None:
            try:
                key = self._cache_key(text)
                cached = self._redis.get(key)
                if cached is not None:
                    embedding = np.frombuffer(base64.b64decode(cached), dtype=np.float32)
                    logger.debug("EmbeddingService: cache hit for key %s", key[:16])
                    return embedding, {"token_count": 0, "latency_ms": 0.0,
                                       "dimensions": self.dimensions, "model_id": self.model_id,
                                       "cache_hit": True}
            except Exception as e:
                logger.warning("EmbeddingService: cache read failed, falling through to Bedrock: %s", e)

        # --- Bedrock call ---------------------------------------------------
        start = time.perf_counter()

        body = json.dumps({
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        })

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
        )

        result = json.loads(response["body"].read())
        elapsed_ms = (time.perf_counter() - start) * 1000

        embedding = np.array(result["embedding"], dtype=np.float32)
        token_count = result.get("inputTextTokenCount", 0)

        # --- Cache store ----------------------------------------------------
        if self._redis is not None:
            try:
                key = self._cache_key(text)
                self._redis.setex(key, _CACHE_TTL, base64.b64encode(embedding.tobytes()))
            except Exception as e:
                logger.warning("EmbeddingService: cache write failed: %s", e)

        metadata = {
            "token_count": token_count,
            "latency_ms": round(elapsed_ms, 2),
            "dimensions": self.dimensions,
            "model_id": self.model_id,
        }

        logger.debug(
            "Embedded %d chars → %d-dim vector (%.1fms, %d tokens)",
            len(text), self.dimensions, elapsed_ms, token_count,
        )

        return embedding, metadata

    def embed_batch(self, texts: list[str]) -> list[tuple[np.ndarray, dict]]:
        """Generate embeddings for multiple texts sequentially.

        Titan V2 has no native batch API, so this calls embed() per text.
        Cache hits skip Bedrock entirely; misses use adaptive retry backoff.
        """
        return [self.embed(text) for text in texts]