"""Embedding Service — generates embeddings via Amazon Titan Text Embeddings V2.

Supports configurable dimensions (256, 512, 1024) for experiment sweeps.
Cost tracking is delegated to BedrockCostTracker — this service only
reports token counts.
"""

import json
import logging
import time
from typing import Optional

import boto3
import numpy as np

from cost.bedrock_pricing import BedrockConfig

logger = logging.getLogger(__name__)

VALID_DIMENSIONS = (256, 512, 1024)


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
        self._client = boto3.client("bedrock-runtime", **client_kwargs)

    @property
    def model_id(self) -> str:
        return self.config.embedding_model_id

    @property
    def dimensions(self) -> int:
        return self.config.embedding_dimensions

    def embed(self, text: str) -> tuple[np.ndarray, dict]:
        """Generate embedding for a single text.

        Returns:
            (embedding, metadata) where embedding is float32 ndarray
            and metadata contains token_count and latency_ms.
            Cost is NOT calculated here — pass token_count to BedrockCostTracker.
        """
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
        """
        return [self.embed(text) for text in texts]