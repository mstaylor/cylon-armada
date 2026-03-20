"""Bedrock pricing and cost tracking.

Pricing resolution follows the same precedence pattern as AWSPricing
in cylon/target/shared/scripts/scaling/costlib/aws_pricing.py:

    Env vars → Event payload → Config file → AWS Pricing API → Static defaults

This ensures the same code works across all execution contexts:
- Lambda: model IDs and config delivered via event payload
- ECS: model IDs and config delivered via environment variables
- Local: model IDs and config delivered via config file or CLI args
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# --- Static default pricing registries ---
# Keys are model ID prefixes — matched via longest prefix against full model IDs.

DEFAULT_LLM_PRICING = {
    "anthropic.claude-3-haiku": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "anthropic.claude-3-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "anthropic.claude-3-5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "anthropic.claude-3-5-haiku": {"input_per_1k": 0.001, "output_per_1k": 0.005},
    "anthropic.claude-3-opus": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "meta.llama3-8b-instruct": {"input_per_1k": 0.0003, "output_per_1k": 0.0006},
    "meta.llama3-70b-instruct": {"input_per_1k": 0.00195, "output_per_1k": 0.00256},
}

DEFAULT_EMBEDDING_PRICING = {
    "amazon.titan-embed-text-v2": {"per_1k": 0.00002},
    "amazon.titan-embed-text-v1": {"per_1k": 0.0001},
    "cohere.embed-english-v3": {"per_1k": 0.0001},
}

# --- Environment variable names ---

ENV_LLM_MODEL_ID = "BEDROCK_LLM_MODEL_ID"
ENV_EMBEDDING_MODEL_ID = "BEDROCK_EMBEDDING_MODEL_ID"
ENV_EMBEDDING_DIMENSIONS = "BEDROCK_EMBEDDING_DIMENSIONS"
ENV_SIMILARITY_THRESHOLD = "SIMILARITY_THRESHOLD"
ENV_PRICING_CONFIG_PATH = "BEDROCK_PRICING_CONFIG"
ENV_AWS_REGION = "AWS_DEFAULT_REGION"
ENV_CONTEXT_BACKEND = "CONTEXT_BACKEND"


@dataclass
class BedrockConfig:
    """Unified configuration for Bedrock models and experiment parameters.

    Resolved from multiple sources with precedence:
        Env vars → Event payload → Config file → Defaults
    """

    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    similarity_threshold: float = 0.85
    region: str = "us-east-1"
    context_backend: str = "cylon"  # "cylon" (Arrow SIMD, default) or "redis" (numpy+bytes)

    @classmethod
    def from_env(cls) -> "BedrockConfig":
        """Resolve config from environment variables. Missing values use defaults."""
        return cls(
            llm_model_id=os.environ.get(ENV_LLM_MODEL_ID, cls.llm_model_id),
            embedding_model_id=os.environ.get(ENV_EMBEDDING_MODEL_ID, cls.embedding_model_id),
            embedding_dimensions=int(os.environ.get(ENV_EMBEDDING_DIMENSIONS, cls.embedding_dimensions)),
            similarity_threshold=float(os.environ.get(ENV_SIMILARITY_THRESHOLD, cls.similarity_threshold)),
            region=os.environ.get(ENV_AWS_REGION, cls.region),
            context_backend=os.environ.get(ENV_CONTEXT_BACKEND, cls.context_backend),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "BedrockConfig":
        """Resolve config from a Lambda event payload or dict."""
        return cls(
            llm_model_id=payload.get("llm_model_id", cls.llm_model_id),
            embedding_model_id=payload.get("embedding_model_id", cls.embedding_model_id),
            embedding_dimensions=int(payload.get("embedding_dimensions", cls.embedding_dimensions)),
            similarity_threshold=float(payload.get("similarity_threshold", cls.similarity_threshold)),
            region=payload.get("region", cls.region),
            context_backend=payload.get("context_backend", cls.context_backend),
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> Optional["BedrockConfig"]:
        """Resolve config from a JSON or YAML config file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file not found: %s", config_path)
            return None
        try:
            data = json.loads(path.read_text())
            bedrock = data.get("bedrock", data)
            return cls.from_payload(bedrock)
        except Exception as e:
            logger.warning("Failed to load config %s: %s", config_path, e)
            return None

    @classmethod
    def resolve(
        cls,
        payload: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> "BedrockConfig":
        """Resolve config with precedence:
        1. Env vars (highest — ECS, Lambda env overrides)
        2. Event payload (Lambda invocation)
        3. Config file (local experiments)
        4. Static defaults (fallback)

        Values from higher-precedence sources override lower ones field by field.
        """
        # Start with defaults
        config = cls()

        # 4 → 3: Config file
        if config_path is None:
            config_path = os.environ.get(ENV_PRICING_CONFIG_PATH)
        if config_path:
            file_config = cls.from_config_file(config_path)
            if file_config:
                config = file_config

        # 3 → 2: Event payload overrides
        if payload:
            if "llm_model_id" in payload:
                config.llm_model_id = payload["llm_model_id"]
            if "embedding_model_id" in payload:
                config.embedding_model_id = payload["embedding_model_id"]
            if "embedding_dimensions" in payload:
                config.embedding_dimensions = int(payload["embedding_dimensions"])
            if "similarity_threshold" in payload:
                config.similarity_threshold = float(payload["similarity_threshold"])
            if "region" in payload:
                config.region = payload["region"]
            if "context_backend" in payload:
                config.context_backend = payload["context_backend"]

        # 2 → 1: Env vars override everything
        if ENV_LLM_MODEL_ID in os.environ:
            config.llm_model_id = os.environ[ENV_LLM_MODEL_ID]
        if ENV_EMBEDDING_MODEL_ID in os.environ:
            config.embedding_model_id = os.environ[ENV_EMBEDDING_MODEL_ID]
        if ENV_EMBEDDING_DIMENSIONS in os.environ:
            config.embedding_dimensions = int(os.environ[ENV_EMBEDDING_DIMENSIONS])
        if ENV_SIMILARITY_THRESHOLD in os.environ:
            config.similarity_threshold = float(os.environ[ENV_SIMILARITY_THRESHOLD])
        if ENV_AWS_REGION in os.environ:
            config.region = os.environ[ENV_AWS_REGION]
        if ENV_CONTEXT_BACKEND in os.environ:
            config.context_backend = os.environ[ENV_CONTEXT_BACKEND]

        return config


@dataclass
class BedrockPricing:
    """Bedrock model pricing — registry-based, model-aware.

    Pricing resolution:
        Env vars → Config file → AWS Pricing API → Static defaults
    """

    llm_pricing: dict = field(default_factory=lambda: dict(DEFAULT_LLM_PRICING))
    embedding_pricing: dict = field(default_factory=lambda: dict(DEFAULT_EMBEDDING_PRICING))
    source: str = "defaults"

    def _match_prefix(self, model_id: str, registry: dict) -> Optional[dict]:
        """Find pricing by longest prefix match against model ID."""
        best_match = None
        best_len = 0
        for prefix, pricing in registry.items():
            if model_id.startswith(prefix) and len(prefix) > best_len:
                best_match = pricing
                best_len = len(prefix)
        return best_match

    def get_llm_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate LLM cost for a given model and token counts."""
        pricing = self._match_prefix(model_id, self.llm_pricing)
        if pricing is None:
            logger.warning("No pricing for model '%s', using max pricing as fallback", model_id)
            pricing = max(
                self.llm_pricing.values(),
                key=lambda p: p.get("input_per_1k", 0) + p.get("output_per_1k", 0),
            )
        return (
            input_tokens * pricing["input_per_1k"] / 1000
            + output_tokens * pricing["output_per_1k"] / 1000
        )

    def get_embedding_cost(self, model_id: str, token_count: int) -> float:
        """Calculate embedding cost for a given model and token count."""
        pricing = self._match_prefix(model_id, self.embedding_pricing)
        if pricing is None:
            logger.warning("No pricing for embedding model '%s', using max pricing", model_id)
            pricing = max(self.embedding_pricing.values(), key=lambda p: p["per_1k"])
        return token_count * pricing["per_1k"] / 1000

    @classmethod
    def from_aws_api(cls, region: str = "us-east-1") -> Optional["BedrockPricing"]:
        """Pull current Bedrock prices from AWS Pricing API.

        Merges live prices into the default registries.
        Returns None on failure so the caller can fall back.
        """
        try:
            client = boto3.client("pricing", region_name="us-east-1")
            pricing = cls()

            response = client.get_products(
                ServiceCode="AmazonBedrock",
                Filters=[],
                MaxResults=100,
            )

            for price_item in response.get("PriceList", []):
                item = json.loads(price_item) if isinstance(price_item, str) else price_item
                attributes = item.get("product", {}).get("attributes", {})
                model_id = attributes.get("modelId", "")

                terms = item.get("terms", {}).get("OnDemand", {})
                for term in terms.values():
                    for dimension in term.get("priceDimensions", {}).values():
                        price = float(dimension["pricePerUnit"].get("USD", 0))
                        desc = dimension.get("description", "").lower()
                        if price <= 0:
                            continue

                        if "embed" in model_id:
                            pricing.embedding_pricing[model_id] = {"per_1k": price}
                        elif "input" in desc:
                            entry = pricing.llm_pricing.setdefault(model_id, {})
                            entry["input_per_1k"] = price
                        elif "output" in desc:
                            entry = pricing.llm_pricing.setdefault(model_id, {})
                            entry["output_per_1k"] = price

            pricing.source = "aws_api"
            logger.info("Resolved Bedrock pricing from AWS Pricing API")
            return pricing

        except (ClientError, Exception) as e:
            logger.warning("Failed to fetch Bedrock pricing from AWS API: %s", e)
            return None

    @classmethod
    def from_config(cls, config_path: str) -> Optional["BedrockPricing"]:
        """Load pricing overrides from a JSON config file.

        Expected format:
        {
            "pricing": {
                "llm": {
                    "anthropic.claude-3-haiku": {"input_per_1k": 0.003, "output_per_1k": 0.015}
                },
                "embedding": {
                    "amazon.titan-embed-text-v2": {"per_1k": 0.00002}
                }
            }
        }
        """
        path = Path(config_path)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            pricing_data = data.get("pricing", data)

            llm = dict(DEFAULT_LLM_PRICING)
            embedding = dict(DEFAULT_EMBEDDING_PRICING)
            llm.update(pricing_data.get("llm", {}))
            embedding.update(pricing_data.get("embedding", {}))

            pricing = cls(llm_pricing=llm, embedding_pricing=embedding, source="config")
            logger.info("Resolved Bedrock pricing from config: %s", config_path)
            return pricing
        except Exception as e:
            logger.warning("Failed to load pricing config %s: %s", config_path, e)
            return None

    @classmethod
    def resolve(
        cls,
        region: str = "us-east-1",
        config_path: Optional[str] = None,
    ) -> "BedrockPricing":
        """Resolve pricing with precedence:
        1. Config file (if provided or via BEDROCK_PRICING_CONFIG env var)
        2. AWS Pricing API (live prices)
        3. Static defaults (hardcoded fallback)
        """
        if config_path is None:
            config_path = os.environ.get(ENV_PRICING_CONFIG_PATH)

        if config_path:
            pricing = cls.from_config(config_path)
            if pricing:
                return pricing

        pricing = cls.from_aws_api(region)
        if pricing:
            return pricing

        logger.info("Using static default Bedrock pricing")
        return cls(source="defaults")


@dataclass
class BedrockCostTracker:
    """Track Bedrock LLM and embedding costs for experiments.

    All cost calculations delegate to BedrockPricing which is model-aware.
    Per-model usage is tracked for detailed breakdowns.
    """

    pricing: BedrockPricing = field(default_factory=BedrockPricing)

    # Per-model counters
    llm_usage: dict = field(default_factory=dict)
    embedding_usage: dict = field(default_factory=dict)
    cache_hits: int = 0
    cache_avoided: list = field(default_factory=list)

    @classmethod
    def create(
        cls,
        region: str = "us-east-1",
        config_path: Optional[str] = None,
    ) -> "BedrockCostTracker":
        """Create a tracker with auto-resolved pricing."""
        pricing = BedrockPricing.resolve(region=region, config_path=config_path)
        return cls(pricing=pricing)

    def record_llm_call(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record an LLM call and return its cost in USD."""
        cost = self.pricing.get_llm_cost(model_id, input_tokens, output_tokens)
        usage = self.llm_usage.setdefault(model_id, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
        })
        usage["calls"] += 1
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        return cost

    def record_embedding_call(self, model_id: str, token_count: int) -> float:
        """Record an embedding call and return its cost in USD."""
        cost = self.pricing.get_embedding_cost(model_id, token_count)
        usage = self.embedding_usage.setdefault(model_id, {
            "calls": 0, "tokens": 0,
        })
        usage["calls"] += 1
        usage["tokens"] += token_count
        return cost

    def record_cache_hit(
        self,
        model_id: str,
        avoided_input_tokens: int,
        avoided_output_tokens: int,
    ) -> float:
        """Record a cache hit and return the avoided cost in USD."""
        avoided_cost = self.pricing.get_llm_cost(
            model_id, avoided_input_tokens, avoided_output_tokens,
        )
        self.cache_hits += 1
        self.cache_avoided.append({
            "model_id": model_id,
            "input_tokens": avoided_input_tokens,
            "output_tokens": avoided_output_tokens,
            "avoided_cost": avoided_cost,
        })
        return avoided_cost

    @property
    def total_llm_cost(self) -> float:
        total = 0.0
        for model_id, usage in self.llm_usage.items():
            total += self.pricing.get_llm_cost(
                model_id, usage["input_tokens"], usage["output_tokens"],
            )
        return total

    @property
    def total_embedding_cost(self) -> float:
        total = 0.0
        for model_id, usage in self.embedding_usage.items():
            total += self.pricing.get_embedding_cost(model_id, usage["tokens"])
        return total

    @property
    def total_cost(self) -> float:
        return self.total_llm_cost + self.total_embedding_cost

    @property
    def total_avoided_cost(self) -> float:
        return sum(entry["avoided_cost"] for entry in self.cache_avoided)

    @property
    def baseline_cost(self) -> float:
        """What the workflow would have cost without any caching."""
        return self.total_cost + self.total_avoided_cost

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost == 0:
            return 0.0
        return 1.0 - (self.total_cost / self.baseline_cost)

    def get_summary(self) -> dict:
        total_llm_calls = sum(u["calls"] for u in self.llm_usage.values())
        total_embedding_calls = sum(u["calls"] for u in self.embedding_usage.values())

        return {
            "total_cost": round(self.total_cost, 6),
            "baseline_cost": round(self.baseline_cost, 6),
            "savings_pct": round(self.savings_pct * 100, 2),
            "pricing_source": self.pricing.source,
            "llm_calls": total_llm_calls,
            "embedding_calls": total_embedding_calls,
            "cache_hits": self.cache_hits,
            "llm_usage_by_model": dict(self.llm_usage),
            "embedding_usage_by_model": dict(self.embedding_usage),
            "cost_breakdown": {
                "llm": round(self.total_llm_cost, 6),
                "embedding": round(self.total_embedding_cost, 6),
                "avoided": round(self.total_avoided_cost, 6),
            },
        }

    def reset(self):
        """Reset all counters for a new experiment run."""
        self.llm_usage.clear()
        self.embedding_usage.clear()
        self.cache_hits = 0
        self.cache_avoided.clear()