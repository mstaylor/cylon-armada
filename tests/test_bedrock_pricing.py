"""Tests for BedrockConfig, BedrockPricing, and BedrockCostTracker."""

import os
import json
import tempfile
import pytest

# Adjust path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from cost.bedrock_pricing import BedrockConfig, BedrockPricing, BedrockCostTracker


# ---------------------------------------------------------------------------
# BedrockConfig
# ---------------------------------------------------------------------------

class TestBedrockConfig:
    def test_defaults(self):
        config = BedrockConfig()
        assert config.embedding_model_id == "amazon.titan-embed-text-v2:0"
        assert config.embedding_dimensions == 1024
        assert config.similarity_threshold == 0.85

    def test_direct_params(self):
        config = BedrockConfig(
            llm_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            embedding_dimensions=256,
            similarity_threshold=0.9,
        )
        assert config.llm_model_id == "anthropic.claude-3-haiku-20240307-v1:0"
        assert config.embedding_dimensions == 256
        assert config.similarity_threshold == 0.9

    def test_resolve_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_LLM_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
        monkeypatch.setenv("BEDROCK_EMBEDDING_DIMENSIONS", "512")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
        config = BedrockConfig.resolve()
        assert config.llm_model_id == "meta.llama3-8b-instruct-v1:0"
        assert config.embedding_dimensions == 512
        assert config.similarity_threshold == 0.75

    def test_resolve_payload_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_LLM_MODEL_ID", "env-model")
        config = BedrockConfig.resolve(payload={"llm_model_id": "payload-model"})
        assert config.llm_model_id == "payload-model"

    def test_resolve_from_config_file(self):
        config_data = {
            "llm_model_id": "file-model",
            "embedding_dimensions": 256,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            config = BedrockConfig.resolve(config_path=f.name)

        assert config.llm_model_id == "file-model"
        assert config.embedding_dimensions == 256
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# BedrockPricing
# ---------------------------------------------------------------------------

class TestBedrockPricing:
    def test_prefix_matching(self):
        pricing = BedrockPricing()
        # Should match the longest prefix
        result = pricing.get_llm_pricing("anthropic.claude-3-haiku-20240307-v1:0")
        assert result is not None
        assert "input_per_1k" in result

    def test_unknown_model_returns_fallback(self):
        pricing = BedrockPricing()
        result = pricing.get_llm_pricing("unknown.model-v1:0")
        # Should return conservative fallback, not crash
        assert result is not None

    def test_embedding_pricing(self):
        pricing = BedrockPricing()
        result = pricing.get_embedding_pricing("amazon.titan-embed-text-v2:0")
        assert result is not None
        assert "per_1k" in result


# ---------------------------------------------------------------------------
# BedrockCostTracker
# ---------------------------------------------------------------------------

class TestBedrockCostTracker:
    def test_record_llm_call(self):
        tracker = BedrockCostTracker()
        cost = tracker.record_llm_call(
            "anthropic.claude-3-haiku-20240307-v1:0",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost > 0
        assert tracker.total_cost == cost

    def test_record_embedding_call(self):
        tracker = BedrockCostTracker()
        cost = tracker.record_embedding_call(
            "amazon.titan-embed-text-v2:0",
            token_count=1000,
        )
        assert cost > 0
        assert cost == pytest.approx(0.00002, abs=1e-6)

    def test_record_cache_hit(self):
        tracker = BedrockCostTracker()
        # First record an LLM call so we have a baseline
        tracker.record_llm_call("anthropic.claude-3-haiku-20240307-v1:0", 1000, 500)
        initial_cost = tracker.total_cost

        # Cache hit avoids cost
        avoided = tracker.record_cache_hit(
            "anthropic.claude-3-haiku-20240307-v1:0",
            avoided_input_tokens=1000,
            avoided_output_tokens=500,
        )
        assert avoided > 0
        assert tracker.total_avoided_cost == avoided
        # Total cost shouldn't change from cache hit
        assert tracker.total_cost == initial_cost

    def test_savings_pct(self):
        tracker = BedrockCostTracker()
        model = "anthropic.claude-3-haiku-20240307-v1:0"

        # 1 real call + 3 cache hits = 75% savings
        tracker.record_llm_call(model, 1000, 500)
        for _ in range(3):
            tracker.record_cache_hit(model, 1000, 500)

        assert tracker.savings_pct == pytest.approx(75.0, abs=1.0)

    def test_get_summary_structure(self):
        tracker = BedrockCostTracker()
        tracker.record_llm_call("anthropic.claude-3-haiku-20240307-v1:0", 100, 50)
        tracker.record_embedding_call("amazon.titan-embed-text-v2:0", 200)

        summary = tracker.get_summary()
        assert "total_cost" in summary
        assert "baseline_cost" in summary
        assert "savings_pct" in summary
        assert "llm_usage" in summary
        assert "embedding_usage" in summary

    def test_multiple_models(self):
        tracker = BedrockCostTracker()
        cost1 = tracker.record_llm_call("anthropic.claude-3-haiku-20240307-v1:0", 1000, 500)
        cost2 = tracker.record_llm_call("meta.llama3-8b-instruct-v1:0", 1000, 500)

        # Different models should have different pricing
        assert cost1 != cost2
        assert tracker.total_cost == pytest.approx(cost1 + cost2, abs=1e-8)

    def test_zero_tokens_cache_hit_warns(self, caplog):
        """Cache hit with 0 tokens should work but log a warning."""
        tracker = BedrockCostTracker()
        avoided = tracker.record_cache_hit("anthropic.claude-3-haiku-20240307-v1:0", 0, 0)
        assert avoided == 0