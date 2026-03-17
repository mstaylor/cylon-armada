/**
 * Tests for CostTracker — Node.js equivalent of BedrockCostTracker.
 *
 * CostTracker is defined inside context_handler.mjs. Since it's not
 * exported, we test it by importing a test-only helper or duplicating
 * the class here. For now we extract and test the logic directly.
 */

import { describe, test, expect } from '@jest/globals';

// Re-implement CostTracker for isolated testing (mirrors context_handler.mjs)
const DEFAULT_LLM_PRICING = {
    'anthropic.claude-3-haiku': { input_per_1k: 0.003, output_per_1k: 0.015 },
    'amazon.nova-lite': { input_per_1k: 0.0006, output_per_1k: 0.0024 },
    'meta.llama3-8b-instruct': { input_per_1k: 0.0003, output_per_1k: 0.0006 },
};

const DEFAULT_EMBEDDING_PRICING = {
    'amazon.titan-embed-text-v2': { per_1k: 0.00002 },
    'amazon.titan-embed-text-v1': { per_1k: 0.0001 },
};

class CostTracker {
    constructor(llmPricing = null, embeddingPricing = null) {
        this.llmPricing = llmPricing || DEFAULT_LLM_PRICING;
        this.embeddingPricing = embeddingPricing || DEFAULT_EMBEDDING_PRICING;
        this.llmUsage = {};
        this.embeddingUsage = {};
        this.cacheHits = {};
    }

    _matchPrefix(modelId, registry) {
        let bestMatch = null;
        let bestLen = 0;
        for (const prefix of Object.keys(registry)) {
            if (modelId.startsWith(prefix) && prefix.length > bestLen) {
                bestMatch = prefix;
                bestLen = prefix.length;
            }
        }
        return bestMatch;
    }

    recordLlmCall(modelId, inputTokens, outputTokens) {
        const prefix = this._matchPrefix(modelId, this.llmPricing);
        const pricing = prefix ? this.llmPricing[prefix] : { input_per_1k: 0.003, output_per_1k: 0.015 };
        const cost = (inputTokens / 1000) * pricing.input_per_1k
                   + (outputTokens / 1000) * pricing.output_per_1k;

        if (!this.llmUsage[modelId]) {
            this.llmUsage[modelId] = { input_tokens: 0, output_tokens: 0, cost: 0, calls: 0 };
        }
        this.llmUsage[modelId].input_tokens += inputTokens;
        this.llmUsage[modelId].output_tokens += outputTokens;
        this.llmUsage[modelId].cost += cost;
        this.llmUsage[modelId].calls += 1;
        return cost;
    }

    recordEmbeddingCall(modelId, tokenCount) {
        const prefix = this._matchPrefix(modelId, this.embeddingPricing);
        const pricing = prefix ? this.embeddingPricing[prefix] : { per_1k: 0.00002 };
        const cost = (tokenCount / 1000) * pricing.per_1k;

        if (!this.embeddingUsage[modelId]) {
            this.embeddingUsage[modelId] = { tokens: 0, cost: 0, calls: 0 };
        }
        this.embeddingUsage[modelId].tokens += tokenCount;
        this.embeddingUsage[modelId].cost += cost;
        this.embeddingUsage[modelId].calls += 1;
        return cost;
    }

    recordCacheHit(modelId, avoidedInputTokens, avoidedOutputTokens) {
        const prefix = this._matchPrefix(modelId, this.llmPricing);
        const pricing = prefix ? this.llmPricing[prefix] : { input_per_1k: 0.003, output_per_1k: 0.015 };
        const avoidedCost = (avoidedInputTokens / 1000) * pricing.input_per_1k
                          + (avoidedOutputTokens / 1000) * pricing.output_per_1k;

        if (!this.cacheHits[modelId]) {
            this.cacheHits[modelId] = { avoided_input: 0, avoided_output: 0, avoided_cost: 0, hits: 0 };
        }
        this.cacheHits[modelId].avoided_input += avoidedInputTokens;
        this.cacheHits[modelId].avoided_output += avoidedOutputTokens;
        this.cacheHits[modelId].avoided_cost += avoidedCost;
        this.cacheHits[modelId].hits += 1;
        return avoidedCost;
    }

    get totalCost() {
        const llm = Object.values(this.llmUsage).reduce((s, u) => s + u.cost, 0);
        const emb = Object.values(this.embeddingUsage).reduce((s, u) => s + u.cost, 0);
        return llm + emb;
    }

    get totalAvoidedCost() {
        return Object.values(this.cacheHits).reduce((s, h) => s + h.avoided_cost, 0);
    }

    get baselineCost() { return this.totalCost + this.totalAvoidedCost; }

    get savingsPct() {
        const baseline = this.baselineCost;
        return baseline > 0 ? ((this.totalAvoidedCost / baseline) * 100) : 0;
    }

    getSummary() {
        return {
            total_cost: this.totalCost,
            baseline_cost: this.baselineCost,
            total_avoided_cost: this.totalAvoidedCost,
            savings_pct: Math.round(this.savingsPct * 100) / 100,
            llm_usage: this.llmUsage,
            embedding_usage: this.embeddingUsage,
            cache_hits: this.cacheHits,
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('CostTracker', () => {
    describe('prefix matching', () => {
        test('matches known model prefix', () => {
            const tracker = new CostTracker();
            const prefix = tracker._matchPrefix('anthropic.claude-3-haiku-20240307-v1:0', DEFAULT_LLM_PRICING);
            expect(prefix).toBe('anthropic.claude-3-haiku');
        });

        test('returns null for unknown model', () => {
            const tracker = new CostTracker();
            const prefix = tracker._matchPrefix('unknown.model-v1:0', DEFAULT_LLM_PRICING);
            expect(prefix).toBeNull();
        });

        test('matches longest prefix', () => {
            const pricing = {
                'anthropic': { input_per_1k: 0.001, output_per_1k: 0.002 },
                'anthropic.claude-3': { input_per_1k: 0.003, output_per_1k: 0.015 },
            };
            const tracker = new CostTracker(pricing);
            const prefix = tracker._matchPrefix('anthropic.claude-3-haiku', pricing);
            expect(prefix).toBe('anthropic.claude-3');
        });
    });

    describe('recordLlmCall', () => {
        test('returns positive cost for known model', () => {
            const tracker = new CostTracker();
            const cost = tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            expect(cost).toBeGreaterThan(0);
        });

        test('accumulates usage per model', () => {
            const tracker = new CostTracker();
            tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 2000, 1000);

            const usage = tracker.llmUsage['anthropic.claude-3-haiku-20240307-v1:0'];
            expect(usage.calls).toBe(2);
            expect(usage.input_tokens).toBe(3000);
            expect(usage.output_tokens).toBe(1500);
        });

        test('uses fallback pricing for unknown model', () => {
            const tracker = new CostTracker();
            const cost = tracker.recordLlmCall('unknown.model', 1000, 500);
            expect(cost).toBeGreaterThan(0);
        });
    });

    describe('recordEmbeddingCall', () => {
        test('returns correct cost for Titan V2', () => {
            const tracker = new CostTracker();
            const cost = tracker.recordEmbeddingCall('amazon.titan-embed-text-v2:0', 1000);
            expect(cost).toBeCloseTo(0.00002, 8);
        });

        test('accumulates embedding usage', () => {
            const tracker = new CostTracker();
            tracker.recordEmbeddingCall('amazon.titan-embed-text-v2:0', 1000);
            tracker.recordEmbeddingCall('amazon.titan-embed-text-v2:0', 2000);

            const usage = tracker.embeddingUsage['amazon.titan-embed-text-v2:0'];
            expect(usage.calls).toBe(2);
            expect(usage.tokens).toBe(3000);
        });
    });

    describe('recordCacheHit', () => {
        test('returns avoided cost', () => {
            const tracker = new CostTracker();
            const avoided = tracker.recordCacheHit('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            expect(avoided).toBeGreaterThan(0);
        });

        test('does not increase totalCost', () => {
            const tracker = new CostTracker();
            tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            const costBefore = tracker.totalCost;

            tracker.recordCacheHit('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            expect(tracker.totalCost).toBe(costBefore);
        });

        test('zero tokens returns zero avoided cost', () => {
            const tracker = new CostTracker();
            const avoided = tracker.recordCacheHit('anthropic.claude-3-haiku-20240307-v1:0', 0, 0);
            expect(avoided).toBe(0);
        });
    });

    describe('savings calculation', () => {
        test('75% savings with 1 call + 3 cache hits', () => {
            const tracker = new CostTracker();
            const model = 'anthropic.claude-3-haiku-20240307-v1:0';

            tracker.recordLlmCall(model, 1000, 500);
            tracker.recordCacheHit(model, 1000, 500);
            tracker.recordCacheHit(model, 1000, 500);
            tracker.recordCacheHit(model, 1000, 500);

            expect(tracker.savingsPct).toBeCloseTo(75.0, 0);
        });

        test('0% savings with no cache hits', () => {
            const tracker = new CostTracker();
            tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            expect(tracker.savingsPct).toBe(0);
        });

        test('baselineCost = totalCost + totalAvoidedCost', () => {
            const tracker = new CostTracker();
            const model = 'anthropic.claude-3-haiku-20240307-v1:0';

            tracker.recordLlmCall(model, 1000, 500);
            tracker.recordEmbeddingCall('amazon.titan-embed-text-v2:0', 500);
            tracker.recordCacheHit(model, 1000, 500);

            expect(tracker.baselineCost).toBeCloseTo(
                tracker.totalCost + tracker.totalAvoidedCost, 10
            );
        });
    });

    describe('getSummary', () => {
        test('returns complete structure', () => {
            const tracker = new CostTracker();
            tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 100, 50);
            tracker.recordEmbeddingCall('amazon.titan-embed-text-v2:0', 200);
            tracker.recordCacheHit('anthropic.claude-3-haiku-20240307-v1:0', 100, 50);

            const summary = tracker.getSummary();
            expect(summary).toHaveProperty('total_cost');
            expect(summary).toHaveProperty('baseline_cost');
            expect(summary).toHaveProperty('total_avoided_cost');
            expect(summary).toHaveProperty('savings_pct');
            expect(summary).toHaveProperty('llm_usage');
            expect(summary).toHaveProperty('embedding_usage');
            expect(summary).toHaveProperty('cache_hits');
            expect(summary.total_cost).toBeGreaterThan(0);
            expect(summary.total_avoided_cost).toBeGreaterThan(0);
        });
    });

    describe('multiple models', () => {
        test('different models have different costs', () => {
            const tracker = new CostTracker();
            const cost1 = tracker.recordLlmCall('anthropic.claude-3-haiku-20240307-v1:0', 1000, 500);
            const cost2 = tracker.recordLlmCall('meta.llama3-8b-instruct-v1:0', 1000, 500);

            expect(cost1).not.toBe(cost2);
            expect(tracker.totalCost).toBeCloseTo(cost1 + cost2, 10);
        });
    });
});