/**
 * armada_aggregate — aggregate Map state results (Node.js / WASM).
 *
 * Receives the full array of per-task results from armada_executor,
 * computes cost/reuse summary, and returns the final workflow output.
 *
 * Input (from Step Functions state):
 *   {
 *     workflow_id,
 *     task_results: [ <armada_executor result>, ... ],
 *     prepare_cost: { ... }   // from armada_init
 *   }
 *
 * CMD: armada_aggregate.handler
 */

import { CostTracker } from './shared.mjs';

export async function handler(event) {
    const workflowId   = event.workflow_id || '';
    const taskResults  = event.task_results || [];
    const prepareCost  = event.prepare_cost || {};

    const aggregated = new CostTracker();
    let cacheHits = 0;
    let llmCalls  = 0;

    for (const result of taskResults) {
        if (result.source === 'cache') {
            cacheHits++;
            if (result.avoided_input_tokens && result.avoided_output_tokens) {
                const modelId = result.model_id || process.env.BEDROCK_LLM_MODEL_ID || 'unknown';
                aggregated.recordCacheHit(modelId, result.avoided_input_tokens, result.avoided_output_tokens);
            }
        } else if (result.source === 'llm') {
            llmCalls++;
            if (result.model_id) {
                aggregated.recordLlmCall(result.model_id, result.input_tokens || 0, result.output_tokens || 0);
            }
        }
    }

    // Add prepare-phase embedding costs
    if (prepareCost?.embedding_usage) {
        for (const [modelId, usage] of Object.entries(prepareCost.embedding_usage)) {
            aggregated.recordEmbeddingCall(modelId, usage.tokens || 0);
        }
    }

    return {
        workflow_id:  workflowId,
        total_tasks:  taskResults.length,
        cache_hits:   cacheHits,
        llm_calls:    llmCalls,
        reuse_rate:   taskResults.length > 0
            ? Math.round((cacheHits / taskResults.length) * 10000) / 100
            : 0,
        cost_summary: aggregated.getSummary(),
    };
}