/**
 * armada_executor — per-task Lambda invoked by the Map state (Node.js / WASM).
 *
 * Mirrors armada_executor.py and the cylon_executor pattern:
 *   - Receives a single task payload (one item from armada_init's body)
 *   - Runs context-reuse routing (similarity search → reuse or LLM call)
 *   - Returns the routing result
 *
 * Infrastructure config (REDIS_HOST, CONTEXT_BACKEND, etc.) comes from
 * Lambda environment variables, not the event payload.
 *
 * CMD: armada_executor.handler
 */

import { StopWatch } from './stopwatch.mjs';
import {
    initWasm,
    getContextBackend,
    CostTracker,
    b64ToNdArray,
    getAllEmbeddings,
    getContext,
    incrementReuseCount,
    storeContext,
    cosineSimilaritySearch,
    invokeLLM,
} from './shared.mjs';

export async function handler(event) {
    const taskDescription   = event.task_description;
    const workflowId        = event.workflow_id;
    const rank              = event.rank || 0;
    const worldSize         = event.world_size || 1;
    const config            = event.config || {};
    const embeddingMetadata = event.embedding_metadata;

    const threshold = parseFloat(
        config.similarity_threshold || process.env.SIMILARITY_THRESHOLD || '0.85'
    );

    const wasm        = getContextBackend() === 'redis' ? null : await initWasm();
    const costTracker = new CostTracker();

    StopWatch.start('route_total');

    // Record embedding cost from prepare step
    if (embeddingMetadata) {
        costTracker.recordEmbeddingCall(embeddingMetadata.model_id, embeddingMetadata.token_count);
    }

    // Decode pre-computed embedding
    const queryEmbedding = b64ToNdArray(event.embedding_b64);

    // Similarity search
    StopWatch.start('search_latency');
    const storedEmbeddings = await getAllEmbeddings(workflowId);
    const matches = cosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold);
    StopWatch.stop('search_latency');

    if (matches.length > 0) {
        // Cache hit
        const bestMatch = matches[0];
        const context   = await getContext(bestMatch.contextId);
        await incrementReuseCount(bestMatch.contextId);

        const avoidedInput  = context.input_tokens  || 0;
        const avoidedOutput = context.output_tokens || 0;
        const llmModelId    = context.model_id || process.env.BEDROCK_LLM_MODEL_ID || 'unknown';

        if (avoidedInput === 0 && avoidedOutput === 0) {
            console.warn('Cache hit but no token counts on original context — avoided cost will be 0');
        }

        const avoidedCost = costTracker.recordCacheHit(llmModelId, avoidedInput, avoidedOutput);
        StopWatch.stop('route_total');

        return {
            response:              context.response,
            source:                'cache',
            similarity:            bestMatch.similarity,
            context_id:            bestMatch.contextId,
            cost_usd:              0,
            avoided_cost_usd:      avoidedCost,
            total_latency_ms:      StopWatch.getMs('route_total'),
            search_latency_ms:     StopWatch.getMs('search_latency'),
            avoided_input_tokens:  avoidedInput,
            avoided_output_tokens: avoidedOutput,
            cost_summary:          costTracker.getSummary(),
            task_description:      taskDescription,
            rank,
            workflow_id:           workflowId,
        };
    }

    // Cache miss — invoke LLM
    StopWatch.start('llm_latency');
    const llmResult = await invokeLLM(taskDescription);
    StopWatch.stop('llm_latency');
    const callCost = costTracker.recordLlmCall(llmResult.model_id, llmResult.input_tokens, llmResult.output_tokens);

    // Store new context
    StopWatch.start('store_latency');
    const contextId = crypto.randomUUID();
    await storeContext(contextId, workflowId, taskDescription, queryEmbedding, llmResult.response, {
        model_id:      llmResult.model_id,
        input_tokens:  llmResult.input_tokens,
        output_tokens: llmResult.output_tokens,
        cost_usd:      callCost,
    });
    StopWatch.stop('store_latency');
    StopWatch.stop('route_total');

    return {
        response:          llmResult.response,
        source:            'llm',
        similarity:        0,
        context_id:        contextId,
        input_tokens:      llmResult.input_tokens,
        output_tokens:     llmResult.output_tokens,
        cost_usd:          callCost,
        total_latency_ms:  StopWatch.getMs('route_total'),
        search_latency_ms: StopWatch.getMs('search_latency'),
        llm_latency_ms:    llmResult.latency_ms,
        store_latency_ms:  StopWatch.getMs('store_latency'),
        model_id:          llmResult.model_id,
        cost_summary:      costTracker.getSummary(),
        task_description:  taskDescription,
        rank,
        workflow_id:       workflowId,
    };
}