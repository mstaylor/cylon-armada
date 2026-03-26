/**
 * Path B — Node.js Lambda handler for context reuse with WASM SIMD128.
 *
 * Uses Cylon's cylon-wasm module for SIMD-accelerated cosine similarity
 * via WASM SIMD128 instructions. Follows the same routing logic as the
 * Python handler but through the Node.js runtime.
 *
 * Actions (matching Python run_action.py):
 *   prepare_tasks     — embed all tasks, generate per-task payloads
 *   route_task        — similarity search + reuse or LLM call for one task
 *   aggregate_results — collect per-task results, compute cost summary
 *   cosmic_ai_infer   — ONNX inference + task generation
 *   embed_and_search  — embed text + similarity search (utility)
 *   simd_benchmark    — pure SIMD throughput measurement
 *
 * Based on: cylon/target/aws/scripts/lambda/wasm_handler.mjs
 *
 * Environment variables:
 *   CYLON_WASM_PATH:     Path to cylon_wasm_bg.wasm
 *   CYLON_WASM_BINDINGS: Path to cylon_wasm.js bindings
 *   REDIS_HOST:          Redis host for context cache
 *   REDIS_PORT:          Redis port (default: 6379)
 *   DYNAMO_TABLE_NAME:   DynamoDB table (default: context-store)
 *   BEDROCK_LLM_MODEL_ID:       LLM model ID
 *   BEDROCK_EMBEDDING_MODEL_ID: Embedding model ID
 *   BEDROCK_EMBEDDING_DIMENSIONS: Embedding dimensions (256/512/1024)
 *   SIMILARITY_THRESHOLD:        Cosine similarity threshold for reuse
 */

import { readFileSync } from 'fs';
import { StopWatch } from './stopwatch.mjs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createClient } from 'redis';
import { DynamoDBClient, PutItemCommand, GetItemCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------------------------------------------------------------------------
// WASM module (lazy-loaded, reused across invocations)
// ---------------------------------------------------------------------------

let wasmModule = null;

async function initWasm() {
    if (wasmModule) return wasmModule;

    const wasmPath = process.env.CYLON_WASM_PATH ||
        join(__dirname, '..', '..', '..', '..', '..', 'cylon', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm_bg.wasm');
    const bindingsPath = process.env.CYLON_WASM_BINDINGS ||
        join(__dirname, '..', '..', '..', '..', '..', 'cylon', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm.js');

    const bindings = await import(bindingsPath);

    // wasm-pack --target nodejs: CJS module auto-initializes WASM on require()
    // wasm-pack --target web/bundler: needs bindings.default({ module_or_path })
    if (typeof bindings.default === 'function') {
        const wasmBytes = readFileSync(wasmPath);
        await bindings.default({ module_or_path: wasmBytes });
    }

    if (typeof bindings.init === 'function') bindings.init();

    wasmModule = bindings;
    return wasmModule;
}

// ---------------------------------------------------------------------------
// AWS clients (lazy-loaded)
// ---------------------------------------------------------------------------

let redisClient = null;
let dynamoClient = null;
let bedrockClient = null;

async function getRedis() {
    if (redisClient) return redisClient;
    const host = process.env.REDIS_HOST || 'localhost';
    const port = parseInt(process.env.REDIS_PORT || '6379');
    redisClient = createClient({ url: `redis://${host}:${port}` });
    await redisClient.connect();
    return redisClient;
}

function getDynamo() {
    if (!dynamoClient) {
        dynamoClient = new DynamoDBClient({
            region: process.env.AWS_DEFAULT_REGION || 'us-east-1',
        });
    }
    return dynamoClient;
}

function getBedrock() {
    if (!bedrockClient) {
        bedrockClient = new BedrockRuntimeClient({
            region: process.env.AWS_DEFAULT_REGION || 'us-east-1',
        });
    }
    return bedrockClient;
}

// ---------------------------------------------------------------------------
// Cost tracker — mirrors Python BedrockCostTracker
// ---------------------------------------------------------------------------

// Default pricing registry — configurable via config_path or env
const DEFAULT_LLM_PRICING = {
    'anthropic.claude-3-haiku': { input_per_1k: 0.003, output_per_1k: 0.015 },
    'anthropic.claude-3-5-sonnet': { input_per_1k: 0.003, output_per_1k: 0.015 },
    'amazon.nova-lite': { input_per_1k: 0.0006, output_per_1k: 0.0024 },
    'amazon.nova-micro': { input_per_1k: 0.00035, output_per_1k: 0.0014 },
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
        this.llmUsage = {};       // { model_id: { input_tokens, output_tokens, cost, calls } }
        this.embeddingUsage = {};  // { model_id: { tokens, cost, calls } }
        this.cacheHits = {};       // { model_id: { avoided_input, avoided_output, avoided_cost, hits } }
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

    get baselineCost() {
        return this.totalCost + this.totalAvoidedCost;
    }

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
// Embedding service
// ---------------------------------------------------------------------------

async function embedText(text) {
    const modelId = process.env.BEDROCK_EMBEDDING_MODEL_ID || 'amazon.titan-embed-text-v2:0';
    const dimensions = parseInt(process.env.BEDROCK_EMBEDDING_DIMENSIONS || '1024');

    const start = performance.now();
    const response = await getBedrock().send(new InvokeModelCommand({
        modelId,
        contentType: 'application/json',
        body: JSON.stringify({
            inputText: text,
            dimensions,
            normalize: true,
        }),
    }));

    const result = JSON.parse(new TextDecoder().decode(response.body));
    const latencyMs = performance.now() - start;

    return {
        embedding: new Float32Array(result.embedding),
        metadata: {
            model_id: modelId,
            dimensions,
            token_count: result.inputTextTokenCount || 0,
            latency_ms: Math.round(latencyMs * 100) / 100,
        },
    };
}

// ---------------------------------------------------------------------------
// Context store operations
// ---------------------------------------------------------------------------

const TABLE_NAME = process.env.DYNAMO_TABLE_NAME || 'context-store';

async function storeContext(contextId, workflowId, taskDescription, embedding, response, costMetadata) {
    const redis = await getRedis();
    const dynamo = getDynamo();

    // DynamoDB
    await dynamo.send(new PutItemCommand({
        TableName: TABLE_NAME,
        Item: {
            context_id: { S: contextId },
            workflow_id: { S: workflowId },
            task_description: { S: taskDescription },
            embedding: { B: Buffer.from(embedding.buffer) },
            embedding_dim: { N: String(embedding.length) },
            response: { S: response },
            model_id: { S: costMetadata.model_id || '' },
            cost_input_tokens: { N: String(costMetadata.input_tokens || 0) },
            cost_output_tokens: { N: String(costMetadata.output_tokens || 0) },
            cost_usd: { N: String(costMetadata.cost_usd || 0) },
            created_at: { S: new Date().toISOString() },
            reuse_count: { N: '0' },
        },
    }));

    // Redis cache
    const pipeline = redis.multi();
    pipeline.set(`embedding:${contextId}`, Buffer.from(embedding.buffer), { EX: 3600 });
    pipeline.set(`context:${contextId}`, JSON.stringify({ response, ...costMetadata }), { EX: 3600 });
    pipeline.sAdd(`workflow:${workflowId}`, contextId);
    pipeline.expire(`workflow:${workflowId}`, 7200);
    await pipeline.exec();
}

async function getAllEmbeddings(workflowId) {
    const redis = await getRedis();
    const contextIds = await redis.sMembers(`workflow:${workflowId}`);

    const results = [];
    for (const contextId of contextIds) {
        const embBytes = await redis.getBuffer(`embedding:${contextId}`);
        if (embBytes) {
            results.push({
                contextId,
                embedding: new Float32Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 4),
            });
        }
    }
    return results;
}

async function getContext(contextId) {
    const redis = await getRedis();
    const cached = await redis.get(`context:${contextId}`);
    if (cached) return JSON.parse(cached);

    const dynamo = getDynamo();
    const result = await dynamo.send(new GetItemCommand({
        TableName: TABLE_NAME,
        Key: { context_id: { S: contextId } },
    }));

    if (!result.Item) return null;
    return {
        response: result.Item.response?.S || '',
        model_id: result.Item.model_id?.S || '',
        input_tokens: parseInt(result.Item.cost_input_tokens?.N || '0'),
        output_tokens: parseInt(result.Item.cost_output_tokens?.N || '0'),
        cost_usd: parseFloat(result.Item.cost_usd?.N || '0'),
    };
}

async function incrementReuseCount(contextId) {
    const dynamo = getDynamo();
    await dynamo.send(new UpdateItemCommand({
        TableName: TABLE_NAME,
        Key: { context_id: { S: contextId } },
        UpdateExpression: 'ADD reuse_count :inc',
        ExpressionAttributeValues: { ':inc': { N: '1' } },
    }));
}

// ---------------------------------------------------------------------------
// SIMD similarity search (WASM SIMD128 via cylon-wasm)
// ---------------------------------------------------------------------------

function simdCosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold, topK = 5) {
    const results = [];

    for (const { contextId, embedding } of storedEmbeddings) {
        const similarity = wasm.cosine_similarity_f32(
            Array.from(queryEmbedding),
            Array.from(embedding),
        );

        if (similarity >= threshold) {
            results.push({ contextId, similarity });
        }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
}

// ---------------------------------------------------------------------------
// LLM invocation via Bedrock
// ---------------------------------------------------------------------------

async function invokeLLM(taskDescription, systemPrompt = null) {
    const modelId = process.env.BEDROCK_LLM_MODEL_ID || 'amazon.nova-lite-v1:0';

    const messages = [];
    if (systemPrompt) {
        messages.push({ role: 'user', content: [{ text: systemPrompt + '\n\n' + taskDescription }] });
    } else {
        messages.push({ role: 'user', content: [{ text: taskDescription }] });
    }

    const start = performance.now();
    const response = await getBedrock().send(new InvokeModelCommand({
        modelId,
        contentType: 'application/json',
        body: JSON.stringify({
            messages,
            inferenceConfig: { temperature: 0.0 },
        }),
    }));

    const result = JSON.parse(new TextDecoder().decode(response.body));
    const latencyMs = performance.now() - start;

    return {
        response: result.output?.message?.content?.[0]?.text || '',
        input_tokens: result.usage?.inputTokens || 0,
        output_tokens: result.usage?.outputTokens || 0,
        latency_ms: Math.round(latencyMs * 100) / 100,
        model_id: modelId,
    };
}

// ---------------------------------------------------------------------------
// Base64 encoding/decoding for embeddings
// ---------------------------------------------------------------------------

function ndArrayToB64(float32Array) {
    return Buffer.from(float32Array.buffer).toString('base64');
}

function b64ToNdArray(b64String) {
    const buf = Buffer.from(b64String, 'base64');
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---------------------------------------------------------------------------
// Action: prepare_tasks
// ---------------------------------------------------------------------------

async function actionPrepareTasks(params, costTracker) {
    const { workflow_id: workflowId, tasks, config = {} } = params;
    const start = performance.now();

    const taskPayloads = [];
    for (let i = 0; i < tasks.length; i++) {
        const { embedding, metadata } = await embedText(tasks[i]);
        costTracker.recordEmbeddingCall(metadata.model_id, metadata.token_count);

        taskPayloads.push({
            task_description: tasks[i],
            embedding_b64: ndArrayToB64(embedding),
            embedding_metadata: metadata,
            workflow_id: workflowId,
            rank: i,
            world_size: tasks.length,
            config,
        });
    }

    return {
        workflow_id: workflowId,
        task_payloads: taskPayloads,
        prepare_cost: costTracker.getSummary(),
        prepare_latency_ms: Math.round((performance.now() - start) * 100) / 100,
    };
}

// ---------------------------------------------------------------------------
// Action: route_task
// ---------------------------------------------------------------------------

async function actionRouteTask(wasm, params, costTracker) {
    const {
        task_description: taskDescription,
        embedding_b64: embeddingB64,
        embedding_metadata: embeddingMetadata,
        workflow_id: workflowId,
        config = {},
    } = params;

    const threshold = parseFloat(config.similarity_threshold || process.env.SIMILARITY_THRESHOLD || '0.85');
    StopWatch.start('route_total');

    // Record embedding cost from prepare step
    if (embeddingMetadata) {
        costTracker.recordEmbeddingCall(embeddingMetadata.model_id, embeddingMetadata.token_count);
    }

    // Decode pre-computed embedding
    const queryEmbedding = b64ToNdArray(embeddingB64);

    // Similarity search
    StopWatch.start('search');
    const storedEmbeddings = await getAllEmbeddings(workflowId);
    const matches = simdCosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold);
    StopWatch.stop('search');

    if (matches.length > 0) {
        // Cache hit
        const bestMatch = matches[0];
        const context = await getContext(bestMatch.contextId);
        await incrementReuseCount(bestMatch.contextId);

        const avoidedInput = context.input_tokens || 0;
        const avoidedOutput = context.output_tokens || 0;
        const llmModelId = context.model_id || process.env.BEDROCK_LLM_MODEL_ID || 'unknown';

        if (avoidedInput === 0 && avoidedOutput === 0) {
            console.warn('Cache hit but no token counts on original context — avoided cost will be 0');
        }

        const avoidedCost = costTracker.recordCacheHit(llmModelId, avoidedInput, avoidedOutput);
        StopWatch.stop('route_total');

        return {
            response: context.response,
            source: 'cache',
            similarity: bestMatch.similarity,
            context_id: bestMatch.contextId,
            cost_usd: 0,
            avoided_cost_usd: avoidedCost,
            total_latency_ms: StopWatch.getMs('route_total'),
            search_latency_ms: StopWatch.getMs('search'),
            avoided_input_tokens: avoidedInput,
            avoided_output_tokens: avoidedOutput,
            cost_summary: costTracker.getSummary(),
        };
    }

    // Cache miss — invoke LLM
    StopWatch.start('llm_call');
    const llmResult = await invokeLLM(taskDescription);
    StopWatch.stop('llm_call');
    const callCost = costTracker.recordLlmCall(llmResult.model_id, llmResult.input_tokens, llmResult.output_tokens);

    // Store new context
    StopWatch.start('store_context');
    const contextId = crypto.randomUUID();
    await storeContext(contextId, workflowId, taskDescription, queryEmbedding, llmResult.response, {
        model_id: llmResult.model_id,
        input_tokens: llmResult.input_tokens,
        output_tokens: llmResult.output_tokens,
        cost_usd: callCost,
    });
    StopWatch.stop('store_context');
    StopWatch.stop('route_total');

    return {
        response: llmResult.response,
        source: 'llm',
        similarity: 0,
        context_id: contextId,
        input_tokens: llmResult.input_tokens,
        output_tokens: llmResult.output_tokens,
        cost_usd: callCost,
        total_latency_ms: StopWatch.getMs('route_total'),
        search_latency_ms: StopWatch.getMs('search'),
        llm_latency_ms: llmResult.latency_ms,
        store_latency_ms: StopWatch.getMs('store_context'),
        model_id: llmResult.model_id,
        cost_summary: costTracker.getSummary(),
    };
}

// ---------------------------------------------------------------------------
// Action: aggregate_results
// ---------------------------------------------------------------------------

function actionAggregateResults(params) {
    const { workflow_id: workflowId, task_results: taskResults, prepare_cost: prepareCost } = params;

    const aggregated = new CostTracker();
    let cacheHits = 0;
    let llmCalls = 0;

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
        workflow_id: workflowId,
        total_tasks: taskResults.length,
        cache_hits: cacheHits,
        llm_calls: llmCalls,
        reuse_rate: taskResults.length > 0 ? Math.round((cacheHits / taskResults.length) * 10000) / 100 : 0,
        cost_summary: aggregated.getSummary(),
    };
}

// ---------------------------------------------------------------------------
// Action: cosmic_ai_infer
// ---------------------------------------------------------------------------

async function actionCosmicAiInfer(params) {
    const { runInferenceFromFile } = await import('./inference.mjs');
    const { generateTasks } = await import('./task_generator.mjs');

    const { data_path: dataPath, model_path: modelPath, max_tasks: maxTasks, seed, config } = params;

    const inferenceResult = await runInferenceFromFile(dataPath, modelPath);

    // Reshape magnitudes from flat to (N, 5)
    const nSamples = inferenceResult.predictions.length;
    const magnitudes = [];
    if (inferenceResult.true_redshifts) {
        // Data file included magnitudes in a flat array
        for (let i = 0; i < nSamples; i++) {
            magnitudes.push(Array.from({ length: 5 }, (_, j) => 0)); // placeholder
        }
    }

    const tasks = generateTasks({
        predictions: inferenceResult.predictions,
        trueRedshifts: inferenceResult.true_redshifts || inferenceResult.predictions,
        magnitudes,
        metrics: inferenceResult.metrics,
        maxTasks: maxTasks || nSamples,
        seed: seed || 42,
        configPath: config?.cosmic_ai_config || null,
    });

    return {
        tasks,
        inference_metrics: inferenceResult.metrics,
        num_predictions: nSamples,
    };
}

// ---------------------------------------------------------------------------
// Lambda handler
// ---------------------------------------------------------------------------

export async function handler(event, context) {
    try {
        const action = event.action || event.ACTION;
        if (!action) {
            return { statusCode: 400, body: JSON.stringify({ error: "Missing 'action' field" }) };
        }

        const wasm = await initWasm();
        const costTracker = new CostTracker();
        const payload = event.action_payload || event;

        switch (action) {
            case 'prepare_tasks':
                return await actionPrepareTasks(payload, costTracker);

            case 'route_task':
                return await actionRouteTask(wasm, payload, costTracker);

            case 'aggregate_results':
                return actionAggregateResults(payload);

            case 'cosmic_ai_infer':
                return await actionCosmicAiInfer(payload);

            case 'embed_and_search': {
                const { text, workflow_id, threshold = 0.85, top_k = 5 } = payload;
                const { embedding, metadata } = await embedText(text);
                costTracker.recordEmbeddingCall(metadata.model_id, metadata.token_count);
                const stored = await getAllEmbeddings(workflow_id);
                const matches = simdCosineSimilaritySearch(wasm, embedding, stored, threshold, top_k);
                return { matches, embedding_length: embedding.length, cost_summary: costTracker.getSummary() };
            }

            case 'simd_benchmark': {
                const { dim = 1024, n = 1000, iterations = 100 } = payload;
                const query = new Float32Array(dim).map(() => Math.random());
                const embeddings = Array.from({ length: n }, () => ({
                    contextId: 'bench',
                    embedding: new Float32Array(dim).map(() => Math.random()),
                }));

                const start = performance.now();
                for (let i = 0; i < iterations; i++) {
                    simdCosineSimilaritySearch(wasm, query, embeddings, 0.0, n);
                }
                const elapsedMs = performance.now() - start;

                return {
                    dim,
                    n_embeddings: n,
                    iterations,
                    total_ms: Math.round(elapsedMs * 100) / 100,
                    avg_search_ms: Math.round((elapsedMs / iterations) * 100) / 100,
                    comparisons_per_sec: Math.round((n * iterations) / (elapsedMs / 1000)),
                };
            }

            default:
                return { statusCode: 400, body: JSON.stringify({ error: `Unknown action: ${action}` }) };
        }
    } catch (error) {
        console.error('Error:', error);
        return { statusCode: 500, body: JSON.stringify({ error: error.message }) };
    }
}

// Local testing
const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
    const testEvent = {
        action: 'simd_benchmark',
        action_payload: { dim: 256, n: 100, iterations: 10 },
    };
    handler(testEvent, {}).then(result => {
        console.log(JSON.stringify(result, null, 2));
    });
}