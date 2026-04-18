/**
 * shared.mjs — shared infrastructure for armada Lambda functions.
 *
 * Extracted from context_handler.mjs so armada_init, armada_executor,
 * and armada_aggregate can each import only what they need without
 * duplicating client setup, pricing tables, or WASM initialization.
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createClient, commandOptions } from 'redis';
import { DynamoDBClient, PutItemCommand, GetItemCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';

export const __filename = fileURLToPath(import.meta.url);
export const __dirname = dirname(__filename);

// ---------------------------------------------------------------------------
// WASM (lazy-loaded, reused across warm invocations)
// ---------------------------------------------------------------------------

let wasmModule = null;

export async function initWasm() {
    if (wasmModule) return wasmModule;

    const wasmPath = process.env.CYLON_WASM_PATH ||
        join(__dirname, '..', '..', '..', '..', '..', 'cylon', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm_bg.wasm');
    const bindingsPath = process.env.CYLON_WASM_BINDINGS ||
        join(__dirname, '..', '..', '..', '..', '..', 'cylon', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm.js');

    const bindings = await import(bindingsPath);

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
let redisUnavailable = false;
let dynamoClient = null;
let bedrockClient = null;

export async function getRedis() {
    if (redisUnavailable) return null;
    if (redisClient) return redisClient;
    const host = process.env.REDIS_HOST || 'localhost';
    const port = parseInt(process.env.REDIS_PORT || '6379');
    try {
        const client = createClient({
            url: `redis://${host}:${port}`,
            socket: { connectTimeout: 3000 },
        });
        await client.connect();
        redisClient = client;
        return redisClient;
    } catch (err) {
        console.warn(`Redis unavailable (${host}:${port}) — continuing without persistence: ${err.message}`);
        redisUnavailable = true;
        return null;
    }
}

export function getDynamo() {
    if (!dynamoClient) {
        dynamoClient = new DynamoDBClient({
            region: process.env.AWS_DEFAULT_REGION || 'us-east-1',
        });
    }
    return dynamoClient;
}

export function getBedrock() {
    if (!bedrockClient) {
        bedrockClient = new BedrockRuntimeClient({
            region: process.env.AWS_DEFAULT_REGION || 'us-east-1',
        });
    }
    return bedrockClient;
}

// ---------------------------------------------------------------------------
// Cost tracker
// ---------------------------------------------------------------------------

const DEFAULT_LLM_PRICING = {
    'anthropic.claude-3-haiku':     { input_per_1k: 0.003,   output_per_1k: 0.015  },
    'anthropic.claude-3-5-sonnet':  { input_per_1k: 0.003,   output_per_1k: 0.015  },
    'amazon.nova-lite':             { input_per_1k: 0.0006,  output_per_1k: 0.0024 },
    'amazon.nova-micro':            { input_per_1k: 0.00035, output_per_1k: 0.0014 },
    'meta.llama3-8b-instruct':      { input_per_1k: 0.0003,  output_per_1k: 0.0006 },
};

const DEFAULT_EMBEDDING_PRICING = {
    'amazon.titan-embed-text-v2': { per_1k: 0.00002 },
    'amazon.titan-embed-text-v1': { per_1k: 0.0001  },
};

export class CostTracker {
    constructor(llmPricing = null, embeddingPricing = null) {
        this.llmPricing       = llmPricing       || DEFAULT_LLM_PRICING;
        this.embeddingPricing = embeddingPricing  || DEFAULT_EMBEDDING_PRICING;
        this.llmUsage       = {};
        this.embeddingUsage = {};
        this.cacheHits      = {};
    }

    _matchPrefix(modelId, registry) {
        let bestMatch = null, bestLen = 0;
        for (const prefix of Object.keys(registry)) {
            if (modelId.startsWith(prefix) && prefix.length > bestLen) {
                bestMatch = prefix; bestLen = prefix.length;
            }
        }
        return bestMatch;
    }

    recordLlmCall(modelId, inputTokens, outputTokens) {
        const prefix  = this._matchPrefix(modelId, this.llmPricing);
        const pricing = prefix ? this.llmPricing[prefix] : { input_per_1k: 0.003, output_per_1k: 0.015 };
        const cost    = (inputTokens / 1000) * pricing.input_per_1k
                      + (outputTokens / 1000) * pricing.output_per_1k;
        if (!this.llmUsage[modelId])
            this.llmUsage[modelId] = { input_tokens: 0, output_tokens: 0, cost: 0, calls: 0 };
        this.llmUsage[modelId].input_tokens  += inputTokens;
        this.llmUsage[modelId].output_tokens += outputTokens;
        this.llmUsage[modelId].cost          += cost;
        this.llmUsage[modelId].calls         += 1;
        return cost;
    }

    recordEmbeddingCall(modelId, tokenCount) {
        const prefix  = this._matchPrefix(modelId, this.embeddingPricing);
        const pricing = prefix ? this.embeddingPricing[prefix] : { per_1k: 0.00002 };
        const cost    = (tokenCount / 1000) * pricing.per_1k;
        if (!this.embeddingUsage[modelId])
            this.embeddingUsage[modelId] = { tokens: 0, cost: 0, calls: 0 };
        this.embeddingUsage[modelId].tokens += tokenCount;
        this.embeddingUsage[modelId].cost   += cost;
        this.embeddingUsage[modelId].calls  += 1;
        return cost;
    }

    recordCacheHit(modelId, avoidedInput, avoidedOutput) {
        const prefix  = this._matchPrefix(modelId, this.llmPricing);
        const pricing = prefix ? this.llmPricing[prefix] : { input_per_1k: 0.003, output_per_1k: 0.015 };
        const cost    = (avoidedInput / 1000) * pricing.input_per_1k
                      + (avoidedOutput / 1000) * pricing.output_per_1k;
        if (!this.cacheHits[modelId])
            this.cacheHits[modelId] = { avoided_input: 0, avoided_output: 0, avoided_cost: 0, hits: 0 };
        this.cacheHits[modelId].avoided_input  += avoidedInput;
        this.cacheHits[modelId].avoided_output += avoidedOutput;
        this.cacheHits[modelId].avoided_cost   += cost;
        this.cacheHits[modelId].hits           += 1;
        return cost;
    }

    get totalCost()      { return Object.values(this.llmUsage).reduce((s, u) => s + u.cost, 0)
                                + Object.values(this.embeddingUsage).reduce((s, u) => s + u.cost, 0); }
    get totalAvoidedCost() { return Object.values(this.cacheHits).reduce((s, h) => s + h.avoided_cost, 0); }
    get baselineCost()   { return this.totalCost + this.totalAvoidedCost; }
    get savingsPct()     { const b = this.baselineCost; return b > 0 ? (this.totalAvoidedCost / b) * 100 : 0; }

    getSummary() {
        return {
            total_cost:         this.totalCost,
            baseline_cost:      this.baselineCost,
            total_avoided_cost: this.totalAvoidedCost,
            savings_pct:        Math.round(this.savingsPct * 100) / 100,
            llm_usage:          this.llmUsage,
            embedding_usage:    this.embeddingUsage,
            cache_hits:         this.cacheHits,
        };
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

export async function embedText(text) {
    const modelId    = process.env.BEDROCK_EMBEDDING_MODEL_ID || 'amazon.titan-embed-text-v2:0';
    const dimensions = parseInt(process.env.BEDROCK_EMBEDDING_DIMENSIONS || '1024');

    const start    = performance.now();
    const response = await getBedrock().send(new InvokeModelCommand({
        modelId,
        contentType: 'application/json',
        body: JSON.stringify({ inputText: text, dimensions, normalize: true }),
    }));

    const result    = JSON.parse(new TextDecoder().decode(response.body));
    const latencyMs = performance.now() - start;

    return {
        embedding: new Float32Array(result.embedding),
        metadata:  {
            model_id:    modelId,
            dimensions,
            token_count: result.inputTextTokenCount || 0,
            latency_ms:  Math.round(latencyMs * 100) / 100,
        },
    };
}

// ---------------------------------------------------------------------------
// LLM invocation
// ---------------------------------------------------------------------------

export async function invokeLLM(taskDescription, systemPrompt = null) {
    const modelId  = process.env.BEDROCK_LLM_MODEL_ID || 'amazon.nova-lite-v1:0';
    const messages = systemPrompt
        ? [{ role: 'user', content: [{ text: systemPrompt + '\n\n' + taskDescription }] }]
        : [{ role: 'user', content: [{ text: taskDescription }] }];

    const start    = performance.now();
    const response = await getBedrock().send(new InvokeModelCommand({
        modelId,
        contentType: 'application/json',
        body: JSON.stringify({ messages, inferenceConfig: { temperature: 0.0 } }),
    }));

    const result    = JSON.parse(new TextDecoder().decode(response.body));
    const latencyMs = performance.now() - start;

    return {
        response:      result.output?.message?.content?.[0]?.text || '',
        input_tokens:  result.usage?.inputTokens  || 0,
        output_tokens: result.usage?.outputTokens || 0,
        latency_ms:    Math.round(latencyMs * 100) / 100,
        model_id:      modelId,
    };
}

// ---------------------------------------------------------------------------
// Context store operations
// ---------------------------------------------------------------------------

export const TABLE_NAME = process.env.DYNAMO_TABLE_NAME || null;

export async function storeContext(contextId, workflowId, taskDescription, embedding, response, costMetadata) {
    const redis = await getRedis();

    if (TABLE_NAME) {
        await getDynamo().send(new PutItemCommand({
            TableName: TABLE_NAME,
            Item: {
                context_id:          { S: contextId },
                workflow_id:         { S: workflowId },
                task_description:    { S: taskDescription },
                embedding:           { B: Buffer.from(embedding.buffer) },
                embedding_dim:       { N: String(embedding.length) },
                response:            { S: response },
                model_id:            { S: costMetadata.model_id || '' },
                cost_input_tokens:   { N: String(costMetadata.input_tokens || 0) },
                cost_output_tokens:  { N: String(costMetadata.output_tokens || 0) },
                cost_usd:            { N: String(costMetadata.cost_usd || 0) },
                created_at:          { S: new Date().toISOString() },
                reuse_count:         { N: '0' },
            },
        }));
    }

    if (redis) {
        const pipeline = redis.multi();
        pipeline.set(`embedding:${contextId}`, Buffer.from(embedding.buffer), { EX: 3600 });
        pipeline.set(`context:${contextId}`, JSON.stringify({
            response,
            input_tokens:  costMetadata.input_tokens  || 0,
            output_tokens: costMetadata.output_tokens || 0,
            model_id:      costMetadata.model_id      || '',
            cost_usd:      costMetadata.cost_usd      || 0,
        }), { EX: 3600 });
        pipeline.sAdd(`workflow:${workflowId}`, contextId);
        pipeline.expire(`workflow:${workflowId}`, 7200);
        await pipeline.exec();
    }
}

export async function getAllEmbeddings(workflowId) {
    const redis      = await getRedis();
    if (!redis) return [];
    const contextIds = await redis.sMembers(`workflow:${workflowId}`);
    const results    = [];
    for (const contextId of contextIds) {
        const embBytes = await redis.get(commandOptions({ returnBuffers: true }), `embedding:${contextId}`);
        if (embBytes) {
            results.push({
                contextId,
                embedding: new Float32Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 4),
            });
        }
    }
    return results;
}

export async function getContext(contextId, workflowId) {
    const redis  = await getRedis();
    const cached = redis ? await redis.get(`context:${contextId}`) : null;
    if (cached) return JSON.parse(cached);

    if (TABLE_NAME && workflowId) {
        const result = await getDynamo().send(new GetItemCommand({
            TableName: TABLE_NAME,
            Key: { context_id: { S: contextId }, workflow_id: { S: workflowId } },
        }));
        if (!result.Item) return null;
        return {
            response:      result.Item.response?.S || '',
            model_id:      result.Item.model_id?.S || '',
            input_tokens:  parseInt(result.Item.cost_input_tokens?.N  || '0'),
            output_tokens: parseInt(result.Item.cost_output_tokens?.N || '0'),
            cost_usd:      parseFloat(result.Item.cost_usd?.N || '0'),
        };
    }
    return null;
}

export async function incrementReuseCount(contextId, workflowId) {
    const redis  = await getRedis();
    const cached = redis ? await redis.get(`context:${contextId}`) : null;
    if (redis && cached) {
        const data   = JSON.parse(cached);
        data.reuse_count = (data.reuse_count || 0) + 1;
        await redis.set(`context:${contextId}`, JSON.stringify(data), { KEEPTTL: true });
    }
    if (TABLE_NAME && workflowId) {
        await getDynamo().send(new UpdateItemCommand({
            TableName: TABLE_NAME,
            Key: { context_id: { S: contextId }, workflow_id: { S: workflowId } },
            UpdateExpression: 'ADD reuse_count :inc',
            ExpressionAttributeValues: { ':inc': { N: '1' } },
        }));
    }
}

// ---------------------------------------------------------------------------
// Similarity search
// ---------------------------------------------------------------------------

export function getContextBackend() {
    return process.env.CONTEXT_BACKEND || 'wasm';
}

export function simdCosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold, topK = 5) {
    const results = [];
    for (const { contextId, embedding } of storedEmbeddings) {
        const similarity = wasm.cosine_similarity_f32(
            Array.from(queryEmbedding), Array.from(embedding),
        );
        if (similarity >= threshold) results.push({ contextId, similarity });
    }
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
}

export function jsCosineSimilaritySearch(queryEmbedding, storedEmbeddings, threshold, topK = 5) {
    const results = [];
    const dim     = queryEmbedding.length;
    let queryNormSq = 0;
    for (let i = 0; i < dim; i++) queryNormSq += queryEmbedding[i] * queryEmbedding[i];
    const queryNorm = Math.sqrt(queryNormSq);
    if (queryNorm === 0) return results;

    for (const { contextId, embedding } of storedEmbeddings) {
        let dot = 0, embNormSq = 0;
        for (let i = 0; i < dim; i++) {
            dot      += queryEmbedding[i] * embedding[i];
            embNormSq += embedding[i] * embedding[i];
        }
        const embNorm = Math.sqrt(embNormSq);
        if (embNorm === 0) continue;
        const similarity = dot / (queryNorm * embNorm);
        if (similarity >= threshold) results.push({ contextId, similarity });
    }
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
}

export function cosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold, topK = 5) {
    if (getContextBackend() === 'redis') {
        return jsCosineSimilaritySearch(queryEmbedding, storedEmbeddings, threshold, topK);
    }
    return simdCosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold, topK);
}

// ---------------------------------------------------------------------------
// Base64 helpers
// ---------------------------------------------------------------------------

export function ndArrayToB64(float32Array) {
    return Buffer.from(float32Array.buffer).toString('base64');
}

export function b64ToNdArray(b64String) {
    const buf = Buffer.from(b64String, 'base64');
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}