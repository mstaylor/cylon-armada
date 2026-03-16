/**
 * Path B — Node.js Lambda handler for context reuse with WASM SIMD128.
 *
 * Uses Cylon's cylon-wasm module for SIMD-accelerated cosine similarity
 * via WASM SIMD128 instructions. Follows the same routing logic as the
 * Python handler but through the Node.js runtime.
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
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createClient } from 'redis';
import { DynamoDBClient, PutItemCommand, GetItemCommand, QueryCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
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
    const wasmBytes = readFileSync(wasmPath);
    await bindings.default({ module_or_path: wasmBytes });

    if (bindings.init) bindings.init();

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

    // Fallback to DynamoDB
    const dynamo = getDynamo();
    const result = await dynamo.send(new GetItemCommand({
        TableName: TABLE_NAME,
        Key: {
            context_id: { S: contextId },
        },
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
// Route task — core context reuse logic
// ---------------------------------------------------------------------------

async function routeTask(wasm, params) {
    const {
        task_description: taskDescription,
        embedding_b64: embeddingB64,
        embedding_metadata: embeddingMetadata,
        workflow_id: workflowId,
        config = {},
    } = params;

    const threshold = parseFloat(config.similarity_threshold || process.env.SIMILARITY_THRESHOLD || '0.85');
    const start = performance.now();

    // Decode pre-computed embedding from base64
    const embBuffer = Buffer.from(embeddingB64, 'base64');
    const queryEmbedding = new Float32Array(embBuffer.buffer, embBuffer.byteOffset, embBuffer.byteLength / 4);

    // Get stored embeddings for this workflow
    const storedEmbeddings = await getAllEmbeddings(workflowId);

    // SIMD similarity search
    const matches = simdCosineSimilaritySearch(wasm, queryEmbedding, storedEmbeddings, threshold);

    if (matches.length > 0) {
        // Cache hit — reuse existing context
        const bestMatch = matches[0];
        const context = await getContext(bestMatch.contextId);
        await incrementReuseCount(bestMatch.contextId);

        return {
            response: context.response,
            source: 'cache',
            similarity: bestMatch.similarity,
            context_id: bestMatch.contextId,
            cost_usd: 0,
            total_latency_ms: Math.round((performance.now() - start) * 100) / 100,
            avoided_input_tokens: context.input_tokens || 0,
            avoided_output_tokens: context.output_tokens || 0,
        };
    }

    // Cache miss — invoke LLM
    const llmResult = await invokeLLM(taskDescription);

    // Store new context
    const contextId = crypto.randomUUID();
    await storeContext(contextId, workflowId, taskDescription, queryEmbedding, llmResult.response, {
        model_id: llmResult.model_id,
        input_tokens: llmResult.input_tokens,
        output_tokens: llmResult.output_tokens,
        cost_usd: 0, // Cost computed by aggregator
    });

    return {
        response: llmResult.response,
        source: 'llm',
        similarity: matches.length > 0 ? matches[0].similarity : 0,
        context_id: contextId,
        input_tokens: llmResult.input_tokens,
        output_tokens: llmResult.output_tokens,
        cost_usd: 0, // Cost computed by aggregator
        total_latency_ms: Math.round((performance.now() - start) * 100) / 100,
        llm_latency_ms: llmResult.latency_ms,
        model_id: llmResult.model_id,
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

        switch (action) {
            case 'route_task':
                return await routeTask(wasm, event.action_payload || event);

            case 'embed_and_search': {
                const { text, workflow_id, threshold = 0.85, top_k = 5 } = event.action_payload || event;
                const { embedding } = await embedText(text);
                const stored = await getAllEmbeddings(workflow_id);
                const matches = simdCosineSimilaritySearch(wasm, embedding, stored, threshold, top_k);
                return { matches, embedding_length: embedding.length };
            }

            case 'simd_benchmark': {
                // Pure SIMD benchmark — no I/O, measures WASM SIMD128 throughput
                const { dim = 1024, n = 1000, iterations = 100 } = event.action_payload || event;
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