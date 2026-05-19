/**
 * armada_executor — per-task Lambda invoked by the Map state (Node.js).
 *
 * Mirrors armada_executor.py:
 *   - Fetches task description + embedding from Redis (offloaded by armada_init)
 *   - Runs context-reuse routing (similarity search → reuse or LLM call)
 *   - Stores full result in Redis, returns only {rank} to Step Functions
 *
 * CMD: armada_executor.handler
 */

import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { StopWatch } from './stopwatch.mjs';
import {
    initWasm,
    CostTracker,
    b64ToNdArray,
    ndArrayToB64,
    getRedis,
    invokeLLM,
    jsCosineSimilaritySearch,
} from './shared.mjs';

// ---------------------------------------------------------------------------
// cylon-node FMI loader
// ---------------------------------------------------------------------------

let _cylonNode = null;

function loadCylonNode() {
    if (_cylonNode !== null) return _cylonNode;
    const binaryPath = process.env.CYLON_NODE_PATH
        ? `${process.env.CYLON_NODE_PATH}/cylon-node.linux-x64-gnu.node`
        : '/app/cylon-node/cylon-node.linux-x64-gnu.node';
    try {
        const _require = createRequire(import.meta.url);
        _cylonNode = _require(binaryPath);
        console.info('cylon-node loaded via binary');
    } catch (e) {
        console.warn(`cylon-node unavailable: ${e.message}`);
        _cylonNode = null;
    }
    return _cylonNode;
}

function createFmiCommunicator(event) {
    const cylon = loadCylonNode();
    if (!cylon) return null;
    const worldSize = parseInt(event.world_size || 1);
    if (worldSize <= 1) return null;
    const workflowId = event.workflow_id || '';
    // Use experiment_name (unique per run) as commName so the Redis INCR rank
    // counter resets each run. workflow_id is shared across runs 1-4 for context
    // reuse — using it as commName causes INCR to accumulate so run 2+ workers
    // get ranks >= world_size and hang waiting at the rendezvous forever.
    const commNameBase = event.experiment_name || workflowId;
    const commName = commNameBase ? `cylon_armada_${commNameBase}` : 'cylon_armada';
    const maxTimeout = parseInt(process.env.FMI_MAX_TIMEOUT || 300000);
    try {
        const comm = cylon.Communicator.createFmi({
            rank:        parseInt(event.rank || 0),
            worldSize,
            host:        process.env.RENDEZVOUS_HOST || '',
            port:        parseInt(process.env.RENDEZVOUS_PORT || 10000),
            maxTimeout,
            commName,
            nonblocking: true,
            redisHost:   process.env.REDIS_HOST || '',
            redisPort:   parseInt(process.env.REDIS_PORT || 6379),
        });
        console.info(`FMI communicator ready: rank=${event.rank} worldSize=${worldSize}`);
        return comm;
    } catch (e) {
        console.warn(`FMI communicator failed: ${e.message}`);
        return null;
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handler(event) {
    const rank           = parseInt(event.rank ?? 0);
    const worldSize      = parseInt(event.world_size ?? 1);
    const workflowId     = event.workflow_id || '';
    const experimentName = event.experiment_name || workflowId;
    const config         = event.config || {};
    const embeddingKey   = event.embedding_key;
    const contextBackend = event.context_backend || process.env.CONTEXT_BACKEND || 'redis';

    const threshold = parseFloat(
        config.similarity_threshold || process.env.SIMILARITY_THRESHOLD || '0.85'
    );

    console.info(`armada_executor rank=${rank}/${worldSize} workflow=${workflowId} backend=${contextBackend}`);

    StopWatch.start('route_total');

    const redis = await getRedis();
    if (!redis) throw new Error('Redis unavailable — required for armada_executor');

    // Fetch task description from Redis (stored by armada_init)
    const taskDescription = await redis.get(`task:${workflowId}:${rank}`) || '';

    // Fetch pre-computed embedding from Redis
    let queryEmbedding;
    if (embeddingKey) {
        const raw = await redis.get(embeddingKey);
        if (!raw) throw new Error(`Embedding key ${embeddingKey} not found in Redis`);
        queryEmbedding = b64ToNdArray(raw);
    } else if (event.embedding_b64) {
        queryEmbedding = b64ToNdArray(event.embedding_b64);
    } else {
        throw new Error('No embedding_key or embedding_b64 in event');
    }

    // FMI communicator (cylon backend, world_size > 1)
    let fmi = null;
    if (contextBackend === 'cylon' && worldSize > 1) {
        fmi = createFmiCommunicator(event);
    }

    const wasm = contextBackend === 'redis' ? await initWasm().catch(() => null) : null;
    const costTracker = new CostTracker();

    if (event.embedding_metadata) {
        costTracker.recordEmbeddingCall(
            event.embedding_metadata.model_id,
            event.embedding_metadata.token_count
        );
    }

    // Fetch stored context embeddings for similarity search
    StopWatch.start('search_latency');
    const contextIds = await redis.sMembers(`workflow:${workflowId}`) || [];
    let storedEmbeddings = [];
    for (const contextId of contextIds) {
        const raw = await redis.get(`embedding:${contextId}`);
        if (raw) storedEmbeddings.push({ contextId, embedding: b64ToNdArray(raw) });
    }

    let matches = [];
    if (fmi && contextBackend === 'cylon') {
        // Broadcast stored embeddings from rank 0 to all workers via FMI
        const payload = rank === 0
            ? storedEmbeddings.map(e => ({ id: e.contextId, emb: ndArrayToB64(e.embedding) }))
            : [];
        const buf = Buffer.from(JSON.stringify(payload));
        console.info(`FMI broadcast: rank ${rank} sending ${payload.length} contexts`);
        try {
            const received = fmi.broadcast(buf, 0);
            const allContexts = JSON.parse(received.toString());
            storedEmbeddings = allContexts.map(c => ({
                contextId: c.id,
                embedding: b64ToNdArray(c.emb),
            }));
        } catch (e) {
            console.warn(`FMI broadcast failed, falling back to Redis: ${e.message}`);
        }
        matches = jsCosineSimilaritySearch(queryEmbedding, storedEmbeddings, threshold);
    } else {
        matches = jsCosineSimilaritySearch(queryEmbedding, storedEmbeddings, threshold);
    }
    StopWatch.stop('search_latency');

    let result;

    if (matches.length > 0) {
        // Cache hit
        const best = matches[0];
        const cached = await redis.get(`context:${best.contextId}`);
        const ctx = cached ? JSON.parse(cached) : null;

        if (ctx) {
            await redis.set(
                `context:${best.contextId}`,
                JSON.stringify({ ...ctx, reuse_count: (ctx.reuse_count || 0) + 1 }),
                { KEEPTTL: true }
            );
        }

        const avoidedInput  = ctx?.input_tokens  || 0;
        const avoidedOutput = ctx?.output_tokens || 0;
        const llmModelId    = ctx?.model_id || config.llm_model_id || 'unknown';
        const avoidedCost   = costTracker.recordCacheHit(llmModelId, avoidedInput, avoidedOutput);
        StopWatch.stop('route_total');

        result = {
            source:                'cache',
            response:              ctx?.response || '',
            context_id:            best.contextId,
            similarity:            best.similarity,
            cost_usd:              0,
            avoided_cost_usd:      avoidedCost,
            avoided_input_tokens:  avoidedInput,
            avoided_output_tokens: avoidedOutput,
            search_latency_ms:     StopWatch.getMs('search_latency'),
            total_latency_ms:      StopWatch.getMs('route_total'),
            task_description:      taskDescription,
            rank,
            workflow_id:           workflowId,
        };
    } else {
        // Cache miss — call LLM
        StopWatch.start('llm_latency');
        const llmResult = await invokeLLM(taskDescription);
        StopWatch.stop('llm_latency');
        const callCost = costTracker.recordLlmCall(
            llmResult.model_id, llmResult.input_tokens, llmResult.output_tokens
        );

        // Store new context
        StopWatch.start('store_latency');
        const contextId = crypto.randomUUID();
        const pipeline = redis.multi();
        pipeline.set(`embedding:${contextId}`, ndArrayToB64(queryEmbedding), { EX: 3600 });
        pipeline.set(`context:${contextId}`, JSON.stringify({
            response:      llmResult.response,
            input_tokens:  llmResult.input_tokens,
            output_tokens: llmResult.output_tokens,
            model_id:      llmResult.model_id,
            cost_usd:      callCost,
            reuse_count:   0,
        }), { EX: 3600 });
        pipeline.sAdd(`workflow:${workflowId}`, contextId);
        pipeline.expire(`workflow:${workflowId}`, 7200);
        await pipeline.exec();
        StopWatch.stop('store_latency');
        StopWatch.stop('route_total');

        result = {
            source:            'llm',
            response:          llmResult.response,
            context_id:        contextId,
            similarity:        0,
            input_tokens:      llmResult.input_tokens,
            output_tokens:     llmResult.output_tokens,
            cost_usd:          callCost,
            avoided_cost_usd:  0,
            llm_latency_ms:    llmResult.latency_ms,
            search_latency_ms: StopWatch.getMs('search_latency'),
            store_latency_ms:  StopWatch.getMs('store_latency'),
            total_latency_ms:  StopWatch.getMs('route_total'),
            model_id:          llmResult.model_id,
            task_description:  taskDescription,
            rank,
            workflow_id:       workflowId,
        };
    }

    // Store full result in Redis for armada_aggregate
    await redis.setEx(`result:${experimentName}:${rank}`, 3600, JSON.stringify(result));

    // Return only rank to Step Functions (keeps SFN state tiny)
    return { rank };
}