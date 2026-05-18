/**
 * armada_init — Step Functions init Lambda (Node.js).
 *
 * Mirrors armada_init.py:
 *   1. Receives {workflow_id, tasks, config, world_size, scaling}
 *   2. Embeds all task descriptions via Bedrock
 *   3. Stores task descriptions + embeddings in Redis (offload from SFN state)
 *   4. Returns minimal body array — one entry per task, no embedding bytes inline
 *
 * CMD: armada_init.handler
 */

import { randomUUID } from 'crypto';
import {
    embedText,
    ndArrayToB64,
    CostTracker,
    getRedis,
} from './shared.mjs';

export async function handler(event) {
    const workflowId   = event.workflow_id || randomUUID();
    const tasks        = event.tasks || [];
    const config       = event.config || {};
    const scaling      = (event.scaling || 'weak').toLowerCase();
    const worldSize    = parseInt(event.world_size || tasks.length || 1);
    const contextBackend = event.context_backend || process.env.CONTEXT_BACKEND || 'redis';

    const rawResultsDir     = event.results_s3_dir  || 'results/lambda-nodejs/{scaling}/';
    const rawExperimentName = event.experiment_name || 'lambda_nodejs_{scaling}_ws{world_size}';
    const resultsS3Dir      = rawResultsDir.replace('{scaling}', scaling).replace('{world_size}', worldSize);
    const experimentName    = rawExperimentName.replace('{scaling}', scaling).replace('{world_size}', worldSize);

    if (tasks.length === 0) {
        throw new Error('tasks array is required');
    }

    const redis = await getRedis();
    const costTracker = new CostTracker();
    const start = performance.now();
    const minimalBody = [];
    const TTL = 7200; // 2 hours

    for (let i = 0; i < tasks.length; i++) {
        const taskDesc = tasks[i];
        const { embedding, metadata } = await embedText(taskDesc);
        costTracker.recordEmbeddingCall(metadata.model_id, metadata.token_count);

        const embKey = `embedding:${workflowId}:${i}`;

        if (redis) {
            // Store task description and embedding in Redis to keep SFN payload tiny
            const pipeline = redis.multi();
            pipeline.setEx(`task:${workflowId}:${i}`, TTL, taskDesc);
            pipeline.setEx(embKey, TTL, ndArrayToB64(embedding));
            await pipeline.exec();

            minimalBody.push({
                rank:               i,
                embedding_key:      embKey,
                embedding_metadata: metadata,
            });
        } else {
            // Fallback: inline embedding (small world sizes only)
            minimalBody.push({
                rank:               i,
                embedding_b64:      ndArrayToB64(embedding),
                embedding_metadata: metadata,
            });
        }
    }

    const prepareLatencyMs = Math.round((performance.now() - start) * 100) / 100;

    console.info(`armada_init done: ${tasks.length} tasks, redis=${!!redis}`);

    return {
        body:               minimalBody,
        workflow_id:        workflowId,
        scaling,
        world_size:         worldSize,
        context_backend:    contextBackend,
        fmi_channel_type:   event.fmi_channel_type || 'direct',
        results_s3_dir:     resultsS3Dir,
        experiment_name:    experimentName,
        config,
        prepare_cost:       costTracker.getSummary(),
        prepare_latency_ms: prepareLatencyMs,
    };
}