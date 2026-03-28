/**
 * armada_init — Step Functions init Lambda (Node.js / WASM runtime).
 *
 * Mirrors armada_init.py and the cylon_init pattern:
 *   1. Receives {workflow_id, tasks, config}
 *   2. Embeds all task descriptions (Bedrock)
 *   3. Builds one payload per task for the Map state
 *   4. Returns {statusCode: 200, body: [...]} — Map state iterates $.body
 *
 * CMD: armada_init.handler
 */

import { randomUUID } from 'crypto';
import {
    embedText,
    ndArrayToB64,
    CostTracker,
} from './shared.mjs';

export async function handler(event) {
    const workflowId  = event.workflow_id || randomUUID();
    const tasks       = event.tasks;
    const config      = event.config || {};

    if (!tasks || tasks.length === 0) {
        return { statusCode: 400, body: JSON.stringify({ error: 'tasks array is required' }) };
    }

    const costTracker  = new CostTracker();
    const start        = performance.now();
    const taskPayloads = [];

    for (let i = 0; i < tasks.length; i++) {
        const { embedding, metadata } = await embedText(tasks[i]);
        costTracker.recordEmbeddingCall(metadata.model_id, metadata.token_count);

        taskPayloads.push({
            task_description:   tasks[i],
            embedding_b64:      ndArrayToB64(embedding),
            embedding_metadata: metadata,
            workflow_id:        workflowId,
            rank:               i,
            world_size:         tasks.length,
            config,
        });
    }

    const prepareLatencyMs = Math.round((performance.now() - start) * 100) / 100;

    return {
        statusCode:         200,
        body:               taskPayloads,        // Map state iterates this
        workflow_id:        workflowId,
        prepare_cost:       costTracker.getSummary(),
        prepare_latency_ms: prepareLatencyMs,
    };
}