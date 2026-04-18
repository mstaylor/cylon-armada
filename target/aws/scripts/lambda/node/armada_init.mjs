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
    const workflowId   = event.workflow_id || randomUUID();
    const tasks        = event.tasks;
    const config       = event.config || {};
    const scaling      = (event.scaling || 'weak').toLowerCase();
    const worldSize    = event.world_size || tasks?.length || 1;

    // File naming — support {scaling} and {world_size} substitutions
    const rawResultsDir     = event.results_s3_dir    || 'results/lambda/{scaling}/';
    const rawExperimentName = event.experiment_name   || 'lambda_{scaling}_ws{world_size}';
    const resultsS3Dir      = rawResultsDir.replace('{scaling}', scaling).replace('{world_size}', worldSize);
    const experimentName    = rawExperimentName.replace('{scaling}', scaling).replace('{world_size}', worldSize);

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
            world_size:         worldSize,
            config,
        });
    }

    const prepareLatencyMs = Math.round((performance.now() - start) * 100) / 100;

    return {
        statusCode:         200,
        body:               taskPayloads,
        workflow_id:        workflowId,
        scaling,
        world_size:         worldSize,
        results_s3_dir:     resultsS3Dir,
        experiment_name:    experimentName,
        prepare_cost:       costTracker.getSummary(),
        prepare_latency_ms: prepareLatencyMs,
    };
}