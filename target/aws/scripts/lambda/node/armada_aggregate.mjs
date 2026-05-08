/**
 * armada_aggregate — aggregate Map state results (Node.js / WASM).
 *
 * Receives the full array of per-task results from armada_executor,
 * computes cost/reuse/latency summary, writes result files to S3,
 * and returns the final workflow output.
 *
 * Input (from Step Functions state):
 *   {
 *     workflow_id, scaling, world_size, results_s3_dir, experiment_name,
 *     task_results: [ <armada_executor result>, ... ],
 *     prepare_cost: { ... }   // from armada_init
 *   }
 *
 * CMD: armada_aggregate.handler
 */

import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { CostTracker } from './shared.mjs';

const s3 = new S3Client({});

function percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, idx)];
}

async function writeResultsToS3(aggregate, taskResults, bucket, resultsS3Dir, experimentName) {
    const dirPrefix = resultsS3Dir ? resultsS3Dir.replace(/\/?$/, '/') : '';

    const scaling   = aggregate.scaling   || '';
    const worldSize = aggregate.world_size || taskResults.length;
    const costSummary = aggregate.cost_summary || {};
    const reuse     = aggregate.reuse_stats   || {};
    const latency   = aggregate.latency       || {};

    // stopwatch.csv
    const rows = taskResults.map(r => ({
        experiment_name:   experimentName,
        workflow_id:       aggregate.workflow_id || '',
        rank:              r.rank ?? 0,
        world_size:        worldSize,
        scaling,
        platform:          'lambda',
        task_description:  r.task_description || '',
        source:            r.source || '',
        search_latency_ms: r.search_latency_ms ?? 0,
        llm_latency_ms:    r.llm_latency_ms    ?? 0,
        total_latency_ms:  r.total_latency_ms  ?? 0,
        cost_usd:          r.cost_usd          ?? 0,
        avoided_cost_usd:  r.avoided_cost_usd  ?? 0,
        similarity:        r.similarity        ?? 0,
        backend:           r.backend           || '',
    }));

    const headers = rows.length > 0 ? Object.keys(rows[0]) : [];
    const csvLines = [
        headers.join(','),
        ...rows.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(',')),
    ];

    await s3.send(new PutObjectCommand({
        Bucket: bucket,
        Key: `${dirPrefix}${experimentName}_stopwatch.csv`,
        Body: csvLines.join('\n'),
        ContentType: 'text/csv',
    }));

    // metrics.json
    const metrics = {
        experiment_name:  experimentName,
        workflow_id:      aggregate.workflow_id || '',
        platform:         'lambda',
        scaling,
        world_size:       worldSize,
        task_count:       taskResults.length,
        cache_hits:       reuse.cache_hits  ?? aggregate.cache_hits  ?? 0,
        llm_calls:        reuse.llm_calls   ?? aggregate.llm_calls   ?? 0,
        reuse_rate:       reuse.reuse_rate  ?? aggregate.reuse_rate  ?? 0,
        // Compute from task results — costSummary.baseline_cost is always 0
        // because Step Functions doesn't propagate avoided costs through Map state
        total_cost:       taskResults.reduce((s, r) => s + (r.cost_usd ?? 0), 0),
        baseline_cost:    taskResults.reduce((s, r) => s + (r.cost_usd ?? 0) + (r.avoided_cost_usd ?? 0), 0),
        savings_pct:      (() => {
            const base = taskResults.reduce((s, r) => s + (r.cost_usd ?? 0) + (r.avoided_cost_usd ?? 0), 0);
            const avoided = taskResults.reduce((s, r) => s + (r.avoided_cost_usd ?? 0), 0);
            return base > 0 ? (avoided / base * 100) : 0;
        })(),
        total_ms:         latency.total_ms  ?? 0,
        avg_latency_ms:   latency.avg_ms    ?? 0,
        p50_latency_ms:   latency.p50_ms    ?? 0,
        p95_latency_ms:   latency.p95_ms    ?? 0,
        p99_latency_ms:   latency.p99_ms    ?? 0,
    };

    await s3.send(new PutObjectCommand({
        Bucket: bucket,
        Key: `${dirPrefix}${experimentName}_metrics.json`,
        Body: JSON.stringify(metrics, null, 2),
        ContentType: 'application/json',
    }));

    // summary.csv (single-row, compatible with pipeline aggregator)
    const summaryHeaders = Object.keys(metrics);
    const summaryLines = [
        summaryHeaders.join(','),
        summaryHeaders.map(h => JSON.stringify(metrics[h] ?? '')).join(','),
    ];

    await s3.send(new PutObjectCommand({
        Bucket: bucket,
        Key: `${dirPrefix}${experimentName}_summary.csv`,
        Body: summaryLines.join('\n'),
        ContentType: 'text/csv',
    }));

    console.log(
        `Results written to s3://${bucket}/${dirPrefix}${experimentName}_* ` +
        `(tasks=${taskResults.length} reuse=${metrics.reuse_rate.toFixed(1)}% cost=$${metrics.total_cost.toFixed(4)})`
    );
}

export async function handler(event) {
    const workflowId    = event.workflow_id    || '';
    const scaling       = (event.scaling       || 'weak').toLowerCase();
    const worldSize     = parseInt(event.world_size || 1);
    const resultsS3Dir  = event.results_s3_dir  || '';
    const experimentName = event.experiment_name || `lambda_${scaling}_ws${worldSize}`;
    const taskResults   = event.task_results   || [];
    const prepareCost   = event.prepare_cost   || {};

    const aggregated = new CostTracker();
    let cacheHits = 0;
    let llmCalls  = 0;

    for (const result of taskResults) {
        if (result.source === 'cache') {
            cacheHits++;
            const modelId = result.model_id || process.env.BEDROCK_LLM_MODEL_ID || 'unknown';
            if (result.avoided_input_tokens || result.avoided_output_tokens) {
                aggregated.recordCacheHit(modelId, result.avoided_input_tokens || 0, result.avoided_output_tokens || 0);
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

    // Latency stats
    const latencies = taskResults.map(r => r.total_latency_ms || 0).sort((a, b) => a - b);
    const totalMs   = latencies.reduce((s, v) => s + v, 0);
    const avgMs     = latencies.length > 0 ? totalMs / latencies.length : 0;
    const latency = {
        total_ms: Math.round(totalMs * 100) / 100,
        avg_ms:   Math.round(avgMs   * 100) / 100,
        p50_ms:   percentile(latencies, 50),
        p95_ms:   percentile(latencies, 95),
        p99_ms:   percentile(latencies, 99),
    };

    const reuse_rate = taskResults.length > 0
        ? Math.round((cacheHits / taskResults.length) * 10000) / 100
        : 0;

    const aggregate = {
        workflow_id:     workflowId,
        scaling,
        world_size:      worldSize,
        results_s3_dir:  resultsS3Dir,
        experiment_name: experimentName,
        total_tasks:     taskResults.length,
        cache_hits:      cacheHits,
        llm_calls:       llmCalls,
        reuse_rate,
        reuse_stats: {
            cache_hits: cacheHits,
            llm_calls:  llmCalls,
            reuse_rate,
        },
        latency,
        cost_summary: aggregated.getSummary(),
    };

    const resultsBucket = (process.env.RESULTS_BUCKET || '').trim();
    if (resultsBucket && resultsS3Dir) {
        await writeResultsToS3(aggregate, taskResults, resultsBucket, resultsS3Dir, experimentName);
    } else {
        console.log(`RESULTS_BUCKET or results_s3_dir not set — skipping S3 write (bucket=${resultsBucket} dir=${resultsS3Dir})`);
    }

    return aggregate;
}