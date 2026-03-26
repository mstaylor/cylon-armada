#!/usr/bin/env node
/**
 * Node.js experiment runner — Path B smoke test and benchmarking.
 *
 * Runs context reuse experiments through the Node.js handler locally,
 * producing cloudmesh-compatible StopWatch CSV + summary CSV output.
 *
 * Usage:
 *   # SIMD benchmark (no AWS dependencies)
 *   node run_experiment.mjs --action simd_benchmark --dim 256 --n 1000
 *
 *   # Full route_task smoke test (requires Redis + Bedrock)
 *   REDIS_HOST=10.211.55.2 node run_experiment.mjs \
 *       --action route_task \
 *       --tasks-file ../../../../experiments/scenarios/hydrology.json \
 *       --tasks 4 \
 *       --output ../../../../experiments/results/smoke_nodejs
 *
 *   # With S3 upload
 *   node run_experiment.mjs --action simd_benchmark --s3-bucket my-bucket
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { StopWatch } from './stopwatch.mjs';
import { handler } from './context_handler.mjs';

function parseArgs() {
    const args = {
        action: 'simd_benchmark',
        dim: 256,
        n: 1000,
        iterations: 100,
        tasksFile: null,
        tasks: 4,
        threshold: 0.85,
        dimensions: 256,
        output: './results',
        s3Bucket: null,
        s3Prefix: 'experiments',
        name: null,
    };

    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--action': args.action = argv[++i]; break;
            case '--dim': args.dim = parseInt(argv[++i]); break;
            case '--n': args.n = parseInt(argv[++i]); break;
            case '--iterations': args.iterations = parseInt(argv[++i]); break;
            case '--tasks-file': args.tasksFile = argv[++i]; break;
            case '--tasks': args.tasks = parseInt(argv[++i]); break;
            case '--threshold': args.threshold = parseFloat(argv[++i]); break;
            case '--dimensions': args.dimensions = parseInt(argv[++i]); break;
            case '--output': args.output = argv[++i]; break;
            case '--s3-bucket': args.s3Bucket = argv[++i]; break;
            case '--s3-prefix': args.s3Prefix = argv[++i]; break;
            case '--name': args.name = argv[++i]; break;
        }
    }

    if (!args.name) {
        args.name = `nodejs_${args.action}_d${args.dim}_n${args.n}`;
    }

    return args;
}

async function runSIMDBenchmark(args) {
    const experimentName = args.name;
    StopWatch.clear();

    console.log(`Running SIMD benchmark: dim=${args.dim}, n=${args.n}, iterations=${args.iterations}`);

    StopWatch.start(`${experimentName}_total`);
    const result = await handler({
        action: 'simd_benchmark',
        action_payload: {
            dim: args.dim,
            n: args.n,
            iterations: args.iterations,
        },
    }, {});
    StopWatch.stop(`${experimentName}_total`);

    StopWatch.record('dim', args.dim);
    StopWatch.record('n_embeddings', args.n);
    StopWatch.record('iterations', args.iterations);
    StopWatch.record('total_ms', result.total_ms);
    StopWatch.record('avg_search_ms', result.avg_search_ms);
    StopWatch.record('comparisons_per_sec', result.comparisons_per_sec);

    return result;
}

async function runRouteTaskExperiment(args) {
    if (!args.tasksFile) {
        console.error('--tasks-file required for route_task experiments');
        process.exit(1);
    }

    const raw = JSON.parse(readFileSync(args.tasksFile, 'utf-8'));
    const allTasks = raw.tasks || raw;
    const tasks = allTasks.slice(0, args.tasks);

    const experimentName = args.name || `nodejs_route_t${args.tasks}_th${args.threshold}_d${args.dimensions}`;
    StopWatch.clear();

    console.log(`Running route_task: tasks=${tasks.length}, threshold=${args.threshold}, dim=${args.dimensions}`);

    // Step 1: Prepare tasks (embed all)
    StopWatch.start(`${experimentName}_total`);
    StopWatch.start(`${experimentName}_prepare`);

    const prepareResult = await handler({
        action: 'prepare_tasks',
        action_payload: {
            workflow_id: `exp-node-${Date.now()}`,
            tasks,
            config: {
                similarity_threshold: String(args.threshold),
                embedding_dimensions: String(args.dimensions),
            },
        },
    }, {});

    StopWatch.stop(`${experimentName}_prepare`);

    if (prepareResult.statusCode) {
        console.error('Prepare failed:', prepareResult.body);
        process.exit(1);
    }

    // Step 2: Route each task
    const taskResults = [];
    for (const payload of prepareResult.task_payloads) {
        StopWatch.start(`${experimentName}_route_${payload.rank}`);

        const routeResult = await handler({
            action: 'route_task',
            action_payload: payload,
        }, {});

        StopWatch.stop(`${experimentName}_route_${payload.rank}`);
        taskResults.push(routeResult);
    }

    // Step 3: Aggregate
    StopWatch.start(`${experimentName}_aggregate`);
    const aggregateResult = await handler({
        action: 'aggregate_results',
        action_payload: {
            workflow_id: prepareResult.workflow_id,
            task_results: taskResults,
            prepare_cost: prepareResult.prepare_cost,
        },
    }, {});
    StopWatch.stop(`${experimentName}_aggregate`);
    StopWatch.stop(`${experimentName}_total`);

    // Record metrics
    StopWatch.record('total_tasks', aggregateResult.total_tasks);
    StopWatch.record('cache_hits', aggregateResult.cache_hits);
    StopWatch.record('llm_calls', aggregateResult.llm_calls);
    StopWatch.record('reuse_rate', aggregateResult.reuse_rate);
    StopWatch.record('total_cost', aggregateResult.cost_summary?.total_cost || 0);
    StopWatch.record('savings_pct', aggregateResult.cost_summary?.savings_pct || 0);
    StopWatch.record('task_count', tasks.length);
    StopWatch.record('similarity_threshold', args.threshold);
    StopWatch.record('embedding_dimensions', args.dimensions);
    StopWatch.record('backend', 'WASM_SIMD128');

    return aggregateResult;
}

async function main() {
    const args = parseArgs();
    let result;

    if (args.action === 'simd_benchmark') {
        result = await runSIMDBenchmark(args);
    } else if (args.action === 'route_task') {
        result = await runRouteTaskExperiment(args);
    } else {
        console.error(`Unknown action: ${args.action}`);
        process.exit(1);
    }

    // Save results
    const saved = await StopWatch.save({
        name: args.name,
        outputDir: args.output,
        tag: args.name,
        s3Bucket: args.s3Bucket,
        s3Prefix: args.s3Prefix,
    });

    console.log('\nResults:');
    console.log(JSON.stringify(result, null, 2));
    console.log('\nFiles saved:', saved);
}

main().catch(err => {
    console.error('Experiment failed:', err);
    process.exit(1);
});