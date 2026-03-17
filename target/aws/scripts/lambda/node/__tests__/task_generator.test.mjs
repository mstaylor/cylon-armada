/**
 * Tests for Node.js task generator.
 */

import { describe, test, expect } from '@jest/globals';
import { generateTasks } from '../task_generator.mjs';

function sampleData(n = 20) {
    const predictions = Array.from({ length: n }, () => Math.random() * 1.5);
    const trueRedshifts = Array.from({ length: n }, () => Math.random() * 1.5);
    const magnitudes = Array.from({ length: n }, () =>
        Array.from({ length: 5 }, () => 17 + Math.random() * 8)
    );
    return { predictions, trueRedshifts, magnitudes };
}

describe('generateTasks', () => {
    test('generates non-empty task list', () => {
        const data = sampleData();
        const tasks = generateTasks(data);
        expect(tasks.length).toBeGreaterThan(0);
        expect(tasks.every(t => typeof t === 'string')).toBe(true);
    });

    test('maxTasks limits per-sample tasks', () => {
        const data = sampleData(50);
        const tasks = generateTasks({ ...data, maxTasks: 5 });
        // 5 per-sample + no batch tasks (no metrics)
        expect(tasks.length).toBeLessThanOrEqual(10);
    });

    test('seed produces reproducible results', () => {
        const data = sampleData();
        const tasks1 = generateTasks({ ...data, maxTasks: 8, seed: 42 });
        const tasks2 = generateTasks({ ...data, maxTasks: 8, seed: 42 });
        expect(tasks1).toEqual(tasks2);
    });

    test('different seeds produce different results', () => {
        const data = sampleData();
        const tasks1 = generateTasks({ ...data, maxTasks: 8, seed: 42 });
        const tasks2 = generateTasks({ ...data, maxTasks: 8, seed: 99 });
        expect(tasks1).not.toEqual(tasks2);
    });

    test('includes batch tasks when metrics provided', () => {
        const data = sampleData();
        const tasks = generateTasks({
            ...data,
            metrics: {
                total_time_ms: 2500,
                num_samples: 20,
                batch_size: 512,
                throughput_bps: 1.3e9,
            },
        });
        const batchTasks = tasks.filter(t => t.includes('batch of') || t.includes('serverless inference'));
        expect(batchTasks.length).toBeGreaterThan(0);
    });

    test('no batch tasks without metrics', () => {
        const data = sampleData();
        const tasks = generateTasks(data);
        const batchTasks = tasks.filter(t => t.includes('batch of') || t.includes('serverless inference'));
        expect(batchTasks.length).toBe(0);
    });

    test('includes outlier analysis for large residuals', () => {
        const data = sampleData(20);
        // Make one prediction a clear outlier
        data.predictions[0] = 5.0;
        data.trueRedshifts[0] = 0.1;

        const tasks = generateTasks(data);
        const outlierTasks = tasks.filter(t => t.includes('prediction error') || t.includes('residual'));
        expect(outlierTasks.length).toBeGreaterThan(0);
    });

    test('custom templates override defaults', () => {
        const data = sampleData(4);
        const tasks = generateTasks({
            ...data,
            templates: {
                redshift_analysis: 'CUSTOM: z={z_pred}',
                color_classification: 'CUSTOM: colors={color_str}',
                outlier_analysis: 'CUSTOM: outlier z={z_pred}',
            },
        });
        expect(tasks.some(t => t.startsWith('CUSTOM:'))).toBe(true);
    });

    test('custom survey types used in batch tasks', () => {
        const data = sampleData();
        const tasks = generateTasks({
            ...data,
            metrics: { total_time_ms: 1000, batch_size: 32, throughput_bps: 1e9 },
            surveyTypes: ['my custom survey'],
        });
        const surveyTasks = tasks.filter(t => t.includes('my custom survey'));
        expect(surveyTasks.length).toBeGreaterThan(0);
    });

    test('tasks contain magnitude data', () => {
        const data = sampleData(4);
        const tasks = generateTasks(data);
        // At least some tasks should reference SDSS band values
        const hasBands = tasks.some(t => t.includes('u=') || t.includes('u-g='));
        expect(hasBands).toBe(true);
    });
});