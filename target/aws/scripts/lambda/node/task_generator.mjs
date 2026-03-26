/**
 * Generate LLM analysis tasks from astronomical inference results (Node.js).
 *
 * Port of target/shared/scripts/cosmic_ai/task_generator.py for Path B.
 * Takes ONNX inference output and generates semantically-clustered
 * LLM tasks for context reuse experiments.
 *
 * Configuration resolution:
 *   1. Direct parameter override
 *   2. Config file (configPath or COSMIC_AI_CONFIG env var)
 *   3. Built-in defaults
 */

import { readFileSync, existsSync } from 'fs';

const BANDS = ['u', 'g', 'r', 'i', 'z'];

const DEFAULT_TEMPLATES = {
    redshift_analysis:
        'Analyze the photometric redshift prediction z={z_pred} ' +
        '(true z={z_true}) for a galaxy with SDSS magnitudes ' +
        '{band_str}. Assess the prediction accuracy and classify ' +
        'the likely galaxy morphological type based on the color profile.',
    color_classification:
        'Given SDSS color indices {color_str} and predicted redshift ' +
        'z={z_pred}, classify this galaxy\'s morphological type ' +
        'and assess whether the colors are consistent with the ' +
        'predicted redshift.',
    outlier_analysis:
        'The AstroMAE model predicted z={z_pred} for a galaxy ' +
        'with true spectroscopic redshift z={z_true} ' +
        '(residual={residual}). The galaxy has magnitudes {band_str}. ' +
        'Analyze whether this prediction error is significant and ' +
        'identify possible causes.',
    batch_summary:
        'Summarize the inference results for a batch of {n} galaxies: ' +
        'mean predicted redshift z={mean_z}, MAE={mae}, ' +
        'bias={bias}, precision(NMAD)={nmad}. ' +
        'Assess whether this accuracy meets the requirements for ' +
        '{survey_type} surveys.',
    cost_analysis:
        'For a serverless inference run processing {n} galaxy images ' +
        'in {time_s} seconds at {throughput} Gbps throughput, ' +
        'analyze the cost-effectiveness compared to traditional HPC. ' +
        'The batch used {batch_size} samples with magnitudes ranging ' +
        '{mag_range}.',
};

const DEFAULT_SURVEY_TYPES = [
    'large-scale structure',
    'cosmological distance measurement',
    'galaxy cluster identification',
    'weak gravitational lensing',
    'baryon acoustic oscillation',
];

/**
 * Load config from JSON file.
 */
function loadConfig(configPath) {
    const data = JSON.parse(readFileSync(configPath, 'utf-8'));
    return {
        templates: { ...DEFAULT_TEMPLATES, ...(data.templates || {}) },
        surveyTypes: data.survey_types || DEFAULT_SURVEY_TYPES,
        bands: data.bands || BANDS,
    };
}

/**
 * Resolve configuration with precedence:
 * 1. Direct parameters → 2. Config file → 3. Env var → 4. Defaults
 */
function resolveConfig(templates = null, surveyTypes = null, configPath = null) {
    let fileConfig = null;

    const resolvedPath = configPath || process.env.COSMIC_AI_CONFIG;
    if (resolvedPath && existsSync(resolvedPath)) {
        fileConfig = loadConfig(resolvedPath);
    }

    return {
        templates: templates || (fileConfig?.templates) || DEFAULT_TEMPLATES,
        surveyTypes: surveyTypes || (fileConfig?.surveyTypes) || DEFAULT_SURVEY_TYPES,
        bands: (fileConfig?.bands) || BANDS,
    };
}

function formatBands(magnitudes, bands = BANDS) {
    return bands.map((b, i) => `${b}=${magnitudes[i].toFixed(2)}`).join(', ');
}

function formatColors(magnitudes, bands = BANDS) {
    const colors = [];
    for (let i = 0; i < bands.length - 1; i++) {
        const diff = magnitudes[i] - magnitudes[i + 1];
        colors.push(`${bands[i]}-${bands[i + 1]}=${diff.toFixed(2)}`);
    }
    return colors.join(', ');
}

/**
 * Simple template formatting — replaces {key} with values.
 */
function fmt(template, vars) {
    return template.replace(/\{(\w+)\}/g, (_, key) => {
        const val = vars[key];
        if (val === undefined) return `{${key}}`;
        return typeof val === 'number' ? val.toFixed(val < 1 ? 4 : 3) : String(val);
    });
}

function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, idx)];
}

/**
 * Seeded pseudo-random number generator (simple LCG).
 */
function seededRng(seed) {
    let s = seed;
    return () => {
        s = (s * 1664525 + 1013904223) & 0xffffffff;
        return (s >>> 0) / 0xffffffff;
    };
}

function sampleN(arr, n, rng) {
    const shuffled = [...arr];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled.slice(0, n);
}

/**
 * Generate LLM analysis tasks from inference results.
 *
 * @param {Object} params
 * @param {number[]} params.predictions - Predicted redshifts
 * @param {number[]} params.trueRedshifts - True redshifts
 * @param {number[][]} params.magnitudes - Magnitude arrays (N x 5)
 * @param {Object} params.metrics - Inference metrics (optional)
 * @param {number} params.maxTasks - Max tasks to generate (optional)
 * @param {number} params.seed - Random seed (optional)
 * @param {Object} params.templates - Template overrides (optional)
 * @param {string[]} params.surveyTypes - Survey type overrides (optional)
 * @param {string} params.configPath - Config file path (optional)
 * @returns {string[]} Array of task description strings
 */
export function generateTasks({
    predictions,
    trueRedshifts,
    magnitudes,
    metrics = null,
    maxTasks = null,
    seed = null,
    templates = null,
    surveyTypes = null,
    configPath = null,
}) {
    const rng = seed !== null ? seededRng(seed) : Math.random;
    const config = resolveConfig(templates, surveyTypes, configPath);
    const nSamples = predictions.length;

    const residuals = predictions.map((p, i) => Math.abs(p - trueRedshifts[i]));
    const tasks = [];

    // Select samples
    let selected;
    if (maxTasks && maxTasks < nSamples) {
        const nOutliers = Math.max(1, Math.floor(maxTasks / 4));
        const nNormal = maxTasks - nOutliers;

        const sortedIndices = residuals.map((r, i) => [r, i])
            .sort((a, b) => a[0] - b[0])
            .map(x => x[1]);
        const outlierIndices = sortedIndices.slice(-nOutliers);
        const outlierSet = new Set(outlierIndices);
        const normalPool = Array.from({ length: nSamples }, (_, i) => i)
            .filter(i => !outlierSet.has(i));
        const normalIndices = sampleN(normalPool, Math.min(nNormal, normalPool.length), rng);
        selected = [...normalIndices, ...outlierIndices];
    } else {
        selected = Array.from({ length: nSamples }, (_, i) => i);
    }

    const outlierThreshold = percentile(residuals, 90);

    for (const idx of selected) {
        const mags = magnitudes[idx];
        const zPred = predictions[idx];
        const zTrue = trueRedshifts[idx];
        const residual = residuals[idx];
        const bandStr = formatBands(mags, config.bands);
        const colorStr = formatColors(mags, config.bands);

        if (residual > outlierThreshold) {
            tasks.push(fmt(config.templates.outlier_analysis || '', {
                z_pred: zPred, z_true: zTrue, residual, band_str: bandStr,
            }));
        } else if (idx % 2 === 0) {
            tasks.push(fmt(config.templates.redshift_analysis || '', {
                z_pred: zPred, z_true: zTrue, band_str: bandStr,
            }));
        } else {
            tasks.push(fmt(config.templates.color_classification || '', {
                z_pred: zPred, color_str: colorStr,
            }));
        }
    }

    // Batch-level tasks
    if (metrics) {
        const deltaZ = predictions.map((p, i) => (p - trueRedshifts[i]) / (1 + trueRedshifts[i]));
        const mae = residuals.reduce((a, b) => a + b, 0) / nSamples;
        const bias = deltaZ.reduce((a, b) => a + b, 0) / nSamples;
        const medianDelta = percentile(deltaZ, 50);
        const nmad = 1.48 * percentile(deltaZ.map(d => Math.abs(d - medianDelta)), 50);
        const meanZ = predictions.reduce((a, b) => a + b, 0) / nSamples;

        const surveyTypeSample = sampleN(config.surveyTypes, Math.min(2, config.surveyTypes.length), rng);
        for (const surveyType of surveyTypeSample) {
            tasks.push(fmt(config.templates.batch_summary || '', {
                n: nSamples, mean_z: meanZ, mae, bias, nmad, survey_type: surveyType,
            }));
        }

        const magFlat = magnitudes.flat();
        const magMin = Math.min(...magFlat);
        const magMax = Math.max(...magFlat);
        tasks.push(fmt(config.templates.cost_analysis || '', {
            n: nSamples,
            time_s: (metrics.total_time_ms || 0) / 1000,
            throughput: (metrics.throughput_bps || 0) / 1e9,
            batch_size: metrics.batch_size || 512,
            mag_range: `${magMin.toFixed(1)}-${magMax.toFixed(1)}`,
        }));
    }

    return tasks;
}