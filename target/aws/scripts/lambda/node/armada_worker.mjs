/**
 * armada_worker — model-parallel Lambda worker (Node.js / ONNX).
 *
 * Handles one stage of partitioned AstroMAE inference. Called by the
 * model_parallel Step Functions workflow:
 *
 *   ParallelStages:
 *     Stage0_ViT       (rank 0) — ViT encoder → vit_features [B, 1096]
 *     Stage1_Inception (rank 1) — Inception   → inception_features [B, 2120]
 *   Stage2_Fusion (after both):  concatenation MLP → redshift [B, 1]
 *
 * Input event:
 *   { action: "model_parallel_stage", action_payload: { ... } }
 *
 * CMD: armada_worker.handler
 */

import { runStageInference } from './inference.mjs';

// ---------------------------------------------------------------------------
// Input normalisation helpers
// ---------------------------------------------------------------------------

/**
 * Decode a tensor value that may be:
 *   - a plain JS array (Step Functions passes numbers directly)
 *   - a base64 string (serialised from a previous stage)
 */
function decodeTensorData(value) {
    if (typeof value === 'string') {
        const buf = Buffer.from(value, 'base64');
        return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
    }
    return value; // already an array
}

/**
 * Normalise inputs and inputShapes into the { name: data } / { name: dims }
 * dicts that runStageInference expects.  Each stage receives different
 * structures from Step Functions:
 *
 *   stage_0_vit:
 *     inputs      = raw image array  (from $.inference_input.image)
 *     input_shapes = raw shape array (from $.inference_input.image_shape)
 *     → { image: data } / { image: [B,5,H,W] }
 *
 *   stage_1_inception:
 *     inputs      = full inference_input object { image, magnitude, shapes, … }
 *     input_shapes = { image: […], magnitude: […] }
 *     → { image: data, magnitude: data } / { image: […], magnitude: […] }
 *
 *   stage_2_fusion:
 *     inputs      = { vit_features: data, inception_features: data }
 *     input_shapes = { vit_features: […], inception_features: […] }
 *     → already in the right format, just decode each tensor
 */
function normaliseInputs(stageName, inputs, inputShapes) {
    if (stageName === 'stage_0_vit') {
        return {
            inputs:      { image: decodeTensorData(inputs) },
            inputShapes: { image: inputShapes },
        };
    }

    if (stageName === 'stage_1_inception') {
        const shapes = inputShapes || inputs?.shapes || {};
        return {
            inputs: {
                image:     decodeTensorData(inputs?.image     || inputs),
                magnitude: decodeTensorData(inputs?.magnitude || []),
            },
            inputShapes: {
                image:     shapes.image     || [1, 5, 224, 224],
                magnitude: shapes.magnitude || [1, 5],
            },
        };
    }

    if (stageName === 'stage_2_fusion') {
        const normInputs = {};
        const normShapes = {};
        for (const [name, val] of Object.entries(inputs || {})) {
            normInputs[name] = decodeTensorData(val);
            normShapes[name] = (inputShapes || {})[name] || [];
        }
        return { inputs: normInputs, inputShapes: normShapes };
    }

    throw new Error(`Unknown stage_name: ${stageName}`);
}

/**
 * Generate a synthetic float32 array of a given shape filled with zeros.
 * Used as a fallback when no ONNX model is available (smoke-test mode).
 */
function syntheticOutput(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    return Array(size).fill(0);
}

/**
 * Run stage inference or return synthetic output when model_path is absent.
 */
async function runStage(stageName, modelPath, inputs, inputShapes) {
    if (!modelPath) {
        // Smoke-test mode — return correct shapes without an ONNX model.
        const outputShapes = {
            stage_0_vit:       { vit_features:       [inputShapes.image?.[0] ?? 1, 1096] },
            stage_1_inception: { inception_features:  [inputShapes.image?.[0] ?? 1, 2120] },
            stage_2_fusion:    { redshift:            [inputShapes.vit_features?.[0] ?? 1, 1] },
        };
        const shapes  = outputShapes[stageName] || {};
        const outputs = Object.fromEntries(
            Object.entries(shapes).map(([k, s]) => [k, syntheticOutput(s)])
        );
        return { outputs, shapes, metrics: { stage: stageName, latency_ms: 0, synthetic: true } };
    }

    return runStageInference({ stageName, modelPath, inputs, inputShapes });
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handler(event) {
    if (event.action !== 'model_parallel_stage') {
        return {
            statusCode: 400,
            error: `Unsupported action: ${event.action}. Expected "model_parallel_stage".`,
        };
    }

    const payload     = event.action_payload || {};
    const stageName   = payload.stage_name;
    const modelPath   = payload.model_path   || null;
    const workflowId  = payload.workflow_id  || '';
    const rank        = payload.rank         ?? 0;
    const worldSize   = payload.world_size   ?? 1;

    const start = performance.now();

    const { inputs, inputShapes } = normaliseInputs(
        stageName,
        payload.inputs,
        payload.input_shapes,
    );

    const { outputs, shapes, metrics } = await runStage(
        stageName,
        modelPath,
        inputs,
        inputShapes,
    );

    const totalMs = Math.round((performance.now() - start) * 100) / 100;

    return {
        stage_name:   stageName,
        workflow_id:  workflowId,
        rank,
        world_size:   worldSize,
        outputs,
        shapes,
        latency_ms:   totalMs,
        inference_ms: metrics?.latency_ms ?? 0,
        synthetic:    metrics?.synthetic  ?? false,
    };
}