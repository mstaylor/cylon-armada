/**
 * AstroMAE ONNX inference for Node.js Lambda (Path B).
 *
 * Runs the exported ONNX model via onnxruntime-node. Supports both
 * full-model inference (single Lambda) and partitioned inference
 * (model parallelism across Lambda functions via FMI).
 *
 * Environment variables:
 *   ONNX_MODEL_PATH:  Path to the ONNX model file (local or /tmp after S3 download)
 *   ONNX_MODEL_S3:    S3 URI for the model (e.g., s3://bucket/astromae.onnx)
 *   INFERENCE_DEVICE:  'cpu' (default, only option on Lambda)
 */

import { readFileSync } from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { writeFileSync, existsSync } from 'fs';

let session = null;
let s3Client = null;

/**
 * Download ONNX model from S3 to /tmp.
 */
async function downloadModelFromS3(s3Uri) {
    if (!s3Client) {
        s3Client = new S3Client({ region: process.env.AWS_DEFAULT_REGION || 'us-east-1' });
    }

    const match = s3Uri.match(/^s3:\/\/([^/]+)\/(.+)$/);
    if (!match) throw new Error(`Invalid S3 URI: ${s3Uri}`);

    const [, bucket, key] = match;
    const localPath = `/tmp/${key.split('/').pop()}`;

    if (existsSync(localPath)) return localPath;

    const response = await s3Client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    const chunks = [];
    for await (const chunk of response.Body) {
        chunks.push(chunk);
    }
    writeFileSync(localPath, Buffer.concat(chunks));
    return localPath;
}

/**
 * Initialize the ONNX inference session.
 */
async function initSession(modelPath) {
    if (session) return session;

    let resolvedPath = modelPath;

    // Download from S3 if needed
    if (resolvedPath && resolvedPath.startsWith('s3://')) {
        resolvedPath = await downloadModelFromS3(resolvedPath);
    }

    if (!resolvedPath) {
        resolvedPath = process.env.ONNX_MODEL_PATH || '/tmp/astromae.onnx';
    }

    session = await InferenceSession.create(resolvedPath, {
        executionProviders: ['CPUExecutionProvider'],
    });

    return session;
}

/**
 * Run inference on a batch of galaxy observations.
 *
 * @param {Float32Array} imageData - Flattened image tensor (B * 5 * H * W)
 * @param {Float32Array} magnitudeData - Magnitude values (B * 5)
 * @param {number} batchSize - Number of samples in batch
 * @param {number} imageSize - Spatial dimension (default: 224)
 * @param {string} modelPath - Optional model path override
 * @returns {Object} { predictions, metrics }
 */
export async function runInference(imageData, magnitudeData, batchSize, imageSize = 224, modelPath = null) {
    const sess = await initSession(modelPath);
    const start = performance.now();

    const imageTensor = new Tensor('float32', imageData, [batchSize, 5, imageSize, imageSize]);
    const magnitudeTensor = new Tensor('float32', magnitudeData, [batchSize, 5]);

    const feeds = {
        image: imageTensor,
        magnitude: magnitudeTensor,
    };

    const results = await sess.run(feeds);
    const predictions = results.redshift.data;
    const elapsedMs = performance.now() - start;

    return {
        predictions: Array.from(predictions),
        metrics: {
            total_time_ms: Math.round(elapsedMs * 100) / 100,
            num_samples: batchSize,
            samples_per_sec: Math.round((batchSize / (elapsedMs / 1000)) * 100) / 100,
            runtime: 'onnxruntime-node',
        },
    };
}

/**
 * Run inference from a data file (JSON format with image and magnitude arrays).
 *
 * Expected JSON format:
 * {
 *   "images": [...],       // Flattened float32 array (B * 5 * H * W)
 *   "magnitudes": [...],   // Flattened float32 array (B * 5)
 *   "redshifts": [...],    // True redshift labels (B,) — optional
 *   "batch_size": 32,
 *   "image_size": 224
 * }
 */
export async function runInferenceFromFile(dataPath, modelPath = null) {
    const raw = readFileSync(dataPath, 'utf-8');
    const data = JSON.parse(raw);

    const imageData = new Float32Array(data.images);
    const magnitudeData = new Float32Array(data.magnitudes);
    const batchSize = data.batch_size || imageData.length / (5 * data.image_size * data.image_size);

    const result = await runInference(imageData, magnitudeData, batchSize, data.image_size || 224, modelPath);

    if (data.redshifts) {
        result.true_redshifts = data.redshifts;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Model parallelism — partitioned inference via FMI
// ---------------------------------------------------------------------------

// Per-stage sessions (independent of the full-model session)
const stageSessions = {};

/**
 * Initialize a stage-specific ONNX session.
 */
async function initStageSession(stageName, modelPath) {
    if (stageSessions[stageName]) return stageSessions[stageName];

    let resolvedPath = modelPath;
    if (resolvedPath && resolvedPath.startsWith('s3://')) {
        resolvedPath = await downloadModelFromS3(resolvedPath);
    }

    stageSessions[stageName] = await InferenceSession.create(resolvedPath, {
        executionProviders: ['CPUExecutionProvider'],
    });

    return stageSessions[stageName];
}

/**
 * Run a single stage of partitioned inference.
 *
 * Each Lambda worker runs its assigned stage and exchanges intermediate
 * tensors via FMI (Direct/Redis/S3 channel).
 *
 * @param {Object} params
 * @param {string} params.stageName - Stage identifier (stage_0_vit, stage_1_inception, stage_2_fusion)
 * @param {string} params.modelPath - Path or S3 URI to the stage's ONNX subgraph
 * @param {Object} params.inputs - Named input tensors { name: Float32Array }
 * @param {number[]} params.inputShapes - Shapes for each input tensor { name: [dims] }
 * @returns {Object} { outputs: { name: Float32Array }, shapes: { name: [dims] }, metrics }
 */
export async function runStageInference({ stageName, modelPath, inputs, inputShapes }) {
    const sess = await initStageSession(stageName, modelPath);
    const start = performance.now();

    const feeds = {};
    for (const [name, data] of Object.entries(inputs)) {
        feeds[name] = new Tensor('float32', new Float32Array(data), inputShapes[name]);
    }

    const results = await sess.run(feeds);
    const elapsedMs = performance.now() - start;

    const outputs = {};
    const shapes = {};
    for (const [name, tensor] of Object.entries(results)) {
        outputs[name] = Array.from(tensor.data);
        shapes[name] = tensor.dims;
    }

    return {
        outputs,
        shapes,
        metrics: {
            stage: stageName,
            latency_ms: Math.round(elapsedMs * 100) / 100,
        },
    };
}

/**
 * Serialize tensor data for FMI exchange.
 * Encodes as base64 for transport over Redis/Direct channels.
 */
export function serializeTensor(data, shape) {
    const float32 = new Float32Array(data);
    const buffer = Buffer.from(float32.buffer);
    return {
        data_b64: buffer.toString('base64'),
        shape,
        dtype: 'float32',
    };
}

/**
 * Deserialize tensor data received from FMI exchange.
 */
export function deserializeTensor(serialized) {
    const buffer = Buffer.from(serialized.data_b64, 'base64');
    return {
        data: new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4),
        shape: serialized.shape,
    };
}