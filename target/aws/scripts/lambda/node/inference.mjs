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