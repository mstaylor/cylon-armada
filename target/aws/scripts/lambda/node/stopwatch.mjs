/**
 * StopWatch for Node.js — produces cloudmesh-compatible CSV output.
 *
 * Matches the output format of cloudmesh.common.StopWatch so Python and
 * Node.js experiment results can be processed by the same results pipeline.
 *
 * CSV format:
 *   # csv,name,status,time_s,sum_s,start,tag,msg,node,user,os,version
 *
 * Usage:
 *   import { StopWatch } from './stopwatch.mjs';
 *
 *   StopWatch.start('embedding');
 *   // ... do embedding ...
 *   StopWatch.stop('embedding');
 *   StopWatch.benchmark({ tag: 'experiment_1', filename: 'results.csv' });
 */

import { writeFileSync } from 'fs';
import { hostname, userInfo, platform, release } from 'os';

const timers = {};
const metrics = {};

export const StopWatch = {
    /**
     * Start a named timer.
     */
    start(name) {
        timers[name] = {
            start: performance.now(),
            startTime: new Date().toISOString().replace('T', ' ').slice(0, 19),
            elapsed: 0,
            status: 'running',
        };
    },

    /**
     * Stop a named timer.
     */
    stop(name) {
        if (!timers[name]) {
            console.warn(`StopWatch: timer '${name}' was never started`);
            return 0;
        }
        const elapsed = (performance.now() - timers[name].start) / 1000; // seconds
        timers[name].elapsed = elapsed;
        timers[name].status = 'ok';
        return elapsed;
    },

    /**
     * Get elapsed time in seconds.
     */
    get(name) {
        return timers[name]?.elapsed || 0;
    },

    /**
     * Get elapsed time in milliseconds.
     */
    getMs(name) {
        return (timers[name]?.elapsed || 0) * 1000;
    },

    /**
     * Record a named metric.
     */
    record(name, value) {
        metrics[name] = value;
    },

    /**
     * Get a recorded metric.
     */
    getMetric(name) {
        return metrics[name];
    },

    /**
     * Get all timers.
     */
    keys() {
        return Object.keys(timers);
    },

    /**
     * Clear all timers and metrics.
     */
    clear() {
        for (const key of Object.keys(timers)) delete timers[key];
        for (const key of Object.keys(metrics)) delete metrics[key];
    },

    /**
     * Write cloudmesh-compatible benchmark CSV.
     *
     * @param {Object} options
     * @param {string} options.tag - Experiment tag
     * @param {string} options.filename - Output CSV path (optional, prints to stdout if omitted)
     */
    benchmark({ tag = '', filename = null } = {}) {
        const node = hostname();
        const user = userInfo().username;
        const os = platform();
        const version = release();

        const header = '# csv,timer,status,time,sum,start,tag,msg,uname.node,user,uname.system,platform.version';
        const lines = [header];

        for (const [name, t] of Object.entries(timers)) {
            const time_s = t.elapsed.toFixed(4);
            lines.push(
                `# csv,${name},${t.status},${time_s},${time_s},${t.startTime},${tag},None,${node},${user},${os},${version}`
            );
        }

        const csv = lines.join('\n') + '\n';

        if (filename) {
            writeFileSync(filename, csv);
        } else {
            process.stdout.write(csv);
        }

        return csv;
    },

    /**
     * Export all timings and metrics as a JSON-serializable dict.
     */
    toDict(experimentName = '') {
        const result = { experiment_name: experimentName };
        for (const [name, t] of Object.entries(timers)) {
            const phase = experimentName ? name.replace(`${experimentName}_`, '') : name;
            result[`${phase}_s`] = parseFloat(t.elapsed.toFixed(6));
            result[`${phase}_ms`] = parseFloat((t.elapsed * 1000).toFixed(4));
        }
        Object.assign(result, metrics);
        return result;
    },

    /**
     * Save benchmark CSV + metrics JSON. Optionally upload to S3.
     *
     * @param {Object} options
     * @param {string} options.name - Experiment name
     * @param {string} options.outputDir - Local output directory
     * @param {string} options.tag - Benchmark tag
     * @param {string} options.s3Bucket - S3 bucket (null = local only)
     * @param {string} options.s3Prefix - S3 key prefix
     * @returns {Object} { csv, json, s3Keys }
     */
    async save({ name, outputDir, tag = '', s3Bucket = null, s3Prefix = 'experiments' } = {}) {
        const { mkdirSync, writeFileSync } = await import('fs');
        mkdirSync(outputDir, { recursive: true });

        const csvPath = `${outputDir}/${name}_stopwatch.csv`;
        this.benchmark({ tag: tag || name, filename: csvPath });

        const jsonPath = `${outputDir}/${name}_metrics.json`;
        writeFileSync(jsonPath, JSON.stringify(this.toDict(name), null, 2));

        const result = { csv: csvPath, json: jsonPath };

        if (s3Bucket) {
            result.s3Keys = await this._uploadToS3(s3Bucket, s3Prefix, name, [csvPath, jsonPath]);
        }

        return result;
    },

    async _uploadToS3(bucket, prefix, name, filePaths) {
        const { S3Client, PutObjectCommand } = await import('@aws-sdk/client-s3');
        const { readFileSync } = await import('fs');
        const { basename } = await import('path');

        const s3 = new S3Client({ region: process.env.AWS_DEFAULT_REGION || 'us-east-1' });
        const uploaded = [];

        for (const localPath of filePaths) {
            const filename = basename(localPath);
            const key = `${prefix}/${name}/${filename}`;
            try {
                await s3.send(new PutObjectCommand({
                    Bucket: bucket,
                    Key: key,
                    Body: readFileSync(localPath),
                }));
                uploaded.push(key);
                console.log(`Uploaded s3://${bucket}/${key}`);
            } catch (e) {
                console.warn(`S3 upload failed for ${filename}: ${e.message}`);
            }
        }

        return uploaded;
    },
};