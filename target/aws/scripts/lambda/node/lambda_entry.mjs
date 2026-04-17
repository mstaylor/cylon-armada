/**
 * lambda_entry — thin entry point for all cylon-armada Node.js Lambda functions.
 *
 * Mirrors the Python lambda_entry.py pattern: downloads scripts from S3 at
 * cold-start time, then delegates to the target handler module. This eliminates
 * Docker image rebuilds for script changes — deploy bug fixes in seconds via:
 *
 *     aws s3 sync target/aws/scripts/lambda/node/ s3://<bucket>/scripts/lambda/
 *
 * Configuration (resolved in priority order: event payload > env var > default):
 *     S3_SCRIPTS_BUCKET   — S3 bucket containing the scripts folder
 *     S3_SCRIPTS_PREFIX   — Key prefix (default: "scripts/")
 *     HANDLER_MODULE      — JS module name to import and call .handler()
 *                           (e.g., "armada_init", "armada_executor",
 *                           "armada_aggregate")
 *                           (env var only — not overridable from event)
 *
 * Warm-start behaviour:
 *     After the first invocation (cold start), scripts are cached in /tmp and
 *     modules are cached by the runtime. Subsequent warm invocations skip the
 *     S3 download entirely — zero overhead.
 */

import { S3Client, ListObjectsV2Command, GetObjectCommand } from '@aws-sdk/client-s3';
import { writeFile, mkdir, symlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { existsSync } from 'node:fs';

const LOCAL_DIR = '/tmp/armada_scripts';
let _downloaded = false;
let _cachedModules = {};

async function downloadScripts(s3Bucket, s3Prefix) {
    if (_downloaded) return true;

    const s3 = new S3Client({});
    let count = 0;

    try {
        let continuationToken;
        do {
            const cmd = new ListObjectsV2Command({
                Bucket: s3Bucket,
                Prefix: s3Prefix,
                ContinuationToken: continuationToken,
            });
            const resp = await s3.send(cmd);

            for (const obj of resp.Contents || []) {
                const key = obj.Key;
                if (key.endsWith('/')) continue;

                const relative = key.startsWith(s3Prefix)
                    ? key.slice(s3Prefix.length)
                    : key;
                const dest = join(LOCAL_DIR, relative);

                await mkdir(dirname(dest), { recursive: true });

                const getCmd = new GetObjectCommand({ Bucket: s3Bucket, Key: key });
                const getResp = await s3.send(getCmd);
                const body = await streamToBuffer(getResp.Body);
                await writeFile(dest, body);
                count++;
            }

            continuationToken = resp.NextContinuationToken;
        } while (continuationToken);

        if (count === 0) {
            console.warn(`No scripts found at s3://${s3Bucket}/${s3Prefix}`);
            return false;
        }

        console.log(`Downloaded ${count} scripts from s3://${s3Bucket}/${s3Prefix} to ${LOCAL_DIR}`);

        // Symlink /app/node_modules into the download dir so ESM package
        // resolution (which walks up from /tmp/armada_scripts/lambda/) finds
        // the native packages already installed in the image.
        const nmLink = join(LOCAL_DIR, 'node_modules');
        if (!existsSync(nmLink)) {
            await symlink('/app/node_modules', nmLink);
        }
    } catch (err) {
        console.error('S3 script download failed:', err);
        return false;
    }

    _downloaded = true;
    return true;
}

async function streamToBuffer(stream) {
    const chunks = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return Buffer.concat(chunks);
}

/**
 * Resolve the handler module path. Prefers S3-downloaded version in /tmp,
 * falls back to baked-in version in /app.
 */
function resolveModulePath(handlerModule) {
    // S3 scripts land in /tmp/armada_scripts/lambda/<module>.mjs
    const s3Path = join(LOCAL_DIR, 'lambda', `${handlerModule}.mjs`);
    if (existsSync(s3Path)) return s3Path;

    // Also check root of S3 scripts dir
    const s3RootPath = join(LOCAL_DIR, `${handlerModule}.mjs`);
    if (existsSync(s3RootPath)) return s3RootPath;

    // Fallback to baked-in image
    return `/app/${handlerModule}.mjs`;
}

export async function handler(event, context) {
    const handlerModule = process.env.HANDLER_MODULE;
    if (!handlerModule) {
        throw new Error(
            'HANDLER_MODULE environment variable not set — ' +
            'configure it in Terraform image_config or Lambda environment'
        );
    }

    const s3Bucket = event.S3_SCRIPTS_BUCKET || process.env.S3_SCRIPTS_BUCKET || '';
    let s3Prefix = event.S3_SCRIPTS_PREFIX || process.env.S3_SCRIPTS_PREFIX || 'scripts/';

    if (s3Prefix && !s3Prefix.endsWith('/')) s3Prefix += '/';

    if (s3Bucket) {
        const ok = await downloadScripts(s3Bucket, s3Prefix);
        if (ok) {
            console.log(`S3 scripts loaded — delegating to ${handlerModule}.handler`);
        } else {
            console.warn(`S3 script load failed — falling back to baked-in ${handlerModule}`);
        }
    } else {
        console.log(`S3_SCRIPTS_BUCKET not set — using baked-in ${handlerModule}`);
    }

    const modulePath = resolveModulePath(handlerModule);

    // Cache modules for warm starts, but invalidate if S3 downloaded new scripts
    if (!_cachedModules[handlerModule] || (_downloaded && modulePath.startsWith(LOCAL_DIR))) {
        // Use file:// URL for dynamic import of absolute paths
        _cachedModules[handlerModule] = await import(`file://${modulePath}`);
    }

    const mod = _cachedModules[handlerModule];
    return mod.handler(event, context);
}