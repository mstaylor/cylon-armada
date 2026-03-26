#!/usr/bin/env bash
##
# Node.js route_task experiment — runs tasks through WASM SIMD + Bedrock.
#
# Requires Redis + Bedrock access.
#
# Usage:
#   ./run_nodejs_route.sh
#   ./run_nodejs_route.sh --tasks 8 --threshold 0.9 --dim 1024
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)/aws/scripts/lambda/node"

# Defaults — override via env vars
: "${CYLON_WASM_BINDINGS:=${HOME}/cylon/rust/cylon-wasm/pkg/cylon_wasm.js}"
: "${CYLON_WASM_PATH:=${HOME}/cylon/rust/cylon-wasm/pkg/cylon_wasm_bg.wasm}"
: "${REDIS_HOST:=10.211.55.2}"

export CYLON_WASM_BINDINGS CYLON_WASM_PATH REDIS_HOST

# Default params
TASKS=4
THRESHOLD=0.8
DIM=256
SCENARIO_FILE="${EXPERIMENT_DIR}/scenarios/hydrology.json"
NAME="nodejs_route_hydrology_t${TASKS}"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_nodejs"

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks) TASKS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --dim|--dimensions) DIM="$2"; shift 2 ;;
        --tasks-file) SCENARIO_FILE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NAME="nodejs_route_hydrology_t${TASKS}"

echo "=== Node.js Route Task Experiment ==="
echo "Scenario:    ${SCENARIO_FILE}"
echo "Tasks:       ${TASKS}"
echo "Threshold:   ${THRESHOLD}"
echo "Dimensions:  ${DIM}"
echo "Redis:       ${REDIS_HOST}"
echo "Output:      ${OUTPUT_DIR}"
echo ""

S3_ARGS=""
if [[ -n "${S3_BUCKET:-}" ]]; then
    S3_ARGS="--s3-bucket ${S3_BUCKET}"
fi

cd "${NODE_DIR}"
node run_experiment.mjs \
    --action route_task \
    --tasks-file "${SCENARIO_FILE}" \
    --tasks "${TASKS}" --threshold "${THRESHOLD}" --dimensions "${DIM}" \
    --name "${NAME}" \
    --output "${OUTPUT_DIR}" \
    ${S3_ARGS}

echo ""
echo "=== Results ==="
METRICS=$(find "${OUTPUT_DIR}" -name "${NAME}*_metrics.json" | head -1)
if [[ -n "${METRICS}" ]]; then
    python -m json.tool "${METRICS}"
fi