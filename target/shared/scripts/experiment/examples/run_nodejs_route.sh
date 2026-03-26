#!/usr/bin/env bash
##
# Node.js route_task experiment — runs tasks through similarity search + Bedrock.
#
# Supports two backends:
#   - wasm (default): WASM SIMD128 via cylon-wasm
#   - redis: Pure JS Float32Array dot product (no WASM required)
#
# Requires Redis + Bedrock access.
#
# Usage:
#   ./run_nodejs_route.sh                        # wasm backend (default)
#   ./run_nodejs_route.sh --context-backend redis # JS dot product
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
CONTEXT_BACKEND=""
TASKS=4
THRESHOLD=0.8
DIM=256
SCENARIO_FILE="${EXPERIMENT_DIR}/scenarios/hydrology.json"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_nodejs"

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --context-backend|--backend) CONTEXT_BACKEND="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --dim|--dimensions) DIM="$2"; shift 2 ;;
        --tasks-file) SCENARIO_FILE="$2"; shift 2 ;;
        --name) NAME_OVERRIDE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BACKEND_LABEL="${CONTEXT_BACKEND:-wasm}"
NAME="${NAME_OVERRIDE:-nodejs_route_${BACKEND_LABEL}_t${TASKS}}"

echo "=== Node.js Route Task Experiment ==="
echo "Backend:     ${BACKEND_LABEL}"
echo "Scenario:    ${SCENARIO_FILE}"
echo "Tasks:       ${TASKS}"
echo "Threshold:   ${THRESHOLD}"
echo "Dimensions:  ${DIM}"
echo "Redis:       ${REDIS_HOST}"
echo "Output:      ${OUTPUT_DIR}"
echo ""

EXTRA_ARGS=""
if [[ -n "${CONTEXT_BACKEND}" ]]; then
    EXTRA_ARGS="--context-backend ${CONTEXT_BACKEND}"
fi
if [[ -n "${S3_BUCKET:-}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --s3-bucket ${S3_BUCKET}"
fi

cd "${NODE_DIR}"
node run_experiment.mjs \
    --action route_task \
    --tasks-file "${SCENARIO_FILE}" \
    --tasks "${TASKS}" --threshold "${THRESHOLD}" --dimensions "${DIM}" \
    --name "${NAME}" \
    --output "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "=== Results ==="
METRICS=$(find "${OUTPUT_DIR}" -name "${NAME}*_metrics.json" | head -1)
if [[ -n "${METRICS}" ]]; then
    python -m json.tool "${METRICS}"
fi