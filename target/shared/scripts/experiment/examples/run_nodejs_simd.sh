#!/usr/bin/env bash
##
# Node.js cosine similarity benchmark — measures search throughput.
#
# Supports two backends:
#   - wasm (default): WASM SIMD128 via cylon-wasm
#   - redis: Pure JS Float32Array dot product (no WASM required)
#
# Prerequisites (wasm backend only):
#   - cylon-wasm built: wasm-pack build --target nodejs --release
#   - cylon_host stub installed (see EXPERIMENT_PLAYBOOK.md)
#
# Usage:
#   ./run_nodejs_simd.sh                        # wasm backend (default)
#   ./run_nodejs_simd.sh --context-backend redis # JS dot product
#   ./run_nodejs_simd.sh --dim 1024 --n 5000
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)/aws/scripts/lambda/node"

# Defaults — override via env vars
: "${CYLON_WASM_BINDINGS:=${HOME}/cylon/rust/cylon-wasm/pkg/cylon_wasm.js}"
: "${CYLON_WASM_PATH:=${HOME}/cylon/rust/cylon-wasm/pkg/cylon_wasm_bg.wasm}"

export CYLON_WASM_BINDINGS CYLON_WASM_PATH

# Default benchmark params
CONTEXT_BACKEND=""
DIM=256
N=1000
ITERATIONS=100
OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_nodejs"

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --context-backend|--backend) CONTEXT_BACKEND="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --name) NAME_OVERRIDE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BACKEND_LABEL="${CONTEXT_BACKEND:-wasm}"
NAME="${NAME_OVERRIDE:-nodejs_simd_${BACKEND_LABEL}_d${DIM}_n${N}}"

echo "=== Node.js Similarity Benchmark ==="
echo "Backend:     ${BACKEND_LABEL}"
echo "Dimensions:  ${DIM}"
echo "Embeddings:  ${N}"
echo "Iterations:  ${ITERATIONS}"
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
    --action simd_benchmark \
    --dim "${DIM}" --n "${N}" --iterations "${ITERATIONS}" \
    --name "${NAME}" \
    --output "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "=== Results ==="
METRICS=$(find "${OUTPUT_DIR}" -name "${NAME}*_metrics.json" | head -1)
if [[ -n "${METRICS}" ]]; then
    python -m json.tool "${METRICS}"
fi