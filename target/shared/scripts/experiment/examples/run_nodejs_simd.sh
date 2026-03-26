#!/usr/bin/env bash
##
# Node.js WASM SIMD benchmark — measures cosine similarity throughput.
#
# No AWS dependencies — pure WASM SIMD performance measurement.
#
# Prerequisites:
#   - cylon-wasm built: wasm-pack build --target nodejs --release
#   - cylon_host stub installed (see EXPERIMENT_PLAYBOOK.md)
#
# Usage:
#   ./run_nodejs_simd.sh
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
DIM=256
N=1000
ITERATIONS=100
NAME="nodejs_simd_d${DIM}_n${N}"
OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_nodejs"

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dim) DIM="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NAME="nodejs_simd_d${DIM}_n${N}"

echo "=== Node.js WASM SIMD Benchmark ==="
echo "Dimensions:  ${DIM}"
echo "Embeddings:  ${N}"
echo "Iterations:  ${ITERATIONS}"
echo "Output:      ${OUTPUT_DIR}"
echo ""

S3_ARGS=""
if [[ -n "${S3_BUCKET:-}" ]]; then
    S3_ARGS="--s3-bucket ${S3_BUCKET}"
fi

cd "${NODE_DIR}"
node run_experiment.mjs \
    --action simd_benchmark \
    --dim "${DIM}" --n "${N}" --iterations "${ITERATIONS}" \
    --name "${NAME}" \
    --output "${OUTPUT_DIR}" \
    ${S3_ARGS}

echo ""
echo "=== Results ==="
METRICS=$(find "${OUTPUT_DIR}" -name "${NAME}*_metrics.json" | head -1)
if [[ -n "${METRICS}" ]]; then
    python -m json.tool "${METRICS}"
fi