#!/usr/bin/env bash
##
# Smoke test — quick validation that the experiment pipeline works.
#
# Runs 4 hydrology tasks with a single configuration (threshold=0.8, dim=256).
# Produces baseline + reuse results in the smoke_test output directory.
#
# Prerequisites:
#   - Redis running (primary persistence layer)
#   - REDIS_HOST set (default: 10.211.55.2)
#   - Optional: DYNAMO_TABLE_NAME set to enable DynamoDB persistence
#
# Usage:
#   ./smoke_test.sh                    # redis backend (default)
#   ./smoke_test.sh --backend cylon    # cylon ContextTable backend
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Defaults — override via env vars
: "${CONTEXT_BACKEND:=redis}"
: "${REDIS_HOST:=10.211.55.2}"
: "${REDIS_PORT:=6379}"
: "${DYNAMO_ENDPOINT_URL:=}"
: "${DYNAMO_TABLE_NAME:=}"
: "${CONDA_ENV:=cylon_dev}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) CONTEXT_BACKEND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CONTEXT_BACKEND REDIS_HOST REDIS_PORT
[[ -n "${DYNAMO_ENDPOINT_URL}" ]] && export DYNAMO_ENDPOINT_URL
[[ -n "${DYNAMO_TABLE_NAME}" ]] && export DYNAMO_TABLE_NAME

OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_test"

PYTHON="conda run -n ${CONDA_ENV} python"

echo "=== Smoke Test ==="
echo "Backend:    ${CONTEXT_BACKEND}"
echo "Redis:      ${REDIS_HOST}:${REDIS_PORT}"
echo "DynamoDB:   ${DYNAMO_TABLE_NAME:-disabled}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

${PYTHON} "${EXPERIMENT_DIR}/runner.py" \
    --context-backend "${CONTEXT_BACKEND}" \
    --tasks-file "${EXPERIMENT_DIR}/scenarios/hydrology.json" \
    --tasks 4 \
    --thresholds 0.8 \
    --dimensions 256 \
    --sampling sequential \
    --output "${OUTPUT_DIR}"

echo ""
echo "=== Results ==="
ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "No JSON results found"
echo ""

# Show summary if available
SUMMARY=$(find "${OUTPUT_DIR}" -name "*_summary.json" -newer "${OUTPUT_DIR}" | head -1)
if [[ -n "${SUMMARY}" ]]; then
    ${PYTHON} -m json.tool "${SUMMARY}"
fi