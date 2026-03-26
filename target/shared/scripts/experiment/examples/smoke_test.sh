#!/usr/bin/env bash
##
# Smoke test — quick validation that the experiment pipeline works.
#
# Runs 4 hydrology tasks with a single configuration (threshold=0.8, dim=256).
# Produces baseline + reuse results in the smoke_test output directory.
#
# Prerequisites:
#   - Redis and DynamoDB Local running (see EXPERIMENT_PLAYBOOK.md)
#   - REDIS_HOST and DYNAMO_ENDPOINT_URL set
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
: "${DYNAMO_ENDPOINT_URL:=http://10.211.55.2:8100}"
: "${DYNAMO_TABLE_NAME:=cylon-armada-context-store}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) CONTEXT_BACKEND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CONTEXT_BACKEND REDIS_HOST REDIS_PORT DYNAMO_ENDPOINT_URL DYNAMO_TABLE_NAME

OUTPUT_DIR="${EXPERIMENT_DIR}/results/smoke_test"

echo "=== Smoke Test ==="
echo "Backend:    ${CONTEXT_BACKEND}"
echo "Redis:      ${REDIS_HOST}:${REDIS_PORT}"
echo "DynamoDB:   ${DYNAMO_ENDPOINT_URL}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

python "${EXPERIMENT_DIR}/runner.py" \
    --tasks-file "${EXPERIMENT_DIR}/scenarios/hydrology.json" \
    --tasks 4 \
    --thresholds 0.8 \
    --dimensions 256 \
    --output "${OUTPUT_DIR}"

echo ""
echo "=== Results ==="
ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "No JSON results found"
echo ""

# Show summary if available
SUMMARY=$(find "${OUTPUT_DIR}" -name "*_summary.json" -newer "${OUTPUT_DIR}" | head -1)
if [[ -n "${SUMMARY}" ]]; then
    python -m json.tool "${SUMMARY}"
fi