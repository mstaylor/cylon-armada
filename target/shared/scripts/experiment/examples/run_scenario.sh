#!/usr/bin/env bash
##
# Run a single experiment scenario with full parameter sweep.
#
# Usage:
#   ./run_scenario.sh hydrology
#   ./run_scenario.sh epidemiology --backend cylon
#   ./run_scenario.sh seismology --tasks 8 16 --thresholds 0.8 0.9
#   ./run_scenario.sh mixed_scientific --no-baseline
#   ./run_scenario.sh hydrology --s3-bucket cylon-armada-results
#
# Available scenarios: hydrology, epidemiology, seismology, mixed_scientific
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults — override via env vars
: "${CONTEXT_BACKEND:=redis}"
: "${REDIS_HOST:=10.211.55.2}"
: "${REDIS_PORT:=6379}"
: "${DYNAMO_ENDPOINT_URL:=http://10.211.55.2:8100}"
: "${DYNAMO_TABLE_NAME:=cylon-armada-context-store}"

export CONTEXT_BACKEND REDIS_HOST REDIS_PORT DYNAMO_ENDPOINT_URL DYNAMO_TABLE_NAME

# First positional arg is the scenario name
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <scenario> [--backend cylon|redis] [runner args...]"
    echo ""
    echo "Available scenarios:"
    ls "${EXPERIMENT_DIR}/scenarios/"*.json 2>/dev/null | xargs -I{} basename {} .json
    exit 1
fi

SCENARIO="$1"; shift

# Parse --backend before passing remaining args to runner
RUNNER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) CONTEXT_BACKEND="$2"; export CONTEXT_BACKEND; shift 2 ;;
        *) RUNNER_ARGS+=("$1"); shift ;;
    esac
done
set -- "${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"}"

SCENARIO_FILE="${EXPERIMENT_DIR}/scenarios/${SCENARIO}.json"

if [[ ! -f "${SCENARIO_FILE}" ]]; then
    echo "Error: scenario file not found: ${SCENARIO_FILE}"
    echo ""
    echo "Available scenarios:"
    ls "${EXPERIMENT_DIR}/scenarios/"*.json 2>/dev/null | xargs -I{} basename {} .json
    exit 1
fi

OUTPUT_DIR="${EXPERIMENT_DIR}/results/${SCENARIO}"

echo "=== Experiment: ${SCENARIO} ==="
echo "Scenario:   ${SCENARIO_FILE}"
echo "Backend:    ${CONTEXT_BACKEND}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

# Default sweep parameters — can be overridden via extra args
DEFAULT_ARGS=(
    --tasks-file "${SCENARIO_FILE}"
    --tasks 8 16 32
    --thresholds 0.70 0.80 0.90
    --dimensions 256 512 1024
    --backends NUMPY
    --output "${OUTPUT_DIR}"
)

# If extra args provided, use them; otherwise use defaults
if [[ $# -gt 0 ]]; then
    python "${EXPERIMENT_DIR}/runner.py" \
        --tasks-file "${SCENARIO_FILE}" \
        --output "${OUTPUT_DIR}" \
        "$@"
else
    python "${EXPERIMENT_DIR}/runner.py" "${DEFAULT_ARGS[@]}"
fi

echo ""
echo "=== Complete ==="
echo "Results: ${OUTPUT_DIR}"
SUMMARY=$(find "${OUTPUT_DIR}" -name "*_summary.json" | sort | tail -1)
if [[ -n "${SUMMARY}" ]]; then
    echo "Latest summary: ${SUMMARY}"
fi