#!/usr/bin/env bash
##
# Run all experiment scenarios sequentially.
#
# Usage:
#   ./run_all_scenarios.sh
#   ./run_all_scenarios.sh --backend cylon
#   ./run_all_scenarios.sh --s3-bucket cylon-armada-results
#
# Runs: hydrology, epidemiology, seismology, mixed_scientific
# Each scenario uses the default parameter sweep (see run_scenario.sh).
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENARIOS=(hydrology epidemiology seismology mixed_scientific)
FAILED=()

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Scenario: ${scenario}"
    echo "================================================================"
    echo ""
    if "${SCRIPT_DIR}/run_scenario.sh" "${scenario}" "$@"; then
        echo "  ✓ ${scenario} complete"
    else
        echo "  ✗ ${scenario} FAILED"
        FAILED+=("${scenario}")
    fi
done

echo ""
echo "================================================================"
echo "  Summary"
echo "================================================================"
echo "  Total:  ${#SCENARIOS[@]}"
echo "  Passed: $(( ${#SCENARIOS[@]} - ${#FAILED[@]} ))"
echo "  Failed: ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed scenarios: ${FAILED[*]}"
    exit 1
fi