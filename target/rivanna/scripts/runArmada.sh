#!/bin/bash --login
# Container entrypoint for cylon-armada on Rivanna via Apptainer.
# Mirrors cylon/target/rivanna/scripts/ucc-ucx-redis/docker/runCylon.sh.
#
# Activates the cylon_dev conda environment, sets PYTHONPATH so the shared
# scripts and armada_ecs_runner.py are importable, then hands off to the
# runner. All experiment config comes in via env vars set by Apptainer --env.

set -euo pipefail

# Activate conda (strict mode must be off while sourcing conda's hook)
set +euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate cylon_dev
set -euo pipefail

# Make shared scripts and the runner importable
export PYTHONPATH=/cylon-armada:/cylon-armada/scripts:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/cylon/cpp/build_gcylon/lib:/cylon/install/lib:/cylon-armada/install/lib:/opt/conda/envs/cylon_dev/lib:${LD_LIBRARY_PATH:-}

echo "=== cylon-armada Rivanna runner ==="
echo "Host:     $(hostname)"
echo "Platform: ${COMPUTE_PLATFORM:-rivanna}"
echo "Backend:  ${SIMD_BACKEND:-numpy}"
echo "Workflow: ${WORKFLOW_ID:-unset}"
echo "==================================="

exec python /cylon-armada/armada_ecs_runner.py