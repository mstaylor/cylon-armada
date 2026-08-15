#!/bin/bash --login
# Container entrypoint for exp_b_collectives.py on Rivanna via Apptainer.
# Mirrors runArmada.sh's conda-activation pattern. All experiment config is
# passed as CLI args (forwarded from the sbatch script's srun invocation).

set +euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate cylon_dev
set -euo pipefail

export PYTHONPATH=/cylon-armada:/cylon-armada/scripts:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/cylon/install/lib:/cylon-armada/install/lib:/opt/conda/envs/cylon_dev/lib:${LD_LIBRARY_PATH:-}

echo "=== exp_b_collectives Rivanna runner ==="
echo "Host: $(hostname)  SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "========================================="

exec python -m experiment.exp_b_collectives "$@"