#!/bin/bash
#SBATCH --job-name=cosmic-pull
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
# Pull the classic Cosmic AI image to a .sif on a compute node, then check its imports.
#
#   DOCKER_IMAGE=qad5gv/cosmic-ai-lambda:arm64 SIF_IMAGE=$SCRATCH/cosmic/cosmic-ai-lambda_arm64.sif \
#   sbatch --account=TG-NAIRR260130 --export=ALL pull_image.slurm
set -euo pipefail

: "${DOCKER_IMAGE:?DOCKER_IMAGE must be set}" "${SIF_IMAGE:?SIF_IMAGE must be set}"
CONTAINER_PYTHON=${CONTAINER_PYTHON:-/opt/conda/envs/cylon_dev/bin/python}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}

set +u
module load tacc-apptainer
set -u
mkdir -p "$APPTAINER_CACHEDIR" "$(dirname "$SIF_IMAGE")"
apptainer pull --force "$SIF_IMAGE" "docker://docker.io/$DOCKER_IMAGE"
apptainer exec "$SIF_IMAGE" "$CONTAINER_PYTHON" -c \
  "import platform, torch, timm; print(platform.machine(), 'torch', torch.__version__, 'timm', timm.__version__)"