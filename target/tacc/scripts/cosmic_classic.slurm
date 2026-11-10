#!/bin/bash
#SBATCH --job-name=cosmic-classic
# One classic Cosmic AI execution on Vista: init, WORLD_SIZE workers, summarize.
# Submitted by sweep_cosmic_classic.sh, which sets the resources and the variables below.
#
# Required: WORLD_SIZE FILE_LIMIT BATCH_SIZE CPUS_PER_WORKER RUN_DIR SIF_IMAGE COSMIC_ROOT
# Optional: PARTITIONS (FILE_LIMIT) INFERENCE_UNIT (rank) DATA_GB DATA_DIR CODE_DIR MODEL_PATH SCRIPTS
#           CONTAINER_PYTHON
set -euo pipefail

: "${WORLD_SIZE:?}" "${FILE_LIMIT:?}" "${BATCH_SIZE:?}" "${CPUS_PER_WORKER:?}"
: "${RUN_DIR:?}" "${SIF_IMAGE:?}" "${COSMIC_ROOT:?}"
PARTITIONS=${PARTITIONS:-$FILE_LIMIT}
INFERENCE_UNIT=${INFERENCE_UNIT:-rank}
DATA_GB=${DATA_GB:-null}
DATA_DIR=${DATA_DIR:-$COSMIC_ROOT/data/25MB}
CODE_DIR=${CODE_DIR:-$COSMIC_ROOT/code/Anomaly Detection}
MODEL_PATH=${MODEL_PATH:-$CODE_DIR/Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt}
SCRIPTS=${SCRIPTS:-$COSMIC_ROOT/cylon-armada/target/tacc/scripts}
CONTAINER_PYTHON=${CONTAINER_PYTHON:-/opt/conda/envs/cylon_dev/bin/python}
export PYTHONNOUSERSITE=1
# XALT preloads the host libcrypto into srun task 0, which breaks the container's _ssl import.
export XALT_EXECUTABLE_TRACKING=no

set +u
module load tacc-apptainer
set -u
mkdir -p "$RUN_DIR"

in_container() {
  apptainer exec --bind "$COSMIC_ROOT" --bind "$(dirname "$RUN_DIR")" "$SIF_IMAGE" \
    "$CONTAINER_PYTHON" "$SCRIPTS/cosmic_classic.py" "$@"
}

t_start=$(date +%s.%N)
in_container init --data-dir "$DATA_DIR" --file-limit "$FILE_LIMIT" --world-size "$WORLD_SIZE" \
  --batch-size "$BATCH_SIZE" --run-dir "$RUN_DIR" --partitions "$PARTITIONS" \
  --inference-unit "$INFERENCE_UNIT"
srun --ntasks="$WORLD_SIZE" --cpus-per-task="$CPUS_PER_WORKER" --cpu-bind=cores \
  --export="ALL,OMP_NUM_THREADS=$CPUS_PER_WORKER" \
  apptainer exec --bind "$COSMIC_ROOT" --bind "$(dirname "$RUN_DIR")" "$SIF_IMAGE" \
  "$CONTAINER_PYTHON" "$SCRIPTS/cosmic_classic.py" worker --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH" --code-dir "$CODE_DIR"
in_container summarize --run-dir "$RUN_DIR" --world-size "$WORLD_SIZE"
t_end=$(date +%s.%N)

end_to_end_s=$(awk -v a="$t_start" -v b="$t_end" 'BEGIN { printf "%.3f", b - a }')
printf '{"job_id": "%s", "world_size": %d, "nodes": %d, "cpus_per_worker": %d, "file_limit": %d, "partitions": %d, "inference_unit": "%s", "data_gb": %s, "batch_size": %d, "t_start": %s, "t_end": %s, "end_to_end_s": %s}\n' \
  "${SLURM_JOB_ID:-}" "$WORLD_SIZE" "${SLURM_JOB_NUM_NODES:-1}" "$CPUS_PER_WORKER" "$FILE_LIMIT" "$PARTITIONS" \
  "$INFERENCE_UNIT" "$DATA_GB" "$BATCH_SIZE" "$t_start" "$t_end" "$end_to_end_s" > "$RUN_DIR/timing.json"
echo "end_to_end_s=$end_to_end_s"