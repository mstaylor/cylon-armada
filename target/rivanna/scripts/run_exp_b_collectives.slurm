#!/bin/bash
# =============================================================================
# Experiment B — collective benchmark Slurm submission script (Rivanna HPC)
#
# Real multi-rank fan-out: --ntasks=N launches N separate processes across the
# SLURM allocation; each runs exp_b_collectives.py --channel ucc, and UCC's
# redis-OOB context (NOT MPI, NOT rendezvous — see docs/EXP_B_DESIGN.md) self
# assigns each process its rank via the redis INCR at REDIS_HOST:REDIS_PORT.
# Mirrors cylon/target/rivanna/scripts/scaling_job.slurm's `srun -n N python
# script.py` pattern, using this project's established Apptainer bind-mount
# conventions (armada-experiment-setup.py's make_slurm_script).
#
# Usage — multi-rank smoke test:
#   sbatch --ntasks=4 \
#     --export=ALL,SIF_IMAGE=/scratch/$USER/cylon-armada/cylon-armada-uccucx.sif \
#     run_exp_b_collectives.slurm
#
# Usage — N sweep:
#   for ws in 4 8 16 32 64; do
#     sbatch --ntasks=$ws \
#       --export=ALL,SIF_IMAGE=/scratch/$USER/cylon-armada/cylon-armada-uccucx.sif,\
# COLLECTIVES="scatter scatterv gather allgather reduce broadcast allreduce barrier",\
# MSG_SIZES="512 4096 32768",WARMUP=3,REPS=20,RUNS=4 \
#       run_exp_b_collectives.slurm
#   done
#
# Required:
#   SIF_IMAGE          path to the cylon-armada-uccucx .sif (built from
#                       Dockerfile.uccucx.python)
#
# Optional (defaults are a fast connectivity smoke test):
#   CHANNEL             collective channel (default: ucc)
#   REDIS_HOST          UCC redis-OOB coordinator (default: dev-cylon-redis1.aws-cylondata.com)
#   REDIS_PORT          (default: 6379)
#   COLLECTIVES         space-separated collective names (default: scatter reduce broadcast barrier)
#   MSG_SIZES           space-separated byte sizes (default: 512 4096)
#   WARMUP / REPS / RUNS  (defaults: 2 / 4 / 1)
#   RESULTS_DIR_HOST    host path for results, bind-mounted to /output
#                       (default: /scratch/$USER/expb-results)
# =============================================================================

#SBATCH --job-name=expb-collectives
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/expb-%j.out
#SBATCH --error=logs/expb-%j.err

set -euo pipefail

: "${SIF_IMAGE:?SIF_IMAGE must be set (path to the cylon-armada-uccucx .sif)}"

CHANNEL="${CHANNEL:-ucc}"
REDIS_HOST="${REDIS_HOST:-dev-cylon-redis1.aws-cylondata.com}"
REDIS_PORT="${REDIS_PORT:-6379}"
COLLECTIVES="${COLLECTIVES:-scatter reduce broadcast barrier}"
MSG_SIZES="${MSG_SIZES:-512 4096}"
WARMUP="${WARMUP:-2}"
REPS="${REPS:-4}"
RUNS="${RUNS:-1}"
RESULTS_DIR_HOST="${RESULTS_DIR_HOST:-/scratch/$USER/expb-results}"

WORLD_SIZE="$SLURM_NTASKS"

module load apptainer

echo "==============================================================="
echo "Experiment B collectives — SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "  world_size  : $WORLD_SIZE"
echo "  channel     : $CHANNEL"
echo "  redis       : $REDIS_HOST:$REDIS_PORT"
echo "  collectives : $COLLECTIVES"
echo "  msg_sizes   : $MSG_SIZES"
echo "  warmup/reps/runs: $WARMUP/$REPS/$RUNS"
echo "  sif         : $SIF_IMAGE"
echo "==============================================================="

mkdir -p logs "$RESULTS_DIR_HOST"

JOB_TMP=/scratch/${USER}/tmp_${SLURM_JOB_ID}
mkdir -p "$JOB_TMP"

srun --exact apptainer exec \
    --bind "${RESULTS_DIR_HOST}:/output" \
    --bind "${SLURM_SUBMIT_DIR}:/rivanna" \
    --bind "${JOB_TMP}:/tmp" \
    --containall \
    "$SIF_IMAGE" \
    /bin/bash /rivanna/runExpBCollectives.sh \
        --channel "$CHANNEL" \
        --world-size "$WORLD_SIZE" \
        --redis-addr "${REDIS_HOST}:${REDIS_PORT}" \
        --collectives $COLLECTIVES \
        --msg-sizes $MSG_SIZES \
        --warmup "$WARMUP" \
        --reps "$REPS" \
        --runs "$RUNS" \
        --output /output

rm -rf "$JOB_TMP"
echo "==============================================================="
echo "Results: ${RESULTS_DIR_HOST}"