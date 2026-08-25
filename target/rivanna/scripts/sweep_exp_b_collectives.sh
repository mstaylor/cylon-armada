#!/bin/bash
# =============================================================================
# Experiment B — collective benchmark N-sweep (Rivanna HPC)
#
# Fires one `sbatch --ntasks=N run_exp_b_collectives.slurm` per world size,
# with the full collective/message-size grid and warmed runs=4. Each N's
# results land in their own subdirectory (RESULTS_DIR_HOST/wsN/) since the
# per-run filenames (run1_..run4_exp_b_collectives_results.csv) aren't
# N-qualified — without separate subdirs, a later N would overwrite an
# earlier N's identically-named files.
#
# Usage:
#   ./sweep_exp_b_collectives.sh [--dry-run] [world_size ...]
#
#   SIF_IMAGE=/scratch/$USER/cylon-armada/cylon-armada-uccucx.sif \
#     ./sweep_exp_b_collectives.sh --dry-run
#
#   SIF_IMAGE=/scratch/$USER/cylon-armada/cylon-armada-uccucx.sif \
#     ./sweep_exp_b_collectives.sh 1 2 4 8 16 32 64
#
# Required:
#   SIF_IMAGE   path to the cylon-armada-uccucx .sif
#
# Optional (same knobs as run_exp_b_collectives.slurm; defaults are the full
# Exp B grid, not the smoke-test subset):
#   CHANNEL, REDIS_HOST, REDIS_PORT, COLLECTIVES, MSG_SIZES,
#   WARMUP, REPS, RUNS, RESULTS_DIR_HOST (the base dir; ws<N>/ is appended per N)
# =============================================================================

set -euo pipefail

DRY_RUN=0
WORLD_SIZES=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        WORLD_SIZES+=("$arg")
    fi
done
if [ ${#WORLD_SIZES[@]} -eq 0 ]; then
    WORLD_SIZES=(1 2 4 8 16 32 64)
fi

: "${SIF_IMAGE:?SIF_IMAGE must be set (path to the cylon-armada-uccucx .sif)}"

CHANNEL="${CHANNEL:-ucc}"
REDIS_HOST="${REDIS_HOST:-dev-cylon-redis1.aws-cylondata.com}"
REDIS_PORT="${REDIS_PORT:-6379}"
COLLECTIVES="${COLLECTIVES:-scatter scatterv gather allgather reduce broadcast allreduce barrier}"
MSG_SIZES="${MSG_SIZES:-8 64 512 4096 32768 262144 1048576}"
WARMUP="${WARMUP:-3}"
REPS="${REPS:-20}"
RUNS="${RUNS:-4}"
RESULTS_BASE="${RESULTS_DIR_HOST:-/scratch/$USER/expb-results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================================="
echo "Experiment B collective sweep"
echo "  world_sizes : ${WORLD_SIZES[*]}"
echo "  channel     : $CHANNEL"
echo "  redis       : $REDIS_HOST:$REDIS_PORT"
echo "  collectives : $COLLECTIVES"
echo "  msg_sizes   : $MSG_SIZES"
echo "  warmup/reps/runs: $WARMUP/$REPS/$RUNS"
echo "  sif         : $SIF_IMAGE"
echo "  results base: $RESULTS_BASE"
echo "  dry_run     : $DRY_RUN"
echo "==============================================================="

for ws in "${WORLD_SIZES[@]}"; do
    ws_results="${RESULTS_BASE}/ws${ws}"
    export_str="ALL,SIF_IMAGE=${SIF_IMAGE},CHANNEL=${CHANNEL},REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},COLLECTIVES=${COLLECTIVES},MSG_SIZES=${MSG_SIZES},WARMUP=${WARMUP},REPS=${REPS},RUNS=${RUNS},RESULTS_DIR_HOST=${ws_results}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] sbatch --ntasks=$ws --export=$export_str $SCRIPT_DIR/run_exp_b_collectives.slurm"
    else
        mkdir -p "$ws_results"
        echo "submitting world_size=$ws -> $ws_results"
        sbatch --ntasks="$ws" --export="$export_str" "$SCRIPT_DIR/run_exp_b_collectives.slurm"
    fi
done

echo "==============================================================="
echo "Done. Check queue: squeue --me"
echo "Results land under: $RESULTS_BASE/ws<N>/ (one subdir per world size)"