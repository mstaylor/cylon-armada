#!/usr/bin/env bash
# Submit the classic Cosmic AI grid on Vista. Each job waits for the previous one,
# so executions never overlap.
#
#   ACCOUNT=TG-NAIRR260130 COSMIC_ROOT=$SCRATCH/cosmic \
#   SIF_IMAGE=$SCRATCH/cosmic/cosmic-ai-lambda_arm64.sif ./sweep_cosmic_classic.sh [--dry-run]
#
# Optional: WORLD_SIZES ("1 2 4 8 16 41"), RUNS (3), WARMUP_RUNS (1), CPUS_PER_WORKER (6),
# CORES_PER_NODE (144), FILE_LIMIT (41), BATCH_SIZE (512), PARTITION (gg), TIME_LIMIT (01:00:00),
# RESULTS_ROOT ($COSMIC_ROOT/results/classic/<timestamp>), DATA_DIR, INFERENCE_UNIT (rank or partition).
#
# GPU runs: DEVICE=cuda PARTITION=gh CORES_PER_NODE=72 with a CUDA image as SIF_IMAGE. Their
# results default to $COSMIC_ROOT/results/gpu/<timestamp>, apart from the CPU results.
#
# Data size sweep: set DATA_SIZES_GB (e.g. "12.6 100 256 512 768 1000") and PARTITION_MB. Each size
# runs ceil(GB * 1024 / PARTITION_MB) partitions, reusing the FILE_LIMIT files cyclically.
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

: "${ACCOUNT:?ACCOUNT must be set to the allocation to charge}"
: "${COSMIC_ROOT:?COSMIC_ROOT must be set}"
: "${SIF_IMAGE:?SIF_IMAGE must be set}"
read -r -a SIZES <<< "${WORLD_SIZES:-1 2 4 8 16 41}"
read -r -a DATA_SIZES <<< "${DATA_SIZES_GB:-}"
if (( ${#DATA_SIZES[@]} )); then
  : "${PARTITION_MB:?PARTITION_MB must be set with DATA_SIZES_GB}"
else
  DATA_SIZES=("")
fi
RUNS=${RUNS:-3}
WARMUP_RUNS=${WARMUP_RUNS:-1}
CPUS_PER_WORKER=${CPUS_PER_WORKER:-6}
CORES_PER_NODE=${CORES_PER_NODE:-144}
FILE_LIMIT=${FILE_LIMIT:-41}
BATCH_SIZE=${BATCH_SIZE:-512}
PARTITION=${PARTITION:-gg}
TIME_LIMIT=${TIME_LIMIT:-01:00:00}
INFERENCE_UNIT=${INFERENCE_UNIT:-rank}
DEVICE=${DEVICE:-cpu}
case "$DEVICE" in
  cpu) results_group=classic ;;
  cuda) results_group=gpu ;;
  *) echo "DEVICE must be cpu or cuda, got $DEVICE" >&2; exit 1 ;;
esac
RESULTS_ROOT=${RESULTS_ROOT:-$COSMIC_ROOT/results/$results_group/$(date +%Y%m%d-%H%M%S)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKERS_PER_NODE=$(( CORES_PER_NODE / CPUS_PER_WORKER ))

previous=""
submitted=0
for n in "${SIZES[@]}"; do
  nodes=$(( (n + WORKERS_PER_NODE - 1) / WORKERS_PER_NODE ))
  for gb in "${DATA_SIZES[@]}"; do
    if [[ -n "$gb" ]]; then
      partitions=$(awk -v gb="$gb" -v mb="$PARTITION_MB" 'BEGIN { p = gb * 1024 / mb; c = int(p); if (c < p) c++; print c }')
      cell="ws$n/gb$gb"
      tag="$DEVICE-ws$n-gb$gb"
      data_gb=$gb
    else
      partitions=$FILE_LIMIT
      cell="ws$n"
      tag="$DEVICE-ws$n"
      data_gb=null
    fi
    labels=()
    for ((i = 1; i <= WARMUP_RUNS; i++)); do labels+=("warmup$i"); done
    for ((i = 1; i <= RUNS; i++)); do labels+=("run$i"); done
    for label in "${labels[@]}"; do
      run_dir="$RESULTS_ROOT/$cell/$label"
      exports="ALL,WORLD_SIZE=$n,FILE_LIMIT=$FILE_LIMIT,PARTITIONS=$partitions,INFERENCE_UNIT=$INFERENCE_UNIT,DEVICE=$DEVICE,DATA_GB=$data_gb,BATCH_SIZE=$BATCH_SIZE,CPUS_PER_WORKER=$CPUS_PER_WORKER,RUN_DIR=$run_dir,SIF_IMAGE=$SIF_IMAGE,COSMIC_ROOT=$COSMIC_ROOT"
      if [[ -n "${DATA_DIR:-}" ]]; then
        exports+=",DATA_DIR=$DATA_DIR"
      fi
      args=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --nodes="$nodes" --ntasks="$n"
            --cpus-per-task="$CPUS_PER_WORKER" --time="$TIME_LIMIT" --job-name="cosmic-$tag-$label"
            --output="$RESULTS_ROOT/logs/$tag-$label-%j.out")
      if [[ -n "$previous" ]]; then
        args+=(--dependency="afterany:$previous")
      fi
      args+=(--export="$exports" "$HERE/cosmic_classic.slurm")
      if (( DRY_RUN )); then
        echo "sbatch ${args[*]}"
        previous="dryrun$submitted"
      else
        mkdir -p "$RESULTS_ROOT/logs"
        output=$(sbatch "${args[@]}")
        previous=$(printf '%s\n' "$output" | { grep -E '^[0-9]+(;[^ ]*)?$' || true; } | tail -n 1 | cut -d';' -f1)
        if [[ -z "$previous" ]]; then
          echo "sbatch returned no job id for $tag $label:" >&2
          printf '%s\n' "$output" >&2
          exit 1
        fi
        echo "submitted $previous $tag $label"
      fi
      submitted=$((submitted + 1))
    done
  done
done
echo "results: $RESULTS_ROOT"