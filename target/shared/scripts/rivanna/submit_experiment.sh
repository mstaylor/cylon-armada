#!/bin/bash
# =============================================================================
# Cylon-Armada Experiment — Rivanna HPC Slurm submission script
#
# Mirrors the ECS experiment pattern: reads all configuration from
# environment variables, runs armada_ecs_runner.py, and writes results
# to S3 (or a local path if RESULTS_BUCKET is not set).
#
# File naming follows the cylon_init.py pattern — RESULTS_DIR and
# EXPERIMENT_NAME are format strings that accept {scaling} and {world_size}
# substitutions. armada_ecs_runner.py performs the substitution at runtime.
#
# Usage — weak scaling sweep:
#   for ws in 1 2 4 8 16; do
#     sbatch \
#       --ntasks=1 --cpus-per-task=$ws \
#       --export=ALL,\
#   SCALING=weak,WORLD_SIZE=$ws,\
#   WORKFLOW_ID=rivanna_weak_$(date +%s),\
#   TASKS_JSON='["task one","task two","task three","task four"]',\
#   RESULTS_DIR="results/rivanna/{scaling}/",\
#   EXPERIMENT_NAME="rivanna_{scaling}_ws{world_size}",\
#   RESULTS_BUCKET=my-results-bucket,\
#   COMPUTE_PLATFORM=rivanna \
#       submit_experiment.sh
#   done
#
# Usage — strong scaling sweep:
#   for ws in 1 2 4 8 16; do
#     sbatch \
#       --ntasks=1 --cpus-per-task=$ws \
#       --export=ALL,\
#   SCALING=strong,WORLD_SIZE=$ws,\
#   WORKFLOW_ID=rivanna_strong_$(date +%s),\
#   TASKS_JSON='["task one","task two","task three","task four"]',\
#   RESULTS_DIR="results/rivanna/{scaling}/",\
#   EXPERIMENT_NAME="rivanna_{scaling}_ws{world_size}",\
#   RESULTS_BUCKET=my-results-bucket,\
#   COMPUTE_PLATFORM=rivanna \
#       submit_experiment.sh
#   done
#
# Required environment variables (set via --export or module defaults):
#   WORKFLOW_ID       unique run identifier
#   TASKS_JSON        JSON array of task description strings
#   SCALING           "weak" or "strong"
#   WORLD_SIZE        number of parallel worker threads (match --cpus-per-task)
#   RESULTS_DIR       S3 key directory; supports {scaling} placeholder
#   EXPERIMENT_NAME   file name stem; supports {scaling} and {world_size}
#   RESULTS_BUCKET    S3 bucket for output (omit to write local files only)
#   COMPUTE_PLATFORM  label written into results (default: rivanna)
#
# Optional:
#   S3_SCRIPTS_BUCKET / S3_SCRIPTS_PREFIX   hot-load scripts from S3
#   REDIS_HOST / REDIS_PORT                 context store
#   CONTEXT_BACKEND                         "redis" or "cylon"
#   SIMD_BACKEND                            "numpy", "cython", or "pycylon"
#   LOG_LEVEL                               "DEBUG", "INFO", etc.
# =============================================================================

#SBATCH --job-name=cylon-armada
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/armada_%j.out
#SBATCH --error=logs/armada_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module purge
module load anaconda

# Activate the cylon_dev conda environment (built by Dockerfile.gpu or
# installed on Rivanna via the build scripts in docs/rivanna_setup.md).
conda activate cylon_dev

# Default COMPUTE_PLATFORM to "rivanna" if not set by the caller
export COMPUTE_PLATFORM="${COMPUTE_PLATFORM:-rivanna}"

# ---------------------------------------------------------------------------
# Derived paths — adjust CYLON_ARMADA_HOME to your allocation's scratch path
# ---------------------------------------------------------------------------

CYLON_ARMADA_HOME="${CYLON_ARMADA_HOME:-/scratch/$USER/cylon-armada}"
SHARED_SCRIPTS_PATH="${SHARED_SCRIPTS_PATH:-${CYLON_ARMADA_HOME}/target/shared/scripts}"
RUNNER="${CYLON_ARMADA_HOME}/target/aws/scripts/lambda/python/armada_ecs_runner.py"

export SHARED_SCRIPTS_PATH
export PYTHONPATH="${CYLON_ARMADA_HOME}/python:${SHARED_SCRIPTS_PATH}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------

: "${TASKS_JSON:?TASKS_JSON must be set (JSON array of task strings)}"
: "${SCALING:?SCALING must be set to weak or strong}"
: "${WORLD_SIZE:?WORLD_SIZE must be set}"

echo "=== cylon-armada Rivanna experiment ==="
echo "  WORKFLOW_ID     : ${WORKFLOW_ID:-auto}"
echo "  SCALING         : $SCALING"
echo "  WORLD_SIZE      : $WORLD_SIZE"
echo "  RESULTS_DIR     : ${RESULTS_DIR:-results/{scaling}/}"
echo "  EXPERIMENT_NAME : ${EXPERIMENT_NAME:-rivanna_{scaling}_ws{world_size}}"
echo "  RESULTS_BUCKET  : ${RESULTS_BUCKET:-(local only)}"
echo "  COMPUTE_PLATFORM: $COMPUTE_PLATFORM"
echo "  SLURM_JOB_ID    : ${SLURM_JOB_ID:-N/A}"
echo "======================================="

mkdir -p logs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

python "$RUNNER"