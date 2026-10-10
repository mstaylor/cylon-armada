#!/bin/bash
#
# Launch N Cosmic AI planning agents locally, one FMI rank each, sharing their plan cache
# through Armada's AllGather over the direct-redis channel.
#
# Usage:
#   run_agents.sh <world_size> [redis_addr] --requests R.json --out-dir DIR --cache-mode MODE --similarity-threshold T

set -u

WORLD_SIZE="${1:?usage: run_agents.sh <world_size> [redis_addr] [agent args...]}"
shift
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    REDIS_ADDR="$1"
    shift
else
    REDIS_ADDR="${REDIS_ADDR:-10.211.55.2:6379}"
fi

CYLON_HOME="${CYLON_HOME:-/home/parallels/cylon}"
CONDA_ENV="${CONDA_ENV:-cylon_dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMADA_SCRIPTS="${ARMADA_SCRIPTS:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ARMADA_PYTHON="${ARMADA_PYTHON:-$(cd "$ARMADA_SCRIPTS/../../.." && pwd)/python}"

cd "$ARMADA_SCRIPTS" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CYLON_HOME/python/pycylon:$ARMADA_SCRIPTS:$ARMADA_PYTHON:${PYTHONPATH:-}"

REDIS_HOST="${REDIS_ADDR%%:*}"
REDIS_PORT="${REDIS_ADDR##*:}"
export REDIS_HOST REDIS_PORT
export FMI_CHANNEL_TYPE="${FMI_CHANNEL_TYPE:-direct-redis}"
export COMM_NAME="cosmic_agents_$$_$(date +%s)"

python - <<'PYEOF' || { echo "ERROR: redis unreachable at $REDIS_ADDR"; exit 1; }
import os, sys, redis
try:
    redis.Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_connect_timeout=3).ping()
except Exception as e:
    print(f"cannot reach redis: {e}")
    sys.exit(1)
PYEOF

OUT_DIR=""
prev=""
for arg in "$@"; do
    [ "$prev" = "--out-dir" ] && OUT_DIR="$arg"
    prev="$arg"
done
if [ -n "$OUT_DIR" ] && compgen -G "$OUT_DIR/plans_*.jsonl" >/dev/null; then
    echo "ERROR: $OUT_DIR already holds plans_*.jsonl from an earlier run; use an empty --out-dir"
    exit 1
fi

LOG_DIR="$(mktemp -d)"
PORT_BASE=$(( 21000 + ($$ % 20000) ))
echo "agents=$WORLD_SIZE  comm_name=$COMM_NAME  logs=$LOG_DIR"

pids=()
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    RANK="$r" \
    WORLD_SIZE="$WORLD_SIZE" \
    FMI_LISTEN_PORT=$((PORT_BASE + r)) \
    ADVERTISE_HOST="${ADVERTISE_HOST:-127.0.0.1}" \
    python -m cosmic_campaign.agent "$@" >"$LOG_DIR/agent_$r.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=$?
done

grep -hiE "error|traceback|exception" "$LOG_DIR"/agent_*.log 2>/dev/null | head -20
echo "agents exit rc=$rc  logs: $LOG_DIR"
exit $rc