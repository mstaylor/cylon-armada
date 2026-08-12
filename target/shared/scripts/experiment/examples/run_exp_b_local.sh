#!/bin/bash
#
# Experiment B — local multi-rank launcher for the collective benchmark (UCC channel).
#
# Spawns N rank processes locally (like python/pycylon/run_ucc_with_redis.py): each
# process builds a UCC communicator via the redis-OOB context (redis INCR assigns
# ranks), runs the collective sweep, and rank 0 writes the per-run CSV. Requires a
# running redis (the OOB coordinator) — start one with `redis-server` if needed.
#
# Usage:
#   run_exp_b_local.sh <world_size> [redis_addr] [extra exp_b args...]
# Example:
#   run_exp_b_local.sh 4 127.0.0.1:6379 --collectives scatter reduce --msg-sizes 512 4096 --reps 10
#
# Env overrides:
#   UCX_HOME   (default /home/parallels/ucx/install)   — from-source UCX install
#   UCC_HOME   (default /home/parallels/ucc/install)   — from-source UCC install
#   CYLON_HOME (default /home/parallels/cylon)          — pycylon in-place build
#   ARMADA_SCRIPTS (default <repo>/target/shared/scripts)

set -u  # NOT -e: a non-zero rank exit at Finalize/teardown is a known artifact.

WORLD_SIZE="${1:?usage: run_exp_b_local.sh <world_size> [redis_addr] [args...]}"
# Redis runs on the Parallels HOST, not in this guest VM. Reach it via the host IP
# (find with `ip neighbor show`; typically 10.211.55.2:6379). Override as arg 2.
REDIS_ADDR="${2:-${REDIS_ADDR:-10.211.55.2:6379}}"
shift || true
shift 2>/dev/null || true   # drop world_size + redis_addr; remaining $@ are exp_b args

UCX_HOME="${UCX_HOME:-/home/parallels/ucx/install}"
UCC_HOME="${UCC_HOME:-/home/parallels/ucc/install}"
CYLON_HOME="${CYLON_HOME:-/home/parallels/cylon}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # .../experiment
ARMADA_SCRIPTS="${ARMADA_SCRIPTS:-$(cd "$SCRIPT_DIR/.." && pwd)}"       # .../shared/scripts

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cylon_dev

# From-source UCX/UCC FIRST on the loader path (the SP1 lib-order lesson), then conda.
# libcylon/UCC RUNPATH already point at $UCX_HOME first, so this just reinforces it.
export LD_LIBRARY_PATH="$UCX_HOME/lib:$UCC_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# LD_PRELOAD of the from-source UCX is a CI-only fix (there, openmpi's mca_pml_ucx.so
# DT_RPATH would otherwise force conda's UCX under mpirun). This launcher runs plain
# processes (no mpirun), so RUNPATH already resolves UCX to $UCX_HOME — preloading is
# redundant and, if $UCX_HOME is an older build than what UCC expects, forces a UCX
# version mismatch warning. Off by default; set EXPB_UCX_PRELOAD=1 to force it.
if [ "${EXPB_UCX_PRELOAD:-0}" = "1" ]; then
    export LD_PRELOAD="$UCX_HOME/lib/libucm.so.0:$UCX_HOME/lib/libucs.so.0:$UCX_HOME/lib/libuct.so.0:$UCX_HOME/lib/libucp.so.0"
fi
# pycylon (in-place build, with the SP1 collective bindings) + the shared scripts.
# The pycylon package lives at $CYLON_HOME/python/pycylon/pycylon, so its parent
# ($CYLON_HOME/python/pycylon) is what goes on PYTHONPATH.
export PYTHONPATH="$CYLON_HOME/python/pycylon:$ARMADA_SCRIPTS:${PYTHONPATH:-}"

# Redis-OOB coordinates ranks for UCC.
export CYLON_UCX_OOB_WORLD_SIZE="$WORLD_SIZE"
export CYLON_UCX_OOB_REDIS_ADDR="$REDIS_ADDR"
# Fresh session id per launch so redis keys never collide across runs.
export CYLON_SESSION_ID="exp_b_$$_$(date +%s 2>/dev/null || echo run)"

REDIS_HOST="${REDIS_ADDR%%:*}"
REDIS_PORT="${REDIS_ADDR##*:}"
# Use the python redis client (cylon_dev has it; redis-cli is not installed, and
# redis-server is on the Parallels host — see CLAUDE.md). Ping + flush stale OOB keys.
echo "checking + flushing redis at $REDIS_ADDR"
REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" python - <<'PYEOF' || {
import os, sys, redis
try:
    r = redis.Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]),
                    socket_connect_timeout=3)
    r.ping(); r.flushall()
    print("  redis OK, flushed")
except Exception as e:
    print(f"  ERROR: cannot reach redis at {os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}: {e}")
    sys.exit(1)
PYEOF
    echo "ERROR: redis unreachable (it runs on the Parallels host ~10.211.55.2:6379)"; exit 1; }

LOG_DIR="$(mktemp -d)"
echo "world_size=$WORLD_SIZE  session=$CYLON_SESSION_ID  logs=$LOG_DIR"
echo "exp_b args: $*"

pids=()
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    RANK="$r" python -m experiment.exp_b_collectives \
        --channel ucc --world-size "$WORLD_SIZE" --redis-addr "$REDIS_ADDR" "$@" \
        >"$LOG_DIR/rank_$r.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=$?
done

echo "=== rank 0 log tail ==="
tail -n 20 "$LOG_DIR/rank_0.log" 2>/dev/null
echo "=== any failures across ranks (excluding the harmless UCX version warning) ==="
grep -hiE "error|traceback|exception|FAIL|Segmentation|before MPI_INIT" "$LOG_DIR"/rank_*.log 2>/dev/null \
    | grep -viE "UCP API version|UCX  WARN" | head -20 || echo "  (none)"
# The Cylon C++ Finalize fix makes teardown clean, so a non-zero rc now signals a
# real failure — no longer the tolerated MPI_Barrier teardown artifact.
echo "launcher exit rc=$rc"
echo "logs: $LOG_DIR"