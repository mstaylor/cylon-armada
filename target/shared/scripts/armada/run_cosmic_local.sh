#!/bin/bash
#
# Cosmic AI agentic pipeline — local multi-rank launcher (SP1 Task 6).
#
# Spawns N rank processes locally, each running one rank of the compiled
# six-operator workflow (Preprocess | Embed | Retrieve | Reason | Bind |
# MemoryUpsert) over the FMI direct-redis channel. Redis is the out-of-band
# coordinator only; the collectives themselves run over direct TCP.
#
# Services are MOCKED unless --live is passed. --live issues one embedding call
# and one LLM call per galaxy on every rank, so it costs real money — dry-run
# without it first.
#
# Usage:
#   run_cosmic_local.sh <world_size> [redis_addr] [extra run_cosmic_local.py args...]
# Examples:
#   run_cosmic_local.sh 4
#   run_cosmic_local.sh 8 10.211.55.2:6379 --galaxies 32
#   run_cosmic_local.sh 4 10.211.55.2:6379 --live --dimensions 256
#
# Env overrides:
#   CYLON_HOME     (default /home/parallels/cylon)          — pycylon in-place build
#   ARMADA_SCRIPTS (default <repo>/target/shared/scripts)
#   CONDA_ENV      (default cylon_dev)

set -u  # NOT -e: a non-zero rank exit during teardown should not mask results.

WORLD_SIZE="${1:?usage: run_cosmic_local.sh <world_size> [redis_addr] [args...]}"
shift
# Redis runs on the Parallels HOST, not in this guest VM (see CLAUDE.md).
# The optional second positional is redis_addr; anything starting with '-' is
# already a flag for the module, so don't swallow it as an address.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    REDIS_ADDR="$1"
    shift
else
    REDIS_ADDR="${REDIS_ADDR:-10.211.55.2:6379}"
fi

CYLON_HOME="${CYLON_HOME:-/home/parallels/cylon}"
CONDA_ENV="${CONDA_ENV:-cylon_dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../armada
ARMADA_SCRIPTS="${ARMADA_SCRIPTS:-$(cd "$SCRIPT_DIR/.." && pwd)}"    # .../shared/scripts
# <repo>/python holds the compiled cylon_armada extension (dag_compiler, ContextTable);
# libcylon_armada.so itself comes from the conda env on LD_LIBRARY_PATH below.
ARMADA_PYTHON="${ARMADA_PYTHON:-$(cd "$ARMADA_SCRIPTS/../../.." && pwd)/python}"

# Never run with armada/ as the working directory: Python puts cwd on sys.path,
# and armada/operator.py would then shadow the stdlib `operator` module, which
# breaks `collections` and therefore almost every import. Running from the
# scripts root keeps `armada` a package and `operator` the stdlib one.
cd "$ARMADA_SCRIPTS" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CYLON_HOME/python/pycylon:$ARMADA_SCRIPTS:$ARMADA_PYTHON:${PYTHONPATH:-}"

REDIS_HOST="${REDIS_ADDR%%:*}"
REDIS_PORT="${REDIS_ADDR##*:}"
export REDIS_HOST REDIS_PORT

# comm_name must be unique per run — a reused one lets the Redis INCR rank
# counter carry over, so a later run's workers get ranks >= world_size and hang
# at the rendezvous (the Experiment B convention).
export COMM_NAME="cosmic_local_$$_$(date +%s 2>/dev/null || echo run)"

echo "checking redis at $REDIS_ADDR"
python - <<'PYEOF' || { echo "ERROR: redis unreachable (it runs on the Parallels host ~10.211.55.2:6379)"; exit 1; }
import os, sys, redis
try:
    redis.Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]),
                socket_connect_timeout=3).ping()
    print("  redis OK")
except Exception as e:
    print(f"  ERROR: cannot reach redis: {e}")
    sys.exit(1)
PYEOF

LOG_DIR="$(mktemp -d)"
PORT_BASE=$(( 21000 + ($$ % 20000) ))
echo "world_size=$WORLD_SIZE  comm_name=$COMM_NAME  logs=$LOG_DIR"
echo "args: $*"

pids=()
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    RANK="$r" \
    WORLD_SIZE="$WORLD_SIZE" \
    FMI_LISTEN_PORT=$((PORT_BASE + r)) \
    ADVERTISE_HOST="${ADVERTISE_HOST:-127.0.0.1}" \
    RESULT_PATH="$LOG_DIR/result_$r.json" \
    python -m armada.run_cosmic_local "$@" \
        >"$LOG_DIR/rank_$r.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=$?
done

echo "=== rank 0 log tail ==="
tail -n 15 "$LOG_DIR/rank_0.log" 2>/dev/null

echo "=== per-rank results ==="
python - "$LOG_DIR" <<'PYEOF'
import glob, json, os, sys
d = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(d, "result_*.json"))):
    try:
        rows.append(json.load(open(path)))
    except Exception:
        pass
if not rows:
    print("  (no result files — see logs)")
else:
    for r in sorted(rows, key=lambda x: x.get("rank", -1)):
        print(f"  rank {r.get('rank'):>3}  establish={r.get('establish_s'):>7}s "
              f"run={r.get('run_s'):>7}s  written={r.get('records_written', '-')}")
PYEOF

echo "=== failures across ranks ==="
grep -hiE "error|traceback|exception|FAIL|Segmentation" "$LOG_DIR"/rank_*.log 2>/dev/null \
    | grep -viE "UCP API version|UCX  WARN|records_failed" | head -20 || echo "  (none)"

echo "launcher exit rc=$rc"
echo "logs: $LOG_DIR"