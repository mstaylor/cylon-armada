#!/usr/bin/env bash
##
# WASM setup — one-time local dev setup for the Node.js WASM backend.
#
# Creates the cylon_host stub required for local testing outside Lambda.
# In Lambda the real cylon_host is provided by the runtime; locally we
# stub it with no-op FMI functions (rank=0, world_size=1).
#
# Prerequisites:
#   - cylon-wasm built: wasm-pack build --target nodejs --release
#     (run from ~/cylon/rust/cylon-wasm)
#
# Usage:
#   ./setup_wasm.sh
#   ./setup_wasm.sh --wasm-pkg /path/to/custom/pkg
##

set -euo pipefail

# Defaults — override via env vars or flags
: "${CYLON_WASM_PKG:=${HOME}/cylon/rust/cylon-wasm/pkg}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wasm-pkg) CYLON_WASM_PKG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

STUB_DIR="${CYLON_WASM_PKG}/node_modules/cylon_host"

echo "=== WASM Local Setup ==="
echo "WASM pkg:   ${CYLON_WASM_PKG}"
echo "Stub dir:   ${STUB_DIR}"
echo ""

# Verify wasm build exists
if [[ ! -f "${CYLON_WASM_PKG}/cylon_wasm_bg.wasm" ]]; then
    echo "ERROR: cylon_wasm_bg.wasm not found in ${CYLON_WASM_PKG}"
    echo "Build it first: wasm-pack build --target nodejs --release"
    exit 1
fi

# Create cylon_host stub
mkdir -p "${STUB_DIR}"

cat > "${STUB_DIR}/index.js" << 'STUB'
// cylon_host stub for local testing — no-op FMI (rank=0, world_size=1).
// In Lambda the real cylon_host is injected by the runtime.
module.exports = {
    host_get_rank:     () => 0,
    host_get_world_size: () => 1,
    host_barrier:      () => {},
    host_broadcast:    (p, l, r) => l,
    host_all_to_all:   (p, l, r) => l,
    host_gather:       (p, l, r, o) => l,
    host_scatter:      (p, l, r, o) => l,
    host_all_gather:   (p, l, o) => l,
};
STUB

# TypeScript declaration for completeness
cat > "${STUB_DIR}/index.d.ts" << 'DTS'
export declare function host_get_rank(): number;
export declare function host_get_world_size(): number;
export declare function host_barrier(): void;
export declare function host_broadcast(ptr: number, len: number, root: number): number;
export declare function host_all_to_all(ptr: number, len: number, root: number): number;
export declare function host_gather(ptr: number, len: number, root: number, out: number): number;
export declare function host_scatter(ptr: number, len: number, root: number, out: number): number;
export declare function host_all_gather(ptr: number, len: number, out: number): number;
DTS

echo "cylon_host stub created: ${STUB_DIR}/index.js"
echo ""
echo "=== Verify ==="
node -e "
const host = require('${STUB_DIR}/index.js');
console.log('rank:', host.host_get_rank());
console.log('world_size:', host.host_get_world_size());
console.log('stub OK');
"