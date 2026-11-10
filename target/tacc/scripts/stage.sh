#!/usr/bin/env bash
# Copy the partitions, the AstroMAE code and weights, and the scripts the jobs run to
# Vista's $SCRATCH/cosmic. Transfers run one at a time.
#
#   ./stage.sh LOCAL_STAGE_DIR      LOCAL_STAGE_DIR holds data/ and code/
#
# Optional: VISTA_HOST (vista), COSMIC_ROOT (remote $SCRATCH/cosmic).
set -euo pipefail

STAGE=${1:?usage: stage.sh LOCAL_STAGE_DIR}
HOST=${VISTA_HOST:-vista}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
REMOTE_ROOT=${COSMIC_ROOT:-$(ssh "$HOST" 'echo $SCRATCH')/cosmic}
REMOTE_REPO=$REMOTE_ROOT/cylon-armada/target

ssh "$HOST" "mkdir -p '$REMOTE_REPO/tacc' '$REMOTE_REPO/aws/scripts/terraform-cosmic-ai'"
rsync -as "$STAGE/data" "$STAGE/code" "$HOST:$REMOTE_ROOT/"
rsync -as --exclude __pycache__ --exclude .claude-flow "$REPO/target/tacc/scripts" "$HOST:$REMOTE_REPO/tacc/"
rsync -as --exclude __pycache__ --exclude .claude-flow "$REPO/target/aws/scripts/terraform-cosmic-ai/lambda_src" \
  "$HOST:$REMOTE_REPO/aws/scripts/terraform-cosmic-ai/"
echo "staged to $HOST:$REMOTE_ROOT"