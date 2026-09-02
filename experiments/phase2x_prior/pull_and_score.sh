#!/bin/bash
# Phase-2X: Pull dumps from Babel and run offline scoring
set -e

SSH_HOST="pengq@login.babel.cs.cmu.edu"
SSH_OPTS="-i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes"
LOCAL_DUMP_DIR="experiments/phase2x_prior/dumps"
RESULT_JSON="experiments/phase2x_prior/result_2x.json"

echo "=== Phase-2X Pull and Score ==="
date

# Check if all 3 seeds are complete on Babel
echo "Checking job status..."
ssh $SSH_OPTS $SSH_HOST "sacct -j 10281123 --format=JobID,State,ExitCode | grep '10281123_[1-3] '"

# Pull tarballs
echo "Pulling dump tarballs..."
mkdir -p "$LOCAL_DUMP_DIR"
for S in 1 2 3; do
  REMOTE="/home/pengq/x_prior_dump_s${S}.tgz"
  LOCAL="$LOCAL_DUMP_DIR/x_prior_dump_s${S}.tgz"
  echo "  Pulling seed $S..."
  scp $SSH_OPTS ${SSH_HOST}:${REMOTE} "$LOCAL" || {
    echo "ERROR: Failed to pull seed $S tarball"
    exit 1
  }
done

# Extract all
echo "Extracting dumps..."
for S in 1 2 3; do
  tar xzf "$LOCAL_DUMP_DIR/x_prior_dump_s${S}.tgz" -C "$LOCAL_DUMP_DIR" && \
    echo "  Extracted seed $S"
done

# Count npz files
NPZ_COUNT=$(find "$LOCAL_DUMP_DIR" -name "*.npz" | wc -l)
echo "Total .npz files: $NPZ_COUNT"

# Score offline (this is the zero-GPU recomputation)
echo "Running offline scorer..."
python experiments/phase2x_prior/score_prior.py \
  --dump-dir "$LOCAL_DUMP_DIR" \
  --out "$RESULT_JSON" \
  --draws 8 \
  --seed 0

# Analyze verdict
echo "Analyzing verdict..."
python experiments/phase2x_prior/analyze_x_verdict.py --result "$RESULT_JSON"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "=== VERDICT: BPO collapses, paper is SAFE ==="
else
  echo "=== VERDICT: BPO survives, paper AT RISK ==="
fi

date
echo "Phase-2X scoring complete."
exit $EXIT_CODE
