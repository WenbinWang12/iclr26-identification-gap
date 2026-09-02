#!/bin/bash
# Poll Phase-2X job until complete, then trigger scoring
SSH_HOST="pengq@login.babel.cs.cmu.edu"
SSH_OPTS="-i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes"

echo "=== Phase-2X Poll Monitor ==="
echo "Started: $(date)"
echo "Monitoring job 10281123 (seeds 1,2,3)..."

while true; do
  # Check if any array task is still running
  RUNNING=$(ssh $SSH_OPTS $SSH_HOST "squeue -u pengq -j 10281123 --noheader | wc -l" 2>/dev/null)

  if [ "$RUNNING" -eq 0 ]; then
    echo "$(date) - All seeds complete!"

    # Verify all 3 completed successfully
    COMPLETED=$(ssh $SSH_OPTS $SSH_HOST "sacct -j 10281123 --format=JobID,State --noheader | grep -E '10281123_[1-3] ' | grep COMPLETED | wc -l" 2>/dev/null)

    if [ "$COMPLETED" -eq 3 ]; then
      echo "All 3 seeds COMPLETED successfully"
      echo "Triggering pull and score..."
      bash experiments/phase2x_prior/pull_and_score.sh
      EXIT_CODE=$?

      if [ $EXIT_CODE -eq 0 ]; then
        echo "==================================="
        echo "PHASE-2X VERDICT: PAPER SAFE"
        echo "BPO collapses under prior skew"
        echo "Next: Implement Phase-2Y composite method"
        echo "==================================="
      else
        echo "==================================="
        echo "PHASE-2X VERDICT: PAPER AT RISK"
        echo "BPO survives skew robustly"
        echo "Next: Revise identification-gap thesis"
        echo "==================================="
      fi

      exit $EXIT_CODE
    else
      echo "WARNING: Not all seeds completed successfully"
      ssh $SSH_OPTS $SSH_HOST "sacct -j 10281123 --format=JobID,State,ExitCode"
      exit 1
    fi
  else
    echo "$(date) - Still running ($RUNNING tasks active)..."
    ssh $SSH_OPTS $SSH_HOST "squeue -u pengq -j 10281123 --format='%.10i %.9P %.30j %.8T %.10M %.6D %R'"
  fi

  sleep 300  # Poll every 5 minutes
done
