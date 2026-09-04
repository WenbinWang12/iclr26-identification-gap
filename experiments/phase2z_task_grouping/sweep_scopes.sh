#!/bin/bash
# Phase-2Z-G (RUNBOOK 3C): the 2x2 on a genuine K_S>=3 multi-task scope.
# Relaxes the eligibility floor so CB (3-way NLI) joins MNLI in one NLI scope,
# giving a genuine d>=2 multi-task scope where Prop 3's q_m is open. Reports
# per-scope q_m and the 2x2 on the K_S>=3 subset. seeds 1-3 over the array.
#SBATCH -J z_scopes
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/iclr26/z_scopes_%A_%a.out
#SBATCH -e /home/pengq/iclr26/z_scopes_%A_%a.err

set -x
SEED=$SLURM_ARRAY_TASK_ID
echo "PHASE-2Z-G: scopes seed=$SEED (K_S>=3 multi-task scope, epochs=7)"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU=$GPU"
case "$GPU" in *RTX_PRO*|*PRO_6000*|*Blackwell*) echo "REFUSE_BLACKWELL"; exit 75 ;; esac

V=""
for c in /data/user_data/pengq/*venv*/bin/python; do
  [ -x "$c" ] || continue
  if "$c" -c "import torch, transformers, peft" 2>/dev/null; then V="$c"; break; fi
done
if [ -z "$V" ]; then echo "NO_WORKING_VENV"; exit 76; fi
echo "VENV_OK $V"

cd /home/pengq/iclr26
export PYTHONPATH=/home/pengq/iclr26
export HF_HOME=/data/user_data/pengq/.cache/huggingface
export HF_ENDPOINT=https://huggingface.co
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p /home/pengq/iclr26/results
OUT=/home/pengq/iclr26/results/z_scopes_s${SEED}

$V -u experiments/phase2z_task_grouping/run_scopes.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --seed $SEED \
  --epochs 7 \
  --cap-per-class 400 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --shared-rank 8 --group-rank 2 --lora-alpha 32 --lora-dropout 0.05 \
  --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4 \
  --scope-min-rarest 12

EXIT=$?
echo "Z_SCOPES_DONE seed=$SEED exit=$EXIT"
exit $EXIT
