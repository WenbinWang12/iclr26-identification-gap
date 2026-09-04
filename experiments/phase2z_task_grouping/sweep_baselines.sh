#!/bin/bash
# Phase-2Z-H (RUNBOOK 3D): published continual-PEFT baselines at MATCHED budget.
# One SLURM array job = one method, seeds 1-3 over the array. Select the method
# with the METHOD env var at submit time:
#   sbatch --export=ALL,METHOD=seqft sweep_baselines.sh
#   sbatch --export=ALL,METHOD=olora sweep_baselines.sh
#   sbatch --export=ALL,METHOD=ewc   sweep_baselines.sh
#SBATCH -J z_basel
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/iclr26/z_basel_%A_%a.out
#SBATCH -e /home/pengq/iclr26/z_basel_%A_%a.err

set -x
SEED=$SLURM_ARRAY_TASK_ID
METHOD=${METHOD:-seqft}
echo "PHASE-2Z-H: baseline seed=$SEED method=$METHOD (matched rank-8, epochs=7)"

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
OUT=/home/pengq/iclr26/results/z_basel

$V -u experiments/phase2z_task_grouping/run_baselines.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --seed $SEED \
  --method "$METHOD" \
  --epochs 7 \
  --cap-per-class 400 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --rank 8 --lora-alpha 32 --lora-dropout 0.05 \
  --olora-lambda 0.5 --ewc-lambda 1.0 \
  --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4

EXIT=$?
echo "Z_BASEL_DONE seed=$SEED method=$METHOD exit=$EXIT"
exit $EXIT
