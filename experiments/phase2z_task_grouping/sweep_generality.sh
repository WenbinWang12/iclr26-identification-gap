#!/bin/bash
# Phase-2Z-F (RUNBOOK 3B): the 2x2 across task orders x backbones.
# One SLURM array job = one (order, backbone) cell, seeds 1-3 over the array.
# Select the cell with ORDER and MODEL env vars at submit time, e.g.
#   sbatch --export=ALL,ORDER=reverse,MODEL=google-t5/t5-large sweep_generality.sh
#   sbatch --export=ALL,ORDER=canonical,MODEL=google-t5/t5-3b  sweep_generality.sh
#   sbatch --export=ALL,ORDER=canonical,MODEL=EleutherAI/pythia-1.4b sweep_generality.sh
# Defaults reproduce the paper cell (canonical / t5-large) if unset.
#SBATCH -J z_general
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/iclr26/z_general_%A_%a.out
#SBATCH -e /home/pengq/iclr26/z_general_%A_%a.err

set -x
SEED=$SLURM_ARRAY_TASK_ID
ORDER=${ORDER:-canonical}
MODEL=${MODEL:-google-t5/t5-large}
echo "PHASE-2Z-F: generality seed=$SEED order=$ORDER model=$MODEL (epochs=7)"

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
OUT=/home/pengq/iclr26/results/z_general

$V -u experiments/phase2z_task_grouping/run_generality.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --seed $SEED \
  --order "$ORDER" \
  --model "$MODEL" \
  --epochs 7 \
  --cap-per-class 400 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --shared-rank 8 --group-rank 2 --lora-alpha 32 --lora-dropout 0.05 \
  --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4

EXIT=$?
echo "Z_GENERAL_DONE seed=$SEED order=$ORDER model=$MODEL exit=$EXIT"
exit $EXIT
