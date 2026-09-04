#!/bin/bash
# Phase-2Z: Task Grouping under fixed budget (shared rank-8 vs 4 groups rank-2)
#SBATCH -J z_group
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/iclr26/z_group_%A_%a.out
#SBATCH -e /home/pengq/iclr26/z_group_%A_%a.err

set -x
SEED=$SLURM_ARRAY_TASK_ID
echo "PHASE-2Z: Task Grouping seed=$SEED (shared r8 vs 4x r2, epochs=7)"

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

mkdir -p /home/pengq/iclr26/results
OUT=/home/pengq/iclr26/results/z_grouping_s${SEED}

$V experiments/phase2z_task_grouping/run_grouping.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --seed $SEED \
  --epochs 7 \
  --cap-per-class 400 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --shared-rank 8 --group-rank 2 --lora-alpha 32 --lora-dropout 0.05 \
  --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4

EXIT=$?
echo "Z_GROUP_DONE seed=$SEED exit=$EXIT"
exit $EXIT
