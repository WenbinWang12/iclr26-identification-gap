#!/bin/bash
# Phase-2Y development sweep: 3 seeds × 3 sensitivities on Babel
#SBATCH -J y_dev
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -a 0-8
#SBATCH -o /home/pengq/y_dev_%A_%a.out

set -x

# Array encoding: idx = seed_slot*3 + lambda_slot
# seed_slot in {0,1,2} → seeds {1,2,3}
# lambda_slot in {0,1,2} → lambda {0.3, 0.5, 0.8}
IDX=$SLURM_ARRAY_TASK_ID
SEED=$((IDX / 3 + 1))
LAMBDA_SLOT=$((IDX % 3))
LAMBDAS=(0.3 0.5 0.8)
LAMBDA=${LAMBDAS[$LAMBDA_SLOT]}

echo "PHASE-2Y DEV: seed=$SEED lambda=$LAMBDA"

# Refuse Blackwell nodes (same as Phase-2W/2X)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU=$GPU"
case "$GPU" in
  *RTX_PRO*|*PRO_6000*|*Blackwell*) echo "REFUSE_BLACKWELL"; exit 75 ;;
esac

# Discover working venv
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

OUT=/data/user_data/pengq/runs/y_dev_s${SEED}_lam${LAMBDA}

# TODO: 当Phase-2Y run_composite.py实现完成后，取消注释并运行
# $V experiments/phase2y_composite/run_composite.py \
#   --data-root /data/user_data/pengq/iclr26_data/order4 \
#   --out $OUT \
#   --seed $SEED \
#   --rank-sensitivity $LAMBDA \
#   --epochs 3 \
#   --cap-per-class 400 --update-cap-per-class 400 \
#   --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
#   --risk-per-class 64 --audit-per-class 64

echo "PLACEHOLDER: Phase-2Y implementation pending"
echo "This job will run the three-stage composite method once implemented"
exit 0
