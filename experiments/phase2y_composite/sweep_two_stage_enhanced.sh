#!/bin/bash
# Phase-2Y-A Enhanced: Two-stage (PSR + SCG) with improved hyperparameters
#SBATCH -J ya_enh
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/iclr26/ya_enh_%A_%a.out
#SBATCH -e /home/pengq/iclr26/ya_enh_%A_%a.err

set -x

SEED=$SLURM_ARRAY_TASK_ID

echo "PHASE-2Y-A ENHANCED: Two-stage (PSR+SCG) seed=$SEED"
echo "Changes from dev: epochs 3->7, PSR budget 16->32, wider tau grid"

# Refuse Blackwell nodes
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

mkdir -p /home/pengq/iclr26/results
OUT=/home/pengq/iclr26/results/y2stage_enhanced_s${SEED}

$V experiments/phase2y_composite/run_two_stage.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --seed $SEED \
  --epochs 7 \
  --cap-per-class 400 --update-cap-per-class 400 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --lora-r 8 --lora-alpha 32 --lora-dropout 0.05 \
  --psr-cover-first 4 --psr-cover-rest 12 --psr-budget 32 \
  --scg-tau-grid "0.4,0.6,0.8,1.0,1.2,1.4,1.6,2.0,2.5,3.0" --scg-worst-threshold -1.0 \
  --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4

EXIT=$?
echo "YA_ENH_DONE seed=$SEED exit=$EXIT"
exit $EXIT
