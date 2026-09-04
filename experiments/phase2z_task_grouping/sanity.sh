#!/bin/bash
#SBATCH -J z_sanity
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /home/pengq/iclr26/z_sanity_%j.out
#SBATCH -e /home/pengq/iclr26/z_sanity_%j.err

set -x
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

SANITY_MODEL=google-t5/t5-small $V experiments/phase2z_task_grouping/sanity_adapters.py
echo "Z_SANITY_DONE exit=$?"
