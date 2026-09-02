#!/bin/bash
# Phase-2W rank sweep. Frozen with the protocol; contains NO scoring logic.
# Config is byte-identical to runs/phase2q_bpo (the local converged r=8 reference)
# except for --lora-r. See protocol Amendment 1(a).
# Array index encodes (rank, seed): idx = rank_slot*3 + (seed-1), rank_slot in 0..5.
#SBATCH -J w_rank
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -a 0-17
#SBATCH -o /home/pengq/w_rank_%A_%a.out
set -x
RANKS=(1 2 4 8 16 32)
IDX=$SLURM_ARRAY_TASK_ID
R=${RANKS[$((IDX / 3))]}
S=$(((IDX % 3) + 1))
EP=${W_EPOCHS:-3}
TAG=${W_TAG:-}

# Blackwell cards in this cluster have no compiled kernels for our torch build and
# fail in ~16s while sacct still reports COMPLETED. Refuse them up front.
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU=$GPU RANK=$R SEED=$S EPOCHS=$EP"
case "$GPU" in
  *RTX_PRO*|*PRO_6000*|*Blackwell*) echo "REFUSE_BLACKWELL"; exit 75 ;;
esac

V=/data/user_data/pengq/sio_venv/bin/python
cd /home/pengq/iclr26
export PYTHONPATH=/home/pengq/iclr26
export HF_HOME=/data/user_data/pengq/.cache/huggingface
export HF_ENDPOINT=https://huggingface.co
export TOKENIZERS_PARALLELISM=false

OUT=/data/user_data/pengq/runs/w_rank${TAG}_r${R}_s${S}
$V experiments/phase2k_qoc/run_qoc.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --lora-r $R \
  --epochs $EP \
  --cap-per-class 400 --update-cap-per-class 400 \
  --m-values 1,4 \
  --risk-per-class 64 --audit-per-class 64 --min-rare-class 40 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --max-source 512 --max-target 8 \
  --cl-method none --n-tasks 15 --seed $S
echo "W_DONE rank=$R seed=$S epochs=$EP EXIT=$?"
