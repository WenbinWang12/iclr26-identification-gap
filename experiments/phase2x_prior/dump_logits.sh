#!/bin/bash
# Phase-2X: one training sweep whose ONLY purpose is to dump audit logits, so the
# entire prior-shift grid becomes a zero-GPU offline recomputation.
#
# Config is byte-identical to experiments/phase2w_rank/sweep_rank.sh at rank 8
# (itself byte-identical to runs/phase2q_bpo) EXCEPT for the two added flags
# --dump-logits and --score-bpo. In particular --risk-per-class and
# --audit-per-class stay at 64: raising them would change which examples land in
# the risk/audit split (see run_qoc.py:66 grow_update_split) and put this run on a
# different evaluation set than the existing 24 runs, destroying comparability.
# Q4's imbalance question is answered offline by resampling the dumped logits, NOT
# by enlarging the split.
#SBATCH -J x_dump
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A bapoczos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -a 1-3
#SBATCH -o /home/pengq/x_dump_%A_%a.out
set -x
S=$SLURM_ARRAY_TASK_ID
R=8
EP=3

# Blackwell cards in this cluster have no compiled kernels for our torch build and
# fail in ~16s while sacct still reports COMPLETED. Refuse them up front.
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU=$GPU RANK=$R SEED=$S EPOCHS=$EP"
case "$GPU" in
  *RTX_PRO*|*PRO_6000*|*Blackwell*) echo "REFUSE_BLACKWELL"; exit 75 ;;
esac

# The venv path frozen into sweep_rank.sh is not guaranteed to survive
# /data/user_data cleanups. Discover it and PROVE the three imports work before
# spending an hour of GPU, rather than dying at import time mid-run.
V=""
for c in /data/user_data/pengq/*venv*/bin/python; do
  [ -x "$c" ] || continue
  if "$c" -c "import torch, transformers, peft" 2>/dev/null; then V="$c"; break; fi
done
if [ -z "$V" ]; then echo "NO_WORKING_VENV"; exit 76; fi
echo "VENV_OK $V"
"$V" -c "import torch, transformers, peft; print('versions', torch.__version__, transformers.__version__, peft.__version__)"

cd /home/pengq/iclr26
export PYTHONPATH=/home/pengq/iclr26
export HF_HOME=/data/user_data/pengq/.cache/huggingface
export HF_ENDPOINT=https://huggingface.co
export TOKENIZERS_PARALLELISM=false

OUT=/data/user_data/pengq/runs/x_dump_r${R}_s${S}
DUMP=/data/user_data/pengq/dumps/x_prior_s${S}
"$V" experiments/phase2k_qoc/run_qoc.py \
  --data-root /data/user_data/pengq/iclr26_data/order4 \
  --out $OUT \
  --dump-logits $DUMP \
  --score-bpo \
  --lora-r $R \
  --epochs $EP \
  --cap-per-class 400 --update-cap-per-class 400 \
  --m-values 1,4 \
  --risk-per-class 64 --audit-per-class 64 --min-rare-class 40 \
  --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
  --max-source 512 --max-target 8 \
  --cl-method none --n-tasks 15 --seed $S
RC=$?

# /data/user_data is compute-node-only, so stage the artefacts into /home for scp.
# Do this inside the job: from the login node the dump directory does not exist.
if [ -d "$DUMP" ]; then
  tar czf /home/pengq/x_prior_dump_s${S}.tgz -C "$DUMP" . \
    && echo "TARRED /home/pengq/x_prior_dump_s${S}.tgz"
  ls -la /home/pengq/x_prior_dump_s${S}.tgz
  echo "NPZ_COUNT=$(ls $DUMP/*.npz 2>/dev/null | wc -l)"
else
  echo "NO_DUMP_DIR $DUMP"
fi
[ -f "$OUT/qoc_seed${S}.json" ] && cp "$OUT/qoc_seed${S}.json" /home/pengq/x_qoc_seed${S}.json
echo "X_DONE rank=$R seed=$S epochs=$EP EXIT=$RC"
