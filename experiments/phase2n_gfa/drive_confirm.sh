#!/usr/bin/env bash
# Phase-2N 确认跑驱动：协议 §A1.4，λ_g=8.0（闸门选出），3 seed，全 15 任务。
# 判据 1-6 只在这个 arm 上评，没有 secondary arm。
#
# 严格非抢占：每卡自等 memory.free >= 6000，永不向别人的进程发信号。
set -u
cd /mnt/data/wenbin/iclr26
PY=/mnt/data/wenbin/phase2_torch_venv/bin/python
LOG=runs/phase2n_confirm_driver.log
mkdir -p runs
echo "N_CONFIRM_START $(date -Is)" >> "$LOG"

# 每 arm: "卡号 seed 输出目录"
ARMS=(
  "0 1 runs/phase2n_gfa_confirm"
  "2 2 runs/phase2n_gfa_confirm"
  "3 3 runs/phase2n_gfa_confirm"
)

run_arm() {
  local card="$1" seed="$2" out="$3"
  local tag="c${card}_s${seed}"
  local waited=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$card" 2>/dev/null | tr -d ' ')
    if [ -n "$free" ] && [ "$free" -ge 6000 ] 2>/dev/null; then break; fi
    if [ $((waited % 600)) -eq 0 ]; then
      echo "N_CWAIT $tag free=${free:-NA} waited=${waited}s $(date -Is)" >> "$LOG"
    fi
    sleep 30; waited=$((waited + 30))
  done
  echo "N_CLAUNCH $tag $(date -Is)" >> "$LOG"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$card" $PY -m experiments.phase2k_qoc.run_qoc \
    --n-tasks 15 --cap-per-class 200 --update-cap-per-class 400 --epochs 3 \
    --risk-per-class 64 --audit-per-class 64 --eval-batch-size 2 \
    --m-values 1,2,4 --cl-method gfa --gfa-lambda 8.0 \
    --seed "$seed" --out "$out" \
    > "runs/phase2n_confirm_${tag}.log" 2>&1
  echo "N_CONFIRM_DONE $tag rc=$? $(date -Is)" >> "$LOG"
}

for spec in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  run_arm $spec &
  sleep 5
done
wait
echo "N_CONFIRM_ALL_DONE $(date -Is)" >> "$LOG"
