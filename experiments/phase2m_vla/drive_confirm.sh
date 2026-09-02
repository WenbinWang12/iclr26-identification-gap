#!/usr/bin/env bash
# Phase-2M 确认跑驱动：6 个 arm（3 seed x 主 λ_a=8.0 / 次 λ_a=0.5），每卡一个。
#
# **严格非抢占**：每个 arm 自己在指定卡上轮询 memory.free，够了才开跑；
# 永远不向任何别人的进程发信号。等不到就一直等，宁可空转也不挤别人。
set -u
cd /mnt/data/wenbin/iclr26
PY=/mnt/data/wenbin/phase2_torch_venv/bin/python
LOG=runs/phase2m_confirm_driver.log
mkdir -p runs
echo "M_CONFIRM_START $(date -Is)" >> "$LOG"

# arm 定义： "卡号 seed λ_a 输出目录"
ARMS=(
  "0 1 8.0 runs/phase2m_vla_confirm"
  "2 2 8.0 runs/phase2m_vla_confirm"
  "3 3 8.0 runs/phase2m_vla_confirm"
  "5 1 0.5 runs/phase2m_vla_lam05"
  "1 2 0.5 runs/phase2m_vla_lam05"
  "4 3 0.5 runs/phase2m_vla_lam05"
)

run_arm() {
  local card="$1" seed="$2" lam="$3" out="$4"
  local tag="c${card}_s${seed}_lam${lam}"
  # 自等：这张卡至少 6000 MiB 空闲才开始。非抢占。
  local waited=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$card" 2>/dev/null | tr -d ' ')
    if [ -n "$free" ] && [ "$free" -ge 6000 ] 2>/dev/null; then break; fi
    if [ $((waited % 600)) -eq 0 ]; then
      echo "M_WAIT $tag card=$card free=${free:-NA} waited=${waited}s $(date -Is)" >> "$LOG"
    fi
    sleep 30; waited=$((waited + 30))
  done
  echo "M_LAUNCH $tag $(date -Is)" >> "$LOG"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$card" $PY -m experiments.phase2k_qoc.run_qoc \
    --n-tasks 15 --cap-per-class 200 --update-cap-per-class 400 --epochs 3 \
    --risk-per-class 64 --audit-per-class 64 --eval-batch-size 2 \
    --m-values 1,2,4 --cl-method vla --vla-lambda "$lam" \
    --seed "$seed" --out "$out" \
    > "runs/phase2m_confirm_${tag}.log" 2>&1
  echo "M_DONE $tag rc=$? $(date -Is)" >> "$LOG"
}

for spec in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  run_arm $spec &
  sleep 5
done
wait
echo "M_CONFIRM_ALL_DONE $(date -Is)" >> "$LOG"
