#!/usr/bin/env bash
# Phase-2N λ_g 闸门驱动：协议 §4，栅格 {0.5, 2.0, 8.0}，seed 1，前 6 个任务。
# `none` arm 复用 Phase-2M 的 runs/phase2m_gate_none（同参数同 seed，不重跑）。
#
# 严格非抢占：每卡自等 memory.free >= 6000，永不向别人的进程发信号。
set -u
cd /mnt/data/wenbin/iclr26
PY=/mnt/data/wenbin/phase2_torch_venv/bin/python
LOG=runs/phase2n_gate_driver.log
mkdir -p runs
echo "N_GATE_START $(date -Is)" >> "$LOG"

ARMS=(
  "0 0.5 runs/phase2n_gate_g05"
  "2 2.0 runs/phase2n_gate_g20"
  "3 8.0 runs/phase2n_gate_g80"
)

run_arm() {
  local card="$1" lam="$2" out="$3"
  local tag="c${card}_lam${lam}"
  local waited=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$card" 2>/dev/null | tr -d ' ')
    if [ -n "$free" ] && [ "$free" -ge 6000 ] 2>/dev/null; then break; fi
    if [ $((waited % 600)) -eq 0 ]; then
      echo "N_WAIT $tag free=${free:-NA} waited=${waited}s $(date -Is)" >> "$LOG"
    fi
    sleep 30; waited=$((waited + 30))
  done
  echo "N_LAUNCH $tag $(date -Is)" >> "$LOG"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$card" $PY -m experiments.phase2k_qoc.run_qoc \
    --n-tasks 15 --cap-per-class 200 --update-cap-per-class 400 --epochs 3 \
    --risk-per-class 64 --audit-per-class 64 --eval-batch-size 2 \
    --m-values 1,2,4 --cl-method gfa --gfa-lambda "$lam" \
    --stop-after-tasks 6 --gate-update-eval \
    --seed 1 --out "$out" \
    > "runs/phase2n_gate_${tag}.log" 2>&1
  echo "N_GATE_DONE $tag rc=$? $(date -Is)" >> "$LOG"
}

for spec in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  run_arm $spec &
  sleep 5
done
wait
echo "N_GATE_ALL_DONE $(date -Is)" >> "$LOG"
