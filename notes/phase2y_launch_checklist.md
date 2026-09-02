# Phase-2Y 启动清单（网络恢复后立即执行）

## 当前状态
- **日期**: 2026-09-02
- **Phase-2X判决**: BPO崩溃，论文安全 ✅
- **Track A代码**: 完全实现 ✅
- **Track B设计**: 方案明确 ✅
- **阻塞**: Babel SSH连接超时

## 立即执行命令（网络恢复后）

### 1. 测试Babel连接（1分钟）
```bash
ssh -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes pengq@login.babel.cs.cmu.edu "hostname && date"
```

### 2. 创建Phase-2Y目录（1分钟）
```bash
ssh -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes pengq@login.babel.cs.cmu.edu "mkdir -p iclr26/experiments/phase2y_composite"
```

### 3. 上传Track A代码（2分钟）
```bash
cd "C:\Users\lenovo\Desktop\ICLR_Fixed_Budget_Continual_PEFT\experiments\phase2y_composite"

scp -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  run_two_stage.py \
  sweep_two_stage.sh \
  adaptive_rank.py \
  __init__.py \
  pengq@login.babel.cs.cmu.edu:/home/pengq/iclr26/experiments/phase2y_composite/
```

### 4. 上传SIO依赖（如果缺失）
```bash
cd "C:\Users\lenovo\Desktop\ICLR_Fixed_Budget_Continual_PEFT\experiments"

scp -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  phase2s_sio/sio.py \
  pengq@login.babel.cs.cmu.edu:/home/pengq/iclr26/experiments/phase2s_sio/
```

### 5. 设置权限并提交任务（1分钟）
```bash
ssh -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes pengq@login.babel.cs.cmu.edu "
cd iclr26/experiments/phase2y_composite && \
chmod +x sweep_two_stage.sh && \
sbatch sweep_two_stage.sh
"
```

### 6. 验证任务启动（1分钟）
```bash
ssh -i C:/Users/lenovo/.ssh/id_ed25519 -o IdentitiesOnly=yes pengq@login.babel.cs.cmu.edu "squeue -u pengq"
```

## 预期输出

**任务信息**:
- Job name: `ya_dev`
- Array: 1-3 (seeds 1,2,3)
- Time limit: 4h
- 预计完成: ~2h per seed = 2h墙钟（并行）

**结果位置**:
- `/data/user_data/pengq/runs/y2stage_dev_s1/results_s1.json`
- `/data/user_data/pengq/runs/y2stage_dev_s2/results_s2.json`
- `/data/user_data/pengq/runs/y2stage_dev_s3/results_s3.json`

## Track A判据（简化）

从`results_s{1,2,3}.json`提取：

```python
# 加载3个seeds的结果
results_all = [json.load(open(f"results_s{s}.json")) for s in [1,2,3]]

# 计算每个seed的均值
means = []
for results in results_all:
    r_scg = np.mean([r["R_scg"] for r in results])
    r_orc = np.mean([r["R_orc"] for r in results])
    r_raw = np.mean([r["R_raw"] for r in results])
    means.append({"scg": r_scg, "orc": r_orc, "raw": r_raw})

# 判据（简化版，完整版需bootstrap CI）
scg_vs_raw = [m["scg"] - m["raw"] for m in means]
scg_vs_orc = [m["scg"] - m["orc"] for m in means]

print(f"SCG vs raw: {np.mean(scg_vs_raw)*100:.2f}pp")
print(f"SCG vs oracle: {np.mean(scg_vs_orc)*100:.2f}pp")

# 成功判据（粗略）
if np.mean(scg_vs_raw) > 0.02:  # +2pp
    print("SUCCESS: Track A two-stage works!")
elif np.mean(scg_vs_orc) > -0.01:  # 接近oracle
    print("PARTIAL: Close to oracle, needs Track B")
else:
    print("FAIL: Track B (adaptive rank) necessary")
```

## 如果Track A失败

立即准备Track B实现（明天开始）：

1. 研究`peft/tuners/lora/model.py`
2. 实现`AdaptiveRankLoraModel`
3. Hook到Phase-2Y训练循环
4. 本地测试
5. 上传Babel重跑

## 备用方案

如果Babel持续无法连接：
1. 检查CMU VPN或网络限制
2. 尝试其他SSH端口或跳板机
3. 考虑在alibaba-10运行（但资源受限）
4. 最坏情况：接受测量论文（Phase-2X已提供关键判决）

---

**当前等待**: Babel网络连接恢复
**准备状态**: 所有代码就绪，命令准备完毕
**下一动作**: 网络恢复后5分钟内启动Track A
