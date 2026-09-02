# 项目当前状态总结（2026-09-02）

## 核心问题
能否发表一篇**正结果的方法论文**，解决固定参数预算下的continual PEFT容量分配？

## 当前瓶颈：Phase-2X判决（进行中，55%完成）

**Phase-2X测什么**：
- Phase-2Q发现：BPO（label-free batch prior offset）R=0.7674 > Oracle R=0.7616
- 如果这个结果站得住，identification-gap论文的根基崩溃
- Phase-2X通过先验网格（10 priors × 5 batches）测试：BPO的优势是真实的还是balanced audit的人工产物

**两种结果**：
1. **BPO崩溃**（crossover exists）→ 论文安全，Δ_id站得住 → 继续Phase-2Y
2. **BPO存活**（no crossover）→ 论文有风险，需要修订理论框架

## 现有方法的失败链

| Phase | 方法 | 主要失败 | 部分成功 |
|---|---|---|---|
| 2K | QOC (offset codebook) | 判据2: m-单调性 0.0000 | 判据1: +2.34pp; 判据3: ρ=+0.646 |
| 2U | SCG (self-confidence gate) | U1: 最坏-2.34pp | U5: ρ=+0.72 (8/8任务) |
| 2W | Rank scaling (32倍) | 全平: ±1.30pp = 噪声底 | 无 |
| 2L/2M/2N | 三个锚 | 全负 | 规范不是机制 |

**关键发现（Phase-2K → 2L）**：
- QOC天花板：m=4 codebook (0.7204) > oracle (0.7146)
- Offset只能修复**45%**的大遗忘（>8pp stratum）
- **55%是表征层的**：MNLI +17.5pp，RTE +11.5pp，BoolQA +6.9pp
- 绑定约束是表征层，不是offset层

## Phase-2Y：三阶段组合方法（已准备）

**假设**：组合三个部分成功的机制可以突破QOC天花板

**Stage 1 (PSR)**：稀有类覆盖缓冲
- Phase-2F remedy已在留出集确认
- 提供训练信号

**Stage 2 (Adaptive Rank)**：根据测量的深度遗忘分配rank
- 核心创新：conditional allocation，不是Phase-2W的uniform scaling
- MNLI +17.5pp残差 → r=15
- Yelp 0.0pp残差 → r=6
- 总预算120（与baseline相同）

**Stage 3 (SCG)**：query-time offset校正
- Task-adaptive thresholds（不是Phase-2U的global τ_m）
- 在PSR样本上拟合

**判据（5个，留出seeds 11-13）**：
- Y1: rank分配与遗忘相关（ρ > 0.5）
- Y2: 突破QOC天花板（> 0.7204）
- Y3: 不伤害baseline（10/12任务）
- Y4: 非空洞（>30%任务rank非均匀）
- Y5: SCG机制存活（ρ > 0.25）

**预计GPU时间**：
- 开发：9 runs × 2h = 18h GPU / 6h墙钟
- 留出：3 runs × 2h = 6h GPU / 2h墙钟
- **总计：24h GPU / 8h墙钟**

## 成功路径

### 路径1：Phase-2Y全通过（目标）
- Y1-Y5全pass → **正结果方法论文** ✅
- 写作重点：三阶段如何协同突破天花板
- 贡献：
  1. 识别间隙Δ_id的测量（+1.10pp，7个多任务scope）
  2. QOC天花板分析（表征层 vs offset层）
  3. 三阶段组合方法突破天花板

### 路径2：Phase-2Y部分成功
- Y2通过但Y1或Y5失败 → 方法有效但机制不清
- 仍可发表，但解释部分存疑
- 写作：报告工作方法+承认机制gap

### 路径3：Phase-2Y失败，降级为测量论文
- Y2失败（ceiling未突破）→ 模型容量是绑定约束
- 或Y3失败（composite有害）
- 论文变为：
  1. Δ_id存在并测量（+1.10pp）
  2. QOC天花板分析（45% offset / 55% 表征层）
  3. **负结果**：在LoRA r=8预算下，表征层遗忘不可修复
  4. 贡献：问题定义+测量+天花板分析

## 应急方案

如果Phase-2Y所有路径失败：
1. **Amendment 2**: constant-scale control arm（memory提到但未执行）
2. **问题pivot**: 从continual learning转向multi-task learning
3. **方法pivot**: 不是容量分配，而是"为什么简单方法（BPO）有效"

## 当前行动

**正在进行**：
- Phase-2X评分（200/360 entries完成，~10分钟剩余）

**Phase-2X完成后立即**：
1. 运行`analyze_x_verdict.py`获得判决
2. 如果BPO崩溃 → 启动Phase-2Y dev（Babel，9 jobs）
3. 如果BPO存活 → 深入分析+决定修订方向

**Phase-2Y时间线（如果启动）**：
- T+0h: 启动dev runs
- T+6h: dev完成，选择最佳sensitivity
- T+6h: 启动held-out runs  
- T+8h: held-out完成，判决
- **T+8-10h: 知道Phase-2Y是否成功**

## 诚实性检查清单

✅ Singleton defect已修正（多任务scope +1.10pp）
✅ Phase-2K converged是epochs=3充分训练
✅ Phase-2W rank全平是真实负结果（epochs=3）
✅ QOC天花板超过oracle是已确认事实
✅ Phase-2X在任何运行前冻结协议
✅ Phase-2Y在任何运行前冻结协议
✅ 所有判据都有预定义的通过/失败阈值
✅ 留出seeds在选择超参数前启动

## 风险评估

**高风险**：
- Phase-2X判BPO存活 → 论文根基需修订（概率：未知）
- Phase-2Y ceiling未突破 → 只能发测量论文（概率：中等）

**中风险**：
- Phase-2Y机制失败但方法有效 → 可发表但解释弱（概率：中等）
- Phase-2Y有害 → 需要Plan B（概率：低）

**低风险**：
- Phase-2Y全成功 → 理想情况（概率：期望但不保证）

## 下一个里程碑

**Phase-2X判决**（~10分钟后）
- 决定论文是否需要修订框架
- 决定是否启动Phase-2Y
- 这是"go/no-go"决策点

---

**更新时间**：2026-09-02 11:15
**状态**：等待Phase-2X判决
**下一步**：根据判决执行对应行动计划
