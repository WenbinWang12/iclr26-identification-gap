# Phase-2X判决后的行动计划

## 当前状态（2026-09-02）

### 已完成
1. ✅ Phase-2X dumps已拉取（3 seeds，360 npz files）
2. ✅ Phase-2Y协议已冻结（三阶段组合方法）
3. ✅ Adaptive rank分配逻辑已实现并测试
4. ✅ Phase-2Y sweep脚本已准备

### 进行中
- Phase-2X离线评分（预计~23分钟，draws=1快速版本）

## 两种结果的应对策略

### 结果A：BPO崩溃（crossover exists at batch=128）
**判决：论文安全，Δ_id站得住**

**立即行动**：
1. 运行Phase-2X判决脚本确认
2. 更新memory记录Phase-2X结果
3. 启动Phase-2Y开发运行（Babel）
   - 9 jobs: 3 seeds × 3 sensitivities
   - 预计18h GPU / 6h墙钟时间
4. 分析开发结果，选择最佳sensitivity
5. 启动Phase-2Y留出验证（3 seeds）
6. 根据5个判据判定结果

**Phase-2Y成功路径**：
- Y1, Y2, Y3, Y4, Y5全通过 → 正结果方法论文 ✅
- Y2失败（ceiling未突破）→ 测量论文，QOC天花板是模型容量极限

**Phase-2Y失败应急**：
- 如果Y3失败（composite有害）→ 回退最佳单一方法
- 如果Y1失败（rank是噪声）→ 降级为PSR+SCG两阶段
- 最坏情况：测量论文（Δ_id存在但无方法）

### 结果B：BPO存活（no crossover at batch=128）
**判决：论文有风险，identification-gap理论需修订**

**立即行动**：
1. 深入分析：为什么label-free offset能击败per-task oracle？
2. 三种可能解释：
   a. BPO实际上隐式读取了task identity（通过batch统计）
   b. Oracle本身有缺陷（64/class risk split不足）
   c. Δ_id定义需要修订（不是"需要task ID的部分"）
3. 设计Phase-2X-followup实验验证解释
4. 决定是否修订论文框架或pivot

**可能的修订方向**：
- 重新定义Δ_id为"需要task-specific信息的部分"（不限于显式ID）
- 承认BPO anomaly，转向"为什么batch statistics有效"的研究
- Pivot到不同问题：multi-task learning而非continual learning

## 无论哪种结果都要做的

### 记录和诚实性
1. 将Phase-2X判决写入memory
2. 如果BPO存活，诚实记录对论文的影响
3. 更新paper roadmap文档

### 准备Plan B
即使Phase-2Y启动，也要准备后备方案：
- Amendment 2: constant-scale control arm（memory中提到）
- 如果所有方法失败，准备"测量论文"框架

## 时间线估算

**如果结果A（BPO崩溃）**：
- 今天：Phase-2X判决 + Phase-2Y dev启动
- +6h：Phase-2Y dev完成，分析结果
- +8h：Phase-2Y held-out启动
- +10h：Phase-2Y held-out完成，判决
- **总计：~10小时墙钟时间到Phase-2Y判决**

**如果结果B（BPO存活）**：
- 今天：Phase-2X判决 + 深入分析
- +1-2天：设计followup实验或决定pivot
- 时间线不确定，取决于用户决策

## 当前等待

Phase-2X评分预计完成时间：10:48 + 23min ≈ **11:11**

准备在评分完成后立即：
1. 运行`analyze_x_verdict.py`
2. 根据判决执行对应的行动计划
3. 向用户报告判决和下一步

---

**下一个checkpoint**：Phase-2X评分完成（~15分钟后）
