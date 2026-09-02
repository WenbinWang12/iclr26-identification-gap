# 项目进展总结 - 2026-09-02

## 重大里程碑：Phase-2X判决完成 ✅

### 判决结果
**BPO在prior=0.85时崩溃 → 论文安全**

关键数据：
- Crossover: prior=0.85 (batch=128)
- Balanced条件下：BPO输给oracle −4.03pp
- 仅2.2%的任务BPO能击败oracle
- **结论**: Identification gap Δ_id站得住

### 意义
Phase-2Q报告的"BPO > oracle"异常是balanced audit的人工产物，不威胁论文根基。可以安全推进Phase-2Y。

## Phase-2Y双轨道设计完成 ✅

### Track A：两阶段（PSR + SCG）
**目标**: 快速验证组合方法
**实现状态**: ✅ 完成
- `run_two_stage.py`: 完整实现
  - Stage 1: PSR coverage buffer (4+12)
  - Stage 2: Task-adaptive SCG thresholds
  - 完整评估流程
- `sweep_two_stage.sh`: Babel array job配置
- 预计时间: 2h/seed × 3 seeds = 2h墙钟

**成功判据**（简化）:
- R_scg > R_raw + 2pp → 成功
- R_scg ≈ R_orc → 部分成功，需Track B
- R_scg < R_raw → 失败，Track B必需

### Track B：三阶段（PSR + Adaptive Rank + SCG）
**目标**: 完整方法，突破QOC天花板
**实现状态**: 设计完成，代码待实现
- 技术方案: Task-conditional parameter masking
- 估计时间: 8-12小时实现 + 6h运行

**触发条件**: Track A失败或仅部分成功

## 已完成的代码

### Phase-2X
- ✅ `score_prior.py`: 完整评分逻辑（360 entries × 50 cells/entry）
- ✅ `analyze_x_verdict.py`: 自动判决分析
- ✅ `verdict_2x.json`: 判决记录

### Phase-2Y Track A
- ✅ `adaptive_rank.py`: Rank分配逻辑（已测试）
- ✅ `run_two_stage.py`: 两阶段训练+评估
- ✅ `sweep_two_stage.sh`: Babel sweep配置

### 文档
- ✅ `phase2y_composite_protocol.md`: 完整协议（5个判据）
- ✅ `phase2y_implementation_status.md`: 实现路径分析
- ✅ `phase2y_launch_checklist.md`: 启动清单
- ✅ `phase2x_post_verdict_plan.md`: 判决后行动计划
- ✅ `project_status_20260902.md`: 总体状态

### Memory
- ✅ `phase2x-bpo-verdict.md`: Phase-2X判决记录

## 当前阻塞

**网络问题**: Babel SSH连接超时
- 影响: 无法上传代码和启动Track A
- 准备: 所有命令已备好，网络恢复后5分钟内可启动
- 备用: 如持续无法连接，可在alibaba-10运行或接受测量论文

## 三条发表路径

### 路径1：Track A成功（最快）
- 时间: ~2h运行 + 1h分析
- 论文: 正结果方法论文
- 贡献: Δ_id测量 + PSR+SCG组合方法

### 路径2：Track B成功（完整）
- 时间: 12h实现 + 6h运行 + 1h分析
- 论文: 正结果方法论文
- 贡献: Δ_id测量 + 三阶段组合 + adaptive rank机制

### 路径3：测量论文（保底）
- 时间: 无需新实验
- 论文: 负结果论文
- 贡献: Δ_id测量 + QOC天花板分析（45% offset / 55%表征层）

## 关键数据总结

### Phase-2X
- 360 entries processed
- 18,000 rows generated (10 priors × 5 batches × 360)
- Crossover: 0.85
- Verdict: PAPER_SAFE

### Phase-2K QOC Converged
- Epochs: 3 (813 gradient steps/seed)
- Criterion 1: +2.34pp [+0.78, +3.91] ✅
- Criterion 2: 0.0000 [0, 0] ❌
- Criterion 3: ρ=+0.646 ✅
- Ceiling: 0.7204 (> oracle 0.7146)

### Phase-2U SCG
- U1: worst −2.34pp ❌ (threshold −1.0pp)
- U5: ρ=+0.72 (8/8 tasks) ✅
- Mechanism works, threshold failed to transfer

### Phase-2W Rank
- 32× scaling: flat (±1.30pp = noise floor)
- Uniform allocation failed

### Δ_id
- Multi-task scope: +1.10pp
- Scorable tasks: 7 scopes (singleton defect corrected)

## 下一步行动

**优先级1**: 恢复Babel连接，启动Track A
**优先级2**: 监控Track A结果（~2h后）
**优先级3**: 根据Track A结果决定Track B实现
**优先级4**: 准备判据验证和bootstrap CI脚本

## 时间线

**如果Track A今晚启动**:
- T+0h: 提交任务
- T+2h: 结果完成，分析判决
- T+3h: 如果成功 → 撰写；如果失败 → 启动Track B
- T+15h (明天): Track B实现完成（如需要）
- T+21h (明天): Track B结果完成
- **T+24h: 知道最终方法是否成功**

**如果网络问题持续**:
- 评估alibaba-10可行性
- 或准备测量论文框架

---

**当前状态**: 代码完备，等待网络恢复
**风险**: 低（三条路径都可发表）
**信心**: Phase-2X判决为论文提供坚实基础
