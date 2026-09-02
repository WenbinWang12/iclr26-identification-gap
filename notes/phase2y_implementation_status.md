# Phase-2Y 实现状态与路径前瞻

**日期**: 2026-09-02
**状态**: 协议冻结，adaptive rank逻辑完成，但LoRA集成受阻

## 已完成

✅ **Phase-2X判决**: BPO在prior=0.85崩溃，论文安全
✅ **Phase-2Y协议**: 完整冻结（`notes/phase2y_composite_protocol.md`）
✅ **Adaptive rank分配逻辑**: 实现并测试（`adaptive_rank.py`）
✅ **Sweep脚本**: Babel array job准备就绪
✅ **Memory更新**: Phase-2X判决已记录

## 核心技术障碍

**问题**: Adaptive rank需要**任务条件化的LoRA rank分配**

PEFT的LoraConfig在模型初始化时设置固定rank（r=8）。Phase-2Y需要：
- Task 1: r=4
- Task 5: r=11 (高遗忘)
- Task 10: r=6
- ...

**三种可能方案**：

### 方案1：任务条件化参数掩码（推荐）
**原理**: 
- 初始化max_rank=16的LoRA（容纳最高分配）
- 每个任务mask掉超出allocated_rank的参数
- 训练时只更新mask内参数

**优点**: 
- 最接近协议设计
- 预算精确控制（Σr_t = 120）

**缺点**: 
- 需要修改PEFT内部机制
- 可能与PEFT版本冲突

**实现复杂度**: ~8-12小时（熟悉PEFT源码+hook实现+测试）

### 方案2：渐进式rank增长
**原理**:
- 每个任务创建独立的LoRA adapter
- Rank根据分配策略设置
- 保留所有adapter，inference时组合

**优点**:
- 不修改PEFT
- 实现相对简单

**缺点**:
- 总参数量可能超预算（所有adapter累加）
- 不是真正的"预算重分配"

**实现复杂度**: ~4-6小时

### 方案3：降级为两阶段（PSR + SCG）
**原理**:
- 放弃Stage 2 (adaptive rank)
- 只保留Stage 1 (PSR) + Stage 3 (SCG)
- Uniform rank=8，与baseline相同

**优点**:
- 立即可实现（<2小时）
- 仍是新组合

**缺点**:
- 失去突破QOC天花板的主要机制
- 如果PSR+SCG不够，回到测量论文

**实现复杂度**: ~2小时

## 推荐路径

### 短期（今天内）：方案3降级版
1. 实现PSR + SCG两阶段（~2小时）
2. 启动Babel dev runs（3 seeds，~6h墙钟）
3. 如果成功 → 快速正结果
4. 如果失败 → 启动方案1的adaptive rank实现

### 中期（明天）：方案1完整版
如果两阶段不够：
1. 深入PEFT源码，实现rank masking
2. 本地测试验证预算约束
3. 重跑Phase-2Y完整三阶段

### 后备：测量论文
如果所有方法失败：
- Δ_id存在（+1.10pp）
- QOC天花板分析（45% offset / 55%表征层）
- 负结果：LoRA r=8预算下表征层遗忘不可修复
- 贡献：问题定义+测量+天花板

## 当前决策点

**需要用户决定**：

**选项A**: 先试PSR+SCG两阶段（快速，今天完成）
- 如果成功 → 正结果论文
- 如果失败 → 再投入adaptive rank

**选项B**: 直接实现adaptive rank完整版（慢，需明天）
- 风险：花12小时实现后仍可能失败
- 回报：如果成功是最完整的方法

**选项C**: 接受测量论文（安全，无需新实验）
- Δ_id已测量
- QOC天花板已分析
- 论文框架完整，只是负结果

## 时间估算

**方案A（PSR+SCG）**:
- 实现: 2h
- Dev运行: 6h (墙钟)
- 分析: 1h
- Held-out: 2h (如果dev成功)
- **总计: ~11h 到判决**

**方案B（完整三阶段）**:
- Adaptive rank实现: 8-12h
- Dev运行: 6h
- 分析: 1h
- Held-out: 2h
- **总计: ~19-23h 到判决**

**方案C（测量论文）**:
- 无需新实验
- 直接撰写: ~数天

---

**我的建议**: 选项A（PSR+SCG两阶段）

理由：
1. 快速验证组合方法是否有效
2. 如果成功，仍是正结果（两个已验证机制的组合）
3. 如果失败，明确adaptive rank是必需的，再投入实现
4. 风险最低，回报仍可观

**等待用户指示下一步行动。**
