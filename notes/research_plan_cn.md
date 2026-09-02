# 针对老板 comments 的论文重构说明

## 一句话版本

新的主张不是“遗忘本质上全部是容量不足”，而是：在局部二次模型下，历史性能下降可以精确拆成分布冲突、固定 rank 的表示误差，以及在线分配/优化误差；当独立的历史敏感方向不断出现时，零二阶干扰空间会收缩，因此固定容量会逐步变成约束。

## 当前 claim 强度

- **理论上可以 claim**：在线最终模型相对逐窗口 oracle 的 gap 可精确拆成 conflict、fixed-rank capacity 和 online error；二次局部模型下 capacity 是曲率白化谱尾；在明确的 non-cancellation/innovation 条件下，旧损失在期望上累积。
- **必须弱化**：不能说全部 forgetting 都由 capacity 导致，也不能说 sequential training 必然产生单调遗忘；top-k 只对固定候选、投影驻点和块对角 screening surrogate 最优。
- **目前只是 tentative**：谱尾和 safe-space 是否能预测真实 LLM forgetting、FCRA 是否优于同预算 baseline、曲率加权 consolidation 是否优于 DELLA/TIES，都还没有实验结果。
- **当前证据状态**：只有定理、反例和实验 protocol，没有 figure/table/result；正文因此不写任何 SOTA、显著提升或数值结论。

## Comments 如何落实

### 1. “Forgetting is a capacity allocation problem?”

改成可证明、不会被反例击穿的版本：

\[
\text{retention gap}
=\text{conflict}+\text{fixed-rank capacity}+\text{online error}.
\]

- `conflict`：即使 rank 无限，一个无条件模型也可能无法同时满足互相矛盾的目标。
- `capacity`：历史混合风险在 rank-
  \(R\) 比较类中的最优值与全空间最优值之差。
- `online error`：没有完整历史数据、曲率估计误差、槽位分配和非凸优化造成的差距。

因此容量是一个可隔离、可测量的来源，但不是遗忘的唯一来源。

### 2. “How can we formalize forgetting? The linear term disappears?”

对旧窗口损失在旧 checkpoint 处展开：

\[
F_{s\to t}=g_s^\top d_{s,t}
+\frac12 d_{s,t}^\top H_s d_{s,t}+r_{s,t}.
\]

只有当旧 checkpoint 对“同一个被评估的旧损失”在可行更新空间内满足投影驻点条件时，线性项才对所有可行位移消失。中间 checkpoint、换了评估分布，或只优化了当前窗口时，不能自动删掉线性项。正文 Lemma 1 和附录给出了三阶余项界。

### 3. “Need a few theorems to prove the capacity problem.”

当前定理链为：

1. 局部遗忘展开与线性项消失的充要条件；
2. 二次旧损失下的 sequential path identity；
3. 非抵消条件下的期望累积定理，以及 `+1,-1` 恢复反例；
4. 共享 Hessian 二次模型下，从 mixture-weight drift 到遗忘的显式桥接；
5. conflict-capacity-online 精确分解；
6. Kronecker 二次模型下 capacity 等于曲率白化谱尾，并扩展到跨层 global rank 分配；
7. 历史曲率零空间和尺度不变近似安全维数单调收缩；
8. 达到固定新任务收益所需支付的最小历史曲率代价；
9. 块对角曲率下 top-k rank 槽位筛选的最优性和近似块对角时的误差因子。

### 4. “Sequential training will gradually cause forgetting.”

不能无条件证明。精确二次旧损失下：

\[
F_{s\to t}-F_{s\to t-1}
=d_{s,t-1}^\top H_s h_t+\frac12\|h_t\|_{H_s}^2.
\]

第一项可能为负并抵消第二项。论文只在 non-cancellation 或独立随机创新条件下证明期望累积；实验必须直接记录 cross term，而不是预设遗忘必然单调。

### 5. “Apart from expanding rank, can we start large and use only part?”

这成为算法主体：

- 训练开始前固定总 rank \(R\)，不再增长；
- 每个窗口最多激活 \(k\ll R\) 个 rank-one atoms；
- 槽位分成 free、reusable、protected、recyclable；
- 按“当前收益 / 历史曲率代价”选择槽位；
- 容量满时先做 curvature-weighted consolidation，再决定是否回收尾部槽位；
- 推理时把所有非零 atoms 合成一个 adapter，不使用 task ID 或 oracle router。

需要强调：参数槽位互不重叠不等于函数输出互不干扰。真正的局部保护来自历史梯度和曲率约束。

### 6. “Can this solve model merging, e.g. DELLA?”

只能证明一个局部特例，不能写成“解决 model merging”。在参数对齐、共享曲率、Kronecker 二次历史损失下，曲率白化后的 truncated SVD 是 rank-
\(R\) merge/consolidation 的最优解。这个结果可以解释和比较 DELLA/TIES，但不覆盖独立训练模型的排列对称、特征不对齐和大步非线性效应。

因此：DELLA 是 consolidation 的次级 baseline，不是主要 continual-learning baseline。

### 7. “Task-by-task PEFT is not a realistic comparison.”

主设置改成 task-identity-free、clocked temporal mixture（窗口由部署时钟给出，但不提供语义 task/source ID）：

\[
P_t=\sum_c\pi_{t,c}Q_{t,c}.
\]

每个窗口可以同时混合多个来源，只改变比例、来源内容或支持集合；损失与输出语义保持一致，不向模型提供 task/source ID。金融情感流只作为低冲突对照；真实容量压力主要由 mixed-skill instruction stream 和 synthetic stream 检验，TemporalWiki/StreamingQA 用于时间知识流。TRACE/Super-NI 只作为补充 task-incremental 压力测试。

## 最近工作带来的定位风险

2026 年的 E2-LoRA 已经做 energy concentration 与动态释放容量；ProCL 已经做 program-memory slot reuse；NSR 已经做压缩、检索和 subspace reallocation；LiteLoRA 已经判断复用还是增加 adapter；SLICE 已经做 replay-aware gradient surgery。

所以不能把“动态分 rank”或“复用空槽”单独当作核心 novelty。论文更有希望的定位是：

1. task-identity-free mixture drift，而非清晰 task sequence；窗口边界固定但不代表语义 task boundary；
2. one-adapter、fixed total rank、no router；
3. conflict-capacity-online 的严格分解；
4. sequential accumulation 的条件与反例；
5. safe-space contraction 和理论量对真实 forgetting 的预测实验。

## 当前尚未完成

- 尚未实现算法；`FCRA` 只是暂定名。
- 尚未运行任何实验，正文没有填结果。
- 需要先做一个线性/小模型 synthetic sanity check，验证谱尾、safe dimension 和 forgetting 的关系。
- 需要检查已有日志中的 requested rank 和实际导出 rank 是否一致；历史上出现过 rank 配置与实际 checkpoint 不一致的风险。
- 只有在严格 resource matching 后，才能讨论比 baseline 更好。

## 下次和老板需要确认的三个决策

1. 是否接受“capacity 是可分离来源而非全部原因”作为核心主张；
2. 是否接受 task-identity-free、clocked mixture stream 作为主要现实设置，并把 task-incremental sequence 降为补充实验；
3. 先投入 synthetic theorem check，还是同时启动 mixed-skill 小模型 pilot 来验证真实数据是否确实产生高 innovation rank。
