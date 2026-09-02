# Phase-2R — Δ_id 的两个敏感性：冻结协议

Frozen 2026-08-31，**在任何 Phase-2R 结果量被计算之前**。本轮只查看了
*设计覆盖*（哪些 (task, stage) 格子存在、scope 活跃成员数序列、任务训练位置），
未读取任何 `R_*` / `q_m` 数值。这一区分是本协议成立的前提，§6 给出可核验的
落地方式。

## 0. 为什么是这两个量，以及为什么不是新方法

§8 的 "What we would do next" 指出，最有价值的下一步不是第四个干预，而是两个
敏感性。2N §5 outcome 4 与 2Q §0 都禁止再造正则项，本协议遵守：**不训练任何
新 arm，不改训练目标，不引入超参**。

两个问题：

* **S-A（规模）**：Δ_id 是否随共享同一 verbalizer 的任务数增长？
* **S-B（陈旧度）**：2P 测到的 staleness 是否随流长增长，如机制所预测？

## 1. 这是重分析，不是重跑

存量 run 的每个 stage 都已记录每个任务的 `R_raw / R_shr_global /
R_shr_scoped / R_orc`（2P 另有 `R_sso1 / R_sso4`）以及 `scope_radii[m]`。
盘上**没有 adapter checkpoint**（只有 HF 原始权重），因此重新打分不可能；
但两个问题所需的量都已在 JSON 里，所以不需要重新打分。

用到的 run，全部已存在、不新跑：

| run | seeds | 用途 |
| --- | --- | --- |
| `runs/phase2p_sso_s{1,2,3}` | 1,2,3 | S-B 主分析（唯一含 `R_sso1`） |
| `runs/phase2k_qoc_converged` | 1,2,3 | S-A 主分析（`--cl-method none` 基线） |
| `runs/phase2q_bpo_s{1,2,3}` | 1,2,3 | S-A 复现臂（同为 `none`，独立 seed 流） |
| `runs/phase2m_vla_lam05` | 1,2,3 | 仅作 S-A 的 arm 不变性检查，不合并 |

GPU 占用为零。这一点本身是判据：若任何分析需要新 run，见 §5 outcome 4。

## 2. 度量定义，逐 stage

记 `t` 为 stage 位置（1..15），`g` 为任务，`pos(g)` 为 `g` 被训练的位置。

* **age** `a = t − pos(g)` ≥ 0。`a=0` 是刚训完。
* **m_live(S, t)** = 该 stage `scope_radii["1"][S]["members"]` 的长度，即
  **当时**已进入该 scope 的任务数。
* **Δ_id_global(g,t)** = `R_orc − R_shr_global`。
* **Δ_id_scoped(g,t)** = `R_orc − R_shr_scoped`。
* **stale(g,t)** = `|R_sso1 − R_orc|`，**仅在单例 scope 上**，此时 Prop 1 保证
  Chebyshev 中心即该元素，故此量是纯陈旧度（2P P5 的逐 stage 版本）。

**一个已发现的陷阱，冻结在此以免复犯。** 记录里的 `scope_size` 字段是该 scope
的**最终**规模（对 `False|True` 恒为 4，即使在只有 WiC 一个成员的 stage 3），
**不是**活跃规模。任何以 `scope_size` 当作 m 的分析都是错的。m 只能从
`scope_radii[m].members` 取。本协议的所有 m 均如此取值，并由 §6 的测试钉住。

只计 `scorable == True` 的任务，与全文一致（Yelp/Amazon/CB/WiC 的排除理由见
附录 app:exclusions）。

## 3. 冻结判据

3 seeds，cluster bootstrap，**任务为 cluster**，10 000 draws，与全文同一套统计。

* **R1（S-A 主判据）.** 在多任务 scope 上，把 `Δ_id_global(g,t)` 对 `m_live`
  回归（斜率为统计量，cluster bootstrap CI）。**R1 通过 = 斜率 CI 排除 0 且为
  正**。这是"gap 随共享任务数增长"的直接检验。
* **R2（S-A 的 scope 对照）.** 同一回归换成 `Δ_id_scoped`。Prop 1 与 2K/2P/2Q
  已测得 scoped gap 在单例上恒 0；若 R2 的斜率也显著为正，则 per-scope offset
  并未随规模饱和，这与 §6 的 "+1.51 pp" 叙事**冲突**，须按 outcome 3 处理。
  **预先声明的预期：R2 不显著。**
* **R3（S-B 主判据）.** 在单例 scope 上，把 `stale(g,t)` 对 `a` 回归。
  **R3 通过 = 斜率 CI 排除 0 且为正**，即陈旧度随后续任务数增长。
* **R4（S-B 的混淆对照，必须与 R3 一起报告）.** 同一回归改用 `pos(g)` 作
  自变量、`a` 作协变量之外的单独模型。设计上 `a` 与 `pos` 强负相关（见 §4），
  故 R3 的斜率可能只是"早训的任务本来就更差"。**R4 通过 = `pos` 单独的斜率
  CI 包含 0**（即位置本身不解释）。R3 通过而 R4 失败时，**不得**宣称随流长
  增长；见 outcome 2。
* **R5（Prop 1 接线检查）.** 所有单例 scope 上 `Δ_id_scoped ≡ 0` 精确成立。
  违反则本协议每个数字作废。与 2K/2M/2P/2Q 的同名检查同口径。
* **R6（arm 不变性，报告不设阈）.** R1 在 `phase2k`、`phase2q` 两个独立 `none`
  臂上分别算，符号一致则记录；不一致则 outcome 3。`phase2m_vla_lam05` 单独
  报告，永不与 `none` 臂合并（§A2.6 的规则）。

## 4. 功效限制，先声明后计算

设计是观测性的，不是随机化的。四个已知限制：

1. **S-A 的 m 只有 {1,2,3,4}，且只来自两个 scope。** `False|True` 走
   1→2→3→4，`Bad|Good` 走 1→2。所有其他 scope 恒为 1。因此 R1 的斜率由两个
   scope 承载，**任务身份与 m 无法完全解耦**。
2. **m_live 与 t 高度共线。** scope 只增员不减员，故 m 随 t 单调不减。R1 的
   斜率无法与"越晚越差"分离。这是 R4 存在的同一问题在 S-A 侧的镜像，我们
   报告它而不假装解决了它。
3. **S-B 的 age 覆盖极不均衡。** 5 个可打分单例任务，训练位置
   COPA p4、RTE p7、DBpedia p12、AGNews p13、Yahoo p15；age 跨度分别为
   0..11、0..8、0..3、0..2、0..0。**a ≥ 9 的格子只由 COPA 一个任务提供**
   （seed1 计数：age 9/10/11 各 1 obs）。因此大 age 端的斜率等于 COPA 的
   个体轨迹。R4 正是为此而设。
4. **单流单序（Order-4）。** 任何"随流长增长"的结论都是这一个任务序上的，
   不是关于流长的一般命题。

这四条会原样进入 §7 limitations，无论结果如何。

## 5. 预先声明的结果

1. **R1 与 R3 通过、R4 通过、R5 通过.** 两个敏感性都成立且不被位置混淆解释。
   Δ_id 随共享规模增长、陈旧度随流长增长，两者都是 §6/§8 已断言但未测的内容，
   届时把"as the mechanism predicts"改为已测，并给出斜率与 CI。
2. **R3 通过但 R4 失败.** 陈旧度与 age 相关，但位置本身也解释它。则
   §6 line 279 的 "the decay grows with the number of subsequent tasks"
   **必须降级**为"在本流上，陈旧度随 age 增长，但与训练位置混淆，我们无法分离"。
   这是最可能的结果，写在此处以免事后被当作发现。
3. **R2 显著为正，或 R6 两臂符号不一致.** 与 §6 现有叙事冲突。按全文惯例：
   保留冲突、appendix 记录、不改旧文本，并在 §7 增加一条。
4. **任何判据需要新 run 才能评估.** 则本协议以"unevaluable without new
   compute"结案，明确写出所缺字段，**不**顺手补跑（补跑属于新协议，需重新冻结）。
5. **R5 失败.** 全部作废，先修接线。

## 6. 反伪影检查，在任何判据被引用之前

* **只读记录，不写记录.** 分析脚本对 `runs/` 只以只读模式打开；不产生任何新
  `runs/` 目录。
* **m 不取自 `scope_size`.** 由测试断言：构造一个 `scope_size=4` 而
  `scope_radii` 只有 1 个 member 的 fixture，断言分析取到 1 而非 4。这是 §2
  记录的那个陷阱的回归测试。
* **age 非负且 `pos` 自 `trained_task` 反推.** 由测试断言，不硬编码任务顺序。
* **单例判定取自 run 自身的 `singleton_scopes`**，不由标签字符串重新推断。
* **bootstrap 以任务为 cluster**，与 `phase2k/decide.py` 同一函数，不另写一份。
* `order4_data.py` 未改动；官方 `test.json` 未读取；无 GPU 作业提交。
* **本轮冻结前只看了设计覆盖.** 具体地：任务训练位置、每 stage 的
  `scope_radii[m].members` 长度、`scorable` 标记、(age, stage) 格子计数。
  **未**读取任何 `R_raw/R_orc/R_shr_*/R_sso*/q_m` 数值。§4 的四条限制全部
  只依赖设计信息，可据此核验。

## 7. 计算

零 GPU。纯 JSON 重分析，本地或远端均可，预计 < 1 分钟。
实现落 `experiments/phase2r_sensitivity/`，判决脚本 `decide_2r.py`，
测试 `tests/test_2r.py`，与前序 phase 同构。

## A1. 确认性结果（修正案，追加于 2026-08-31）

冻结文本一字未改。以下为 §3 判据的逐条求值，判决脚本
`experiments/phase2r_sensitivity/decide_2r.py`，输入为四个存量臂，零 GPU。

### A1.1 判据表（主臂 = `phase2p_sso`，3 seeds，279 obs）

| 判据 | 量 | 斜率 | CI | 判定 |
| --- | --- | --- | --- | --- |
| R1 | Δ_id_global vs m_live | −0.278 pp/成员 | [−1.721, +1.017] | **FAIL**（CI 跨零） |
| R2 | Δ_id_scoped vs m_live | +0.401 pp/成员 | [−0.542, +0.945] | 不显著（= 预期） |
| R3 | stale vs age | +0.387 pp/后继任务 | [−0.051, +3.379] | **FAIL**（CI 跨零，下界擦零） |
| R4 | stale vs pos(g) | −0.262 pp/位置 | [−1.045, +0.207] | PASS（CI 含零） |
| R5 | 单例 Δ_id_scoped ≡ 0 | — | — | PASS，87 obs，0 违反 |

corr(age, pos) = **−0.556**，即 §4 限制 3 预告的混淆确实存在但不极端。

### A1.2 §5 无任何预声明结果逐字命中

主臂上 R3 **失败**，故 outcome 1 与 outcome 2 都不适用（两者都以 R3 通过为
前提）；R2 在主臂不显著，故 outcome 3 也不触发；R5 通过，故 outcome 5 不触发；
所有判据都能求值，故 outcome 4 不触发。判决脚本据此输出 `outcome: null` 并按
§3 逐条报告，这是协议留下的空档，记录在此而不事后补一个分支。

**这意味着两个敏感性都是负结果。** §8 的
"whether the staleness we measured grows with stream length as the mechanism
predicts" 与 §6 line 279 的 "the decay grows with the number of subsequent
tasks" 都**未被本流数据支持**：点估计方向对（+0.387 pp/任务），但 CI 含零。
§6 line 279 必须降级，措辞见 A1.5。

R3 的 CI 下界 −0.0511 极其贴近零。按判据文字这是失败，我们不做"接近显著"的
解读；这类解读正是本项目反复自查要避免的。功效不足是最可能的解释（§4 限制 3：
age ≥ 9 的格子只由 COPA 一个任务提供），但功效不足不等于效应存在。

### A1.3 R6：`phase2k` 与 `phase2p` 不是两个独立臂 —— 协议缺陷

§1 的表把 `phase2k_qoc_converged` 与 `phase2p_sso_*` 列为 S-A 的两个来源，
R6 又要求"两个独立 `none` 臂符号一致"。**这是错的，而且是我在冻结时就该核对
出来的。** 逐位比较两者末 15 stage 的 `R_raw/R_shr_global/R_shr_scoped/R_orc`：
**480 个字段全部逐位相同**（差异 0 个），`n_steps` 序列相同，seed/lr/epochs/
model 相同，2P 只是多了 `record_sso=True` 与 λ=0 的开关。也就是说 2P 是 2K 的
同一训练轨迹加挂 SSO 打分，逐位可复现。

后果：R6 实际只有 **一个** 独立的 `none` 复现臂（`phase2q`），加一个不同 arm
的 `phase2m_vla_lam05`。两者与主臂的 R1 符号一致（全为负，全不显著）：

| 臂 | R1 斜率 | CI | R2 斜率 | CI |
| --- | --- | --- | --- | --- |
| `phase2p`(=`phase2k`) | −0.278 | [−1.721, +1.017] | +0.401 | [−0.542, +0.945] |
| `phase2q` | −0.092 | [−0.659, +0.856] | **+0.613** | **[+0.120, +1.173]** |
| `phase2m_vla05` | −0.188 | [−2.468, +1.619] | +1.267 | [−0.151, +1.950] |

R1 三臂符号一致（负），无一显著。这一致性是真的，但"三臂"实为"两臂 + 一个
不同 arm"，本表按此口径读。

### A1.4 `phase2q` 的 R2 显著性是 m=1 机械零点造成的

`phase2q` 臂上 R2 的 CI 排除零（+0.613, [+0.120, +1.173]），按 §5 outcome 3
本应触发"与 §6 叙事冲突"。做完剔除检验后它不成立：**m=1 时 Δ_id_scoped 恒等
于 0 是 Prop 1 的定义结论，不是测量**（scope 只有一个成员时 Chebyshev 中心即
该成员，故 scoped = orc）。把这些结构性零点放进对 m 的回归，等于给回归塞进一
个必然在 m=1 处为 0 的锚点，人为造出正斜率。

只用 m ≥ 2 重算，三臂全部不显著：

| 臂 | R2 (m≥2) 斜率 | CI |
| --- | --- | --- |
| `phase2p` | +0.119 | [−1.446, +1.270] |
| `phase2q` | +0.714 | [−0.319, +1.889] |
| `phase2m_vla05` | +0.584 | [−1.728, +2.118] |

因此 **outcome 3 不触发**，§6 的现有叙事不受冲突。但这是一个诊断，作出于看到
`phase2q` 显著之后，属事后分析，故：(i) 不用它去救任何判据，(ii) 冻结的 R2
判定按原文写在 A1.1（主臂不显著），(iii) 此处只记录 `phase2q` 的那一处显著性
不可当作发现。同一缺陷也影响 R1 的口径，m≥2 的 R1 三臂仍全不显著
（−0.494 / −0.756 / −0.990，CI 均跨零）。

**给后续协议的教训：** 任何以 m（或任何在某取值处被定义强制为常数的量）作自
变量的回归，必须在冻结时就声明是否剔除该取值。本协议 §2 定义了 Δ_id_scoped
却没说这件事，是同一类疏漏的第二次出现（第一次是 2Q §A1.4 的 `Delta_id_scoped`
未落盘）。

### A1.5 本修正案允许说什么

* 可以说：在 Order-4 上，Δ_id 随共享同一 verbalizer 的任务数增长**未被测到**
  （R1 点估计为负且 CI 跨零，三臂同向）；陈旧度随流长增长**未被测到**
  （R3 点估计为正但 CI 含零），且位置本身也不解释陈旧度（R4 通过）。
* 必须改：§6 line 279 "the decay grows with the number of subsequent tasks,
  which is the one quantity a continual method cannot bound" —— 前半句是未测
  断言，须改为"我们测了它，点估计方向如此但 CI 含零，在本流上无法确立"。
  §8 的 "as the mechanism predicts" 同样须改。
* 不可以说：R3 "接近显著"；或用 m≥2 的重算去替换冻结的判据结果；或把 R1 的
  负斜率读作"gap 随规模下降"（CI 跨零，方向不可解释）。
* 不可以说：本结果否证了 Prop 2/3。R5 通过（87 obs、0 违反）说明几何侧接线
  正确；未测到的是**accuracy 侧的规模依赖**，而 §5 早已声明 κ≈0.018 使
  accuracy 侧的界在本 benchmark 上很弱。R3/R1 的阴性与那一节自述的弱界一致，
  不是新矛盾。

### A1.6 统计实现上一个非冻结的决定

`slope_ci` 对"退化重采样"（池内 x 无变异 ⇒ 斜率无定义）的处理**不在冻结文本
内**：§3 只写了"cluster bootstrap，任务为 cluster，10 000 draws"。首次运行时
它确实发生（全 5 簇都抽中只有单一 age 的 Yahoo），`np.percentile` 于是把所有
CI 变成 nan。

选择是**丢弃退化抽样并报告其比例**，而不是记作斜率 0（后者会把"无定义"伪装成
"无效应"并把 CI 拉向零）。此决定作出于第一次运行之后、已知点估计之后，因此不
是冻结内容。实测比例可忽略：主臂各判据 0.01%–0.20%。输出里
`n_degenerate_draws / frac_degenerate / ci_usable` 逐判据可查，且 1% 以上即
判 CI 不可用（`_verdict` 因此不得判通过）。四条回归测试钉住此行为，其中一条
专门断言"退化不得记作 0"（否则该 fixture 的 CI 下界会从正变负）。
