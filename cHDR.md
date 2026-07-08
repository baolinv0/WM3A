# Experiment Plan

## Counterfactual Exposure Valuation for Adaptive Multi-Exposure Acquisition

按照当前 ARIS 流程，我们现在位于：

[
\text{research-refine}
\rightarrow
\boxed{\text{experiment-plan}}
\rightarrow
\text{run-experiment}
\rightarrow
\text{auto-review-loop}
]

`experiment-plan` 的原则不是列出大量实验，而是建立 **Claim → Evidence → Run Order → Decision Gate**；默认控制核心 claim 数量和实验 block 数量，并且每个实验都要明确成功标准与失败解释。

结合我们与导师的讨论，我建议把实验严格分成：

[
\boxed{
\text{Stage 0: Problem Feasibility}
\rightarrow
\text{Stage 1: Value Predictability}
\rightarrow
\text{Stage 2: Novelty Isolation}
\rightarrow
\text{Stage 3: Generalization}
}
]

其中 **Stage 0 不通过，项目立即停止**。

---

# 1. Frozen Problem and Method Thesis

## Problem

现有 flexible-frame fusion 可以处理不同数量的已采集曝光帧，但没有回答：

> 当前已有 measurements 下，哪一个 candidate exposure 值得继续采集？

FreeMEF 的 RSSM 确实支持 arbitrary input length，并递归积累全局 hidden state (H_t)，这使它在概念上非常适合作为 acquisition state 的候选来源；但论文的目标仍然是 flexible fusion，而不是主动 acquisition。([arXiv][1])

---

## Method Thesis

> **The value of a future exposure should be estimated relative to the information already accumulated in the current fusion state, rather than directly predicting the next exposure action.**

形式：

[
C_t
\rightarrow
H_t,
]

[
(H_t,a)
\rightarrow
\widehat{\Delta Q}(a|C_t),
]

最终：

[
a_t^*
=====

\arg\max_a
[
\widehat{\Delta Q}(a|C_t)-c(a)
].
]

第二阶段候选方法进一步为：

[
(H_t,a)
\rightarrow
\widehat G_t^a(x),
]

其中 (G_t^a(x)) 是空间 counterfactual complementarity field。

---

# 2. Claim Map

| Claim                                                 | 类型               | 最小可信证据                                             | 对应实验  |
| ----------------------------------------------------- | ---------------- | -------------------------------------------------- | ----- |
| **C0**：Adaptive acquisition 存在足够 headroom             | feasibility      | Oracle 显著优于 strong heuristic / best fixed policy   | B1    |
| **C1**：Candidate value 可由 current state + action 预测   | primary claim    | Fusion-state predictor 显著降低 action regret          | B3    |
| **C2**：Spatial complementarity 比 scalar value 更有价值    | supporting claim | 相同 backbone 下，在 OOD/variable budget 中稳定优于 scalar Q | B4/B5 |
| **Anti-claim A**：收益只是来自更大 action pool                 | 排除项              | 所有方法共享完全相同 action set 和 frame budget               | B1    |
| **Anti-claim B**：Spatial map 只是可视化 head               | 排除项              | 必须提升 ranking/regret/rollout，而非只提供好看的 map           | B4    |
| **Anti-claim C**：Continuous action query 自动 zero-shot | 排除项              | 必须在 unseen EV interpolation 上实验证明                  | B5    |

这里我刻意把：

> Spatial ECF 是核心贡献

放到了**待证明状态**。

当前真正冻结的 primary claim 仍然只是：

[
\boxed{
\text{candidate measurement value is predictable}
}
]

---

# 3. Experimental Contract

在所有实验开始前，先冻结以下条件。

## 3.1 Action Space

Stage 0 先研究 exposure-level acquisition，而不是同时优化完整 ISP action space。

初始：

[
\mathcal A
==========

{-4,-3,-2,-1,+1,+2,+3,+4}\ EV
]

固定：

[
I^0
]

作为 initial base exposure。

总 frame budget：

[
B\in{1,2,3,4}.
]

只有 Stage 0 通过后，才扩展：

[
a=
[
\Delta EV,\log T,\log G
].
]

原因是先回答：

> adaptive acquisition 是否有价值？

再回答：

> shutter / gain 如何联合控制？

不要同时解决两个问题。

---

## 3.2 Dataset Strategy

### Primary：HDR video / HDR scene source + exposure simulation

优先复用 AdaptiveAE 已经使用的 HDRV 与 DeepHDRVideo 体系，并复用其 blur/noise synthesis 思路构建 candidate exposure pool。AdaptiveAE 已经在这两个 benchmark 上进行评估，并明确设计了同时考虑 motion blur 和 noise 的 synthesis pipeline。

这比直接采用普通静态 MEF dataset 更适合当前问题，因为我们需要：

[
\text{same underlying scene}
+
\text{multiple candidate measurements}.
]

### Stage 0

先生成 EV pool。

### Stage 1 后

再引入：

[
(T,ISO)
]

分解，从而研究：

* noise；
* motion blur；
* same EV but different capture quality。

---

## 3.3 Fusion Backend

原则：

# **Stage 0 必须冻结 downstream fusion backend。**

不能一边修改 acquisition，一边修改 fusion。

理想主 backend 是 variable-frame fusion model，因为 FreeMEF 明确支持可变输入长度，并将 exposure features 递归积累到 (H_t)。([arXiv][1])

但当前 arXiv 页面没有直接列出官方代码链接，因此**不能让 Stage 0 被完整复现 FreeMEF 阻塞**。([arXiv][1])

建议：

### Stage 0

使用一个已跑通、冻结、能够处理不同 exposure subset 的 fusion evaluator。

### Stage 1

涉及：

[
H_t
]

的 representation claim 时，再使用 FreeMEF/RSSM 类 recurrent fusion state。

这是一个很重要的工程纪律：

> **Oracle headroom test 不依赖新方法 backbone；state representation test 才依赖。**

---

# 4. Ground-Truth Counterfactual Value Construction

给定完整 exposure pool：

[
\mathcal E
==========

{I^{a_1},...,I^{a_K}}.
]

当前已采集集合：

[
C_t\subset \mathcal E.
]

当前结果：

[
Y_t=\mathcal F(C_t).
]

加入 candidate (a)：

[
Y_{t,a}
=======

\mathcal F(C_t\cup I^a).
]

---

## Scalar Ground Truth

[
v_t^{a*}
========

## Q(Y_{t,a},Y^*)

Q(Y_t,Y^*).
]

---

## Spatial Ground Truth

建议不要直接使用 local PSNR，而使用稳定的 pixel-wise perceptual-domain error difference：

[
E_t(x)
======

\rho
\left(
\mu(Y_t(x))-\mu(Y^*(x))
\right),
]

[
E_{t,a}(x)
==========

\rho
\left(
\mu(Y_{t,a}(x))-\mu(Y^*(x))
\right),
]

然后：

[
\boxed{
G_t^{a*}(x)
===========

E_t(x)-E_{t,a}(x)
}
]

其中：

* (G>0)：candidate 有帮助；
* (G\approx0)：冗余；
* (G<0)：加入该 measurement 可能损害结果。

AdaptiveAE 的 HDR evaluation 已使用 PSNR-(\mu)、SSIM-(\mu)、PU-PSNR、PU-SSIM 和 HDR-VDP-2；这些指标可以作为最终 quality evaluation，而不是全部混入训练标签。

---

# 5. Experiment Block 1 — Oracle Headroom

## B1. Oracle Acquisition Feasibility

### Claim tested

[
\boxed{
\text{Adaptive selection has meaningful headroom beyond strong heuristics}
}
]

---

## Compared Systems

### 1. Best Fixed-(B)

不是随便选择：

[
{-2,0,+2}
]

而是在 training split 搜索：

[
S_B^*
=====

\arg\max_{|S|=B}
\mathbb E_{\text{train}}Q(\mathcal F(S)),
]

然后固定到 test。

这是更强的 fixed baseline。

---

### 2. Random-(B)

相同 action pool、相同 frame budget。

---

### 3. Strong Histogram/Coverage Heuristic

维护 current reliable coverage：

[
M_t(x).
]

根据 current observations 估计 candidate exposure 后的可靠区间：

[
\tilde M_a(x).
]

定义：

[
S_{\mathrm{heur}}(a)
====================

\sum_x
(1-M_t(x))
\tilde M_a(x)
-------------

## \lambda_s R_{\mathrm{sat}}

\lambda_n R_{\mathrm{noise}}.
]

它回答：

> 哪个 EV 可以覆盖最多尚未可靠观测的区域？

AdaptiveAE 的 related-work section 也将 histogram-based exposure strategy 和 SNR-based optimization 视为重要传统强基线，而不是仅与 random/fixed 比较。

---

### 4. Oracle Greedy

[
a_t^*
=====

\arg\max_a
v_t^{a*}.
]

---

### 5. Oracle Sequence/Subsets

在 budget (B) 下：

[
S_B^*
=====

\arg\max_{|S|\le B}
Q(\mathcal F(S)).
]

这是 theoretical upper bound。

---

## Primary Metrics

### Quality–Capture Pareto

横轴：

[
\text{Number of Captures}
]

纵轴：

[
Q_{\mathrm{fusion}}.
]

主指标：

* Pareto dominance；
* Area Under Quality–Cost Curve；
* quality at fixed (B)；
* frames required for fixed quality threshold。

---

## Decision Gate A

### STOP

如果：

[
Oracle\ Greedy
\approx
Strong\ Heuristic
]

且没有：

* quality gain；
* frame reduction；
* latency advantage。

特别是如果差距只有约 (0.1\sim0.2) dB，同时 frame count 没有优势，继续发展复杂 learning model 的价值很低。

### STRONG GO

导师建议的一个很强 signal 是：

[
Oracle\ Greedy

>

Strong\ Heuristic
]

并且在严格 frame budget 下出现明显差距，例如 2-frame 限制下接近或超过约 1 dB 的质量优势。

但正式判据应该采用 **Pareto dominance**，而不是机械固定一个 dB 数字。

---

# 6. Experiment Block 2 — Utility Structure and Horizon

## B2. Is the Acquisition Problem Structurally Learnable?

这一块不是训练网络，而是分析 oracle tensor：

[
V^*[scene,C_t,a].
]

---

## Analysis A — Scene Dependence

计算不同 scene 的：

[
a^*=\arg\max_a V^*(C,a).
]

看 optimal action distribution 是否足够多样。

如果 80% 以上场景最终都选择同一个 EV：

> deep adaptive model 的价值值得怀疑。

---

## Analysis B — State Dependence

对同一个 scene：

[
V(a|C_1)
]

和：

[
V(a|C_2)
]

是否显著变化？

关键问题不是：

> 不同 scene 是否不同。

而是：

> **随着 measurement 已经被加入，下一步 action preference 是否会发生变化。**

---

## Analysis C — Approximate Diminishing Returns

不要预设 submodularity，而是测量：

对于：

[
A\subset B,
]

统计是否：

[
\Delta(a|A)
\ge
\Delta(a|B).
]

定义 violation ratio：

[
R_{\mathrm{viol}}
=================

P[
\Delta(a|A)<\Delta(a|B)
].
]

并统计 violation magnitude。

这能回答导师提出的：

> exposure acquisition 是否近似满足 diminishing returns？

---

## Analysis D — Greedy Horizon Gap

定义：

[
Gap_B
=====

## Q(S_B^{oracle})

Q(S_B^{greedy}).
]

### Outcome A

[
Gap_B\approx0.
]

结论：

> CUT World Model。

### Outcome B

[
Gap_B
]

持续显著：

> one-step valuation 不充分，返回 `/research-refine`，重新研究 multi-step prediction。

---

# 7. Experiment Block 3 — Predictability

只有 Gate A 通过才启动。

## B3. Can Future Measurement Value Be Predicted?

### Target

先预测 scalar：

[
(H_t,a)
\rightarrow
\hat v(a).
]

不要直接做 Spatial ECF。

---

## Compared Representations

### P1 — Histogram + Action

[
[h(C_t),\phi(a)]
\rightarrow
\hat v.
]

---

### P2 — Image Feature + Action

[
[z(C_t),\phi(a)]
\rightarrow
\hat v.
]

---

### P3 — Fusion State + Action

[
[H_t,\phi(a)]
\rightarrow
\hat v.
]

---

### P4 — Fusion State + Completeness Mask

仅作为条件性 ablation：

[
[H_t,M_t,\phi(a)]
\rightarrow
\hat v.
]

注意：

> Dual-path completeness representation 不是默认方法。

只有 P3 明显不足、P4 明显改善时，才保留。

---

## Action Query

第一版：

[
\phi(a)=MLP(\Delta EV).
]

第二版：

[
\phi(a)
=======

MLP[
\Delta EV,\log T,\log ISO
].
]

使用 FiLM 或轻量 cross-attention 均可，但第一阶段应使用最简单实现。

---

## Metrics

### Primary

#### Action Regret

[
R_{\mathrm{action}}
===================

V^*(a^*)-V^*(\hat a).
]

这是最关键指标。

### Secondary

* Spearman rank correlation；
* Top-1 accuracy；
* Top-2 recall；
* rollout final quality；
* average frame count。

---

## Decision Gate B

### REFRAME

如果：

[
Oracle\gg Heuristic
]

但：

[
P3\approx P1
]

说明 fusion state 没有 acquisition-state 优势。

---

### SIMPLE GO

如果 scalar：

[
Q(H,a)
]

已经高度可预测 measurement value：

继续做简单 valuation method。

不要因为 spatial map 看起来漂亮而强行复杂化。

---

# 8. Experiment Block 4 — Novelty Isolation

## B4. Direct Policy vs Scalar Value vs Spatial Complementarity

这是整篇论文最关键的 method experiment。

所有模型必须共享：

* 相同 state encoder；
* 相同 training scenes；
* 相同 action pool；
* 相近 parameter budget；
* 相同 acquisition rollout。

---

## M1 — Direct Policy

[
\pi(a|H_t).
]

建议先使用 oracle action imitation supervision，而不是 RL。

这样可以排除：

> Ours 只是因为 RL 更难训练所以赢了。

---

## M2 — Scalar Value

[
Q(H_t,a)
\rightarrow
\hat v_a.
]

---

## M3 — Spatial Complementarity

[
G(H_t,a,x)
\rightarrow
\hat v_a.
]

训练：

[
\mathcal L
==========

\mathcal L_{\mathrm{field}}
+
\lambda_r
\mathcal L_{\mathrm{rank}}.
]

---

## Success Criterion

Spatial model 必须至少在一个关键维度明显超过 scalar Q：

* lower action regret；
* better unseen-action interpolation；
* stronger unseen-subset generalization；
* better variable-budget rollout；
* better motion-shift robustness。

如果只有：

```text
Spatial map 更漂亮
Scalar ranking 基本一样
```

则：

# CUT Spatial ECF.

---

# 9. Experiment Block 5 — Compositional Generalization

## B5. Does Valuation Learn Measurement Relation or Memorize Action Preference?

这一块决定论文是否具有真正的 representation claim。

---

## OOD-1 — Unseen Exposure Actions

Train：

[
{-4,-2,0,+2,+4}.
]

Test：

[
{-3,-1,+1,+3}.
]

更强版本测试：

[
\pm1.5EV.
]

注意：

> continuous physical query 不自动等于 zero-shot generalization。

实验必须证明。

---

## OOD-2 — Unseen Initial EV

训练：

[
I^0
]

作为 base。

测试：

[
I^{-1}
\quad\text{or}\quad
I^{+1}
]

作为 initial measurement。

---

## OOD-3 — Unseen Current Subset Composition

训练主要看到：

[
{0},
{0,-2},
{0,+2}.
]

测试：

[
{0,-3},
{0,-1,+2},
...
]

---

## OOD-4 — Variable Budget

Train：

[
B\le3.
]

Test：

[
B=1,2,4.
]

比较 direct policy 与 valuation model 是否能够自然重新评估 candidate value。

---

## OOD-5 — Motion Shift

利用 HDR video + blur/noise simulation，将测试场景按 motion magnitude 分组。

AdaptiveAE 已经证明其 exposure-control evaluation 会随 motion level 变化，并专门分析不同 motion magnitude 下的 robustness，因此这是一个合理的物理 OOD 维度。

---

# 10. Paper Storyline

## Main paper 必须证明

1. **Oracle headroom 存在**；
2. **Candidate value 可预测**；
3. **Explicit valuation 优于 direct policy**；
4. 如果使用 spatial ECF，必须证明其优于 scalar Q；
5. 至少一个 meaningful compositional/OOD advantage。

---

## Appendix 可以放

* completeness mask；
* alternate fusion backend；
  -更多 action embedding；
  -更多 quality metrics；
  -更多 spatial visualization；
  -小规模 real-camera cases。

---

## 当前明确 CUT

* World Model；
* RL；
* HDR generation；
* diffusion；
* tone mapping co-design；
* AE/AWB/AF joint control；
* generic camera agent。

---

# 11. Run Order and Decision Gates

| Milestone | 内容                                     |       时间 | Gate              |
| --------- | -------------------------------------- | -------: | ----------------- |
| **M0**    | 数据、exposure pool、fusion、metric sanity  |     2–3天 | pipeline 正确       |
| **M1**    | Oracle Headroom + Strong Heuristic     |     5–7天 | Gate A            |
| **M2**    | Greedy vs Sequence + utility structure |     3–5天 | 决定是否需要 multi-step |
| **M3**    | Scalar Predictability                  |    7–10天 | Gate B            |
| **M4**    | Policy vs Scalar vs Spatial            |    7–10天 | Gate C            |
| **M5**    | OOD / variable budget                  |     5–7天 | Paper GO          |
| **M6**    | alternate backend / real capture       | optional | polish            |

因此：

# Stage 0 = M0 + M1 + M2

符合导师提出的约两周 feasibility window。

---

# 12. Compute Budget

以下以：

* 单张 24–48 GB GPU；
* 256–512 crop；
* frozen fusion backbone；
* lightweight value predictor；

为规划假设。

| Stage                      | 粗略 GPU-hours |
| -------------------------- | -----------: |
| M0 sanity                  |         5–10 |
| Stage 0 oracle enumeration |        20–60 |
| Scalar predictability      |        30–80 |
| Novelty isolation          |       60–150 |
| OOD                        |        30–80 |

完整进入 paper development 前：

[
\boxed{
145\sim380\ GPU\ hours
}
]

这个范围高度依赖 fusion backbone 推理速度和 exposure subset 数量，因此仅作为资源规划上限，不是结果承诺。

学习模型阶段建议使用 3 seeds；Stage 0 的 deterministic oracle 不需要 seed，random baseline 则应多次重复采样。`experiment-plan` skill 本身也建议随机方差重要时默认使用 3 seeds。

---

# 13. Experiment Tracker

| Run ID | Milestone | Purpose                                   | Priority          | Status            |
| ------ | --------- | ----------------------------------------- | ----------------- | ----------------- |
| R001   | M0        | 1 scene exposure pool sanity              | MUST              | TODO              |
| R002   | M0        | subset → fusion → ΔQ validation           | MUST              | TODO              |
| R003   | M1        | 20–50 scene pilot: Fixed/Heuristic/Oracle | MUST              | TODO              |
| R004   | M1        | full Oracle Pareto                        | MUST              | BLOCKED by R003   |
| R005   | M2        | Greedy vs Sequence                        | MUST              | BLOCKED           |
| R006   | M2        | diminishing-return analysis               | MUST              | BLOCKED           |
| R007   | M3        | Histogram scalar predictor                | MUST              | BLOCKED by Gate A |
| R008   | M3        | Fusion-state scalar predictor             | MUST              | BLOCKED           |
| R009   | M4        | Direct policy baseline                    | MUST              | BLOCKED           |
| R010   | M4        | Spatial ECF                               | CONDITIONAL       | BLOCKED           |
| R011   | M5        | unseen action test                        | MUST if M4 passes | BLOCKED           |
| R012   | M5        | variable-budget test                      | MUST if M4 passes | BLOCKED           |

---

# 14. First Three Runs to Launch

当前最合理的前三个 run 是：

## R001 — Single-Scene Sanity

选择 1 个 scene：

[
I^0
+
{-4,-2,+2,+4}.
]

手动验证：

* exposure ordering 正确；
* fusion result 合理；
* (\Delta Q) sign 正确；
* candidate gain 不存在 metric bug。

---

## R002 — Small Subset Enumeration

选择约 10 个 scene。

计算：

[
V^*[scene,C,a].
]

先回答：

> optimal candidate 是否真的随 scene/current subset 改变？

不训练任何模型。

---

## R003 — Small Oracle Headroom Pilot

选择约 20–50 个 scene。

比较：

[
Best\ Fixed
]

[
Strong\ Heuristic
]

[
Oracle\ Greedy
]

[
Oracle\ Sequence.
]

只有这张 pilot table 出现明显 headroom，才扩大 Stage 0。

---

# 15. 与 ARIS 后续流程的衔接

当前只完成 `/experiment-plan`。

进入 `/run-experiment` 时，应先进行环境检测、GPU pre-flight 和代码/环境一致性检查，再部署实际 job；该 skill 明确把 environment detection 和 pre-flight 放在实验启动之前。

而 `/auto-review-loop` 应在已经有：

* claims；
* method/code；
* raw experiment results；
* logs；

之后再启动，因为其 review loop 明确要求 reviewer 直接检查这些 artifacts，并在后续轮次持续验证修改是否真正解决问题。

---

# 最终执行判断

当前项目不应该从：

[
\text{ECF network design}
]

开始。

应该从：

[
\boxed{
\text{Oracle Value Tensor Construction}
}
]

开始。

整个项目的第一条真正实验链是：

[
\boxed{
\begin{aligned}
&\text{Exposure Pool}\
\rightarrow;&V^*(scene,C,a)\
\rightarrow;&Oracle\ vs\ Heuristic\
\rightarrow;&Greedy\ vs\ Sequence\
\rightarrow;&\text{Gate A}
\end{aligned}
}
]

**Gate A 通过后，才允许训练第一个 scalar predictor；scalar predictability 通过后，才允许开发 Spatial ECF。**

这是目前最符合导师意见、ARIS pipeline 和我们前面所有 novelty/review/refine 结论的实验顺序。

[1]: https://arxiv.org/abs/2606.27905 "There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion"
