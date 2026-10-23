# 直接偏好优化家族

> Rafailov 等人（2023）表明，RLHF 的最优解可以写成偏好数据的闭式表达，因此可以跳过显式奖励模型，直接优化策略。这个洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每一个都在修复 DPO 的一种失败模式。2026 年，直接对齐算法在前沿后训练任务中的交付量已超过 PPO。但第 2 课的过度优化曲线仍然适用：DAA 并不能逃出古德哈特，只是改变了它咬人的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 的闭式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复 DPO 的哪种失败模式。
- 区分「隐式奖励差距」与「偏好强度」，并解释 IPO 的恒等映射为何重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明：即使没有显式 RM，DAA 仍会过度优化。

## 问题

RLHF 目标（第 1 课）：

```math
\max_{\pi} \; \mathbb{E}_{x, y \sim \pi} \big[ r(x, y) \big] - \beta \, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
```

有一个已知的最优解：

```math
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)
```

于是奖励由最优策略与参考策略的比值隐式定义：

```math
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
```

把它代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 只依赖 `x`，因此相消。剩下的是一个只涉及策略参数的损失——不需要奖励模型。这就是 DPO。

麻烦在于：推导假设最优解可达、偏好数据在分布内、参考策略是真正的模式锚点。这些都不精确成立。家族中的每个成员修复一个被违反的假设。

## 概念

### DPO（Rafailov 等，2023）

```math
L_{\text{DPO}} = -\log \sigma\!\left(
  \beta \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
  - \beta \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
```

可能出什么问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个极小的偏好可以产生任意大的差距。
- 损失把选中与拒绝的对数概率推向相反方向。只要拒绝下降得更快，它可以把选中的绝对对数概率也压下去。这就是退化选中回复（Degraded Chosen Response）现象。
- 分布外的偏好（罕见对 vs 罕见对）会产生任意的隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化（Identity Preference Optimization）用恒等映射代替 log-sigmoid 作用于偏好概率。损失变成有界目标上的平方误差：

```math
L_{\text{IPO}} = \left( \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta} \right)^2
```

间隔被限制在 `1/(2 beta)` 以内。偏好强度与隐式奖励差距成正比。不会爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky 优化完全放弃成对结构。给定单个带标签的输出和二元「可取」或「不可取」信号，它映射到前景理论效用：

```math
v(x, y) = \sigma\!\left(\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}}\right)
```

收益和损失使用不同权重（损失厌恶）。好处：可以使用非成对数据，而后者要充足得多。

### SimPO（Meng 等，2024）

Simple Preference Optimization 让训练信号与生成时的度量对齐。完全移除参考策略，并按长度归一化对数似然：

```math
L_{\text{SimPO}} = -\log \sigma\!\left(
  \frac{\beta}{|y_w|} \log \pi(y_w \mid x)
  - \frac{\beta}{|y_l|} \log \pi(y_l \mid x)
  - \gamma
\right)
```

用间隔 `gamma` 来稳定。长度归一化去除了利用 DPO 长度偏置失败模式的动机（更长的 `y_w` 按构造会给出更大的对数概率差距）。

### ORPO（Hong 等，2024）

优势比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然上加一个偏好项：

```math
\begin{aligned}
L_{\text{ORPO}} &= L_{\text{NLL}}(y_w) + \lambda L_{\text{OR}} \\
L_{\text{OR}} &= -\log \sigma\!\left(\log \frac{\text{odds}(y_w)}{\text{odds}(y_l)}\right)
\end{aligned}
```

不需要参考策略——SFT 项就是正则项。从基础模型到对齐模型单阶段训练。不需要单独的 SFT checkpoint。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别出退化选中回复问题：DPO 保持了 `y_w > y_l` 的排序，但 `y_w` 的绝对对数概率可以下降。BPO 加入一行修正，惩罚选中回复的向下移动。在 Llama-3.1-8B-Instruct 的数学推理上，报告相对 DPO 提升 +10.1% 准确率。

### 普遍结论：DAA 仍会过度优化

Rafailov 等人 “Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）在多个数据集上用 DPO、IPO、SLiC 训练策略，跨越不同的 KL 预算。黄金奖励-对-KL 曲线呈现出与 Gao 等人相同的先达峰后崩塌形状。训练中隐式奖励会查询到分布外的样本；KL 正则无法稳定这一点。

DAA 并没有逃出古德哈特。它们把咬人的表面从「奖励模型被过度优化」变成「参考策略比值被过度优化」。通用修正——更好的数据、集成、早停——对两者都适用。

### 如何在其中选择（2026）

- 如果有大规模成对偏好数据：DPO 配保守的 beta；如果长度偏置明显，用 SimPO。
- 如果有非成对二元反馈：KTO。
- 如果要从基础模型做单阶段 pipeline：ORPO。
- 如果在 DPO 日志中看到选中的对数概率退化：BPO。
- 如果偏好强度差异很大且 DPO 已饱和：IPO。

每家实验室都会在一组任务上跑全部五种，再按任务挑赢家。没有理由认为数学推理和安全的最优解是同一个。

```figure
dpo-margin
```

## 使用它

`code/main.py` 在一个玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每种损失都在同样的 500 对样本上针对一个小 softmax 策略优化。绘制每种方法的最终胜率、选中对数概率漂移和隐式奖励散布。

## 交付它

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 非成对、偏好强度可变 vs 均匀、长度分布）和目标（单阶段，或先 SFT 再做偏好训练），推荐一种偏好损失，并报告它防范的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选中对数概率下降。BPO 应保留更高的选中绝对概率——验证这一点。

2. 修改偏好数据，使所有对的强度相等。六种方法中哪一种最稳健？哪一种会退化？解释 IPO 在这方面的优势。

3. 让被拒回复平均比选中回复长 2 倍。其他都不改，用数值展示 DPO 的长度利用和 SimPO 的修正。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过度优化。复现一个单点版本：绘制选中减拒绝的 KL 散度，观察大 beta 下 DPO 的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 加在 DPO 上的一行修正。对照 `code/main.py` 中的实现确认。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “没有奖励模型的 RLHF” | 由 RLHF 闭式最优解导出的损失；只涉及策略参数 |
| 隐式奖励 | “对数比值” | `beta * log(pi(y\|x) / pi_ref(y\|x))`——DPO 隐含的奖励 |
| IPO | “有界 DPO” | 用恒等映射代替 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 封顶 |
| KTO | “非成对 DPO” | 在单个标签上前景理论效用，带损失厌恶 |
| SimPO | “无参考 DPO” | 长度归一化的对数似然加间隔；没有参考策略 |
| ORPO | “单阶段 DPO” | NLL 加优势比偏好项；从基础模型一趟训练完成 |
| BPO | “保选中的 DPO” | DPO 加上对选中回复绝对对数概率下降的惩罚 |
| 退化选中 | “选中的也掉了” | 只要拒绝降得更快，DPO 就会压低选中对数概率 |
| DAA | “直接对齐算法” | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov 等人 —— Direct Preference Optimization（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等人 —— A General Theoretical Paradigm to Understand Learning from Human Preferences（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) —— IPO
- [Ethayarajh 等人 —— KTO: Model Alignment as Prospect Theoretic Optimization（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen —— SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne —— ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO —— Behavior Preservation Optimization（ICLR 2026，OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 —— Scaling Laws for RM Overoptimization in DAAs（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
