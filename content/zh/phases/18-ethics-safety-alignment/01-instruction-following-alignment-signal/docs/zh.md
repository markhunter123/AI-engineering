# 指令遵循作为对齐信号

> 此后每一篇对 RLHF 的批评，针对的都是这条流水线。在研究优化压力如何扭曲代理指标之前，你必须先看清这个代理本身。InstructGPT（Ouyang 等，2022）定义了参考架构：在指令-回复对上做监督微调，用在成对偏好排序上训练的奖励模型，再以 PPO 针对奖励模型优化，并施加对 SFT 策略的 KL 惩罚。一个 1.3B 的 InstructGPT 被标注者偏好的程度超过了 175B 的 GPT-3。正是这一个结果，让 2026 年的每一家前沿实验室仍在交付 RLHF 形态的后训练流水线。

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## 学习目标

- 说出 InstructGPT 流水线的三个阶段，以及每个阶段所用的损失。
- 解释为什么一个 1.3B 的指令微调模型能在人类偏好评估上胜过原始 175B 的 GPT-3。
- 说明第 3 阶段的 KL 惩罚在防范什么，以及去掉它为什么会退化为模式寻求（mode-seeking）行为。
- 描述对齐税（alignment tax），以及 Ouyang 等人用来抵消它的 PPO-ptx 缓解方案。

## 问题

预训练语言模型做的是补全文本，不是回答问题。问 GPT-3「写一个反转列表的 Python 函数」，你常常会得到另一个 prompt，因为训练分布大部分是网页文本，而网页文本的延续仍是更多的网页文本。模型在履行它的职责——只是这个职责本身错了。

每家严肃实验室用来修正这一点的代理指标，都是人类偏好。两段补全交给评分员，评分员挑出更好的那个，奖励模型学习评分员。随后 RL 循环把策略推向奖励模型打高分的那类输出。这就是 InstructGPT 的全部论点，三句话说完。论文其余部分都是工程。

## 概念

### 阶段 1：监督微调（SFT）

收集 prompt-回复对，其中回复是一个有善意的人会写出的内容。Ouyang 等人使用了来自标注者和 OpenAI API 的 13k 条 prompt。用标准交叉熵损失在这份数据上微调基础模型。

SFT 给你什么：模型现在会回答问题，而不是续写问题。它不给你什么：当多个答案都说得通时，评分员更偏好哪一个的信号。

### 阶段 2：奖励模型（RM）

对每条 prompt，从 SFT 模型采样 K 个补全。标注者给它们排序。训练一个奖励模型，为任意 prompt-回复对打分，使得对于 `y_w` 优于 `y_l` 的对：

```math
L_{\text{RM}} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
```

这是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，把 LM head 换成标量 head。

奖励模型很小：对 175B 的 InstructGPT 来说，6B 就够用。它们也很脆弱——论文第 5 节大部分讨论的是小规模下就出现的奖励黑客行为。

### 阶段 3：带 KL 惩罚的 PPO

定义目标：

```math
J(\pi) = \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] - \beta \, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\big)
```

用 PPO 最大化它。KL 项让 `pi` 不会偏离 SFT 策略太远。没有它，优化器会找到对抗样本——这些字符串在 RM 下得分很高，原因不是人类真的偏好它们，而是 RM 从没见过它们。

KL 系数 `beta` 是 RLHF 中最重要的单个超参数。太低：奖励黑客。太高：相对 SFT 毫无改进。

### 对齐税

RLHF 之后，模型更受人类偏爱，但在标准基准（SQuAD、HellaSwag、DROP）上退步。Ouyang 等人称之为对齐税，并用 PPO-ptx 修正：把预训练梯度混入 RL 目标，使模型不会忘记那些它从未因之获得奖励的下游任务。

```math
J_{\text{ptx}}(\pi) = J(\pi) + \gamma \, \mathbb{E}_{x \sim D_{\text{pretrain}}} \big[ \log \pi(x) \big]
```

PPO-ptx 成了标准做法。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在约 70% 的情况下被标注者偏好于 175B 的基础 GPT-3。在生产流量中的隐藏测试 prompt 上，差距还会进一步拉大。这个数字可以读出两件事：

1. 对齐是与能力不同的维度。175B 模型能力更强，1.3B 模型对齐更好，标注者偏好对齐的那个。
2. 能力下限由基础模型决定。你无法通过 RLHF 让基础模型知道它从未见过的事实。

### 为什么这是第 18 阶段的参照点

后续每一课中的批评——奖励黑客（第 2 课）、DPO（第 3 课）、谄媚（第 4 课）、CAI（第 5 课）、睡眠智能体（第 7 课）、对齐伪装（第 9 课）——都在攻击这条流水线的某个部分。奖励黑客攻击阶段 2。DPO 把阶段 2 和 3 合并成一步。CAI 替换人类标注者。谄媚说明标注者本身就是有偏的信号。对齐伪装说明策略可以完全绕过阶段 3。脑子里没有这条流水线，任何批评都跟不上。

```figure
al-instruct-pipeline
```

## 使用它

`code/main.py` 在玩具偏好数据上模拟这三个阶段。基础「策略」是动作 {A, B, C} 上的有偏硬币。阶段 1 的 SFT 在 200 条 prompt 上模仿标注者动作。阶段 2 从 500 个成对排序中拟合 Bradley-Terry 奖励模型。阶段 3 运行简化的 PPO 更新，并对 SFT 策略施加 KL 惩罚。你可以观察奖励爬升、KL 散度增长、策略漂移——也可以关掉 KL 项，看奖励黑客在 50 个更新步内出现。

要看什么：

- `beta = 0.1` 与 `beta = 0.0` 下的奖励轨迹。
- 训练步上的 $\mathrm{KL}(\pi \Vert \pi_{\text{SFT}})$。
- 最终动作分布与标注者偏好的对比。

## 交付它

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一段 RLHF 流水线描述或论文摘要，它识别出被修改的是三个阶段中的哪一个、每个阶段使用什么损失，以及是否存在 KL 惩罚或等价的正则项。

## 练习

1. 运行 `code/main.py`。设 `beta = 0.0`，报告 200 个 PPO 步后的动作分布。用一段话解释其中的模式寻求行为。

2. 修改奖励模型，给动作 B 加 +0.5 的偏置（模拟奖励 bug）。以 `beta = 0.1` 运行 PPO。KL 惩罚能阻止策略利用这个偏置吗？在什么 `beta` 下这种利用会变得可见？

3. 阅读 Ouyang 等人（arXiv:2203.02155）图 1。运行 PPO 1、5、20、100 步，测量相对 SFT 模型的偏好，复现标注者偏好曲线。

4. 论文 4.3 节报告 1.3B InstructGPT 在约 70% 的情况下胜过 175B GPT-3。为什么在隐藏的生产 prompt 上，这个比例会比标注者自己的 prompt 更高？

5. 在同一份偏好数据上，把 PPO 损失换成 DPO（第 10 阶段 · 08）。比较最终策略漂移（相对 SFT 的 KL）与最终奖励。在奖励相同的情况下，哪个方法漂移得更远？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| SFT | “指令微调” | 阶段 1：在 prompt-回复对上做交叉熵微调 |
| 奖励模型 | “RM” | 在（prompt, 回复）上的标量回归器，用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | “成对偏好损失” | $-\log \sigma(r_w - r_l)$；把成对排序归约为二分类 |
| KL 惩罚 | “正则项” | `beta * KL(pi \|\| pi_SFT)`——让 RL 策略靠近 SFT 锚点 |
| PPO-ptx | “混入预训练的 PPO” | 在 PPO 目标中加入一部分预训练对数似然，抵消对齐税 |
| 对齐税 | “RLHF 带来的退步” | RLHF 未针对的标准基准在 RLHF 后的下降 |
| 标注者偏好 | “真值” | 人类排序的样本；RM 是对它的统计代理，而不是对“人类价值观”的代理 |

## 延伸阅读

- [Ouyang 等人 —— Training language models to follow instructions with human feedback（arXiv:2203.02155）](https://arxiv.org/abs/2203.02155) —— InstructGPT 论文，此后每条 RLHF 流水线的基础
- [Stiennon 等人 —— Learning to summarize from human feedback（arXiv:2009.01325）](https://arxiv.org/abs/2009.01325) —— 用于摘要任务的 RLHF 前身
- [Christiano 等人 —— Deep reinforcement learning from human preferences（arXiv:1706.03741）](https://arxiv.org/abs/1706.03741) —— 最初的基于偏好的 RL 表述
- [Bai 等人 —— Training a Helpful and Harmless Assistant with RLHF（arXiv:2204.05862）](https://arxiv.org/abs/2204.05862) —— Anthropic 对 InstructGPT 流水线的 HH（有用且无害）扩展
