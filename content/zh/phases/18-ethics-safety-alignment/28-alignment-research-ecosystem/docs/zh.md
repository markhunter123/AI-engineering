# 对齐研究生态——MATS、Redwood、Apollo、METR

> 五家组织定义了 2026 年实验室之外的对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来有 527+ 名研究者，180+ 篇论文，10K+ 次引用，h-index 47；2024 年夏季队列注册为 501(c)(3)，约 90 名学者和 40 名导师；2025 年前的校友中 80% 从事安全/安保工作，其中 200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；提出 AI Control（第 10 课）；与英国 AISI 合作撰写控制安全案例。Apollo Research：为前沿实验室做部署前的图谋（scheming）评估；著有 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估、自主任务时间跨度研究；「Common Elements of Frontier AI Safety Policies」比较各实验室的框架。Eleos AI Research：模型福祉的部署前评估（第 19 课）；执行了 Claude Opus 4 的福祉评估。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## 学习目标

- 识别非实验室对齐研究生态中的五家组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的作用。
- 描述 Redwood 的 AI Control 议程及其与英国 AISI 的合作。
- 描述 METR 基于任务的评估方法。

## 问题

前沿实验室（第 18 课）在内部做安全评估，并公开部分结果。实验室之外的生态，则是评估被验证、新失败模式被首次发现、人才被培养的地方。理解这个生态，有助于判断哪些研究发现被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

始于 2021 年底。这是一个研究导师计划；学者与一位资深研究者就某个具体的对齐问题共事 10–12 周。

规模（2026）：
- 成立以来 527+ 名研究者。
- 发表 180+ 篇论文。
- 10K+ 次引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业去向：2025 年前的校友约 80% 从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。提出 AI Control 议程（第 10 课）。与英国 AISI 合作撰写控制安全案例。为 DeepMind 和 Anthropic 提供评估设计建议。

代表论文：Greenblatt、Shlegeris 等，「AI Control」（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等，arXiv:2412.14093，与 Anthropic 合作）。

风格：具体的威胁模型、最坏情况的对手、可做压力测试的具体协议。

### Apollo Research

为前沿实验室做部署前的图谋评估。著有 In-Context Scheming（第 8 课，arXiv:2412.04984）。是 2025 年 OpenAI 反图谋训练合作的伙伴。产出 Towards Safety Cases for AI Scheming（2024）。

风格：让欺骗得以显现的 Agent 环境评估；三支柱分解（失准、目标导向性、情境感知）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间跨度研究。「Common Elements of Frontier AI Safety Policies」（metr.org/common-elements，2025）比较各实验室的框架。

与 Apollo 共同署名 AI Scheming 安全案例草稿。

风格：长时程任务评估、经验性能力测量、框架综合。

### Eleos AI Research

模型福祉的部署前评估。执行了系统卡 5.3 节记录的 Claude Opus 4 福祉评估。为第 19 课中与福祉相关的论断提供外部方法学检查。

### 流动关系

MATS 培养研究者。毕业生去 Anthropic、DeepMind、OpenAI（实验室安全团队），或去 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及英国 AISI / CAISI 合作。发表成果又反哺生态，为 MATS 的下一批队列提供素材。

### 这一层为什么重要

单一来源的评估不可靠：实验室评估自己的模型，存在结构性利益冲突。外部评估者可以提出并验证实验室可能低报的失败模式。2024 年的 Sleeper Agents 论文（第 7 课）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多组织结构就是质量控制。

### 本课在 Phase 18 中的位置

第 7–11 课引用 Redwood 和 Apollo 的工作；第 18 课引用 METR 的框架比较；第 19 课引用 Eleos。第 28 课是本 Phase 其余课程所依赖的生态的显式组织地图。

```figure
sae-features
```

## 使用它

无代码。阅读 METR 的「Common Elements of Frontier AI Safety Policies」，把它作为外部综合如何为实验室内部政策工作增值的范例。

## 交付它

本课产出 `outputs/skill-ecosystem-map.md`。给定一项对齐主张或评估，它识别所属组织、发表场所与方法风格，并与已知的对应组织交叉核对。

## 练习

1. 从第 7–15 课中选一篇论文，识别参与的组织。把作者与 MATS 校友及当前生态归属交叉核对。

2. 阅读 METR 的「Common Elements of Frontier AI Safety Policies」。指出他们强调的三处跨实验室趋同和两处最大分歧。

3. MATS 的职业去向约 80% 是安全/安保。论证这种选择压力是适应性的（训练了这个领域）还是有偏的（滤掉了异见立场）。

4. Redwood 和 Apollo 都做控制/图谋工作，但风格不同。选一个失败模式，描述两者会如何调查它。

5. Eleos AI 是唯一纯粹的模型福祉组织。设计一个假想的第二家组织，聚焦另一个与福祉相邻的问题（认知自由、机器人具身化等），并阐明其方法论。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| MATS | 「那个导师计划」 | ML Alignment & Theory Scholars；自 2021 年以来 527+ 名研究者 |
| Redwood Research | 「控制实验室」 | 应用对齐；AI Control 的作者；英国 AISI 伙伴 |
| Apollo Research | 「图谋评估」 | 为前沿实验室做部署前图谋评估 |
| METR | 「任务时程评估」 | 基于任务的能力评估；框架综合 |
| Eleos AI | 「福祉实验室」 | 模型福祉的部署前评估 |
| Talent pipeline | 「MATS -> 实验室」 | MATS 毕业生流向 Anthropic、DeepMind、OpenAI、Redwood、Apollo、METR |
| External evaluation | 「实验室外检查」 | 不由模型生产者自己做的评估；提供额外可信度 |

## 延伸阅读

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) — 该导师计划
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) — 图谋评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福祉方法学
