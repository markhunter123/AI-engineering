# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年的政策。它采用两级缓解结构：一类是 Anthropic 会单方面采取的行动，另一类被框定为全行业建议（包括 RAND SL-4 安全标准）。新增「前沿安全路线图」和「风险报告」作为常设文件，而不再是一次性交付物。取消 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦越过，Anthropic 必须发布一份肯定性论证（affirmative case），指明失调风险与缓解措施。Claude Opus 4.6 并未越过该阈值。Anthropic 在 v3.0 公告中表示，「有把握地排除这一点正变得越来越困难」。SaferAI 曾给 2023 年的 RSP 打 2.2 分；他们将 v3.0 下调到 1.9，使 Anthropic 与 OpenAI、DeepMind 一同落入「弱」RSP 类别。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最严重的倒退。

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## 问题

前沿实验室发布的扩展政策，部分是技术文档，部分是治理文档，部分是对监管者的信号。RSP v3.0 是 Anthropic 当前的文档。仔细阅读它之所以重要，并不是因为遵守它具有约束力（它并没有约束力），而是因为其中的框架塑造了一家实验室如何设想灾难性风险，以及如何向公众传达权衡。

v3.0 与 v2.0 的差异是有用的分析单元。新增了什么：前沿安全路线图、风险报告、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重新框定了什么：两级缓解时间表，拆分为 Anthropic 单方面行动与全行业建议。外部评审——SaferAI——将评分从 2.2（v2）下调到 1.9（v3.0）。扩展政策就是这样在看起来更精致的同时变得更不严谨。

## 概念

### 两级缓解时间表

- **Anthropic 单方面行动**：无论其他实验室怎么做，Anthropic 都会采取的行动。超过阈值就停止训练、特定安全措施、特定部署门槛。
- **全行业建议**：Anthropic 认为全行业应集体采取的行动。包括 RAND SL-4 安全标准。这些不是 Anthropic 一方的承诺，而是政策倡导。

两级结构在 v2 中并不存在。这意味着读者需要看清每一项承诺落在哪一栏。「全行业建议」栏中的安全措施不是 Anthropic 的承诺，而是 Anthropic 的期望。

### AI R&D-4 阈值

这是 RSP v3.0 指认为下一个重要阈值的能力水平。具体而言：一个能够以有竞争力的成本自动化相当大一部分 AI 研究的模型。一旦 Anthropic 认为某个模型越过该阈值，他们必须在继续扩展之前发布肯定性论证，指明失调风险与缓解措施。

根据 v3.0 公告，Claude Opus 4.6 并未越过该阈值。文档补充道：「有把握地排除这一点正变得越来越困难。」这种措辞很重要；它承认该阈值已经足够近，是现实关切，而不是推测性的极限。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接指向这一阈值。自动化对齐研究员越过研究质量门槛，就是 AI R&D-4 阈值正在逼近的证据。

### 前沿安全路线图与风险报告

v3.0 将两类产物提升为常设文件：

- **前沿安全路线图**：前瞻性文档，描述计划中的安全工作、能力预期和缓解研究。
- **风险报告**：模型发布后针对具体模型的回顾性文档，描述观察到的能力和残余风险。

两者都公开，都按声明好的节奏更新。其用处在于：读者可以对照 Anthropic 在路线图中说他们会做什么，与他们在风险报告中报告了什么。

### 移除暂停条款

2023 年的 RSP 包含明确的暂停承诺：如果模型越过特定能力阈值，训练将暂停，直到缓解措施到位。v3.0 用更温和的表述取代了明确的暂停（发布肯定性论证，若缓解措施足够则继续）。SaferAI 和其他分析人士直接指出，这是新文档中最严重的倒退。

支持这一变化的政策论点：2023 年的定量阈值到 2026 年时代的能力基准下已无法企及，因为基准本身被重新调整了。反驳论点：扩展政策中的暂停条款是一种承诺机制；移除它就移除了政策的可信度。

### SaferAI 的下调

SaferAI 是一家对 RSP 类文档进行评级的独立组织。他们的公开评级：2023 年 Anthropic RSP 得分 2.2（在 4.0 为当前最佳 RSP、1.0 为名义值的量表上）。v3.0 得分 1.9。这使 Anthropic 从「中等」降至「弱」，与 OpenAI 和 DeepMind 同属弱类别。

据 SaferAI，下调因素包括：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值的缓解措施被描述为「肯定性论证」，而非具体措施。
- 评审机制依赖 Anthropic 的安全顾问组（Safety Advisory Group），独立监督有限。

### 本课不是什么

这不是一堂合规课。RSP v3.0 不是法规；没有任何力量强迫 Anthropic 遵守它。本课要培养的是以它应得的精确和怀疑去阅读这份文档。扩展政策是前沿实验室就灾难性风险态势发出的主要公开信号。对于任何工作依赖前沿能力的人来说，读懂它们是实用技能。

```figure
a5-rsp-ladder
```

## 使用它

`code/main.py` 实现了一个小型决策引擎，镜像 RSP 阈值评估的形态：给定一个候选模型和一组能力测量值，返回是否越过 AI R&D-4 阈值、所需的肯定性论证章节，以及部署能否继续。它刻意保持简单；重点是让文档的逻辑显式化。

## 交付它

`outputs/skill-scaling-policy-review.md` 依据 v3.0 参考标准评审一份扩展政策（Anthropic、OpenAI、DeepMind 或内部）：两级结构、阈值、暂停承诺、独立评审。

## 练习

1. 运行 `code/main.py`。输入三个处于不同能力水平的合成模型。确认阈值评估器的行为符合预期，并生成正确的肯定性论证模板。

2. 完整阅读 RSP v3.0（32 页）。找出所有落在「全行业建议」层级的承诺。其中哪些在 v2 中本会是「Anthropic 单方面」？

3. 阅读 SaferAI 的 RSP 评分方法。用他们的评分表对文档逐项打分，复现 v3.0 的 1.9 分。评分表中哪一行对下调影响最大？

4. 2023 年的暂停承诺已被移除。提出一项替代承诺，既能保持政策的可信度，又能承认 2026 年基准重新调整的问题。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）比较。选出一个 v3.0 更强的领域。选出一个 Preparedness Framework 更强的领域。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| RSP | 「Anthropic 的扩展政策」 | 负责任扩展政策（Responsible Scaling Policy）；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | 「研究自动化阈值」 | 以有竞争力的成本自动化大量 AI 研究的能力 |
| 肯定性论证（affirmative case） | 「安全论证」 | 公开发布的论证，说明风险已识别且缓解措施充分 |
| 前沿安全路线图 | 「前瞻计划」 | 关于计划中安全工作与预期能力的常设文档 |
| 风险报告 | 「模型回顾」 | 关于发布后观察到的能力与残余风险的常设文档 |
| 两级缓解 | 「单方面 vs 行业」 | 将 Anthropic 承诺与行业建议分列 |
| 暂停承诺 | 「2023 条款」 | 明确承诺暂停训练；在 v3.0 中被移除 |
| SaferAI 评级 | 「独立 RSP 评分」 | 第三方评分表；v3.0 得 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策。
- [Anthropic — RSP v3.0 公告](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 相对 v2 的变化摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/responsible-scaling-policy/roadmap) — 从 RSP v3.0 链接的常设文档。
- [Anthropic — Risk Report: February 2026](https://www.anthropic.com/feb-2026-risk-report) — 对发布时前沿模型 Claude Opus 4.6 的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与实测自主性联系起来。
