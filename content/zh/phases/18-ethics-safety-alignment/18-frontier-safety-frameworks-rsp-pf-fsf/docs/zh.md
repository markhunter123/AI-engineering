# 前沿安全框架——RSP、PF、FSF

> 三大实验室的框架定义了 2026 年前沿能力的行业治理。Anthropic 的 Responsible Scaling Policy v3.0（2026 年 2 月）引入了分级的 AI Safety Levels（ASL-1 到 ASL-5+），以生物安全等级为蓝本，其中 ASL-3 于 2025 年 5 月针对 CBRN 相关模型激活。OpenAI 的 Preparedness Framework v2（2025 年 4 月）定义了跟踪能力的五项标准，并把能力报告（Capabilities Reports）与保障报告（Safeguards Reports）分开。DeepMind 的 Frontier Safety Framework v3.0（2025 年 9 月）引入了关键能力等级（Critical Capability Levels），包括一个新的有害操纵 CCL。三个框架现在都包含竞争对手调整条款，允许在同行实验室没有同等保障就发布时推迟要求。跨实验室的对齐仍是结构性的，而非术语性的：「Capability Thresholds」「High Capability thresholds」和「Critical Capability Levels」指的是类比构造。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## 学习目标

- 描述 Anthropic 的 ASL 等级结构，以及是什么触发了 ASL-3 的激活。
- 说出 OpenAI Preparedness Framework v2 跟踪能力的五项标准。
- 描述 DeepMind 的关键能力等级结构以及有害操纵 CCL。
- 解释竞争对手调整条款，以及它们为何对竞赛动态很重要。
- 定义安全论证（safety case），并描述三支柱结构（监控、不可读性、无能力）。

## 问题

第 7-17 课确立了：欺骗是可能的、两用能力确实存在、评估有其局限。拥有前沿能力模型的实验室需要一种内部治理结构，它要：

- 定义何时需要新保障措施的阈值。
- 定义扩展前所需的评估。
- 描述安全论证长什么样。
- 处理竞赛动态问题（如果竞争对手在没有保障的情况下发布，你怎么办？）。

这三套 2025-2026 年的框架是当前最先进的实践——不完美、仍在演进，但跨实验室已经足够一致，以至于治理问题已不再是「框架是否存在」，而是「框架是否足够」。

## 概念

### Anthropic 的 Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：

- ASL-1：不是前沿模型（被「弱于前沿」的基线涵盖）。
- ASL-2：当前前沿基线；以常规保障措施部署。
- ASL-3：灾难性误用风险显著更高；CBRN 相关能力。2025 年 5 月激活。
- ASL-4：越过 AI R&D-2 阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高级 AI R&D；能大幅加速有效扩展的模型。

v3.0 新增：

- 前沿安全路线图（Frontier Safety Roadmaps，以脱敏形式公开）。
- 风险报告（Risk Reports，每季度，部分接受外部审查）。
- AI R&D 被拆分为 AI R&D-2 和 AI R&D-4。
- 一旦越过 AI R&D-4，就需要肯定性安全论证，指出模型追求错位目标所带来的错位风险。

### OpenAI 的 Preparedness Framework v2（2025 年 4 月 15 日）

跟踪能力的五项标准：

- **合理（Plausible）。** 存在合理的威胁模型。
- **可测量（Measurable）。** 可以进行实证评估。
- **严重（Severe）。** 危害很大。
- **净新增（Net-new）。** 不是既有风险的放大。
- **瞬时或不可逆（Instantaneous-or-irremediable）。** 危害发生得很快，或无法撤销。

五项全部满足的能力才会被跟踪，其他则不会。

PF v2 的其他结构：

- 把能力报告（模型能做什么）与保障报告（存在哪些控制）分开。
- 安全咨询小组（Safety Advisory Group）审查。
- 领导层批准；董事会的安全与安保委员会监督。
- 「调整条款」：如果另一个实验室在没有同等保障的情况下发布，OpenAI 可以降低要求。

### DeepMind 的 Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的关键能力等级（CCL）：

- 生物武器提升（Bioweapon Uplift）
- 网络提升（Cyber Uplift）
- ML R&D 加速（ML R&D Acceleration）
- 有害操纵（Harmful Manipulation，v3.0 新增）：可能在高风险情境中实质性改变信念/行为的模型。

v2.0（2025 年 2 月）增加了欺骗性对齐（Deceptive Alignment）章节，并为 ML R&D CCL 增加了更高的安全等级。

### 跨实验室对齐

- Anthropic：「Capability Thresholds」。
- DeepMind：「Critical Capability Levels」。
- OpenAI：「High Capability thresholds」。

没有行业标准术语。但在结构上是对齐的：前沿能力的三个层级，配有公开的评估标准。三者从 2025 年起都包含竞争对手调整条款。

### 安全论证

安全论证（safety case）是一份书面论证，说明在最坏情况假设下，某次部署可以被接受地安全。标准结构针对三个支柱：

- **监控（Monitoring）。** 如果出现不良行为，我们能检测到吗？
- **不可读性（Illegibility）。** 模型是否缺乏执行一个连贯的有害计划的能力？
- **无能力（Incapability）。** 模型是否缺乏造成相关危害的能力？

不同的安全论证针对不同的支柱。对于 ASL-3 的 CBRN 案例，主要目标是无能力（通过遗忘实现）。对于欺骗性对齐，目标是监控与不可读性。对于网络提升，三个支柱都相关。

### 竞赛动态问题

竞争对手调整条款颇具争议。批评者认为它们会造成逐底竞争：如果三家实验室都会在竞争对手违规时降低要求，均衡就会向违规偏移。辩护者则认为，如果违规实验室的安全意识更弱，替代方案（单边保障）会带来更糟的结果。

英国 AISI、美国 CAISI 和欧盟 AI Office（第 24 课）是外部治理的对应机构。实验室框架是自愿的；监管框架仍在形成中。

### 这一课在 Phase 18 中的位置

第 17-18 课是建立在欺骗与红队分析之上的测量与治理层。第 19-24 课涵盖福利、偏见、隐私、水印和监管结构。第 28 课梳理把这些评估付诸实践的研究生态（MATS、Redwood、Apollo、METR）。

```figure
al-asl-ladder
```

## 使用它

本课没有代码。阅读三份一手文献：RSP v3.0、PF v2、FSF v3.0。把每家实验室的等级结构映射到其他两家，并找出每家实验室定义了而其他两家没有的一个阈值。

## 交付它

本课产出 `outputs/skill-framework-diff.md`。给定一个安全框架或发布说明，它会把该框架的阈值定义、所需评估和安全论证结构，与 RSP v3.0、PF v2、FSF v3.0 逐一比较，并标出跨实验室的差距。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制一张表，列出每家实验室的 CBRN 阈值、AI R&D 阈值，以及各自的部署前必需评估。

2. 竞争对手调整条款出现在三套框架中（2025 年及以后）。写一段支持它的论述，再写一段反对它的论述。指出每种立场依赖的假设。

3. 为一个越过 Anthropic AI R&D-4 阈值的模型设计安全论证。列出三个支柱（监控、不可读性、无能力）各自需要的证据。

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三项实证测量，用以表明模型已越过该阈值。

5. 阅读 METR 的《Common Elements of Frontier AI Safety Policies》（2025）。指出跨实验室最强的三处趋同和最大的两处分歧。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| RSP | 「Anthropic 的框架」 | Responsible Scaling Policy；ASL 等级；v3.0 于 2026 年 2 月 |
| PF | 「OpenAI 的框架」 | Preparedness Framework；五项标准；v2 于 2025 年 4 月 |
| FSF | 「DeepMind 的框架」 | Frontier Safety Framework；CCL；v3.0 于 2025 年 9 月 |
| ASL-3 | 「生物安全 3 级的类比」 | Anthropic 针对 CBRN 相关能力的层级；2025 年 5 月激活 |
| CCL | 「关键能力等级」 | DeepMind 的阈值构造；按领域划分 |
| 安全论证（Safety case） | 「正式论证」 | 书面论证：在最坏情况假设下，部署可以被接受地安全 |
| 调整条款 | 「竞争对手违规许可」 | 框架中允许在竞争对手没有同等保障就发布时降低要求的条款 |

## 延伸阅读

- [Anthropic——Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy)——ASL 等级、路线图、AI R&D 拆分
- [OpenAI——Updating the Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/)——五项标准、调整条款
- [DeepMind——Strengthening our Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)——CCL v3.0、有害操纵
- [METR——Common Elements of Frontier AI Safety Policies（2025）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)——跨实验室比较
