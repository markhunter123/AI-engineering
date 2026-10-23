# 失败模式：智能体为何会崩溃

> MASFT（Berkeley，2025）将 14 种多智能体失败模式归入 3 个类别。Microsoft 的分类法记录了现有 AI 失败在智能体环境中如何被放大。行业一线数据收敛于五种反复出现的模式：幻觉动作、范围蔓延、级联错误、上下文丢失、工具误用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出 MASFT 的三大失败类别，以及每类中至少四种具体模式。
- 解释为什么智能体失败会放大现有的 AI 失败模式（偏见、幻觉）。
- 描述五种行业反复出现的模式及其缓解措施。
- 实现一个 stdlib 检测器，为智能体轨迹打上失败模式标签。

## 问题

团队交付的智能体在 90% 的轨迹上工作正常。剩下 10% 的失败不是随机噪声——它们集中在少数几类反复出现的类别中。一旦你能叫出它们的名字，就能监控并修复它们。

## 概念

### MASFT（Berkeley，arXiv:2503.13657）

多智能体系统失败分类法（Multi-Agent System Failure Taxonomy）。14 种失败模式归入 3 个类别。标注者间 Cohen's Kappa 为 0.88——这些类别可以被可靠地区分。

核心主张：失败是多智能体系统中的根本设计缺陷，而不是靠更好的基座模型就能修复的 LLM 局限。

### Microsoft 的智能体 AI 系统失败模式分类法

- 现有 AI 失败（偏见、幻觉、数据泄露）在智能体环境中会被放大。
- 新的失败源于自主性：大规模的非预期动作、工具误用、任务漂移。
- 该白皮书是智能体产品的风险登记册。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- 失败源于编排、内部状态演化以及与环境交互。
- 不仅仅是「坏代码」或「坏模型输出」。

### LLM 智能体幻觉综述（arXiv:2509.18970）

两种主要表现：

1. **指令遵循偏离（Instruction-following Deviation）**——智能体不遵循 system prompt。
2. **长程上下文误用（Long-range Contextual Misuse）**——智能体遗忘或误用前几轮的上下文。

子意图错误：遗漏（Omission，漏掉步骤）、冗余（Redundancy，重复步骤）、错序（Disorder，步骤顺序错误）。

### 五种行业反复出现的模式

Arize、Galileo、NimbleBrain 2024-2026 年的一线分析收敛于：

1. **幻觉动作。** 智能体调用了不存在的工具，或捏造参数。
2. **范围蔓延。** 智能体把任务扩展到用户要求之外（创建多余的 PR、发送多余的邮件）。
3. **级联错误。** 一次错误调用触发下游连锁反应。一个虚构 SKU 的幻觉触发四次 API 调用——演变为多系统事故。
4. **上下文丢失。** 长程任务遗忘了早先轮次的约束。
5. **工具误用。** 用错误的参数调用正确的工具，或完全用错工具。

级联是致命杀手。智能体无法区分「我失败了」和「任务不可能完成」，并且常常在收到 400 错误时幻觉出一条成功消息来闭合循环。

### 缓解：每一步都设关卡

在推理链的每一步设置自动验证关卡，对照环境状态检查事实依据。具体地：

- 每步安全分类器（第 21 课）。
- 工具调用参数校验（第 06 课）。
- 将检索到的内容与已知事实交叉核对（第 05 课，CRITIC）。
- 通过重新探测状态来检测成功幻觉（文件是否真的创建了？）。

### 失败监控容易出错的地方

- **只给崩溃打标签。** 大多数智能体失败产生的是看似有效的输出。需要内容级检查。
- **没有基线。** 漂移检测需要一个 last-known-good；没有它，你无法判断「情况正在变差」。
- **告警过度。** 每次失败都触发一页告警。需要聚类与速率限制。

```figure
failure-cascade
```

## 构建它

`code/main.py` 实现一个 stdlib 失败模式标注器：

- 一个覆盖五种模式的合成轨迹数据集。
- 按模式划分的检测器函数（针对工具调用、输出、重复动作的签名模式）。
- 一个标注器，为每条轨迹打标签并报告模式分布。

运行它：

```
python3 code/main.py
```

输出：每条轨迹的标签 + 汇总分布，以低成本复现 Phoenix 轨迹聚类所呈现的结果。

## 使用它

- **Phoenix** 用于生产漂移聚类（第 24 课）。
- **Langfuse** 用于会话回放 + 标注。
- **自定义实现** 用于你的可观测平台无法检测的领域特定签名。

## 交付它

`outputs/skill-failure-detector.md` 生成贴合你领域的失败模式检测器，并接入轨迹存储。

## 练习

1. 添加一个「成功幻觉」检测器：智能体返回成功，但目标状态未改变。
2. 为你构建过的产品标注 100 条真实轨迹。哪种模式占主导？修复它的成本是多少？
3. 实现「级联半径」指标：给定第 N 步的失败，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种失败模式。挑选三种适用于你产品的模式，编写检测器。
5. 将一个检测器接入 CI 作业：若 >=5% 的轨迹命中某模式，则让构建失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MASFT | 「多智能体失败分类法」 | Berkeley 的 14 模式分类 |
| 级联错误（Cascading error） | 「涟漪式失败」 | 早期的一个错误沿 N 个步骤传播 |
| 上下文丢失（Context loss） | 「忘了约束」 | 长程轮次丢弃早期事实 |
| 工具误用（Tool misuse） | 「错工具 / 错参数」 | 调用有效，但调用方式错误 |
| 成功幻觉（Success hallucination） | 「伪造完成」 | 智能体在 400 上声称成功；状态未变 |
| 范围蔓延（Scope creep） | 「越权」 | 智能体做的事超出要求 |
| 指令遵循偏离（Instruction-following deviation） | 「不服从」 | 无视 system prompt 或用户约束 |
| 子意图错误（Sub-intention errors） | 「计划缺陷」 | 计划执行中的遗漏、冗余、错序 |

## 进一步阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) —— 14 种失败模式，3 个类别
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) —— 风险登记册
- [Arize Phoenix](https://docs.arize.com/phoenix) —— 实践中的漂移聚类
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 更简单的模式如何完全避开这些失败模式
