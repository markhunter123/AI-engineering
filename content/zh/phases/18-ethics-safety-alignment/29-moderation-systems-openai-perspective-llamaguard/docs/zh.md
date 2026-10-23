# 内容审核系统——OpenAI、Perspective、Llama Guard

> 生产环境的内容审核系统把第 12–16 课定义的安全政策落到了实处。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，在一次调用中对文本 + 图像分类；在多语言测试集上比此前版本好 42%；响应模式返回 13 个类别布尔值——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用可以隐藏延迟；命中时返回占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、Code Interpreter Abuse、8 种语言（v3）、多图（v4）。Perspective API（Google Jigsaw）：早于「LLM 当审核员」浪潮的毒性评分；主要是单维度毒性，带 severe-toxicity/insult/profanity 变体；是内容审核研究的基线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 取代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法，以及它与 Llama Guard 3 的 MLCommons 类别集有何不同。
- 描述三层审核模式（输入、输出、自定义），并为每一层各举一个失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的地位，以及它为何仍被用于研究。
- 说明 Azure 的弃用时间线。

## 问题

第 12–16 课讲攻击与防御工具。第 29 课讲已部署的审核系统——它们在用户接触产品的那个面上把防御落到实处。三层模式是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次调用中分类文本 + 图像。对多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余类别仅支持文本。

为教学简洁起见，`code/main.py` 中的代码试样把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到它们的顶层父类。生产代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点好 42%。按类别给分；由应用设定阈值。

### Llama Guard 3/4

第 16 课已介绍。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B。

OpenAI 与 Llama Guard 的分类法有重叠也有分歧。OpenAI 有宽泛的「illicit」类别；Llama Guard 则把「暴力犯罪」和「非暴力犯罪」分开。部署方按自身政策分类法的契合度来选择。

### Perspective API（Google Jigsaw）

早于「LLM 当审核员」浪潮的毒性评分系统（2020 年前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。以单维度主分数（TOXICITY）为主，辅以子维度变体。

它被广泛用作内容审核研究的基线，因为 API 稳定、有文档，且积累了多年的校准数据。对于现代与 LLM 相邻的用例，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户的 prompt 分类。命中则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型输出分类。命中则替换为拒答。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定的规则（正则、允许列表、业务政策）。在输入或输出端运行。

三层的顺序是设计使然：输入审核必须在生成前完成，输出审核在生成后运行。并行只发生在层内——对同一段文本并发运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏各分类器的延迟。作为可选优化，输入审核完成前可以显示占位响应（「稍等，正在检查……」），并推迟 token-1 的流式输出。命中后的行为可配置：拒绝、净化、升级人工审核。

### 失败模式

- **只有输入。** 无法捕获输出侧的幻觉（第 12–14 课的编码攻击可绕过输入分类器）。
- **只有输出。** 任何输入都能到达模型；成本上升；并把内部推理暴露给攻击者。
- **只有自定义。** 跨类别不够稳健；正则是脆弱的。

分层是默认做法。多一道保险总没错。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 取代，后者基于 LLM，并与 Azure OpenAI 集成。对 Azure 部署而言，这次迁移是 2024–2027 年的一项字段级工程。

### 本课在 Phase 18 中的位置

第 16 课在红队测试的语境下讲审核工具。第 29 课讲已部署的审核。第 30 课以当前的双用途能力证据收尾。

```figure
an-moderation-layers
```

## 使用它

`code/main.py` 构建一个三层审核试样：输入审核器（关键词 + 类别分）、输出审核器（对输出使用同一分类器）、自定义审核器（领域规则）。你可以把输入跑过去，观察哪一层拦住了什么。

## 交付它

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署，它给出审核栈配置建议：输入端用哪个分类器、输出端用哪个、哪些自定义规则，以及边缘情况交给什么判定器。

## 练习

1. 运行 `code/main.py`。让一个良性、一个边界和一个有害输入分别通过三层。报告每一层对哪个输入触发。

2. 扩展试样，为某个特定类别加入 Perspective API 风格的毒性评分。把它与类别分数的阈值行为作比较。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。把每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法干净映射的类别。

4. 为一个代码助手部署（例如 GitHub Copilot）设计审核栈。指出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 于 2027 年 2 月退役。规划向 Azure AI Content Safety 的迁移。指出迁移中风险最高的环节。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | 「omni-moderation-latest」 | 基于 GPT-4o 的 13 类别（文本）分类器，部分支持多模态 |
| Perspective API | 「Google Jigsaw 毒性」 | 前 LLM 时代的毒性评分基线 |
| Llama Guard | 「MLCommons 14 类别」 | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| Input moderation | 「生成前过滤」 | 在模型调用前对用户 prompt 分类 |
| Output moderation | 「生成后过滤」 | 在交付前对模型输出分类 |
| Custom moderation | 「领域规则」 | 部署特定的规则（正则、允许列表、政策） |
| Layered moderation | 「三层全上」 | 标准的生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 的替代方案
