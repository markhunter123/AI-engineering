# 红队工具——Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年的红队工具栈。Llama Guard（Meta）——一个在 14 个 MLCommons 危害类别上微调的 Llama-3.1-8B 分类器；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，带有静态、动态和自适应探针，覆盖幻觉、数据泄露、提示注入、毒性和越狱。PyRIT（Microsoft）——多轮红队行动，用 Crescendo、TAP 和自定义转换器链进行深度利用。Llama Guard 3 记录在 Meta 的《Llama 3 Herd of Models》（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的探针架构见 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（第 12-15 课）与部署（第 17 课及以后）之间的生产接口。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器，或两者兼有。
- 说出 14 个 MLCommons 危害类别，并指出一个不那么直观的类别（Code Interpreter Abuse）。
- 描述 Garak 的探针架构：probes、detectors、harnesses。
- 描述 PyRIT 的多轮行动结构，以及它如何与 Garak 探针组合。

## 问题

第 12-15 课呈现了攻击面。生产部署需要可重复、可扩展的评估。2026 年有三款工具占据主导：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（行动编排器）。它们各自针对红队生命周期的不同层面。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B 模型，针对 MLCommons AILuminate 的 14 个类别微调为输入/输出分类器：

- 暴力犯罪、非暴力犯罪、性相关、儿童性虐待材料（CSAM）、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨
- 自杀/自伤、性内容、选举、代码解释器滥用

支持 8 种语言。用法：放在 LLM 之前（输入审核）、之后（输出审核），或两者兼有。两种用途会产生不同的训练分布——Llama Guard 3 以单一模型发布，同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动 CPU 上约 30 tokens/s）是量化边缘版本。

Llama Guard 4（2025 年 4 月）为 12B，原生多模态，从 Llama 4 Scout 剪枝而来。它用一个同时接收文本和图像的分类器，取代了此前的 8B 文本版和 11B 视觉版。

### Garak（NVIDIA）

开源漏洞扫描器。架构：

- **探针（Probes）。** 针对幻觉、数据泄露、提示注入、毒性、越狱的攻击生成器。分为静态（固定 prompt）、动态（生成的 prompt）、自适应（响应目标输出）。
- **检测器（Detectors）。** 按预期失效模式给输出打分——有毒、泄露、已越狱。
- **实验装置（Harnesses）。** 管理探针—检测器配对、运行行动、生成报告。

TrustyAI 将 Garak 与 Llama-Stack shields（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，用于端到端的带防护目标评估。基于分级的评分（TBSA）取代二元通过/失败——同一探针下，模型可以在严重度 3 级通过，却在严重度 5 级失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。多轮红队行动。围绕以下组件构建：

- **转换器（Converters）。** 转换种子 prompt——复述、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 运行行动：Crescendo（升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 更重的表亲。Garak 运行成千上万个单轮探针；PyRIT 运行深度多轮行动，专门设计来击破特定的失效模式。

### 工具栈

在模型两侧都放上 Llama Guard。每晚运行 Garak 做回归。发布前运行 PyRIT 行动。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **评判者身份。** 三款工具都可以使用 LLM 评判者；评判者的校准会决定报告的 ASR（第 12 课）。要连同工具一起说明评判者。
- **探针陈旧。** 随着模型针对探针打补丁，Garak 探针会老化。自适应探针（PAIR 形态）比静态探针老化得慢。
- **Llama Guard 在良性内容上的误报率。** 早期 Llama Guard 版本对政治和 LGBTQ+ 内容过度标记；Llama Guard 3/4 的校准有所改善，但并未针对每个部署单独校准。

### 这一课在 Phase 18 中的位置

第 12-15 课是攻击家族。第 16 课是生产工具。第 17 课（WMDP）是两用能力的评估。第 18 课是把这些工具纳入政策结构的前沿安全框架。

```figure
al-guard-stack
```

## 使用它

`code/main.py` 构建了一个玩具版 Llama Guard 式分类器（针对 14 个类别的关键词 + 语义特征）、一个玩具 Garak 实验装置（探针—检测器循环），以及一条 PyRIT 式多轮转换器链。你可以对模拟目标运行这三款工具，观察不同的覆盖特征。

## 交付它

本课产出 `outputs/skill-red-team-stack.md`。给定一份部署说明，它指出三款工具中哪些适用、各自需要配置什么，以及应运行怎样的回归节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 式分类器在单轮与多轮攻击上的检测率。

2. 实现一个新 Garak 探针：base64 编码的有害请求。测量 Llama Guard 式分类器对它的检测。

3. 用「先翻译成法语，再复述」的转换器扩展 PyRIT 式转换器链。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个类别，其训练数据在合法开发者内容上现实中会产生高误报率。

5. 比较 Garak 与 PyRIT 的设计原则。论证在什么部署中各自是合适的工具。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | 「分类器」 | 在 14 个危害类别上微调的 Llama-3.1-8B/4-12B 安全分类器 |
| Garak | 「扫描器」 | NVIDIA 开源漏洞扫描器；probes、detectors、harnesses |
| PyRIT | 「行动工具」 | Microsoft 多轮红队编排器；converters、orchestrators、scoring |
| Prompt-Guard | 「小分类器」 | Meta 的 86M 提示注入分类器，与 Llama Guard 配对使用 |
| TBSA | 「分级评分」 | Garak 基于分级的通过/失败，取代二元结果 |
| 转换器链 | 「paraphrase + encode + ...」 | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons 危害类别 | 「14 类分类法」 | Llama Guard 所针对的行业标准分类法 |

## 延伸阅读

- [Meta——Llama Guard 3（收录于 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783)——8B 分类器
- [Meta——Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713)——量化移动分类器
- [NVIDIA Garak——GitHub](https://github.com/NVIDIA/garak)——扫描器仓库与文档
- [Microsoft PyRIT——GitHub](https://github.com/Azure/PyRIT)——行动工具包
