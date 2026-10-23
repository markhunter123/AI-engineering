# 工具使用与 Function calling

> Toolformer（Schick 等，2023）开创了自监督的工具标注。Berkeley Function Calling Leaderboard V4（Patil 等，2025）设定了 2026 年的标准：40% Agent、30% 多轮、10% 实时、10% 非实时、10% 幻觉。单轮已经解决。记忆、动态决策和长时程工具链还没有。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## 学习目标

- 解释 Toolformer 的自监督训练信号：只有当执行工具能降低下一 token 的损失时，才保留工具标注。
- 说出 BFCL V4 的五个评估类别，以及每一类衡量什么。
- 实现一个 stdlib 工具注册表，包含 schema 校验、参数强制转换和执行沙箱。
- 诊断 2026 年的三个开放问题：长时程工具链、动态决策和记忆。

## 问题

早期的工具使用问的是：模型能否预测出正确的 function call？现代的工具使用问的是：模型能否在 40 步中串联工具，带记忆、带部分可观测性、能从工具失败中恢复，而且不会幻觉出不存在的工具？

Toolformer 奠定了基线：模型可以通过自监督学会何时调用工具。BFCL V4 定义了 2026 年的评估目标。两者之间的差距，就是生产 Agent 所处的空间。

## 概念

### Toolformer（Schick 等，NeurIPS 2023）

思路：让模型在自己的预训练语料上标注候选 API 调用。对每个候选执行它。只有当把工具结果纳入后能降低下一 token 的损失时，才保留该标注。在筛选后的语料上做微调。

覆盖的工具：计算器、问答系统、搜索引擎、翻译器、日历。自监督信号完全取决于工具是否有助于预测文本——不需要人工标签。

规模结果：工具使用在规模上涌现。较小的模型会因工具标注受损；较大的模型则从中获益。这就是为什么 2026 年的前沿模型内置了很强的工具使用能力，而大多数 7B 模型需要显式的工具使用微调才可靠。

### Berkeley Function Calling Leaderboard V4（Patil 等，ICML 2025）

BFCL 是 2026 年事实上的评估基准。V4 的构成：

- **Agent（Agentic，40%）**——完整的 Agent 轨迹：记忆、多轮、动态决策。
- **多轮（Multi-Turn，30%）**——带工具链的交互式对话。
- **实时（Live，10%）**——用户提交的真实 prompt（分布更难）。
- **非实时（Non-Live，10%）**——合成测试用例。
- **幻觉（Hallucination，10%）**——检测本不应调用任何工具的情况。

V3 引入了基于状态的评估：在一串 tool call 之后，检查 API 的实际状态（例如「文件是否已创建？」），而不是匹配 tool call 的 AST。V4 增加了网页搜索、记忆和格式敏感性类别。

2026 年的关键发现：单轮 function calling 接近解决。失败集中在记忆（跨轮携带上下文）、动态决策（根据先前结果选择工具）、长时程链条（20 步后漂移）和幻觉检测（没有合适工具时拒绝调用）。

### 工具 schema

每个供应商都有自己的 schema。细节不同，但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是承重的——模型靠读它来选对工具。工具描述写得差，是选错工具这类失败的头号根因。

### 参数校验

不要信任任何 tool call。要校验：

1. **类型强制转换。** 模型可能在 schema 要求 int 的地方返回字符串 "5"。无歧义时转换；有歧义时拒绝。
2. **枚举校验。** 如果 schema 要求 `status in {"open", "closed"}` 而模型输出了 `"in_progress"`，就用带有说明的错误拒绝。
3. **必填字段。** 缺少必填字段 -> 立即把错误观察回传给模型，而不是崩溃。
4. **格式校验。** 日期、电子邮件、URL——用具体的解析器校验，不要用正则。

每一次校验失败都应返回结构化的观察，好让模型用正确的形状重试。

### 并行 tool calls

现代供应商支持在同一个助手轮中并行调用工具。循环是：

1. 模型发出 3 次 tool call，各自带有不同的 `tool_use_id`。
2. 运行时执行它们（若相互独立则并行）。
3. 每个结果作为一个 `tool_result` 块返回，通过 `tool_use_id` 关联。

工程守则：把关联 ID 当作承重结构。互换它们，你得到的就是工具与结果错配的路由。

### 沙箱

工具执行就是沙箱边界。细节见第 09 课。简版：每个工具都应明确读写范围、网络访问、超时和内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## 构建它

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集校验器（仅用 stdlib）。
- 工具注册，包含描述、输入 schema、超时和执行器。
- 参数强制转换和枚举校验。
- 带关联 ID 的并行工具分派。
- 以结构化字符串形式返回的错误观察。

运行它：

```
python3 code/main.py
```

轨迹展示一个迷你 Agent 在一轮中调用三个工具，其中一个调用被故意写成畸形，被拒绝并返回模型可据以行动的描述性错误。

## 使用它

每个供应商都有自己的工具 schema——Anthropic、OpenAI、Gemini、Bedrock。如果你需要多供应商支持，就使用一层转换（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准——如果工具使用是产品的核心，上线前拿它跑一遍你的 Agent。

## 交付它

`outputs/skill-tool-registry.md` 为给定的任务领域生成工具目录、schema 和注册表。包含描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 增加一个「no-op」工具，让模型可以显式拒绝使用任何其他工具。在类 BFCL 的幻觉测试上度量效果。
2. 为 int 写成字符串和 float 写成字符串实现参数强制转换。强制转换从什么时候开始掩盖真正的 bug？
3. 增加按工具的超时和熔断器（连续 3 次失败后 60 秒内拒绝该工具）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 的说明。选一个类别（例如「多轮」），用你的 Agent 跑 10 个示例 prompt。报告通过率。
5. 把 stdlib 校验器移植到 Pydantic 或 Zod。Pydantic/Zod 抓住了哪些玩具实现漏掉的问题？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|----------------|------------------------|
| Function calling | 「工具使用」 | 带校验 schema 的结构化输出式 tool call |
| Toolformer | 「自监督工具标注」 | Schick 2023——只保留结果能降低下一 token 损失的调用 |
| BFCL | 「Berkeley Function Calling Leaderboard」 | 2026 年基准：40% Agent、30% 多轮、10% 实时、10% 非实时、10% 幻觉 |
| Tool schema | 「给模型看的函数签名」 | 名称、描述、参数的 JSON Schema |
| tool_use_id | 「关联 ID」 | 把 tool call 与结果绑定；并行分派必不可少 |
| Hallucination detection | 「知道何时不该调用」 | V4 类别：没有合适工具时拒绝调用 |
| Argument coercion | 「字符串转 int 修复」 | 针对可预测的 schema 不匹配做窄范围修复；有歧义时拒绝 |
| Sandboxing | 「工具执行边界」 | 按工具限定读写范围、网络、超时、内存上限 |

## 进一步阅读

- [Schick 等，Toolformer（arXiv:2302.04761）](https://arxiv.org/abs/2302.04761)——自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)——2026 年评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Agent SDK 中的生产级工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)——函数工具类型与 Guardrails
