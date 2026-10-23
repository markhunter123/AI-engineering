# 转交与例行程序 —— 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把多 Agent 编排浓缩成两个原语：**例行程序**（routine，把指令与工具组织成一个 system prompt）与**转交**（handoff，一个返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的转交工具来完成路由。OpenAI Agents SDK（2025 年 3 月）是生产级的继任者。Swarm 本身则始终是最干净的概念参考——它的全部源码只用几百行就能放下。这个模式之所以流行，是因为 API 表面大致就是「agent = prompt + tools；handoff = 返回 agent 的函数」。局限：无状态，所以记忆是调用方的问题。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

每个多 Agent 框架都想让你学它自己的 DSL：LangGraph 的节点与边、CrewAI 的 crew 与 task、AutoGen 的 GroupChat 与 manager。这些 DSL 是真实的抽象，但它们让整件事显得比实际需要的更沉重。

Swarm 反其道而行：直接使用模型本来就具备的 tool calling 能力。转交变成 tool call。编排器就是当前持有这段对话的那个 Agent。状态机隐含在各 Agent 的 system prompt 里。

## 概念

### 两个原语

**例行程序（Routine）。** 一段 system prompt，定义 Agent 的角色与可用工具。可以把它想成一组有范围的指令：「你是分诊 Agent；如果用户询问退款，就转交给退款 Agent。」

**转交（Handoff）。** Agent 可以调用的一个工具，它返回一个新的 Agent 对象。Swarm 运行时检测到返回的是 Agent，就在下一轮切换活跃 Agent。

整个抽象就只有这些。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊 Agent 的 system prompt 让它根据用户消息选择正确的转交。真正完成路由的是 LLM 的 tool calling 能力。

### 为什么它流行

- **API 小。** 只需学两个概念。
- **利用模型已有的能力。** 各家提供商的 tool calling 早已达到生产可用水平。
- **没有状态机负担。** 你不需要描述图；Agent 的 prompt 自己说明要转交给谁。

### 无状态的取舍

Swarm 在两次运行之间明确是无状态的。框架在一次运行期间保留消息历史，但不持久化任何东西。记忆、连续性、长时间运行的任务——全都是调用方的问题。

到了生产环境（OpenAI Agents SDK，2025 年 3 月），这是最主要的改变之一：SDK 增加了内置的会话管理、防护栏与追踪，同时保留了转交原语。

### Swarm/转交适用的情况

- **分诊模式。** 一线 Agent 把用户路由给专家。
- **基于技能的转交。** 「任务需要写代码就叫程序员；需要调研就叫研究员。」
- **简短、有边界的对话。** 客户支持、FAQ 转工单、简单工作流。

### Swarm 吃力的情况

- **带共享记忆的长会话。** 转交会把对话状态重置为新 Agent 的 prompt 加历史。若调用方不自行管理记忆，Agent 之间就没有持久状态。
- **并行执行。** 转交是逐个进行的——活跃 Agent 会切换。要并行，需要调用方编排多次 Swarm 运行。
- **审计与重放。** 无状态运行很难精确重放；LLM 的转交选择并不确定。

### OpenAI Agents SDK（2025 年 3 月）

生产级继任者增加了：

- **会话状态。** 跨运行持久的 thread。
- **防护栏。** 输入/输出校验钩子。
- **追踪。** 每次 tool call 与转交都有日志。
- **转交过滤器。** 控制转交时传递哪些上下文。

转交原语保留下来，生产级的易用能力则围绕它补齐。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，区别在于**由谁挑选下一个**：

- GroupChat：由外部选择器（函数或 LLM）挑选下一个发言者。
- Swarm：当前 Agent 通过调用转交工具来挑选它的后继者。

Swarm 是「Agent 决定下一步」；GroupChat 是「管理者决定下一步」。Swarm 的决策存在于活跃 Agent 的 tool call 中；GroupChat 的决策存在于 `GroupChatManager` 中。

```figure
sw-handoff-routing
```

## 构建它

`code/main.py` 从零实现 Swarm：一个 Agent 数据类，一套转交机制（工具返回 Agent），以及一个检测 Agent 切换的运行循环。

演示：一个分诊 Agent 把请求路由给退款、销售或支持专家。每位专家都有自己的工具。运行循环会打印每一次转交。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md` 为给定任务设计转交拓扑：存在哪些 Agent，它们可以调用哪些转交，传递什么上下文。

## 交付它

检查清单：

- **转交日志。** 每次转交都写入一条追踪事件，包含来源 Agent、目标 Agent、上下文快照。
- **上下文传递规则。** 决定转交时传递什么：完整历史（昂贵）、最近 N 条消息，还是一份摘要。
- **转交处的防护栏。** 转交给拥有不同工具权限的专家时必须经过认证——否则提示注入可以强行触发不需要的转交。
- **循环检测。** 两个 Agent 来回转交是常见故障；用一个简单的「最近 K 次」环形检查就能发现。
- **兜底 Agent。** 如果转交目标不存在，就回退到一个安全的默认选项。

## 练习

1. 运行 `code/main.py`，从分诊转到退款 Agent。确认第二轮的活跃 Agent 就是退款。
2. 增加一条循环检测规则：如果同样的两个 Agent 连续转交 3 次，就强制退出。设计兜底方案。
3. 阅读 OpenAI Agents SDK 关于转交过滤器的文档。实现一个「转交时摘要」版本：在接手的 Agent 开始工作前，交出的 Agent 把上下文压缩成要点摘要。
4. 对比 Swarm 的转交与 GroupChatManager 的选择器。哪种模式会让提示注入的危害更大，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 做出的一项明确设计决策，看 OpenAI Agents SDK 是改变了它还是保留了它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 例行程序（Routine） | 「Agent 的提示词」 | System prompt + 工具列表。定义角色与可用的转交。 |
| 转交（Handoff） | 「移交给另一个 Agent」 | 活跃 Agent 可以调用的工具，返回一个新的 Agent。运行时会切换活跃 Agent。 |
| 无状态（Stateless） | 「两次运行之间没有记忆」 | Swarm 不持久化任何东西；记忆由调用方负责。 |
| 活跃 Agent（Active agent） | 「现在谁在说话」 | 当前持有对话的 Agent。转交会改变它。 |
| 上下文传递（Context transfer） | 「转交时带走什么」 | 关于接手 Agent 能看到哪些历史的策略：完整、最近 N 条，还是摘要。 |
| 转交循环（Handoff loop） | 「Agent 打乒乓」 | 两个 Agent 不断把任务转回给对方的失效模式。 |
| OpenAI Agents SDK | 「生产版 Swarm」 | 2025 年 3 月的继任者；在转交原语之上增加了会话、防护栏与追踪。 |
| 转交过滤器（Handoff filter） | 「转交处的关卡」 | SDK 特性，在转交边界检查并修改上下文。 |

## 延伸阅读

- [OpenAI cookbook —— 编排 Agent：Routines 与 Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 参考性阐述
- [OpenAI Swarm 仓库](https://github.com/openai/swarm) —— 原始实现，作为概念参考保留
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— 带会话与追踪的生产级继任者
- [Anthropic 关于 Claude 中转交的说明](https://docs.anthropic.com/en/docs/claude-code) —— Claude Code 子 Agent 如何通过 `Task` 使用类似转交的模式
