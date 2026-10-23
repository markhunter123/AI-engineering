# 行动预算、迭代上限与成本治理器

> 一个中型电商 Agent 的月度 LLM 成本在团队启用「订单追踪」技能后从 1,200 美元涨到 4,800 美元。这不是定价 bug。这是一个 Agent 找到了新的循环，并在其中持续花钱。微软的 Agent Governance Toolkit（2026 年 4 月 2 日）把针对这类情况的防御手段制度化：按请求的 `max_tokens`、按任务的 token 与美元预算、按日/月上限、迭代上限、分层模型路由、prompt 缓存、上下文窗口化、昂贵操作上的 HITL checkpoint、预算突破时的终止开关。Anthropic 的 Claude Code Agent SDK 以不同名称提供同样的原语。财务速度限制——例如 10 分钟内花费超过 50 美元就切断访问——比月度上限更快地捕捉循环。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## 问题

自主 Agent 每一轮都在花真金白银。聊天机器人的糟糕输出只是一条糟糕的回复；Agent 的糟糕循环则是一张账单。业界记录这种失效模式的术语是「钱包拒绝服务」（Denial of Wallet）——Agent 不断推理、不断调用工具、不断计费，而没有任何东西阻止它，因为根本没有设计任何东西来阻止。

修复方案不是一个数字。它是一叠不同时间尺度和粒度上的限制：按请求、按任务、按小时、按天、按月。设计良好的限制栈能在几分钟内捕捉失控循环，在几小时内捕捉缓慢泄漏，在一天内捕捉糟糕的发布。同样的限制栈能让长视野、自主运行的 Agent 始终守住预算。

这是一节工程课：数学很简单，纪律才是团队失败的地方。下面列出的限制都出自微软 Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 的文档。

## 概念

### 成本治理器栈

1. **按请求的 `max_tokens`。** 简单。防止任何单次调用产生无界补全。
2. **按任务的 token 预算。** 整个运行期间不超过 N 个 token。达到上限即硬停止。
3. **按任务的美元预算。** 与 token 相同，但以货币计。Claude Code 中的 `max_budget_usd`。
4. **按工具的调用上限。** 不超过 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** Agent 循环的总迭代数；防止无限推理循环。
6. **按分钟 / 小时 / 天 / 月的上限。** 滚动窗口。在不同时间尺度上捕捉泄漏。
7. **财务速度限制。** 例如「如果 10 分钟内花费超过 50 美元，就切断访问」。在月度上限触发之前捕捉基于循环的燃烧。
8. **分层模型路由。** 默认使用较小模型；仅当分类器判断任务值得时才升级到较大模型。
9. **Prompt 缓存。** 系统 prompt 和稳定上下文存入提供商缓存；重新发送的 token 成本几乎为零。
10. **上下文窗口化。** 通过压缩 / 摘要让活跃上下文低于阈值；直接降低 token 成本。
11. **昂贵操作上的 HITL checkpoint。** 在执行已知昂贵的操作（长时间 tool call、大量下载、昂贵的模型升级）之前，要求人工点击确认。
12. **预算突破时的终止开关。** 任何上限触发时会话中止。上限被记录；需要单独的重新启用路径。

### 为什么用栈，而不是单一上限

单个月度上限只有在钱包被掏空之后才能捕捉失控 Agent。单一的按请求上限在会话层面什么也捕捉不到。不同的失效模式需要不同的时间尺度：

- **失控循环**（Agent 卡在 5 秒重试中）：由速度限制捕捉。
- **缓慢泄漏**（Agent 每个任务做约 2 倍预期的工作）：由每日上限捕捉。
- **糟糕发布**（新版本使用 5 倍 token）：由每周 / 每月上限捕捉。
- **合理激增**（真实需求，不是 bug）：由小时 / 天上限捕捉，并留下清晰日志。

### 一个 harness 预算接口

Claude Code Agent SDK 暴露（公开文档）：

- `max_turns` —— 迭代上限。
- `max_budget_usd` —— 美元上限；突破时会话中止。
- `allowed_tools` / `disallowed_tools` —— 工具允许列表和拒绝列表。
- 工具使用前的 hook 点，用于自定义成本核算。

与权限模式阶梯（第 10 课）结合使用。没有 `max_budget_usd` 的 `autoMode` 会话是无治理的自主性。Anthropic 明确将 Auto Mode 描述为需要预算控制；分类器与成本是正交的。

### EU AI Act 与 OWASP Agentic Top 10

微软的 Agent Governance Toolkit 覆盖 OWASP Agentic Top 10 和 EU AI Act 第 14 条（人类监督）的要求。在欧盟生产部署时，日志记录和上限执行不是可选项。

### 观察到的 1,200 → 4,800 美元案例

微软文档中的真实案例：一个电商 Agent 在添加新工具后月度成本翻了三倍。该工具允许 Agent 在每个会话中轮询订单状态。没有循环检测。没有按工具上限。没有对周环比增长的告警。修复方案是按工具上限加每日增长告警。这是一个模板：每一个新的工具面都是一个潜在的新循环；每一个新工具都需要自己的上限和自己的告警。

```figure
cost-governor-stack
```

## 使用它

`code/main.py` 模拟一个带和不带分层成本治理器栈的 Agent 运行。模拟 Agent 在若干轮后漂移进轮询循环；分层栈在速度窗口内捕捉到它，而单个月度上限要到几天后才会触发。

## 交付它

`outputs/skill-agent-budget-audit.md` 审计拟议 Agent 部署的成本治理器栈，并标出缺失的层。

## 练习

1. 运行 `code/main.py`。确认在轮询循环轨迹上，速度限制先于迭代上限触发。现在禁用速度限制，测量在迭代上限捕捉到它之前 Agent「花费」了多少。

2. 为浏览器 Agent（第 11 课）设计一组按工具上限。哪个工具需要最紧的上限？哪个工具可以在无风险的情况下无界运行？

3. 阅读微软 Agent Governance Toolkit 文档。列出该工具包命名的每一种上限类型。把每一种映射到一种失效模式（失控循环、缓慢泄漏、糟糕发布、激增）。

4. 为一项现实任务（例如「为仓库中的 50 个 issue 分类」）估算一次通宵无人值守运行的价格。把 `max_budget_usd` 设为点估计的 2 倍。论证为什么是 2 倍。

5. Claude Code 的 `max_budget_usd` 在会话聚合成本上触发。设计一个你在外部执行的补充速度限制。什么触发切断？重新启用又是什么样子？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 钱包拒绝服务（Denial of Wallet） | 「失控账单」 | Agent 循环产生花费，却没有上限阻止 |
| max_tokens | 「按请求上限」 | 单次补全大小的上限 |
| max_turns | 「迭代上限」 | 会话中 Agent 循环迭代的上限 |
| max_budget_usd | 「美元终止开关」 | 会话成本上限；突破即中止 |
| 速度限制 | 「速率上限」 | 短时间窗口内花费的限制（例如 50 美元 / 10 分钟） |
| 分层路由 | 「小模型优先」 | 默认使用廉价模型；仅在分类器判定需要时升级 |
| Prompt 缓存 | 「缓存系统 prompt」 | 提供商侧缓存将重发 token 成本降至接近零 |
| HITL checkpoint | 「人工审批门」 | 昂贵操作前需要人工点击 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK —— Agent 循环与预算](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— `max_turns`、`max_budget_usd`、工具允许列表。
- [Microsoft Agent Framework —— 人在回路与治理](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— 成本治理器 checkpoint。
- [Anthropic —— Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) —— 提供商侧成本控制。
- [Anthropic —— Prompt 缓存（Claude API 文档）](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) —— 缓存机制。
- [Anthropic —— 在实践中衡量 Agent 自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长视野 Agent 的成本画像。
