# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 提供商锁定的代价很高。不同的工具调用工作负载适合不同的模型。路由网关提供统一的 API 表面、重试、故障转移、成本跟踪和防护栏。2026 年有三种典型形态占据主导：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课给出决策标准，并走读一个用标准库实现的路由网关。

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## 学习目标

- 区分自托管、托管和生产级三种路由方案。
- 实现一条回退链，按既定优先级顺序在提供商故障时重试。
- 跨提供商跟踪每次请求的成本和 token 用量。
- 针对给定的生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题

需要提供商路由的场景：

1. **成本。** Claude Sonnet 的成本是 Haiku 的 3 倍。对于分诊任务，Haiku 已经足够；对于综合任务，Sonnet 物有所值。按请求路由。

2. **故障转移。** OpenAI 出现一小时故障，每个请求都失败。你希望自动回退到 Anthropic，而无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速的首 token 时间；批量摘要器则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上对两个模型做 A/B 测试。按测试分桶路由。

为每个集成手写这些逻辑十分重复。路由网关提供统一的 OpenAI 兼容 API，其余事情由它处理。

## 概念

### OpenAI 兼容的代理形态

大家都说 OpenAI 的格式。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。客户端不需要关心这些差异。

### 模型别名

你的代码不写死某个快照 id，而是使用 `our_smart_model`。网关把别名映射到真实模型。当提供商发布新一代模型时，你在服务端修改别名；你的代码完全不用动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这条链。重试会消耗预算，因此级联回退不会让成本失控。

### 语义缓存

完全相同或近乎相同的 prompt 会命中缓存，而不是发给提供商。在重复的智能体循环中，节省可达 30% 到 60%。缓存键基于 embedding；近乎相同的 prompt 共享同一个缓存槽位。

### 防护栏

网关级防护栏：

- **PII 脱敏。** 在发送 prompt 之前执行正则或基于 ML 的脱敏。
- **策略违规。** 拒绝包含违禁内容的 prompt。
- **输出过滤器。** 清洗补全结果中的泄漏。

Portkey 和 Kong 都自带成体系的防护栏。LiteLLM 将其保留为可选项。

### 按 key 的速率限制

一个 API key 对应一个团队。按 key 的预算防止某个团队耗尽共享配额。大多数网关支持这一点。

### 自托管与托管的权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署一个代理 | 注册账号 | 两者皆可 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 使用自己的 key | OpenRouter 额度 | 使用自己的 key |
| 可观测性 | OpenTelemetry | 仪表板 | 完整 OTel + PII 脱敏 |
| 最适合 | 希望完全掌控的团队 | 快速原型验证 | 有合规要求的生产环境 |

当你拥有 SRE 团队并希望数据主权时，LiteLLM 是首选。当你想要单一订阅且不想管理基础设施时，OpenRouter 是首选。当你需要开箱即用的防护栏与合规能力时，Portkey 是首选。

### 成本跟踪

每个请求都携带 `provider`、`model`、`input_tokens`、`output_tokens`。将其乘以每个模型的单 token 价格（来自网关维护的价格表），再按用户 / 团队 / 项目聚合。

### MCP 与路由

网关既可以路由 LLM 调用，也可以路由 MCP 采样（sampling）请求。当采样请求的 modelPreferences 偏好某个特定模型时，网关会将其转换到正确的后端。这正是 Phase 13 · 17（MCP 网关）与本课的路由网关有时会合并为一个服务的地方。

### 路由策略

- **静态优先级。** 使用列表中的第一个；出错时回退。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟 / 质量要求的最便宜模型。
- **延迟感知。** 选择最近 N 分钟内最快的模型。
- **任务感知。** prompt 分类器把编码任务路由到一个模型，把摘要任务路由到另一个。

```figure
tp-router-failover
```

## 使用它

`code/main.py` 用约 150 行实现了一个路由网关：接受 OpenAI 格式的请求，转换为各提供商的桩（stub），运行优先级回退链，跟踪每次请求的成本，并对输入应用一遍 PII 脱敏。用三个场景运行它：正常请求、主提供商故障触发回退、PII 泄漏被脱敏拦截。

重点看：

- `ROUTES` 字典：别名 -> 按优先级排序的具体提供商列表。
- 回退循环在 5xx 时重试。
- 成本跟踪器把 token 用量乘以每个模型的费率。
- PII 脱敏器在转发前清洗 SSN 格式的模式。

## 交付它

本课产出 `outputs/skill-routing-config-designer.md`。给定工作负载画像（延迟、成本、合规），该技能会选择 LiteLLM / OpenRouter / Portkey，并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景；确认回退落在第二个提供商上，且成本归属正确。

2. 加入语义缓存：用 prompt 的 SHA256 作为查找键；缓存命中时立即返回。测量重复调用节省的成本。

3. 加入一个 prompt 分类器：把「code ...」类 prompt 路由到偏重智能程度的别名，把「summarize ...」类 prompt 路由到偏重速度的别名。

4. 设计按团队预算：每个团队有月度支出上限；一旦达到上限，网关拒绝请求。选择一种执行粒度（按请求或按时间窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 的文档。指出每个产品独有、另外两个没有的一个功能。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 路由网关 | 「LLM 代理」 | 位于众多提供商之前、提供统一 API 表面的层 |
| OpenAI 兼容 | 「说 OpenAI schema」 | 接受 `/v1/chat/completions` 格式，并转换到任意后端 |
| 模型别名 | 「our_smart_model」 | 代码中的名称，由网关映射到具体模型 |
| 回退链 | 「重试列表」 | 失败时按顺序尝试的提供商列表 |
| 语义缓存 | 「prompt embedding 缓存」 | 键是 prompt 的 embedding；近似重复的 prompt 共享缓存命中 |
| 防护栏 | 「输入 / 输出过滤器」 | 脱敏 PII、拒绝策略违规 |
| 按 key 的速率限制 | 「团队预算」 | 作用于某个 API key 的配额 |
| 成本跟踪 | 「每次请求的花费」 | 按模型聚合 token 用量 x 单价 |
| LiteLLM | 「开源代理」 | 可自托管的开源路由网关 |
| OpenRouter | 「托管 SaaS」 | 按额度计费的托管网关 |
| Portkey | 「生产级选项」 | 开源 + 托管，内置防护栏 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — 文档](https://portkey.ai/docs) — 带防护栏的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 选型指南
- [Relayplane — 2026 年 LLM 网关对比](https://relayplane.com/blog/llm-gateway-comparison-2026) — 厂商调研
