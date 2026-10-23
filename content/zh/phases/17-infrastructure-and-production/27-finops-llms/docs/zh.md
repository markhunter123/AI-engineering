# LLM 的 FinOps —— 单位经济与多租户归因

> 传统的 FinOps 在 LLM 支出上行不通。成本是 token 交易，而不是资源占用时长。标签对不上——一次 API 调用是一笔交易，不是一项资产。工程决策（prompt 设计、上下文窗口、输出长度）就是财务决策。2026 年的打法要求在第一天就打点三个归因维度：按用户（`user_id`），用于席位定价与扩容；按任务（`task_id` + `route`），用于产品界面成本与优先级；按租户（`tenant_id`），用于单位经济与续约。四层 token——prompt、tool、memory、response——混在一个桶里会掩盖支出。多租户产品的执行阶梯：按租户限流（预期峰值的 2-3 倍，清晰的 429 + retry-after）；每日支出上限（合同上限的 1.5-3 倍；触发限流收紧 + 告警）；当支出 z-score > 4 时触发熔断（自动暂停 + 呼叫值班）。归因模式：打标签再聚合、遥测关联（trace-ID → 账单；准确度最高）、采样与外推、基于模型的分配、事件溯源、实时流式。单位指标：每个已解决查询的成本、每件生成产物的成本——而不是 $/M token。事后补标签总会漏；在请求创建时就打点。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## 学习目标

- 解释为什么传统 FinOps（标签 + 分层）在 LLM 支出上失效，并说出三个新的归因维度。
- 列举四层 token（prompt、tool、memory、response），以及为什么单桶计费会掩盖成本。
- 为多租户产品设计执行阶梯（限流 → 支出上限 → 熔断）。
- 选择单位指标（每个已解决查询 / 每件产物的成本），而不是 $/M token。

## 问题

你的账单显示 40,000 美元。你不知道：
- 哪个租户花的。
- 哪个产品功能驱动的。
- 是否有某个用户滥用。
- 是 prompt 膨胀、tool 调用还是 memory 放大造成的。

在提供商侧打标签再聚合，对云资源（EC2、S3）有效，因为标签会传播到账单条目。LLM API 调用不会自动打标签——你必须在调用点打上 user/task/tenant 并一路传递。事后归因总会漏掉边界情况。

## 概念

### 三个归因维度

**按用户**（`user_id`）：谁在花多少钱。驱动席位定价、扩容对话，识别重度用户。

**按任务**（`task_id` + `route`）：哪个产品界面花多少钱。驱动功能优先级、砍掉昂贵功能的决策。

**按租户**（`tenant_id`）：哪个客户在赚钱。驱动单位经济、续约定价、分层阈值。

第一天就在调用点给三者都打点。事后补永远更差。

### 四层 token

| 层 | 示例 | 通常占总量的百分比 |
|-------|---------|---------------------|
| Prompt | 系统提示 + 用户输入 | 40-60% |
| Tool | 回填的 tool 调用结果 | 20-40%（Agent 负载） |
| Memory | 历史对话 / 检索到的文档 | 10-30% |
| Response | 模型输出 | 10-30% |

四层混在一个桶里会让优化变成盲人摸象。在归因 schema 中把它们拆开。

### 执行阶梯

1. **按租户限流**。预期峰值的 2-3 倍。返回 429 并带 `Retry-After`。租户感受到摩擦；不会收到意外账单。

2. **按租户每日支出上限**。合同上限的 1.5-3 倍。触发动作：收紧限流 + 通知客户成功团队。

3. **熔断**：当支出相对租户基线的 z-score > 4 时触发。自动暂停租户；呼叫值班；升级到运维 + 客户成功。

### 归因模式

- **打标签再聚合**：在元数据头里打标记；稍后聚合。简单；粗略。
- **遥测关联**：通过 trace ID 把追踪记录和账单关联起来。准确度最高。成熟团队的做法。
- **采样 + 外推**：采样 5-10%，再乘以倍数。对粗略支出估算性价比高；会漏掉长尾。
- **基于模型的分配**：用回归推断成本驱动因素。适用于没有标签的历史数据。
- **事件溯源**：把成本作为事件写入流（Kafka / Kinesis）。实时。
- **实时流式**：仪表盘亚秒级更新。

### 每个 X 的成本才是单位指标

$/M token 是厂商话术。产品指标应该是：

- 每个已解决支持工单的成本。
- 每篇生成文章的成本。
- 每个成功 Agent 任务的成本。
- 每用户会话分钟的成本。

把成本绑到产品结果上。否则优化就没有锚点。

### 成本归因 trace 的形态

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用都发出。存进数据湖。按维度聚合。Phase 17 · 13 的可观测性栈就是它该待的地方。

### 复合节省栈

栈：缓存 + batch + 路由 + 网关。四者齐上：
- 缓存 L2（Phase 17 · 14）：输入便宜约 10 倍。
- Batch（Phase 17 · 15）：打五折。
- 路由到便宜模型（Phase 17 · 16）：成本降低 60%。
- 网关效率（Phase 17 · 19）：冗余 + 重试。

最佳情况叠加：约为朴素基线的 5-10%。大多数团队只启用 2-3 个杠杆；很少四个全上。

### 应该记住的数字

- 归因维度：按用户、按任务、按租户。
- 四层 token：prompt、tool、memory、response。
- 熔断：支出 z-score > 4。
- 单位指标：每个已解决查询的成本，而不是 $/M token。
- 叠加优化：最低可到基线的 5-10%。

```figure
i4-spend-ladder
```

## 使用它

`code/main.py` 模拟一个带三级执行阶梯的多租户 LLM 服务。注入一个滥用租户，演示熔断触发。

## 交付它

本课产出 `outputs/skill-finops-plan.md`。给定产品与规模，设计归因 schema 和执行阶梯。

## 练习

1. 运行 `code/main.py`。熔断在哪个 z-score 触发？你如何选择阈值？
2. 设计一个按租户、按任务的成本仪表盘。你最先构建的 5 个视图是什么？
3. 你最大的租户单位经济为负。提出三项干预措施，按客户影响排序。
4. 为一个支持类产品计算每个已解决工单的成本：每工单 300 万 token，每天约 800 个工单，使用 GPT-5 缓存费率。
5. 论证事后打标签是否可能奏效。什么情况下可以接受？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Per-user attribution | 「用户级成本」 | 每次调用都打上 `user_id` |
| Per-task attribution | 「功能成本」 | `task_id` + `route` 标识产品界面 |
| Per-tenant attribution | 「客户成本」 | `tenant_id`；驱动单位经济 |
| Four token layers | 「成本层」 | prompt + tool + memory + response |
| Rate limit | 「429 防护」 | 网关执行的按租户上限 |
| Daily spend cap | 「每日上限」 | 租户范围预算，超限告警 |
| Kill switch | 「自动暂停」 | 支出 z-score > 4 触发自动停用 |
| Cost per resolved | 「产品单位指标」 | 成本绑定产品结果，而非 token |
| Telemetry joiner | 「trace 到账单」 | 准确度最高的归因模式 |
| Stacked optimization | 「cache+batch+route+gateway」 | 复合节省至基线的约 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
