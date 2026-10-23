# 项目 11 —— LLM 可观测性与评估仪表盘

> Langfuse 走向了 open-core。Arize Phoenix 发布了 2026 年的 GenAI semconv 映射。Helicone 和 Braintrust 都加码每用户成本归因。Traceloop 的 OpenLLMetry 成了事实上的 SDK 埋点方案。生产形态是：ClickHouse 存 trace，Postgres 存元数据，Next.js 做 UI，再加一大批评估任务（DeepEval、RAGAS、LLM-judge）跑在采样的 trace 上。自托管一个，接入至少四个 SDK 家族，并演示在五分钟内抓到一次注入的回归。

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 hours

## 问题

2026 年，每个跑生产流量的 AI 团队都会在模型旁边维护一个可观测性平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄漏告警。开源参考实现——Langfuse、Phoenix、OpenLLMetry——都收敛到 OpenTelemetry GenAI 语义约定作为摄取 schema。现在你可以用一套 SDK 给 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 埋点，产出彼此兼容的 span。

你会构建一个自托管的仪表盘，接入至少四个 SDK 家族，在采样的 trace 上跑一小批评估任务，检测漂移并告警。衡量标准是：给定一次故意注入的回归（某个提示开始产出 PII），仪表盘能在五分钟内抓到它并触发告警。

## 概念

摄取走 OTLP HTTP。SDK 产出 GenAI-semconv span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。span 落入 ClickHouse 做列式分析；元数据（用户、会话、应用）落入 Postgres。

评估以批处理任务的形式跑在采样的 trace 上。DeepEval 打忠实度、毒性和答案相关性分数。当 trace 携带检索上下文时，RAGAS 打检索指标。自定义 LLM-judge 跑领域特定检查（PII 泄漏、违反策略的回复）。评估结果作为 eval span 写回同一个 ClickHouse，并链接到父 trace。

漂移检测随时间观察 embedding 空间的分布（对 prompt embedding 计算 PSI 或 KL 散度），外加评估分数趋势。告警送入 Prometheus Alertmanager，再到 Slack / PagerDuty。UI 是 Next.js 15 加 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- Collector：OpenTelemetry Collector，带尾部采样处理器（用于控制成本）
- 存储：ClickHouse 存 span，Postgres 存元数据，S3 存原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、自定义 LLM-judge
- 漂移：对池化的 prompt embedding（sentence-transformers）每周计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

```figure
ce-otel-drift
```

## 构建它

1. **Collector 配置。** OpenTelemetry Collector 配 OTLP HTTP receiver，一个尾部采样器保留 100% 的错误 trace 和 10% 的成功 trace，并配置到 ClickHouse 和 S3 的 exporter。

2. **ClickHouse schema。** 表 `spans`，列对应 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，再加一个 JSON 字段装长载荷。按 user_id 和 app_id 建二级索引。

3. **SDK 覆盖测试。** 写一个小客户端应用，分别使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM），并启用 OpenLLMetry 自动埋点。验证每一种都产出规范的 GenAI span 并落入 ClickHouse。

4. **评估任务。** 一个定时任务读取最近 15 分钟的采样 trace，跑 DeepEval 的忠实度、毒性和答案相关性。输出是 eval span，链接到父 trace。

5. **自定义 LLM-judge。** 一个 PII 泄漏判定器：给定一条回复，调用防护 LLM 给 PII 泄漏的可能性打分。高分回复进入分诊队列。

6. **漂移检测。** 每周任务计算本周池化 prompt embedding 与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，就告警。

7. **仪表盘。** Next.js 15，页面包括：总览（spans/秒、每用户成本、p95 延迟）、trace（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链路。** Prometheus exporter 读取评估分数聚合和延迟分位数；Alertmanager 把警告路由到 Slack，把严重违规路由到 PagerDuty。

9. **回归探针。** 注入一个 bug：受评估的聊天机器人开始以 1% 的概率泄漏伪造的 SSN。测量 MTTR：从 bug 部署到 Slack 告警。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付它

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表盘摄取它的 trace，跑评估，对漂移告警，并在 Next.js 里展示每用户成本拆分。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | trace schema 覆盖 | 产出规范 GenAI span 的 SDK 家族数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数对比人工标注集 |
| 20 | 仪表盘 UX | 注入回归上的 MTTR（目标 5 分钟以内） |
| 20 | 成本 / 规模 | 持续以 1k spans/秒 摄取且无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端跑通 |
| **100** | | |

## 练习

1. 为 Haystack 框架加自定义埋点。验证规范的 span 带着忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在同一批 trace 上把 DeepEval 换成 Phoenix evaluator。测量两个评估引擎之间的分数差异。

3. 让漂移检测更精细：按 app-id 分别计算 PSI，而不是全局计算。展示每个应用的漂移轨迹。

4. 加一个「用户影响」页面：每用户成本和每用户失败率，带迷你走势图。

5. 构建一个尾部采样策略：保留 100% 毒性 > 0.5 的 trace，再加上其余 trace 的 10% 分层样本。测量引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | 「OTel LLM 属性」 | 2025 年 OpenTelemetry 规范，定义 LLM span 属性（system、model、tokens） |
| Tail sampling | 「trace 后采样」 | Collector 在 trace 完成后决定保留或丢弃（可以偷看错误） |
| PSI | 「群体稳定性指数」 | 比较两个分布的漂移指标；> 0.2 通常表示显著漂移 |
| LLM-judge | 「用模型做评估」 | 一个 LLM 按评分标准给另一个 LLM 的输出打分（忠实度、毒性、PII） |
| Tail-sampling policy | 「保留规则」 | 决定哪些 trace 持久化、哪些丢弃的规则；错误优先 + 采样率 |
| Eval span | 「关联评估 trace」 | 携带评估分数的子 span，链接到原始 LLM 调用 span |
| Cost per user | 「单位经济」 | 一个窗口内归因到某个 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) —— open-core 可观测性平台的参考
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) —— 另一个参考，漂移支持很强
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) —— 自动埋点 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 摄取 schema
- [Helicone](https://www.helicone.ai) —— 另一种托管可观测性方案
- [Braintrust](https://www.braintrust.dev) —— 另一种评估优先的平台
- [ClickHouse 文档](https://clickhouse.com/docs) —— 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) —— evaluator 库
