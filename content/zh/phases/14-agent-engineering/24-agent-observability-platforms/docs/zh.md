# Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年，三个开源 Agent 可观测性平台占据主导。Langfuse（MIT）——每月 600 万+ 次安装，追踪 + prompt 管理 + 评估 + 会话重放。Arize Phoenix（Elastic 2.0）——深度 Agent 特定评估、RAG 相关性、OpenInference 自动 instrumentation。Comet Opik（Apache 2.0）——自动 prompt 优化、防护栏、LLM-judge 幻觉检测。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## 学习目标

- 说出三个顶级开源 Agent 可观测性平台及其许可证。
- 区分每个平台最强的部分：Langfuse（prompt 管理 + 会话）、Phoenix（RAG + 自动 instrumentation）、Opik（优化 + 防护栏）。
- 解释为什么到 2026 年有 89% 的组织报告已部署 Agent 可观测性。
- 实现一个标准库的 trace 到仪表盘 pipeline，带 LLM-judge 评估。

## 问题

OTel GenAI（第 23 课）给了你 schema。你仍然需要一个平台来摄取 span、运行评估、存储 prompt 版本并暴露回归。三个竞争者各自强调生命周期中的不同部分。

## 概念

### Langfuse（MIT）

- 每月 600 万+ SDK 安装，19k+ GitHub star。
- 功能：追踪、带版本管理和 playground 的 prompt 管理、评估（LLM-as-judge、用户反馈、自定义）、会话重放。
- 2025 年 6 月：原商业模块（LLM-as-a-judge、标注队列、prompt 实验、Playground）以 MIT 开源。
- 最擅长：端到端可观测性，配合紧密的 prompt 管理闭环。

### Arize Phoenix（Elastic License 2.0）

- 更深的 Agent 特定评估：trace 聚类、异常检测、RAG 的检索相关性。
- 原生 OpenInference 自动 instrumentation。
- 与托管版 Arize AX 搭配用于生产。
- 没有 prompt 版本管理——定位为漂移/行为回归工具，与更广泛的平台并列。
- 最擅长：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验进行自动 prompt 优化。
- 防护栏（PII 脱敏、主题约束）。
- LLM-judge 幻觉检测。
- Comet 自己的测量基准：Opik 的日志 + 评估耗时 23.44s，Langfuse 为 327.15s（约 14 倍差距）——厂商基准只作方向性参考。
- 最擅长：优化循环、自动化实验、防护栏执行。

### 行业数据

据 Maxim（2026 年现场分析）：89% 的组织已部署 Agent 可观测性；质量问题是最主要的生产障碍（32% 的受访者提到）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 带 prompt 管理的一体化方案 | Langfuse |
| 深度 RAG 评估 + 漂移 | Phoenix |
| 自动化优化 + 防护栏 | Opik |
| 开放许可，不要 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 都可以——它们都导出 OTel |

### 这种模式容易出错的地方

- **没有评估策略。** 只追踪不评估，只是昂贵的日志。
- **自研 LLM-judge 缺乏依据。** CRITIC 模式（第 05 课）适用——judge 需要外部工具做事实核查。
- **prompt 版本没有与 trace 绑定。** 生产回归时，你无法二分定位到导致问题的 prompt。

```figure
wb-trace-ingest
```

## 构建它

`code/main.py` 实现一个标准库 trace collector + LLM-judge 评估器：

- 摄取 GenAI 形状的 span。
- 按会话分组，标记失败的运行（防护栏触发、低置信度评估）。
- 一个脚本化的 LLM-judge，按评分标准给 Agent 回复打分。
- 类似仪表盘的摘要：失败率、主要失败原因、评估分数分布。

运行它：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 会展示的内容一致。

## 使用它

- **Langfuse** 自托管或云版；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；自动 instrumentation OpenInference。
- **Comet Opik** 自托管或云版；自动化优化循环。
- **Datadog LLM Observability** 适合同时运行 Datadog 的运维 + ML 混合团队。

## 交付它

`outputs/skill-obs-platform-wiring.md` 选择一个平台，把 trace + 评估 + prompt 版本接入现有 Agent。

## 练习

1. 把一周的 OTel trace 导出到 Langfuse 云版（免费层）。哪些会话失败了？为什么？
2. 为你的领域写一份 LLM-judge 评分标准（事实正确性、语气、范围遵循）。在 50 条 trace 上测试。
3. 比较 Langfuse 的 prompt 版本管理和 Phoenix 的 trace 聚类。哪一个更快告诉你哪里坏了？
4. 阅读 Opik 的防护栏文档。给你的一个 Agent 运行接入 PII 脱敏防护栏。
5. 在你的语料上对三者做基准测试。忽略厂商发布的数字；测量你自己的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 追踪 | 「span 收集器」 | 摄取 OTel / SDK span；按会话索引 |
| Prompt 管理 | 「prompt CMS」 | 与 trace 绑定的版本化 prompt |
| LLM-as-judge | 「自动评估」 | 另一个 LLM 按评分标准给 Agent 输出打分 |
| 会话重放 | 「trace 回放」 | 逐步回放历史运行以调试 |
| RAG 相关性 | 「检索质量」 | 检索到的上下文是否与查询匹配 |
| Trace 聚类 | 「行为分组」 | 对相似运行聚类以检测漂移 |
| 防护栏执行 | 「日志时策略」 | 对记录内容做 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、prompt 管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动 instrumentation、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 防护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者共同消费的 schema
