# Prompt 缓存与语义缓存经济学

> **定价快照日期 2026-04。** 下文的数字结论反映的是本课发布时抓取的供应商价目表；在下游引用之前，请对照链接的文档核实。

> 缓存发生在两个层面。L2（供应商级）的 prompt/前缀缓存会为重复的前缀复用注意力的 KV——Anthropic 的 prompt caching 文档宣称，在长 prompt 上可降低最多 90% 的成本、85% 的延迟；对 Claude 3.5 Sonnet，缓存读取为 $0.30/M，未缓存为 $3.00/M，TTL 为 5 分钟，选择 1 小时 TTL 需支付 2 倍写入溢价（docs.anthropic.com，2026-04）。OpenAI 的 prompt caching 对 ≥1024 token 的 prompt 自动生效，缓存输入的价格约为未缓存的 90% 折扣（platform.openai.com，2026-04）；每个模型的确切缓存费率取决于实时价目表。L1（应用级）语义缓存则在 embedding 相似度命中时完全跳过 LLM。供应商所说的「95% 准确率」指的是匹配正确率，而不是命中率——已报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；两家供应商都没有发布官方基线，因此应把这些数字视为社区遥测数据，而非保证。生产环境中的陷阱：并行化会杀死缓存（在首次缓存写入完成之前发出的 N 个并行请求可能让支出膨胀数倍），而前缀中的动态内容会彻底阻止缓存命中。ProjectDiscovery 报告称，通过把动态文本移出可缓存前缀，命中率从 7% 提升到了 74%（2025-11）。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分 L2 prompt/前缀缓存（在供应商侧复用 KV）与 L1 语义缓存（对相似 prompt 绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记，以及两种 TTL 选项（5 分钟 vs 1 小时）及其价格倍数。
- 在给定命中率、prompt/响应构成和 token 价格的情况下，计算预期的月度节省。
- 说出让账单膨胀 5-10 倍的并行化反模式，以及让命中率崩塌的动态内容反模式。

## 问题

你给 RAG 服务加上了 prompt 缓存。账单纹丝不动。你测了命中率：7%。你的 prompt 看起来是静态的，其实不是——系统 prompt 里包含精确到分钟的当前时间、一个请求 ID，以及为增加多样性而随机重排的示例。每个请求都写入一条新的缓存条目，读取为零。

另外，你的智能体对每个用户问题运行十次并行工具调用。十次调用都在首次缓存写入完成之前到达供应商。十次写入，零次读取。你的账单是「启用缓存后」应有成本的 5-10 倍。

缓存是一种协议，不是一个开关。两个层面，两种不同的失效模式。

## 概念

### L2——供应商 prompt/前缀缓存

供应商为可缓存前缀保存注意力的 KV，并在下一个匹配该前缀的请求中复用它。你支付一次写入成本，读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：在请求中使用显式的 `cache_control` 标记。你标记哪些块可缓存。TTL：5 分钟（写入成本为基础价的 1.25 倍）或 1 小时（写入成本为基础价的 2 倍）。缓存读取：Claude 3.5 Sonnet 上为 $0.30/M，未缓存为 $3.00/M——便宜 10 倍（docs.anthropic.com，截至 2026-04）。费率因模型而异（Opus/Haiku 单独发布）；务必对照实时定价页面核实。

**OpenAI**：对 ≥1024 token 的 prompt 自动缓存（platform.openai.com，2026-04）。没有显式开关。在当前 gpt-4o/gpt-5 价目表上，缓存输入比未缓存输入便宜约 10 倍。文档和发布说明都没有公布官方命中率基线；社区报告在精心设计 prompt 的情况下集中在 30–60%。监控 `usage.cached_tokens` 来测量你自己的命中率。

**Google（Gemini）**：通过显式 API 做上下文缓存；100 万 token 的上下文意味着缓存收益更大。

**自托管（vLLM、SGLang）**：Phase 17 · 06 介绍了 RadixAttention——同样的模式，跑在你自己的算力上。

### L1——应用级语义缓存

在调用 LLM 之前，先对 prompt 做哈希、求 embedding，然后寻找相似的缓存请求（余弦相似度高于阈值，通常 0.95 以上）。命中则直接返回缓存响应。未命中则调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

供应商宣称的准确率，指的是返回的缓存响应在语义上是否恰当——而不是你的命中频率。生产环境中的命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 客服：40-70%。
- 代码问题：20-30%（细微的变体就会杀死命中）。
- 重复固定 prompt 的语音智能体：50-80%（语音归一化处理的是固定集合）。

### 并行化反模式

你的智能体并行发起 10 次工具调用。10 次调用都有相同的 4K token 系统 prompt。Anthropic 的缓存写入是按请求进行的；首次缓存写入大约在供应商看到 prompt 后 300 ms 完成。第 2-10 个请求在同一个毫秒窗口内到达，每一个都看到缓存未命中。你支付了 10 份写入溢价，0 份读取折扣。

修复方法：先顺序、再批量——只发第 1 个请求，等它的缓存填好后再发第 2-10 个。第一次工具调用多花 300 ms；账单省下 5-10 倍。

### 动态内容反模式

你的系统 prompt 长这样：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都独一无二。每个请求都写入。零命中。

修复方法：把所有真正静态的内容移到可缓存前缀，把动态内容追加到缓存边界之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 就这样把缓存命中率从 7% 提到了 74%，并公开了其中的结构剖析。

### 为夜间负载叠加 batch + 缓存

Batch API（Phase 17 · 15）提供 50% 折扣，周转时间 24 小时。在其之上再叠加缓存输入，还能再省约 10 倍。夜间分类、标注和报告生成类负载，通过叠加可降至同步未缓存成本的约 10%。

### 你应该记住的数字

以下价格数据于 2026-04 从链接的供应商文档抓取，每隔几个月就会变动——依赖之前请重新核实。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为未缓存输入的 1/10（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于 ≥1024 token 的 prompt；在当前价目表上，缓存输入定价约为未缓存输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 最高约 70%。并非供应商文档化的基线。
- ProjectDiscovery：把动态内容移出前缀后，命中率从 7% 提升到 74%（项目博客，2025-11）。
- 并行化反模式：当 N 个并行请求都错过首次缓存写入时，典型报告显示账单膨胀 5–10 倍。

```figure
semantic-cache-hit
```

## 使用它

`code/main.py` 在混合负载上模拟 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## 交付它

本课产出 `outputs/skill-cache-auditor.md`。给定 prompt 模板和流量，审计其可缓存性并建议重构。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统 prompt 里有一个日期。把它移出去。展示前后命中率的计算。
3. 给定你的请求到达率，计算 1 小时 TTL（2 倍写入）与 5 分钟 TTL（1.25 倍写入）的盈亏平衡点。
4. 语义缓存在 0.95 阈值下命中 20%。降到 0.85 时命中 50%，但你会看到错误的缓存响应。选出正确的阈值并说明理由。
5. 你对每个用户问题批量发起 10 个并行子查询。在不增加端到端延迟的前提下，改写成对缓存友好的形式。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| L2 prompt 缓存 | 「prefix cache」 | 供应商为重复前缀保存 KV |
| `cache_control` | 「Anthropic 缓存标记」 | 显式标记可缓存块的属性 |
| 缓存写入溢价 | 「写入税」 | 首次未命中转缓存时的额外成本（1.25 倍或 2 倍） |
| L1 语义缓存 | 「embedding 缓存」 | 调用 LLM 前在应用层做哈希与 embedding |
| GPTCache | 「LLM 缓存库」 | 流行的开源 L1 缓存库 |
| 缓存命中率 | 「命中数 / 总数」 | 由缓存直接服务的请求比例 |
| 并行化反模式 | 「N 次写入陷阱」 | N 个并行请求错过缓存 N 次 |
| 动态内容陷阱 | 「prompt 里塞时间陷阱」 | 前缀中的动态字节杀死命中率 |
| RadixAttention | 「副本内缓存」 | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——官方的 `cache_control` 语义与 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)——自动缓存行为与适用条件。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
