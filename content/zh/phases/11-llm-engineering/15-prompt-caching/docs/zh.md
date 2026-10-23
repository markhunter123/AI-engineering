# Prompt 缓存与上下文缓存

> 你的 system prompt 有 4,000 个 token。你的 RAG 上下文有 20,000 个 token。每次请求你都会把它们一起发出去，也每次都为它们付全价。Prompt caching 让提供商把这段前缀留在自己一侧保持温热，复用时只按正常费率的 10% 计费。用对了，它能把推理成本降低 50–90%，把首 token 延迟降低 40–85%。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## 问题

一个编码智能体在对话的每一轮都把同一段 15,000 token 的 system prompt 发给 Claude。二十轮，按每百万输入 token 3 美元计算，光输入成本就是 0.90 美元——这还没算用户实际发送的任何消息。乘以每天 10,000 次对话，这些从不改变的文字每天要烧掉 9,000 美元。

你没法在不损害质量的前提下缩短 prompt，也没法不发它——模型每一轮都需要它。唯一的出路，是不要再为提供商已经见过的前缀付全价。

这条出路就是 prompt caching。Anthropic 在 2024 年 8 月推出它（2025 年增加了 1 小时延长 TTL 版本），OpenAI 在那年晚些时候实现了自动缓存，Google 随 Gemini 1.5 推出了显式上下文缓存，如今三家都在各自的前沿模型上把它作为一等特性提供。

## 概念

![Prompt 缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近某次请求的前缀匹配时，提供商直接复用上一轮运行留下的 KV cache，而不是重新编码这些 token。第一次你支付一笔较小的写入溢价，之后每次读取都享受大幅折扣。

**2026 年的三种提供商方案。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式标记 `cache_control` | 输入费用减免 90% | 加价 25% | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用减免 50% | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取约为正常的 25% | 按 token·小时收存储费 | 用户设定（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变式。** 三家都只缓存前缀。只要请求之间有一个 token 不同，第一个不同 token 之后的内容全部未命中。把**稳定**的部分放在最上面，把**可变**的部分放在最下面。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

打乱这个顺序——把用户消息放到 system prompt 上面、在 few-shot 之间穿插动态检索结果——缓存就永远不会命中。

### 盈亏平衡计算

Anthropic 25% 的写入溢价意味着，一个缓存块至少要读取两次才能在净额上省钱。一次写入加一次读取，平均每次请求成本为 0.675x（节省 32%）；一次写入加十次读取，平均为 0.205x（节省 80%）。经验法则：只要预期在 TTL 内至少复用 3 次，就值得缓存。

```figure
prompt-cache-hit
```

## 构建它

### 步骤 1：使用显式标记的 Anthropic prompt 缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 把这个块保存 5 分钟。窗口内复用会命中；窗口外复用则过期并重新写入。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查这两个字段——如果多次请求之间 `cache_read_input_tokens` 一直是零，说明你的缓存键在漂移。

### 步骤 2：一小时延长 TTL

对于长时间运行的批处理任务，5 分钟的默认值会在两个任务之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2 倍（相对基线加价 50%，而非 25%），但只要某个批次在 TTL 内复用前缀超过 5 次，就很快回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 没有任何东西需要你配置。任何超过 1,024 token 且与近期请求匹配的前缀，都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

缓存友好的布局规则同样适用。有两件事会破坏 OpenAI 的缓存，却不影响 Anthropic：改变 `user` 字段（它被用作缓存键的一部分），以及重排工具顺序。

### 步骤 4：Gemini 显式上下文缓存

Gemini 把缓存当作一等对象，由你创建并命名：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时收取存储费，只要缓存存在就一直计费；读取约为正常输入费率的 25%。当你需要在多天内、跨大量会话复用同一段超长 prompt 时，这是正确的形态。

### 步骤 5：在生产中测量命中率

参见 `code/main.py`，其中有一个模拟的三提供商记账器，追踪 write/read/miss 次数，并计算每 1K 请求的综合成本。把目标命中率作为部署门禁——大多数生产环境的 Anthropic 配置在预热后应达到 80% 以上的读取比例。

## 2026 年仍在踩的坑

- **顶部放动态时间戳。** system prompt 开头写着 `"Current time: 2026-04-22 15:30:02"`，每次请求都会未命中。把时间戳移到缓存断点下方。
- **工具重排。** 以稳定的顺序序列化工具——部署之间一次 dict 重排就会破坏所有命中。
- **自由文本的近重复。** "You are helpful." 与 "You are a helpful assistant."——一个字节之差就是完全未命中。
- **块太小。** Anthropic 强制执行 1,024 token 的下限（Haiku 为 2,048）。更小的块会静默地不被缓存。
- **盲目的成本仪表盘。** 把「输入 token」拆成已缓存和未缓存两部分。否则流量下降看起来会像缓存收益。

## 使用它

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 智能体，稳定的 10k+ system prompt，多轮对话 | Anthropic `cache_control`，5 分钟 TTL |
| 批处理任务，前缀复用 30 分钟以上 | Anthropic，`ttl: "1h"` |
| GPT-5 上的无服务器端点，无自建基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 多天复用大型代码/文档语料 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 让可缓存前缀的布局在各提供商之间保持一致，这样任何一处命中都有效 |

与语义缓存（Phase 11 · 11）结合，覆盖用户消息层：prompt 缓存处理 **token 完全一致**的复用，语义缓存处理**语义相同**的复用。

## 交付它

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 针对 Claude 跑一段 10 轮对话，system prompt 为 5,000 token。先不用 `cache_control` 跑一遍，再启用跑一遍。报告各自的输入 token 账单。
2. **中等。** 编写一个测试框架：给定 prompt 模板和请求日志，计算每种提供商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）下的预期命中率和美元节省额。
3. **困难。** 构建一个布局优化器：给定一个 prompt 和一组标记为 `stable=True/False` 的字段，重写 prompt，在不丢失信息的前提下把唯一的缓存断点放在最利于缓存的位置。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | 「让长 prompt 变便宜」 | 对匹配前缀复用提供商侧的 KV cache；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | 「Anthropic 的标记」 | 内容块属性，声明「直到这里的内容都可缓存」；`{"type": "ephemeral"}`。 |
| Cache write | 「支付溢价」 | 第一次填充缓存的请求；Anthropic 按约 1.25x 输入费率计费，OpenAI 免费。 |
| Cache read | 「折扣」 | 后续匹配该前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | 「缓存能活多久」 | 缓存保持温热的秒数；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力维持最长 1 小时，Gemini 由用户设定。 |
| Extended TTL | 「Anthropic 的 1 小时缓存」 | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价 2 倍，但批处理复用时值得。 |
| Prefix match | 「我的缓存为什么没命中」 | 只有当从开头到断点的每一个 token 都字节级一致时，缓存才会命中。 |
| Context caching (Gemini) | 「显式的那种」 | Google 的具名、按存储计费的缓存对象；最适合大型语料的多天复用。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) —— 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) —— `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) —— 原始发布文章，含延迟数据。
- Phase 11 · 05（Context Engineering）—— 在哪里切分 prompt，才能让缓存落地。
- Phase 11 · 11（Caching and Cost）—— 把 prompt 缓存与用户消息上的语义缓存配合使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) —— prompt 缓存暴露给用户的 KV-cache 内存模型；解释了为什么缓存前缀的重新读取成本约为重新计算的 1/10。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) —— prefill 正是 prompt 缓存所绕过的阶段；这篇论文解释了为什么命中缓存时 TTFT 大幅下降，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) —— prompt 缓存与推测解码、Flash Attention、MQA/GQA 一样，都是弯折推理成本曲线的杠杆；读这篇可以了解另外三个。
