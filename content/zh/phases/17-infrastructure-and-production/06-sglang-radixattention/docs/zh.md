# 前缀缓存服务——RadixAttention 与 KV 复用

> 把 KV cache 当作一等、可复用的资源，存放在基数树（radix tree）中，调度也随之改变：不再像 vLLM 那样按 FCFS（先到先服务）调度，缓存感知调度器优先处理共享前缀更长的请求——实际上就是深度优先的基数树遍历，让热分支常驻 HBM。SGLang 就是围绕这一理念构建服务的引擎。在类似 ShareGPT 的 1K prompt 上，Llama 3.1 8B 的 SGLang 达到约 16,200 tok/s，而 vLLM 约为 12,500，领先约 29%。在前缀密集的 RAG 工作负载上，优势可达 6.4 倍。在语音克隆形态的工作负载上，缓存命中率超过 86%。2026 年部署在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 的 40 万+ 块 GPU 上。要注意的是，当前缀顺序不一致时，6.4 倍这个数字会蒸发——顺序是工程师手里的杠杆。

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## 学习目标

- 画出 RadixAttention：前缀如何存储在基数树中，KV block 如何在以同一分支为根的序列之间共享。
- 解释缓存感知调度，以及为什么 FCFS 对前缀密集的流量是错误的。
- 给定前缀缓存命中率和 prompt 长度分布，计算工作负载的预期加速比。
- 说出能让 6.4 倍这个数字成真、而不是只留作潜在收益的 prompt 排序纪律。

## 问题

经典服务把每个请求的 prompt 当作不透明的。即便 5,000 个 RAG 请求都以同样的 2,000 token system prompt 加同样的检索前导开头，vLLM 仍会把那 2,000 token 的前缀预填充 5,000 次。GPU 一遍又一遍地做同样的工作。

观察到的现象是：智能体和 RAG 工作负载中的 prompt 几乎总是共享长前缀。system prompt、tool schema、few-shot 示例、检索头、对话历史——全都在请求之间重复。如果你把那部分前缀的 KV cache 存一次并复用，就不必再次预填充它。

RadixAttention 正是这么做的。token 被索引进一棵基数树；每个节点拥有从其根路径上 token 序列对应的 KV block。新请求遍历这棵树：任何 token 匹配的节点都复用该节点的 KV block。预填充成本变成与「新增」后缀成正比，而不是与完整 prompt 成正比。

挑战在调度。如果两个请求共享 2,000 token 前缀，而第三个请求只共享该前缀中的 200 个 token，你希望把两个长共享请求放在一起服务，让长前缀留在 HBM 中。FCFS 恰恰相反——它服务先到者，可能在下一个长前缀请求到来之前就把热分支驱逐了。

## 概念

### 基数树作为 KV 索引

基数树（紧凑 trie）存储 token 序列。每个节点拥有一个 token 区间以及为这个区间计算出的 KV block。子节点把序列扩展一个或多个 token。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

新请求到来，内容是 system prompt + 「Context: <doc A>」+ 「Question: Carol」。调度器遍历：system 前缀匹配（复用 124 个 block），doc-A 分支匹配（复用 31 个 block），然后只为「Question: Carol」分配新 block（4 个 block）。预填充成本：4 个 block 的新 token。没有这棵树：160 个 block。预填充约省 40 倍。

### 缓存感知调度

如果缓存不停抖动，基数树支撑的复用毫无意义。两条关键策略：

1. **深度优先派发**。从队列中挑选下一个请求时，优先选与当前运行集合根植于同一分支的请求。这样热分支就能保持常驻。
2. **按分支而非按 block 做 LRU**。整条分支地驱逐（从最少使用的叶节点开始），而不是单个 block，使缓存形状与基数形状一致。

FCFS 同时违反这两条。一个共享 2,000 token 的请求排在一个只共享 50 个 token 的请求之后，接着 2,000 token 的分支为了接纳 50 token 的请求被驱逐。

### 应该背下来的基准数字

- Llama 3.1 8B，H100，ShareGPT 1K prompt：SGLang 约 16,200 tok/s，vLLM 约 12,500（领先约 29%）。
- 前缀密集 RAG（相同 system + 相同文档，问题不同）：SGLang 上最高 6.4 倍。
- 语音克隆工作负载：86.4% 的前缀缓存命中率。
- SGLang 客户的生产命中率：依 prompt 纪律在 50-99% 之间。
- 2026 年部署在 40 万+ 块 GPU 上。

### 顺序陷阱

6.4 倍这个数字依赖一致的 prompt 模板顺序。如果你的客户端在某些请求中把 prompt 构造成 `[system, tools, context, history, question]`，在另一些请求中构造成 `[system, context, tools, history, question]`，树就找不到共享前缀。人眼里看起来相同的前缀，对基数树来说是两个不同的序列。

工程师的杠杆：你的 prompt 模板就是缓存键。把顺序固定下来。把一切不可变内容（system、tools、schema）放最前，接着放检索上下文，用户问题放最后。不要把动态内容插入前缀中间。

研究中的一个真实案例：把动态内容移出可缓存前缀，只用一次改动就把某次部署的缓存命中率从 7% 提升到 74%。

### RadixAttention 在哪里赢、在哪里输

赢：
- RAG（相同检索前导，问题不同）。
- 智能体（相同 tool schema，查询不同）。
- 带长 system prompt 的聊天。
- 有重复前导的语音/视觉工作负载。

输（回落到 vLLM 级吞吐量）：
- 一次性生成且 prompt 唯一（代码补全、没有 system prompt 的开放式聊天）。
- 每个请求都把唯一内容插入前缀的动态 prompt。

### 为什么这是调度器问题，而不只是 kernel 问题

你可以把 KV 复用实现为一个 kernel 技巧。SGLang 的洞见是：只有当调度器让热分支常驻时，复用才划算。朴素的「有就复用」策略在混合负载下会让缓存抖动。基数树索引的调度器，才是把 kernel 技巧变成 29% 生产优势的关键。

### 与 vLLM 的关系

两个系统并非严格意义上的竞争对手。2026 年，vLLM 加入了前缀缓存（`--enable-prefix-caching`）和一个缓存感知路由（用 Rust 写的 vLLM Router）。差距缩小了，但没有完全消失——SGLang 的整个栈是基数树优先；vLLM 是后来嫁接上去的。对前缀复用占主导的工作负载，SGLang 仍是默认。对没有强前缀模式的通用服务，vLLM 仍然相当或更好。

```figure
roofline
```

## 使用它

`code/main.py` 实现了一个玩具式基数树 KV cache，以及一个带两种策略的调度器：FCFS 和缓存感知。它在同一工作负载上跑两者，报告前缀缓存命中率和吞吐量差。然后跑一个「乱序」工作负载，展示 6.4 倍的崩塌。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?workload=shared&shared=4096) 这台机器展示了真实 L4 运行上的基数缓存。四组各有自己的 4,096 token system prompt。某组的第一个请求计算它的 prompt。当该 prompt 留在缓存中时，该组后续请求只计算 128 token 的问题。把基数缓存关掉，可以看到首 token 中位数从 479 ms 变成 14,706 ms。

## 交付它

本节课产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（prompt 模板形态、检索模式、并发租户数），它会产出一份 prompt 排序处方，以及是否采用 SGLang 的 go/no-go 判断。

## 练习

1. 运行 `code/main.py`。在同一工作负载上比较 FCFS 与缓存感知。差异从何而来——预填充节省、解码节省，还是排队延迟？
2. 修改工作负载，让 prompt 随机置换 `[system, tools, context]`。重跑。命中率会发生什么？为什么？
3. 计算在 Llama 3.1 8B 上把 2,000 token 的 system prompt 作为一个基数分支常驻的 HBM 成本。与没有前缀复用的 16 序列 batch 的成本比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么在前缀密集负载下，树形 LRU 驱逐胜过 block 形 LRU。
5. 一位客户报告缓存命中率只有 8%。说出三个可能原因，以及你为每个原因会跑的诊断。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| RadixAttention | 「SGLang 那一套」 | 把 KV cache 索引为基数树，让共享前缀复用 block |
| Radix tree | 「紧凑 trie」 | 每个节点拥有一个 token 区间及其 KV block 的树 |
| Cache-aware scheduler | 「热分支优先」 | 优先选择共享常驻分支的请求的调度器 |
| Prefix-cache hit rate | 「prompt 有多少是免费的」 | 从复用 KV block 中提供的 prompt token 比例 |
| FCFS | 「先到先服务」 | 破坏前缀局部性的默认调度 |
| Branch-level LRU | 「驱逐叶子」 | 与基数形状匹配的驱逐策略 |
| Prompt template ordering | 「缓存键」 | prompt 各组成部分的顺序决定了树能共享什么 |
| System prompt pinning | 「常驻前缀」 | 让不可变的 system 部分保持常驻，避免驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang)——源码与文档。
- [SGLang 文档](https://sgl-project.github.io/)——RadixAttention 与调度细节。
- [SGLang 论文——高效编程大型语言模型（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104)——设计参考。
- [LMSYS 博客——带 RadixAttention 的 SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/)——基准数字与调度器理由。
- [vLLM——前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)——vLLM 自己的类基数树实现，供比较。
