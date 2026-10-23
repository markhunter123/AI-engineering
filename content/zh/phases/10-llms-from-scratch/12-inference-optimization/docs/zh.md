# 推理优化

> 两个阶段定义了 LLM 推理。Prefill 并行处理你的 prompt——计算受限。Decode 一次生成一个 token——内存受限。每一种优化都瞄准其中一个阶段，或两者兼顾。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-08 (Transformer architecture, attention)
**Time:** ~120 minutes

## 学习目标

- 实现 KV cache，消除自回归 token 生成过程中的冗余计算
- 解释 LLM 推理的 prefill 与 decode 两个阶段，以及为什么两者的瓶颈不同（计算受限 vs 内存受限）
- 实现 continuous batching 与 PagedAttention 的核心概念，在并发请求下最大化 GPU 利用率
- 比较各类推理优化技术（KV cache、投机解码、flash attention）及其吞吐量/延迟权衡

## 问题

你在 4×A100 GPU 上部署 Llama 3 70B。单个用户大约能拿到每秒 50 个 token，感觉很快。然后 100 个用户同时打到这个端点，吞吐量跌到每用户每秒 3 个 token。你每月 2.5 万美元的 GPU 账单，服务的速度比人打字还慢。

从 1 个用户到 100 个用户，模型本身没有任何变化：同样的权重、同样的架构、同样的数学。变化的是你如何调度这些工作。朴素推理会浪费 90% 以上的可用 GPU 算力。一个等待第 47 个 token 的用户，占着一个完整的 batch 槽位，而 GPU 的内存总线在两次 matmul 之间空转。与此同时，一个新用户的 2,000 token prompt 本可以用这些空闲时间做有用的计算。

这不是扩展性问题，而是调度问题。本节课讲的技术——KV cache、continuous batching、PagedAttention、投机解码、前缀缓存——正是一个每月 2.5 万美元的推理账单和同样流量下每月 5 千美元的账单之间的差别。

vLLM 在 4×A100-80GB 上服务 Llama 3 70B，低并发时约每用户每秒 50 token；在 100 个并发请求下，借助 continuous batching 和 PagedAttention，仍能维持每用户每秒 15-25 TPS。没有这些优化，同样的硬件在该并发下只能提供每用户 5 TPS。同样的 GPU，同样的模型，4 倍的吞吐量。

## 概念

### Prefill 与 Decode

每一次 LLM 推理请求都包含两个截然不同的阶段。

**Prefill** 处理整个输入 prompt。所有 token 都是已知的，因此注意力可以在整个序列上并行计算。这是一次大规模的矩阵乘法——GPU 核心保持忙碌。瓶颈是计算：你的硬件每秒能交付多少 FLOPS。一块 A100 的算力是 312 TFLOPS（BF16）。在单块 A100 上，70B 模型处理 4,096 token 的 prompt，prefill 耗时约 400ms。

**Decode** 一次生成一个输出 token。每个新 token 都要关注此前所有的 token，但每次前向传播只产出一个 token。权重矩阵的大小和 prefill 时相同，但你是在用单个向量而不是矩阵去乘它们。GPU 核心几微秒就算完了，然后等待下一批权重从内存运来。瓶颈是内存带宽：你从 HBM 向计算单元流式传输模型权重的速度。一块 A100 的带宽是 2 TB/s。FP16 的 70B 模型有 140 GB，完整读一遍模型需要 70ms——这就是单步 decode 的下限。

```mermaid
graph LR
    subgraph "Prefill (compute-bound)"
        P1["All prompt tokens"] --> P2["Parallel attention"]
        P2 --> P3["Full matmul utilization"]
    end

    subgraph "Decode (memory-bound)"
        D1["One token at a time"] --> D2["Sequential generation"]
        D2 --> D3["Waiting on memory reads"]
    end

    P3 --> D1
```

**ops:byte 比值**（也叫算术强度，arithmetic intensity）刻画了这种权衡。它衡量你每从内存读入一个字节，执行了多少次运算。

```
ops:byte ratio = FLOPs per token / bytes read from memory
```

Prefill 时 batch 为 4,096 个 token，每加载一个权重你会执行约 4,096 次乘加运算。比值很高——你受计算约束。Decode 时 batch size 为 1，每加载一个权重你只执行约 1 次运算。比值很低——你受内存约束。

根本洞见：*decode 之所以内存受限，是因为为了产出一个 token，你要读取整个模型*。下面的每一种优化，要么减少读取量，要么增加每次读取处理的 token 批量，要么完全避免读取。

### KV Cache

在注意力计算中，每个 token 的 query 都要关注之前所有 token 的 key 和 value 向量。如果不做缓存，生成第 N 个 token 就需要重新计算前 N-1 个 token 的 key 和 value 投影。token 1 在生成 token 2 时被投影一次，生成 token 3 时再投影一次，生成 token 4 时又来一次。到第 1,000 个 token 时，token 1 已经被投影了 999 次。

KV cache 保存此前所有 token 的 key 和 value 投影。生成第 N 个 token 时，你只需计算该 token 的 key 和 value，再把它们与缓存的、来自 token 1 到 N-1 的 K/V 拼接起来。

```mermaid
graph TD
    subgraph "Without KV Cache"
        A1["Token 5: recompute K,V for tokens 1-4"]
        A2["Token 6: recompute K,V for tokens 1-5"]
        A3["Token 7: recompute K,V for tokens 1-6"]
    end

    subgraph "With KV Cache"
        B1["Token 5: compute K5,V5, read K1-4,V1-4 from cache"]
        B2["Token 6: compute K6,V6, read K1-5,V1-5 from cache"]
        B3["Token 7: compute K7,V7, read K1-6,V1-6 from cache"]
    end
```

**KV cache 的内存公式：**

```
KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param
```

以 Llama 3 70B 为例（80 层、GQA 下的 8 个 KV head、head_dim=128、BF16）：

```
per token: 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB
at 4,096 tokens: 320 KB * 4,096 = 1.28 GB
at 128K tokens: 320 KB * 131,072 = 40 GB
```

Llama 3 70B 单次 128K 上下文的对话要占用 40 GB 的 KV cache——半块 A100 的显存。如果 100 个并发用户各用 4K token，仅 KV cache 就需要 128 GB。这就是为什么 KV cache 管理是推理优化的核心挑战。

### Continuous Batching

静态批处理（static batching）会等到凑齐 N 个请求，把它们一起处理，并且在*所有*请求完成之前不再接收新请求。如果一个请求需要 500 个 token，另一个只需 10 个，那么短请求完成后还要空等 490 个 decode 步。

Continuous batching（也叫 iteration-level batching，迭代级批处理）在任何请求完成的瞬间，就把新请求插入到 batch 中。每个 decode 步都会重新评估 batch。一个 10 个 token 就结束的请求，会立刻被一个等待中的请求替换。

```mermaid
sequenceDiagram
    participant GPU
    participant R1 as Request 1 (50 tokens)
    participant R2 as Request 2 (10 tokens)
    participant R3 as Request 3 (30 tokens)
    participant R4 as Request 4 (waiting)

    Note over GPU: Static batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    Note over R2: Wasting 40 steps...
    Note over R3: R3 done at step 30
    Note over R3: Wasting 20 steps...
    GPU->>R4: Finally start R4 at step 50

    Note over GPU: Continuous batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    GPU->>R4: Insert R4 at step 11
    Note over R3: R3 done at step 30
```

吞吐量的提升取决于输出长度的差异有多大。长度齐平时，continuous batching 和静态批处理打平。长度参差时（这是常态），continuous batching 能带来 2-5 倍的吞吐量，因为 GPU 的槽位从不会空着。

### PagedAttention

每个请求的 KV cache 是一块连续内存。随着请求的到来与离开，内存会产生碎片——就像操作系统里的 RAM 碎片一样。一个 4K token 的请求需要 1.28 GB 连续空间。即使你总共还有 2 GB 空闲，也可能凑不出一块 1.28 GB 的*连续*空间。于是你要么浪费内存，要么拒绝请求。

PagedAttention（来自 vLLM）把操作系统式的虚拟内存引入 KV cache。它不为每个请求分配一整块连续内存，而是分配固定大小的「页」（通常每页 16 个 token）。页可以位于物理 GPU 内存的任何位置。页表把每个请求的逻辑序列位置映射到物理页位置。

```mermaid
graph TD
    subgraph "Contiguous allocation"
        C1["Request A: 2GB block"]
        C2["[free: 0.5GB]"]
        C3["Request B: 1GB block"]
        C4["[free: 1.5GB -- but fragmented]"]
    end

    subgraph "PagedAttention"
        P1["Page pool: 256 pages of 16 tokens each"]
        P2["Request A: pages 3,7,12,45,88..."]
        P3["Request B: pages 1,4,9,22,67..."]
        P4["No fragmentation, no waste"]
    end
```

PagedAttention 还让共享前缀可以使用**写时复制**（copy-on-write）。如果 50 个请求共享同一段 system prompt，这段 system prompt 对应的 KV cache 页只存一份，被 50 个请求共同引用。只有当某个请求开始分叉（用户消息不同）时，它才获得自己的页。对于带共享 system prompt 的应用，这能大幅削减内存占用。

vLLM 报告称，借助 PagedAttention，内存浪费接近零（约 4%，而朴素分配是 60-80%）。

### 投机解码

Decode 慢是因为它是串行的——生成一个 token，喂回去，再生成下一个。但如果你能廉价地猜出接下来的 5 个 token，然后一次性验证它们呢？

投机解码（speculative decoding）用一个小而快的**草稿模型**（draft model）生成 K 个候选 token。大**目标模型**（target model）在一次前向传播中处理全部 K 个候选（这次前向很像 prefill——并行、计算受限、高效）。如果目标模型认可草稿模型的预测，你就在一次目标前向传播的时间里接受了全部 K 个 token。如果在第 j 个位置出现分歧，你接受第 1 到 j-1 个 token，丢弃其余部分。

```mermaid
graph LR
    D["Draft model (1B)"] -->|"Generate 5 tokens<br/>~5ms"| C["Candidates: the cat sat on the"]
    C --> T["Target model (70B)"]
    T -->|"Verify all 5 in one pass<br/>~70ms"| V{"Match?"}
    V -->|"4 of 5 match"| A["Accept 4 tokens in 75ms<br/>vs 280ms sequential"]
    V -->|"Mismatch at pos 5"| R["Reject token 5<br/>Resample from target"]
```

加速幅度取决于**接受率**（acceptance rate）——草稿模型的预测与目标模型一致的比例。用 Llama 3 8B 给 Llama 3 70B 做草稿，自然语言上的接受率通常在 70-85%。折算下来是 2-3 倍的 decode 加速。

投机解码的三种方案：

| 方法 | 草稿来源 | 接受率 | 额外开销 |
|--------|-------------|-----------------|----------|
| Draft-target（Leviathan 等） | 独立的小模型 | 70-85% | 草稿模型的内存 |
| EAGLE（Li 等） | 目标模型上的轻量头 | 75-90% | 约 1% 的额外参数 |
| N-gram 查表 | token n-gram 表 | 40-60% | 可忽略 |

**EAGLE** 在目标模型的隐藏状态之上训练一个小的自回归头。它用目标模型倒数第二层的特征预测下一个 token 的 embedding。由于它作用于目标模型自身的表示（而不是另一个模型的表示），能以极小的额外内存获得更高的接受率。EAGLE-2 增加了动态草稿树，根据上下文调整候选数量。

**N-gram 投机解码**维护一张 n-gram 续写表，来源可以是当前上下文或预先构建的语料。如果草稿与同一段对话中出现过的内容吻合（重复模式、代码、结构化输出），它就能以零神经网络开销触发。平均接受率较低，但每次投机的成本几乎为零。

投机解码在*数学上是精确的*——输出分布与目标模型的分布完全相同。它不是近似。验证步骤保证每一个被接受的 token 都恰好具有目标模型会赋予的概率。

### 前缀缓存

许多请求共享相同的前缀：聊天机器人的 system prompt、一段 RAG 上下文、一组 few-shot 示例。没有前缀缓存，每个请求都要从头重算这些共享 token 的 KV cache。

前缀缓存（prefix caching）保存常见前缀的 KV cache，并在请求之间复用。当一个带有已知前缀的新请求到来时，系统复制（或引用）缓存的 KV 条目，只计算独有后缀部分的 KV。

对于一段被所有请求共享的 2,000 token system prompt，前缀缓存能省去每个请求约 400ms 的 prefill。按每秒 100 个请求算，每秒就省下 40 秒的 GPU 计算——超过一块 GPU 的算力。

SGLang 的 RadixAttention 用一棵基数树（radix tree，即 trie）实现前缀缓存，按 token 内容索引前缀。任何命中已存前缀的请求都能免费获得对应的 KV cache。这棵树支持部分前缀匹配——如果你与缓存条目共享 2,000 个前缀 token 中的 1,500 个，就复用这 1,500 个，只重算剩下的 500 个。

### 推理引擎

生产环境中的 LLM 服务由三个引擎主导：

| 引擎 | 核心创新 | 适用场景 |
|--------|---------------|----------|
| vLLM | PagedAttention、continuous batching | 通用服务，兼容性最好 |
| SGLang | RadixAttention（前缀缓存）、结构化生成 | 多轮聊天机器人、约束解码 |
| TensorRT-LLM | NVIDIA kernel 融合、FP8 量化 | NVIDIA 硬件上的单 GPU 最大吞吐量 |

**vLLM** 是默认的起点。它支持的模型范围最广，可在任何厂商的 GPU 上运行（NVIDIA、AMD、Intel），并借助 PagedAttention + continuous batching 取得很强的吞吐量。它提供 OpenAI 兼容 API，可以直接替换任何 OpenAI API 调用。

**SGLang** 建立在与 vLLM 相同的基础之上，但增加了用于前缀缓存的 RadixAttention，以及一门用于结构化 LLM 程序的领域特定语言。如果你的负载涉及多轮对话、工具使用或约束解码（JSON 输出、正则引导的生成），SGLang 往往能凭借前缀复用比 vLLM 快 2-5 倍。

**TensorRT-LLM** 把模型编译成优化过的 NVIDIA GPU kernel。它融合算子（把注意力 + 线性层 + 激活合并进一个 kernel），在 H100 GPU 上使用 FP8，并与 NVIDIA Triton Inference Server 集成用于生产部署。它在 NVIDIA 硬件上能达到最高的单 GPU 吞吐量，但配置更繁琐，且只能在 NVIDIA GPU 上运行。

Llama 3 70B 的真实数据（4×A100-80GB、BF16）：

| 指标 | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| 吞吐量（1 用户） | ~50 TPS | ~55 TPS | ~65 TPS |
| 吞吐量（100 用户） | 合计 ~2,500 TPS | 合计 ~3,200 TPS | 合计 ~3,000 TPS |
| 首 token 延迟 | ~400ms | ~300ms（前缀命中） | ~350ms |
| 最大上下文 | 128K | 128K | 128K |

### Ops:Byte 框架

你无法优化你未曾度量的东西。ops:byte 比值告诉你是计算受限还是内存受限，而这一点决定了哪些优化才有意义。

```
Compute roof: peak FLOPS of the GPU
Memory roof:  peak bandwidth * ops:byte ratio
```

当 ops:byte 较低（decode、小批量）时，你会撞上内存带宽的屋顶。增加算力（更高频率、更多核心）没有用。你需要减少内存读取（量化、KV cache 压缩），或者增大 batch size，把读取成本摊薄到更多有用的工作上。

当 ops:byte 较高（prefill、大批量）时，你会撞上计算屋顶。内存带宽优化帮不上忙。你需要更快的 GPU、kernel 融合或更低的精度来榨出更多 FLOPS。

| 场景 | ops:byte | 瓶颈 | 优化手段 |
|----------|----------|-------|---------------|
| Prefill，batch=1 | ~4,096 | 计算 | Kernel 融合、FP8 |
| Decode，batch=1 | ~1 | 内存 | 量化、KV 压缩 |
| Decode，batch=32 | ~32 | 内存 | 更大的 batch、continuous batching |
| Decode，batch=256 | ~256 | 过渡区 | 两者都重要 |
| Decode，batch=1024 | ~1,024 | 计算 | Kernel 融合、张量并行 |

A100 上的交叉点大约在 ops:byte = 156（312 TFLOPS / 2 TB/s）。低于 156 你是内存受限，高于 156 你是计算受限。Continuous batching 通过每次迭代塞入更多 token，把 decode 推向这个交叉点。

```figure
context-window-slide
```

## 构建它

### 第 1 步：从零实现 KV Cache

我们实现一个多头 KV cache，按层、按头保存 key 和 value 投影，并展示内存增长的模式。

```python
import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len
```

### 第 2 步：带 KV Cache 的注意力

一个简化版的多头注意力，在 decode 步中使用 KV cache。

```python
def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)
```

### 第 3 步：Continuous Batching 模拟器

下面模拟静态批处理与 continuous batching 的调度差异。

```python
import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }
```

### 第 4 步：前缀缓存

一个基于 trie 的前缀缓存，为共享前缀保存 KV 条目。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 第 5 步：投机解码模拟器

我们模拟草稿-目标（draft-target）投机解码，接受率可配置。

```python
class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results
```

### 第 6 步：KV Cache 内存分析器

计算真实模型配置下 KV cache 的内存需求。

```python
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }
```

## 使用它

使用 vLLM：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain inference optimization in one paragraph."], params)
```

使用 SGLang 实现前缀缓存 + 结构化输出：

```python
import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("You are a classifier. Output JSON only.")
    s += sgl.user(f"Classify this text: {text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "This product is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It was okay I guess."},
])
```

使用 TensorRT-LLM：

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Explain KV caching.")],
    max_new_tokens=256,
    temperature=0.7,
)
```

## 交付它

本节课产出：
- `outputs/skill-inference-optimization.md`——一个用于诊断和优化 LLM 推理服务的 skill

## 练习

1. 修改 KV cache 分析器，比较 FP16、FP8 与 INT4 的 KV cache 量化。对 4K 上下文的 Llama 3 70B，计算在 4×A100-80GB 上每种方案的最大并发用户数。KV 量化到 INT4 应能把用户容量提升约 4 倍。

2. 扩展 continuous batching 模拟器，跟踪 GPU 利用率（每一步被填充的 batch 槽位比例）。用 50 个输出长度服从帕累托分布（shape=1.5，scale=20）的请求，绘制静态批处理与 continuous batching 的利用率随时间变化。Continuous batching 应保持 80% 以上的利用率。

3. 实现分组查询注意力（GQA）版本的 KV cache，其中 `num_kv_heads < num_query_heads`。Llama 3 70B 使用 64 个 query head，但只有 8 个 KV head。计算相对于完整多头注意力的内存节省（KV cache 大小缩减 8 倍）。

4. 构建一个使用 LRU 淘汰的前缀缓存。把 max_entries 设为 500，生成 1,000 个请求，其中 60% 共享 5 个常见前缀之一。测量命中率并与不限容量的缓存比较。淘汰策略得当时，命中率应保持在 55% 以上。

5. 扩展投机解码模拟器，实现基于树的投机（EAGLE-2 风格）。不是单条 K 个草稿 token 的链，而是生成一棵候选树（例如 3 层、每层 2 个分支 = 8 个叶子候选）。比较每轮验证接受的总 token 数与线性投机。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|----------------------|
| Prefill | 「处理 prompt」 | 并行计算所有输入 token 的注意力——计算受限，因为完整的矩阵乘法让 GPU 核心保持忙碌 |
| Decode | 「生成 token」 | 每次前向传播产出一个 token，每次都读取完整模型权重——内存受限，因为计算在下一批权重到达之前就完成了 |
| KV cache | 「缓存注意力状态」 | 保存此前所有 token 的 key 和 value 投影，使它们在每个 decode 步不必重算——用内存换计算 |
| Continuous batching | 「动态批处理」 | 任何请求一完成就把新请求插入正在运行的 batch，在每个 decode 迭代评估，而不是等整批结束 |
| PagedAttention | 「KV cache 的虚拟内存」 | 以固定大小的页而非连续块分配 KV cache，消除内存碎片，并为共享前缀启用写时复制 |
| Speculative decoding | 「草稿加验证」 | 用快速草稿模型提出多个 token，再在一次目标模型前向传播中全部验证——数学上精确，2-3 倍加速 |
| EAGLE | 「自投机解码」 | 一种投机解码变体，在目标模型自身的隐藏状态上训练轻量头，比独立的草稿模型获得更高接受率 |
| Prefix caching | 「复用 system prompt 的 KV」 | 为常见前缀（system prompt、few-shot 示例）保存已算好的 KV cache 条目，在请求间复用，跳过冗余 prefill |
| Ops:byte ratio | 「算术强度」 | 计算操作数与读取的内存字节数之比——决定一个负载是计算受限（比值高）还是内存受限（比值低） |
| Time to first token | 「TTFT」 | 从收到请求到产出第一个输出 token 的延迟——长 prompt 下主要由 prefill 时间决定 |

## 延伸阅读

- Kwon 等，"Efficient Memory Management for Large Language Model Serving with PagedAttention"（2023）——vLLM 论文，提出了分页式 KV cache 管理，现已成为推理服务的行业标准
- Leviathan 等，"Fast Inference from Transformers via Speculative Decoding"（2023）——奠基性论文，证明草稿-验证式投机在产出精确目标模型分布的同时实现 2-3 倍加速
- Li 等，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"（2024）——在目标模型自身特征上训练头，而非使用独立的草稿模型，从而获得更高的接受率
- Zheng 等，"SGLang: Efficient Execution of Structured Language Model Programs"（2024）——引入用于前缀缓存的 RadixAttention，以及面向多次调用 LLM 程序的编程模型
- Williams 等，"Roofline: An Insightful Visual Performance Model for Multicore Architectures"（2009）——roofline 原始论文，形式化了用于推理计算与内存瓶颈的 ops:byte 框架
