# 推理引擎内部机制——PagedAttention、连续批处理、分块预填充

> 现代推理引擎的吞吐量建立在三个相互叠加的默认机制上，而不是某一个技巧。PagedAttention 始终开启。连续批处理在解码迭代之间把新请求注入活跃 batch。分块预填充把长 prompt 切片，让解码 token 永远不会挨饿。三者全开时，单张 H100 SXM5 上 FP8 的 Llama 3.3 70B 在 128 并发下可达 2,200-2,400 tok/s——大约比 vLLM 自身默认值高 25%，是朴素 PyTorch 循环的 3-4 倍。本节课拆解 vLLM 的调度器与注意力 kernel——这三种技术的参考引擎——拆到你能画出来的程度，并以 `code/main.py` 中一个玩具式连续批处理器收尾，它像 vLLM 那样调度预填充和解码。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 把 PagedAttention 解释为一个 KV cache 分配器：block、block table，以及为什么生产负载下碎片率保持在 4% 以下。
- 在迭代层面画出连续批处理：已完成的序列如何离开 batch，新序列如何在不排空的情况下加入。
- 用一句话描述分块预填充，并说出它保护的是哪个延迟指标（提示：是 TTFT 尾部，不是平均吞吐量）。
- 在一次开启全部优化之前，对照你所用版本的兼容性矩阵检查 vLLM 的特性组合。

## 问题

一个朴素的 PyTorch 服务循环一次只处理一个请求：tokenize、预填充、解码直到 EOS、返回。一个用户时没问题。一百个用户时，它就是一群耐心排队的人。显而易见的修法——静态批处理——把每个请求 padding 到窗口内最长的 prompt，把每次解码 padding 到最长的预期输出，并让整个 batch 卡在最慢的序列上。你为永远用不上的 padding 付费，而快请求要等慢请求。

vLLM 一次解决了三个问题。PagedAttention 阻止 KV cache 碎片像经典连续分配那样吃掉 60-80% 的 GPU 内存。连续批处理让请求在每个解码迭代之间加入和离开 batch，因此 batch 始终装满真实工作。分块预填充把一个 32k token 的 prompt 切成约 512 token 的片，与解码交错执行，这样长 prompt 不会冻结 GPU 上的每个解码 token。

2026 年的生产默认是三者全开。你需要理解每一个在做什么，因为故障模式全都出在调度器上，而不在模型上。

## 概念

### PagedAttention 作为虚拟内存系统

每个序列的 KV cache 为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对 8192 token 的 Llama 3.3 70B，在 BF16 下每个序列约 1.25 GB。如果你为每个请求都预留 8192 个槽位，而平均请求只用了 1500 个 token，你就浪费了所预留 HBM 的约 82%。经典批处理就要付这份浪费。

PagedAttention 借用了操作系统虚拟内存的思路。KV cache 不再按序列连续存放，而是以固定大小的 block 分配（默认 16 token）。每个序列有一张 block table，把它的逻辑 token 位置映射到物理 block ID。当序列增长超出已分配的 block 时，就再加一个 block。当它结束时，其 block 归还到池中。

碎片率从 60-80%（经典方式）降到 4% 以下（PagedAttention）。你不需要用某个开关来启用 PagedAttention——它是 vLLM 交付的唯一分配器。可调的是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活值之后，为 KV block 预留多少 HBM。

### 迭代层面的连续批处理

旧的「动态批处理」等待一个窗口（比如 10 ms）来凑满一个 batch，然后跑预填充 + 解码 + 解码 + 解码，直到每个序列结束。快序列早早离开，在 GPU 完成慢序列时干等。

连续批处理在每个解码步骤之间运作。把正在运行的序列集合称为 `RUNNING` 列表。每次迭代时：

1. `RUNNING` 中任何刚命中 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV block，就接纳新序列（预填充或恢复）。
3. 前向传播在当前 `RUNNING` 中的内容上运行，每个序列产出一个新 token。

batch 大小从不 padding 到固定数字。处于不同输出位置的序列共享一次融合前向。在 2026 年的 vLLM 中，这被称为 `V1 scheduler`。关键不变量：调度器每个解码迭代运行一次，而不是每个请求运行一次。

### 分块预填充保护 TTFT 尾部

预填充是计算受限的。Llama 3.3 70B 上一个 32k token 的 prompt，在单张 H100 上要花约 800 ms 的纯预填充。预填充运行时，batch 中其它所有序列的解码 token 都在等待。在服务循环中，一个长 prompt 的首 token 延迟（TTFT）会变成数十个其它用户的 token 间延迟（ITL）尖峰。

分块预填充把预填充切成固定大小的块（默认 512 token），并把每一块作为一个调度单元。块与块之间，调度器可以推进解码序列一个 token。你付出一点绝对的预填充延迟代价（每块几毫秒），换来低得多的解码期抖动。在已发表的基准中，混合负载下的 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认机制相互作用

这三种特性彼此依赖。PagedAttention 给调度器提供细粒度的 KV 资源来做权衡。连续批处理需要这种细粒度资源，这样接纳新序列不会强制全局重排。分块预填充是调度器在同一个 `RUNNING` 列表上做的决策——它是又一条调度策略，而不是另一套系统。

你不需要知道每一个 flag。你需要知道调度器在优化什么：KV block 预算下的 goodput，并受分块预填充切片的约束。

### 检查兼容性矩阵

在一次全部启用之前，对照你所用 vLLM 确切版本的兼容性矩阵检查每一个特性组合，因为哪些能组合会随版本变化。在 v0.18.0 中，特性矩阵把投机解码标记为与分块预填充和前缀缓存兼容；投机解码页面列出两个已知不兼容项：截至 v0.15.0 的 pipeline 并行，以及截至 v0.10.0 的草稿模型投机。至于草稿方法本身，2026 年的默认通常是 EAGLE-3（`"method": "eagle3"`），在 Phase 17 · 05 中讲解。

### 应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同一模型，vLLM 默认（无分块预填充）：约 1,800 tok/s。
- 同一模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：分块预填充约 15 ms，不用约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个循环的 stdlib Python 版本，使用假的 token 数和假的前向延迟。运行它可以看到分块预填充如何在长预填充期间让解码序列保持存活。

```figure
tensor-parallel
```

## 使用它

`code/main.py` 模拟一个 vLLM 风格、可切换特性的调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：padding 并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级接纳与释放。
- `CONTINUOUS + CHUNKED` 模式：预填充片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 那一行应该占优。

## 交付它

本节课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一份服务配置（batch 大小、KV 内存利用率、分块预填充大小、投机配置），它会产出一份调度器诊断，指出三个默认机制中哪一个成了瓶颈，以及该调什么。

## 练习

1. 运行 `code/main.py`。在短长请求混合的工作负载上比较 `STATIC` 与 `CONTINUOUS`。吞吐量差距从何而来——预填充效率、解码效率，还是尾部延迟？
2. 修改玩具调度器，加入 `--max-num-batched-tokens`。对运行 Llama 3.3 70B FP8 的 H100，正确的值是多少？（提示：它是 KV block 大小和空闲 block 数的函数，而不是原始 HBM。）
3. 重读 vLLM v0.18.0 的发布说明。哪些 flag 组合互斥？列出来。
4. 对一条 1,000 请求、输出均值 1,500 token、标准差 600 token 的 trace，计算 KV cache 碎片浪费，分别在 (a) 按请求连续分配、最大 8192 的情况下，(b) 使用 16 token block 的 PagedAttention 下。
5. 用一段话解释为什么分块预填充改善 P99 ITL，但单独看并不改善吞吐量。实践中吞吐量的收益从何而来？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | 「KV 技巧」 | KV cache 的固定大小 block 分配器；碎片 <4% |
| Block table | 「页表」 | 每个序列从逻辑 token 位置到物理 KV block 的映射 |
| Continuous batching | 「动态批处理，但做对了」 | 每个解码迭代都做接纳/释放决策 |
| Chunked prefill | 「预填充切分」 | 把长预填充切成 512 token 的片，与解码交错 |
| TTFT | 「首 token 时间」 | 预填充 + 排队 + 网络；长 prompt 下由预填充主导 |
| ITL | 「token 间延迟」 | 连续解码 token 之间的时间；由 batch 大小主导 |
| Goodput | 「满足 SLO 的吞吐量」 | 每个请求仍达到 TTFT 和 ITL 目标的 token/秒 |
| V1 scheduler | 「新调度器」 | vLLM 的 2026 调度器；运行连续批处理加分块预填充 |
| `--gpu-memory-utilization` | 「内存旋钮」 | 权重和激活值之后，为 KV block 预留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档——投机解码](https://docs.vllm.ai/en/latest/features/spec_decode/)——关于分块预填充与投机解码兼容性的官方来源。
- [vLLM 发布说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)——2026 年发布节奏与版本特定行为。
- [vLLM 博客——PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)——最初的文章，至今仍定义着如何理解这个分配器。
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180)——碎片分析与调度器设计。
- [Aleksa Gordic——Inside vLLM](https://www.aleksagordic.com/blog/vllm)——带火焰图的 V1 调度器详细剖析。
