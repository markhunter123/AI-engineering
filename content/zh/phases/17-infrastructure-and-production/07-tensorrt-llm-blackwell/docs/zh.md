# 硬件专用推理编译——Blackwell 上的 FP8 与 NVFP4

> 硬件专用推理编译用可移植性换取吞吐量，而 TensorRT-LLM——只支持 NVIDIA、为 Blackwell 调优——是这笔交易划算的最清晰例子。在 GB200 NVL72 上配合 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 0.012 美元，而 H100 + vLLM 为 0.09 美元/M——7 倍的经济差距。这套栈是三种浮点体制的叠加：FP8 对 KV cache 和注意力 kernel 仍然至关重要，因为它具备所需的动态范围；NVFP4（4 bit 微缩放）负责权重和激活；多 token 预测（MTP）与分离式 prefill/decode 再叠加 2-3 倍。day-0 模型支持直接加载 FP4 权重，无需训练后转换。对 2026 年的工程团队而言，症结在于：TRT-LLM 是开源的，但专属于 NVIDIA——为 CUDA 和 Blackwell 专门优化——采用它就是用可移植性换吞吐量。承诺之前，先针对你的模型与硬件组合算一笔账。

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## 学习目标

- 解释为什么即使权重采用 NVFP4，FP8 对 KV cache 和注意力仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省从何而来。
- 说出 TRT-LLM 利用的 Blackwell 专属特性（day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断何时 TRT-LLM 的 NVIDIA 锁定值得相对 Hopper 上 vLLM 的 7 倍成本差距。

## 问题

2026 年推理经济学的核心问题是「每美元多少 token」。答案取决于四个叠加选择：硬件代际（Hopper H100/H200 对 Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM 对 SGLang 对 TRT-LLM）以及编排（普通对分离式对 Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 每百万 token 约 0.09 美元。在 Blackwell + TRT-LLM + Dynamo 上，同一模型约 0.012 美元——便宜 7 倍。差距的一部分来自硬件（Blackwell 的单 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。一部分来自栈：FP4 权重、MTP 草稿、分离式 prefill/decode，以及用于 MoE 专家通信的 NVLink 5 all-to-all。

你无法在 NVIDIA 栈之外复现这一点。这就是权衡——用可移植性换经济效益。理解哪些栈选择贡献了差距的哪一份额，正是本课的重点。

## 概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年常见的错误：以为 NVFP4 可以到处套用。并非如此。KV cache 需要 FP8（8 位浮点），因为它存储的注意力 key 和 value 跨越很宽的动态范围。把 KV 量化到 FP4 会造成灾难性的精度损失——分布尾部被削掉，注意力分数随之崩溃。FP8 的指数位给了 KV cache 所需的范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每个权重块都有自己的缩放因子，因此小块可以覆盖不同的动态范围，而不会出现按张量缩放带来的损失。对激活而言，FP4 能够支撑，因为激活在层内的范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 bit 微缩放）。
- 激活：NVFP4。
- KV cache：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专属原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 加载时无需训练后转换。FP4 不需要 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（Phase 17 · 05）思路相同，但集成进 TRT-LLM 构建。
- **分离式服务**：prefill 与 decode 位于不同的 GPU 池，KV cache 通过 NVLink 或 InfiniBand 传输。思路与 Dynamo 相同（Phase 17 · 20）。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟相对 Hopper 降低 3 倍。TRT-LLM 的 MoE kernel 为此调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上为 0.02 美元/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）为 0.012 美元/M token。
- 可比工作负载下 H100 + vLLM 约 0.09 美元/M token。
- 三个月的 TRT-LLM 更新带来 2.8 倍吞吐量提升（2026）。
- 单 GPU LLM 吞吐量，Blackwell 相对 Hopper 为 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每一个提交任务上领先。

### FP4 在质量上究竟要付什么代价

NVFP4 很激进。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重会出现可见的质量退化。按块校准能缓解，但无法消除。发布推理模型的团队常用 FP8 权重 + FP4 激活作为折中，或继续使用全程 FP8 的 H200。

规则：在承诺 NVFP4 权重之前，务必在你的评测集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源 kernel。模型需要针对特定 GPU SKU 编译。没有 AMD，没有 Intel，没有 ARM。如果你的基础设施策略是多供应商，TRT-LLM 对于 TRT-LLM 服务层就是不可选项——你仍然可以在混合硬件上通过 vLLM 提供服务。如果你只用 NVIDIA，7 倍差距足以支付锁定的代价。

### 2026 实用配方

对于每年 1 亿美元以上的推理账单，跑在 Hopper + vLLM 上等于把 7-10 倍留在桌上。把成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。实验层保留在 H100 + vLLM 上以维持模型迭代速度。每个转换到 NVFP4 的模型都要在生产前验证质量。

### 分离式带来的额外收益

TRT-LLM 的分离式服务（独立的 prefill 和 decode 池）在 Phase 17 · 20 有深入讲解。在 Blackwell 上，乘数会叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍这个数字假设了完整栈。

```figure
pipeline-parallel
```

## 使用它

`code/main.py` 为模型在三种栈上计算 HBM 占用、解码吞吐量（内存受限区间）和 $/M token：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察复利效应以及每个变化贡献的差距份额。

## 交付它

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型规模和年 token 量，它判断 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对一个激活参数占 30% 的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上受内存带宽限制的解码吞吐量。最大的跃升来自哪里？
2. 某客户每年在 H100 + vLLM 上花费 200 万美元。考虑到 7 倍的经济差距，他们需要购买多少块 Blackwell GPU，才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 权重转换为 NVFP4 后，你看到 MATH 上的准确率下降 3 个点。说出两条恢复路径：一条质量优先（保留 FP8 权重），一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务上 Blackwell 相对 Hopper 的差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV cache、128k 上下文下所需的 HBM。它能放进单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| FP8 | 「八位浮点」 | 8 位浮点；因其动态范围而用于 KV cache 和注意力 |
| NVFP4 | 「四位微缩放」 | NVIDIA 的 4 bit 微缩放 FP 格式；Blackwell 上用于权重和激活 |
| MXFP8 | 「MX 八位」 | 微缩放 FP8 变体；在 Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | 「直接发布 FP4 权重」 | 模型提供商发布时权重已是 FP4；无需训练后转换步骤 |
| MTP | 「多 token 预测」 | TRT-LLM 集成的投机解码草稿模型（Phase 17 · 05） |
| 分离式服务 | 「拆分 prefill/decode」 | prefill 与 decode 在独立 GPU 池；KV 通过 NVLink/IB 传输 |
| All-to-all | 「MoE 专家通信」 | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍 |
| InferenceX | 「SemiAnalysis 推理基准」 | 2026 年业界认可的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)——2026 年 4 月的 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)——NVLink 5 all-to-all 与 MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)——官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)——TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)——发布 Blackwell 数字的基准套件。
