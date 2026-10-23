# 生产环境量化——AWQ、GPTQ、GGUF K-quant、FP8、MXFP4/NVFP4

> 量化格式不是放之四海而皆准的选择——它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 统治 CPU 与边缘，通过 llama.cpp 和 Ollama 交付。当需要在同一基础模型上运行多个 LoRA 时，GPTQ 在 vLLM 内胜出。AWQ 配合 Marlin-AWQ kernel，在 7B 级模型上以 INT4 达到约 741 tok/s，Pass@1 在 INT4 中最佳——这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上仍是中间路线——近乎无损且支持广泛。NVFP4 和 MXFP4（Blackwell 微缩放）更激进，需要按块验证。有两个陷阱会咬到团队：校准数据集必须与部署领域匹配；KV cache 与权重量化相互独立——AWQ 那课「我的模型现在只有 4 GB 了」忘记了生产 batch 规模下 10-30 GB 的 KV cache。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## 学习目标

- 说出 2026 年六种生产量化格式及其最佳适用场景。
- 给定硬件（CPU 对 GPU、Hopper 对 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（日常聊天、推理、多 LoRA），选择一种格式。
- 计算所选格式节省的权重内存，以及保持原样的 KV cache。
- 说出会让量化模型在领域流量上退化的校准数据集陷阱。

## 问题

量化减少内存和 HBM 带宽，而这正是 decode 所需要的。一个 FP16 的 70B 模型权重有 140 GB。把权重量化到 INT4（AWQ 或 GPTQ），模型变成 35 GB——能放进一块 H100，并为 KV cache 留出空间。这一点很重要，因为在 128 个并发序列、2k 上下文时，仅 KV cache 就有 20-30 GB。

但量化并非没有代价。激进量化会损害质量，尤其在推理密集型任务上。不同格式适配不同引擎。不同硬件原生支持不同精度。2026 年的格式动物园是真实存在的，你不能照搬别人的选择——必须根据自己的栈来定。

## 概念

### 六种格式

| 格式 | 位数 | 最佳场景 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF——CPU/边缘的默认选择

GGUF 是一种文件格式，本身并非量化方案——它把 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包进一个容器。Q4_K_M 和 Q5_K_M 是生产默认选择——在 4-5 bit 下接近 BF16 质量。它是 CPU 或边缘服务的最佳选择，因为 llama.cpp 是迄今为止最快的 CPU 推理引擎。

在 vLLM 中的吞吐代价：7B 上约 93 tok/s——该格式并未针对 GPU kernel 优化。只有部署目标是 CPU/边缘时才用 GGUF，其他情况不要用。

### GPTQ——vLLM 中的多 LoRA

GPTQ 是一种带校准过程的后训练量化算法。Marlin kernel 让它在 GPU 上很快（相对非 Marlin 的 GPTQ 有 2.6 倍加速）。7B 上约 712 tok/s。

它独有的优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你服务的是一个基础模型外加 10-50 个微调变体（每个都是一个 LoRA），GPTQ 就是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ——数据中心 GPU 的默认选择

Activation-aware Weight Quantization（激活感知权重量化）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ kernel：相对朴素实现有 10.9 倍加速。7B 上约 741 tok/s，在 INT4 格式中 Pass@1 最佳。

新的 GPU 服务选 AWQ，除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4）。

### FP8——可靠的中间路线

8 位浮点。近乎无损。支持广泛。Hopper Tensor Core 原生加速 FP8，Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年安全的默认选择。内存节省只有 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4——Blackwell 的激进选项

微缩放 FP4。每个权重块都有自己的缩放因子。激进，但在 Blackwell Tensor Core 上硬件加速。相对 FP8 把每 token 字节数减半——这正是 Phase 17 · 07 中的经济收益。

注意事项：
- 尚不支持 LoRA（2026 年初）。
- 在推理密集型工作负载上可见质量下降。
- 对每个模型在你的评测集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），用通用网页文本校准会让算法对「该保护哪些权重」做出错误判断。HumanEval 上的 Pass@1 可能下降几个点。

解决办法：用领域内数据校准。通常几百条领域样本就足够。上线前在评测集上测试。

### KV cache 陷阱

AWQ 把权重压到 4 bit。KV cache 是独立的，仍保持在 FP16/FP8。对于一个 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 降到 INT4）。
- 128 并发 × 2k 上下文下的 KV cache：约 20 GB。
- 激活：约 5 GB。
- 总计：约 60 GB——能放进 H100 80GB。

天真地说「我把模型量化到 4 GB」会忘掉另外那 30-50 GB。要整体规划 HBM 预算。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是另一个独立选择，有其自己的权衡——它直接影响注意力精度，并非免费的午餐。

### AWQ INT4 对推理任务是危险的

思维链、数学、长上下文代码生成——这些任务会明显受到激进量化的影响。AWQ INT4 在 MATH 上损失约 3-5 个点。对推理密集型工作负载，交付 FP8 或 BF16，接受内存成本。

### 2026 选用指南

- CPU/边缘服务：GGUF Q4_K_M。就这样。
- GPU 服务、日常聊天、无 LoRA：AWQ。
- GPU 服务、多 LoRA：GPTQ 配 Marlin。
- 推理型工作负载：FP8。
- Blackwell 数据中心、质量已验证：NVFP4 + FP8 KV。
- 拿不准：对每个候选格式跑 1,000 样本的评测。

```figure
gpu-memory-breakdown
```

## 使用它

`code/main.py` 计算一系列模型规模下六种格式的内存占用（权重 + KV + 激活）和相对吞吐量。它展示 KV cache 在何处占主导、权重压缩在何处划算，以及 FP8 在何处是安全选择。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?dtype=fp8&users=64) 这台机器展示了 NVIDIA L4 上的 FP8。权重从 15.28 GB 降到 8.81 GB，KV 池从 24,094 个 token 增长到 142,369 个 token，64 用户下的实测输出从每秒 250.85 token 提升到 735.91 token。

## 交付它

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型规模、工作负载类型和质量容忍度，它选择一种格式并给出校准/验证计划。

## 练习

1. 运行 `code/main.py`。对 70B 模型、128 并发、2k 上下文，计算每种格式的总 HBM。哪种格式能让你放进一块 H100 80GB？
2. 你有一个 7B 的代码模型。选一种格式并给出理由。如果你对质量容忍度判断错了，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集规模。为什么数据更多不一定更好？
4. 阅读 Marlin-AWQ kernel 的论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而原始 GPTQ 只有约 712。
5. 何时适合把 AWQ 权重与 FP8 KV cache 组合，而不是让 KV 保持 BF16？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GGUF | 「llama.cpp 格式」 | 打包 K-quant 变体的文件格式；CPU/边缘默认选择 |
| Q4_K_M | 「Q4 K M」 | 4 bit K-quant medium；GGUF 的生产默认 |
| GPTQ | 「gee pee tee q」 | 带校准的后训练 INT4；在 vLLM 中支持 LoRA |
| AWQ | 「a w q」 | 激活感知 INT4；Marlin kernel；INT4 中 Pass@1 最佳 |
| Marlin kernel | 「快速 INT4 kernel」 | 为 Hopper 上 INT4 定制的 CUDA kernel；10 倍加速 |
| FP8 | 「八位浮点」 | Hopper/Ada/Blackwell 上安全的精度默认 |
| MXFP4 / NVFP4 | 「微缩放四位」 | Blackwell 4 bit FP，带按块缩放因子 |
| 校准数据集 | 「cal data」 | 用于选取量化参数的输入文本；必须匹配领域 |
| KV cache 量化 | 「KV INT8」 | 与权重分开的选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)——对比基准。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)——各格式吞吐量数字。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)——逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html)——支持的格式与参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)——AWQ 原始公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323)——GPTQ 原始公式。
