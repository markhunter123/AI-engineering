# 项目 14 — 投机解码推理服务器

> 投机解码——由廉价的草稿模型提出 token，目标模型用一次前向完成校验——如今已是生产可用的优化技术，而不是研究把戏。vLLM 0.7 中的 EAGLE-3 在真实流量上带来 2.5-3 倍吞吐。AWS 2026 的 P-EAGLE 把并行投机推得更远。SGLang 的 SpecForge 支持大规模训练草稿头。Red Hat 的 Speculators 中心为常见开源模型发布了对齐草稿。TensorRT-LLM 让投机解码在 NVIDIA 上成为一等能力。2026 年的生产服务栈是 vLLM 或 SGLang 加 EAGLE 系列草稿、FP8 或 INT4 量化，以及基于队列等待的 HPA。本项目的目标是以 2.5 倍以上的基线吞吐服务两个开源模型，并给出完整的尾延迟报告。

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## 问题

到 2026 年，投机解码已成为通用能力。EAGLE-3 草稿头在目标模型的隐藏状态上训练，向前预测 N 个 token；目标模型用一次前向完成校验。60-80% 的接受率可以转化为 2-3 倍的端到端吞吐。vLLM 0.7 原生集成了这一能力。SGLang + SpecForge 提供训练 pipeline。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐草稿。

真正的工夫在服务运维，而不在模型。接受率会随流量分布漂移（ShareGPT、代码、领域数据各不相同）。发生拒绝时的尾延迟比不用投机更差——你必须报告多个 batch 规模下的 p99，而不只是稳态 tokens/sec。与 Anthropic / OpenAI API 相比的每百万 token 成本，是可信度的关键杠杆。

## 概念

投机解码有两层。**草稿（draft）**模型（EAGLE-3 头、ngram，或与目标对齐的较小模型）在每个步骤提出 k 个候选 token。**目标（target）**模型用一次前向校验全部 k 个；任何被接受的前缀都会替换贪心路径。接受率取决于草稿与目标的对齐程度以及输入分布。

在大多数流量上，EAGLE-3 优于 ngram 草稿。P-EAGLE 运行并行投机，构成更深的草稿树。代价是：拒绝时的 P99 延迟更高，因为校验过程更大。服务配置必须按 batch 规模分桶报告延迟，才能暴露这一点。

部署平台是 Kubernetes。vLLM 0.7 在每个 GPU 或张量并行分片上运行一个副本。HPA 依据队列等待而非 CPU 自动扩缩。FP8（Marlin）与 INT4（AWQ）量化把 GPU 显存控制在 H100 / H200 的范围内。端到端报告包括吞吐量、接受率、batch 1/8/32 下的 p50/p99，以及每百万 token 的成本（$/1M tokens）。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行投机、ngram 回退
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于队列等待指标做 HPA
- 评测：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于衡量不同领域分布下的接受率
- 参考：TensorRT-LLM 投机解码，作为厂商基线

```figure
cf-spec-decode
```

## 构建它

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2 路张量并行）上用 vLLM 0.7 部署。

2. **草稿来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或用 SpecForge 训练一个）。加载到 vLLM 的投机解码配置中。

3. **基线数字。** 启用投机之前：batch 1/8/32 下的 tokens/s、p50/p99 延迟、GPU 利用率。公布结果。

4. **启用 EAGLE-3。** 切换配置；重跑相同基准。报告加速比、接受率与 p99 尾延迟变化。

5. **P-EAGLE。** 启用并行投机；测量更深的草稿树与串行 EAGLE-3 的对比。报告 P-EAGLE 从有益转为有害的拐点。

6. **领域流量。** 让 ShareGPT、HumanEval 与领域特定流量经过同一个服务器。按分布测量接受率。找出草稿发生漂移的情形。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行同样的 pipeline。草稿更难做（MoE 路由噪声）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载增至三倍时的扩容。

9. **成本比较。** 在同一评测上计算每百万 token 成本，与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 比较。公布结果。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付它

`outputs/skill-inference-server.md` 描述交付物：一个带投机解码、经过实测的服务栈，一份完整的基准报告，以及一个 K8s 部署。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 相对基线的实测加速 | 两个模型在质量匹配时达到 2.5 倍以上吞吐 |
| 20 | 真实流量上的接受率 | 按分布给出接受率报告 |
| 20 | P99 尾延迟纪律 | 有、无投机时 batch 1/8/32 下的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、发布过程平滑 |
| 15 | 报告与方法论 | 清楚解释改了什么、为什么 |
| **100** | | |

## 练习

1. 测量草稿比目标落后一个版本时的接受率下降（例如 Llama 3.3 -> 3.4 的漂移）。建立监控告警。

2. 实现 ngram 回退：当 EAGLE-3 接受率低于阈值时，切换到 ngram 草稿。报告可靠性提升。

3. 运行受控 MoE 实验：同一个 Qwen3-Coder-30B，在注入路由噪声与不注入的情况下对比。测量草稿接受率的敏感度。

4. 扩展到 H200（141 GB）。报告每个副本可容纳模型规模的余量，以及能否服务未量化的 Llama 3.3 70B。

5. 在同一块 H100 硬件上对 TensorRT-LLM 投机解码做基准测试。报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | 「投机模型」 | 提议 N 个 token 供目标模型校验的小模型 |
| EAGLE-3 | 「2026 草稿架构」 | 在目标模型隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | 「并行投机」 | 由目标模型一次前向校验的草稿分支树 |
| 接受率 | 「命中率」 | 草稿 token 中无需重采样即被接受的比例 |
| 量化 | 「FP8 / INT4」 | 用更低精度权重在 GPU 显存中容纳更多模型 |
| 队列等待 | 「HPA 指标」 | 请求在推理开始前于待处理队列中等待的时间 |
| Speculators 中心 | 「对齐草稿」 | Red Hat Neural Magic 为常见开源模型提供的 EAGLE 草稿中心 |

## 延伸阅读

- [vLLM 的 EAGLE 与 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练 pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿中心
- [TensorRT-LLM 投机解码](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码与基准
