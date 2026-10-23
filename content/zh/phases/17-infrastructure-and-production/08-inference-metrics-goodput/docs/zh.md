# 推理指标——TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一个推理部署是否正常工作。TTFT 由 prefill、排队和网络组成。TPOT（等价于 ITL）是每个 token 的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求占比。低 goodput 下的高吞吐量意味着你处理的 token 从未及时送达用户。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数字：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。始终报告 P50、P90、P99——绝不要只报均值。还要当心测量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 则包含；同一次运行，两个工具给出的 TPOT 不一致。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个指标测量的是哪个环节。
- 解释为什么均值是 LLM 服务中错误的统计量，以及如何解读 P50/P90/P99。
- 构造一个多约束 SLO（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并据此计算 goodput。
- 说出两个对同一次运行给出不同 TPOT 的基准工具，并解释原因。

## 问题

「我们的吞吐量是每秒 15,000 token。」那又怎样？如果 40% 的请求端到端超过 2 秒，用户已经放弃了会话。单看吞吐量并不能说明产品是否可用。

推理有多个延迟维度，每个维度的失效方式都不同。prefill 是计算受限的，随 prompt 长度增长。decode 是内存受限的，随 batch 大小变化。排队延迟是运维问题。网络是物理距离问题。你需要为每一项设立独立指标，需要百分位数，还需要一个能回答「用户是否得到了预期体验」的复合指标——那就是 goodput。

## 概念

### TTFT——首个 token 的时间

`TTFT = queue_time + network_request + prefill_time`

当 prompt 很长时，prefill 占主导。在 H100 上运行 Llama-3.3-70B FP8，32k 的 prompt 需要约 800 ms 的纯 prefill。排队时间是负载下的调度器行为。网络请求是包含 TLS 的链路时间。TTFT 是用户在任何内容流回之前看到的延迟。

### TPOT / ITL——token 间延迟

同一个量有很多名字。`TPOT`（每个输出 token 的时间）、`ITL`（token 间延迟）、`decode latency per token`——全都一样。它指的是首个 token 之后，连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同样的 Llama-3.3-70B H100 栈上配合分块 prefill（chunked prefill），TPOT 均值约 7 ms。没有分块 prefill 时，当邻近序列正在进行长 prefill，TPOT 可能飙到 50 ms。要盯 P99，而不是均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对长输出（>500 token），E2E 由 TPOT 主导。对带长 prompt 的短输出，E2E 由 TTFT 主导。报告 E2E 时要按输出长度分桶。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率，但不告诉你单个请求的健康状况。

### Goodput——你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束。只有当每个约束都成立时，一个请求才算「好」。goodput 就是这类请求的占比。60% goodput 下的高吞吐量是失败。99% goodput 下的较低吞吐量才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交以及 AI 平台提供商内部 SLA 跟踪所用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个 decode batch 里如果邻居正在长 prefill，可能输出 500 个 token 时 TPOT 约 7 ms，另有 20 个 token 时 TPOT 约 60 ms。TPOT 均值为 9 ms。TPOT P99 为 65 ms。用户会经常撞上 P99——这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。就用户体验而言，P99 才是你要优化的那个。

### 参考数字——Llama-3.1-8B-Instruct 在 TRT-LLM 上，2026

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：根据分块 prefill 配置不同，在 10-25 ms 之间变化。

这些是 NVIDIA 公布的参考点。它们会随模型规模（70B 会显示 3-5 倍）、硬件（H100 对 B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准工具对同一次运行的 TPOT 各执一词：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从第 2 个 token 算起。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 算起。

对于一个 TTFT 500 ms、100 个输出 token、总解码时间 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

始终说明用的是哪个工具，始终公布定义。

### 构建 SLO

2026 年一个 70B 聊天模型面向消费者的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 对 <300 token 的输出，E2E P99 <= 3 s。
- goodput 目标 >= 99%。

企业级 SLO 会收紧 TTFT（200-400 ms）并放宽 E2E。关键是写下来、三项都测、把 goodput 作为单一复合指标来跟踪。

### 如何测量

- 跑真实流量，或足够真实的合成流量（LLMPerf 搭配 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准运行的目标并发设为峰值的 2 倍。
- 跑 30-50 轮，取合并样本的百分位数。
- 发布时附上工具名、工具版本、模型、硬件、并发度和 prompt 分布。

```figure
throughput-latency
```

## 使用它

`code/main.py` 是一个简易的 goodput 计算器。生成合成延迟分布，套用 SLO，计算 goodput。它还会在同一个 trace 上展示 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?users=1) 这台机器展示了 NVIDIA L4 上每个输出 token 的时间从何而来。一个解码步读取 16.41 GB 权重，以 300 GB/s 计算至少需要 54.7 ms，而实测该步耗时 60.16 ms。

## 交付它

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它生成一套可直接用于 CI/CD 的基准测试配方，用 goodput 而非吞吐量来卡部署。

## 练习

1. 运行 `code/main.py`。生成一个带 1% 尾部尖峰的分布。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 某供应商宣称「Llama 3.3 70B 在 H100 上 15,000 tok/s」。在相信它之前，说出三个该问的问题。
3. 为什么分块 prefill 能保护 P99 TPOT，却保护不了 TPOT 均值？
4. 为语音助手构造一套消费者 SLO（首个 token 是被听到的，而不是读到的）。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出另外三个两个工具会不一致的指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| TTFT | 「首个 token 的时间」 | 排队 + 网络 + prefill；长 prompt 时由 prefill 主导 |
| TPOT | 「每个输出 token 的时间」 | 首个 token 之后每个 token 的内存受限解码成本 |
| ITL | 「token 间延迟」 | 在大多数工具中与 TPOT 相同（并非全部——见 GenAI-Perf） |
| E2E | 「端到端」 | TTFT + TPOT * output_len；再加上响应侧网络 |
| 吞吐量 | 「tok/s」 | 集群效率；没有延迟百分位数就没有意义 |
| Goodput | 「SLO 达标率」 | 同时满足每一项 SLO 约束的请求占比 |
| P99 | 「尾部」 | 百分之一最差情况的延迟；用户体验指标 |
| SLO 多约束 | 「联合约束」 | 三个延迟上限全部取 AND；违反任何一个即请求失败 |
| GenAI-Perf 对 LLMPerf | 「工具陷阱」 | 两个工具对 ITL 是否包含 TTFT 意见不一 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)——TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)——替代定义与测量配方。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)——真实部署上的实用测量。
- [LLMPerf](https://github.com/ray-project/llmperf)——基于 Ray 的开源基准工具。
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md)——NVIDIA 的基准工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)——业界认可、基于 goodput 的基准。
