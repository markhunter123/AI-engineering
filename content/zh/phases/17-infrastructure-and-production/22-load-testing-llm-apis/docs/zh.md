# LLM API 负载测试 —— 为什么 k6 和 Locust 会说谎

> 传统负载测试工具并不是为流式响应、可变输出长度、token 级指标或 GPU 饱和设计的。有两个陷阱会咬住大多数团队。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行分词，高并发时会与请求生成争抢 CPU；分词积压随后抬高了所报告的 token 间延迟——瓶颈在你的客户端，不在服务端。prompt 均匀性陷阱：循环里发送相同 prompt 只测试了 token 分布上的一个点；真实流量长度可变，前缀匹配也各不相同。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这一点。2026 年工具对照：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）提供 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知，通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式，最适合 CI/CD 闸门；Vegeta 用于 Go 的恒定速率饱和测试；Locust 2.43.3 仅在配合 LLM-Locust 扩展时用于流式。负载模式：稳态、爬坡、尖刺（自动扩缩容测试）、浸泡（内存泄漏）。

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## 学习目标

- 解释两个反模式（GIL 陷阱、prompt 均匀性陷阱），它们让通用负载测试工具在 LLM API 上给出错误结论。
- 按用途选工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 闸门）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考实现）。
- 设计四种负载模式（稳态、爬坡、尖刺、浸泡），并说出每种能抓住的故障模式。
- 基于输入 token 的均值 + 标准差构造真实的 prompt 分布，而不是固定长度。

## 问题

你用 k6 以 500 并发用户压测了 LLM 端点。它撑住了。你发布了。生产上只有 200 个真实用户时服务就倒了——P99 TTFT 暴涨，GPU 满载。

发生了两件事。第一，k6 发送了 500 个相同的 prompt——你的请求合并（request coalescing）和前缀缓存让它看起来像在处理 500 个并发解码，实际上只处理了一个。第二，k6 不会按人眼感受的方式跟踪流式响应上的 token 间延迟；它看到的是一个 HTTP 连接，而不是 500 个以不同间隔到达的 token。

LLM 的负载测试是一门独立的学问。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下于客户端侧运行分词。高并发时，分词器排在请求生成之后排队。报告的 token 间延迟包含客户端的分词积压。你以为服务端慢；其实是测试框架。

修复：LLM-Locust 扩展把分词移到独立进程，或改用编译型语言的测试框架（k6、使用 tokenizers.rs 的 LLMPerf）。

### prompt 均匀性陷阱

所有已知负载测试工具都允许你配置一个 prompt。在 10,000 次迭代的循环测试中，每次发送的都是完全相同的 prompt。服务端每次都看到相同前缀——前缀缓存命中率接近 100%，吞吐量看起来很棒。

修复：从 prompt 分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样，内容也多样。

### 四种负载模式

1. **稳态** —— 30-60 分钟恒定 RPS。抓：基线性能回归。
2. **爬坡** —— 15 分钟内 RPS 从 0 线性升到目标值。抓：容量拐点、预热异常。
3. **尖刺** —— 2 分钟内 RPS 突增 3-10 倍再回落。抓：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡** —— 稳态运行 4-8 小时。抓：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具对照

**LLMPerf**（Anyscale）—— Python 编写，但分词由 Rust 支撑。均值/标准差 prompt。支持流式感知。性能测试的默认首选。

**NVIDIA GenAI-Perf** —— NVIDIA 的参考实现。使用 Triton 客户端；指标覆盖全面。注意它的 ITL 不含 TTFT；LLMPerf 的包含。两个工具对同一服务端会给出不同的 TPOT。

**LLM-Locust**（TrueFoundry）—— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** —— 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本体（Go 编写，编译型，无 GIL）新增了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 做 Kubernetes 原生分布式测试。
- 最适合 CI/CD 闸门和 SLA 测试。

**Vegeta** —— Go 编写，比 k6 简单。恒定速率 HTTP 饱和测试。不感知 LLM，但适合网关 / 速率限制测试。

**Locust 2.43.3 原版** —— 对 LLM 存在 GIL 陷阱。只在配合 LLM-Locust 扩展时可用。

### CI 中的 SLA 闸门

在 PR 上运行 k6，配置：

- 每个基线 RPS 下跑 30-50 次迭代。
- 闸门：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 违反即中断构建。

### 真实的 prompt 分布

从真实流量样本构建（如果有），或从公开分布构建（例如聊天用 ShareGPT prompts，代码用 HumanEval）。把均值和标准差喂给 LLMPerf。无论如何都要避免「一个 prompt 跑循环」。

### 应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：并发 X 下 100-1000 个请求。
- 典型 CI 闸门：每个 PR 30-50 次迭代。
- 四种模式：稳态、爬坡、尖刺、浸泡。

```figure
load-pattern-waves
```

## 使用它

`code/main.py` 用真实 prompt 分布模拟一次负载测试，测量有效 TPOT，并演示均匀 prompt 陷阱。

## 交付它

本节课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。对比均匀分布与真实分布——差距在哪里？
2. 为 CI 闸门写 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出三种原因，以及用来区分它们的插桩手段。
4. 从 10 RPS 到 100 RPS 的尖刺测试。如果已有 Karpenter + vLLM production-stack（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务端报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | 「那个 LLM 测试框架」 | Anyscale 的基准工具，支持流式感知 |
| GenAI-Perf | 「NVIDIA 工具」 | NVIDIA 参考测试框架 |
| LLM-Locust | 「给 LLM 用的 Locust」 | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | 「合成基准」 | 大规模合成测试工具 |
| k6 Operator | 「K8s 上的 k6」 | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | 「Python 客户端开销」 | 分词积压抬高了报告的延迟 |
| prompt 均匀性陷阱 | 「单 prompt 谎言」 | 循环用同一 prompt 命中缓存，虚高吞吐 |
| 稳态（steady-state） | 「恒定负载」 | N 分钟内 RPS 持平 |
| 爬坡（ramp） | 「线性加压」 | 在持续时间内从 0 升到目标 |
| 尖刺（spike） | 「突发测试」 | 突然倍数增长再回落 |
| 浸泡（soak） | 「长时间测试」 | 数小时，用于检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
