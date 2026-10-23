# 自托管服务选型 —— 让引擎匹配硬件与规模

> 引擎选型取决于硬件、规模和生态，而不是读一份排行榜。2026 年有四个引擎主导自托管推理：llama.cpp、Ollama、vLLM、SGLang，TGI 则处于维护模式、逐渐掉队。**llama.cpp** 在 CPU 上最快——模型支持最广，对量化和线程有完全控制。**Ollama** 是开发笔记本上一条命令就能装好的选择，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在类生产负载下吞吐量相差 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——只修 bug，原始吞吐量比 vLLM 慢约 10%，但历史上可观测性一流、HF 生态集成最好。这种维护状态使它成为有风险的长期押注——对新项目来说，SGLang 或 vLLM 是更稳妥的默认选择。**vLLM** 是通用生产默认——v0.15.1（2026 年 2 月）加入 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 Agent 多轮 / 前缀密集型场景的专家——生产环境部署 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：CPU 优先 → llama.cpp。AMD / 非 NVIDIA → vLLM 是支持最好的路径（TRT-LLM 锁定 NVIDIA）。2026 年的 pipeline 模式：开发 = Ollama，预发 = llama.cpp，生产 = vLLM 或 SGLang。引擎接受不同的权重格式——llama.cpp 家族用 GGUF，GPU 引擎用 HF safetensors——所以阶段之间可能需要一次格式转换。

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## 学习目标

- 给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和负载（通用聊天 / Agent / 长上下文），选出合适的引擎。
- 说出 TGI 在 2025 年 12 月 11 日进入维护模式这一状态，以及为什么它会让新项目偏向 vLLM 或 SGLang。
- 描述开发/预发/生产 pipeline，包括 GGUF 到 safetensors 的格式转换在阶段之间的位置。
- 解释为什么「CPU 优先」指向 llama.cpp，而「AMD」排除 TRT-LLM。

## 问题

你的团队启动一个新的自托管 LLM 项目。一位工程师说 Ollama，另一位说 vLLM，第三位说「TGI 不是开箱即用吗？」三人在各自的语境下都对。但没有一个放之四海而皆准。

2026 年的选择树很重要：硬件第一，规模第二，负载第三。而 2025 年的一件具体事件——TGI 在 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘 / 依赖最少 / 模型支持最广 | CPU 上最快，完全可控 |
| **Ollama** | 开发笔记本、单用户、一条命令安装 | 比 llama.cpp 慢 15-30%；生产吞吐差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛的生产默认；v0.15.1 于 2026 年 2 月发布 |
| **SGLang** | Agent 多轮、前缀密集型负载 | 生产环境 400,000+ GPU |

### 硬件优先决策

**CPU 优先** → llama.cpp。Ollama 也能用，但更慢。在 CPU 上没有其他引擎有竞争力。

**AMD GPU** → vLLM 是支持最好的路径（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 锁定 NVIDIA，因此出局。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了它。

### 规模第二决策

**1 用户 / 本地开发** → Ollama。一条命令，首 token 只需几秒。

**10-100 用户 / 小型团队** → vLLM 单 GPU。

**100-1 万用户 / 生产** → vLLM 生产栈（Phase 17 · 18）或 SGLang。

**1 万+ 用户 / 企业** → vLLM 生产栈 + 分离式部署（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 负载第三决策

**通用聊天 / 问答** → vLLM 凭借广泛默认胜出。

**Agent 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占主导。

**前缀大量复用的 RAG** → SGLang。

**代码生成** → vLLM 也可以；SGLang 在缓存上略好。

**长上下文（128K+）** → vLLM + chunked prefill；SGLang + 分层 KV。

### TGI 的维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后只修 bug。历史上它拥有：顶级可观测性、一流的 HF 生态集成（模型卡、安全工具），原始吞吐量略逊于 vLLM。

对 2026 年的新项目：默认避开 TGI。现有 TGI 部署可以继续运行，但最终应迁移。SGLang 和 vLLM 是更稳妥的默认。

### Pipeline 模式

开发（Ollama）→ 预发（llama.cpp）→ 生产（vLLM）。引擎接受不同的权重格式——llama.cpp 家族用 GGUF，GPU 引擎用 HF safetensors——所以阶段之间可能需要一次格式转换。工程师在笔记本上快速迭代；预发镜像生产的量化配置；生产是服务目标。

### Ollama 的注意事项

Ollama 做开发很棒。做共享生产就不太行：Go HTTP 序列化带来额外开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在它擅长的场景用 Ollama——一个用户、一条命令——共享场景换 vLLM。

### 自托管还是托管是另一个决策

Phase 17 · 01（托管超大规模厂商）、· 02（推理平台）讲的是托管方案。本课假设你已经决定自托管。自托管的原因：数据驻留、自定义微调、大规模下的总拥有成本、托管端拿不到的领域模型。

### 应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产规模：400,000+ GPU。
- Ollama 相对 llama.cpp 的吞吐差距：慢 15-30%；生产负载下 3 倍。

```figure
data-parallel
```

## 使用它

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 负载，选出一个引擎并解释原因。

## 交付它

本课产出 `outputs/skill-engine-picker.md`。给定约束条件，选择引擎并写出迁移计划。

## 练习

1. 用你的硬件 / 规模 / 负载运行 `code/main.py`。输出符合你的直觉吗？
2. 你的基础设施是 12 块 H100 和 8 块 MI300X AMD。选哪个引擎？为什么 TRT-LLM 出局？
3. 某团队想用 TGI，因为「我们只熟悉它」。论证迁移的理由。
4. 从 Ollama 开发到 vLLM 生产：量化、配置和可观测性方面各有什么变化？
5. 一个 RAG 产品，P99 前缀长度 8K，租户间复用率高。选一个引擎，并结合 Phase 17 · 11 + 18 搭建栈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | 「那个 CPU 的」 | 模型支持最广，CPU 上最快 |
| Ollama | 「那个笔记本的」 | 一条命令安装，开发级吞吐 |
| TGI | 「HF 的服务框架」 | 自 2025 年 12 月起进入维护模式 |
| vLLM | 「默认选择」 | 2026 年广泛的生产基线 |
| SGLang | 「Agent 那个」 | 前缀密集，RadixAttention |
| TRT-LLM | 「锁定 NVIDIA」 | Blackwell 吞吐领先，仅限 NVIDIA |
| GGUF | 「llama.cpp 格式」 | 打包带 K-quant 变体 |
| Production-stack | 「vLLM K8s」 | Phase 17 · 18 的参考部署 |
| Pipeline pattern | 「dev→stage→prod」 | Ollama → llama.cpp → vLLM；权重格式随引擎而变 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) —— 发布说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
