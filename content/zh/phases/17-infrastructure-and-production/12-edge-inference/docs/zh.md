# 边缘推理——Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘的核心约束是内存带宽，而不是算力。移动端 DRAM 处于 50-90 GB/s；数据中心 HBM3 超过 2-3 TB/s——30-50 倍差距。decode 是内存受限的，所以这个差距是决定性的。2026 年的格局分为四块。Apple M4/A18 Neural Engine 峰值 38 TOPS，采用统一内存（无 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B（Q4）（约为原生性能的 70-80%）；17.6k GitHub star，OpenAI 兼容 API，移动端覆盖约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可跑 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块 prefill——Bosch、ThunderSoft、MediaTek 在 CES 2026 上做了展示。

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，算力是次要的。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并为每个匹配一个用例。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶），以及 Safari iOS 26 的落地情况。
- 为每个目标选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

一位客户想要设备端聊天机器人：语音优先、默认私密、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 约 55 tok/s——没问题。在 iPhone 16 Pro 上，同一模型只有 3 tok/s——不行。在一台搭载 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。在 Chrome Android v121+ 的浏览器里通过 WebGPU，视设备不同为 4-8 tok/s。

吞吐量的差异不是移植问题。它是带宽差距乘以量化格式，再乘以 NPU 是否可以从用户态访问。2026 年的边缘推理是四个不同的问题，各有四种不同的解法。

## 概念

### 带宽才是真正的天花板

decode 每生成一个 token 都要读取全部权重。一个 Q4 的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms——理论上限约 14 tok/s。在 90 GB/s（高端移动 DRAM）下，上限变为约 25 tok/s。低于这个数字，再多算力也帮不上忙。

数据中心 HBM3 以 3 TB/s 读取同样的 3.5 GB 只需 1.2 ms——上限是 830 tok/s。同样的模型，同样的权重，不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 与 ANE 共享同一内存池）——没有拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 走 Metal Performance Shaders（MPS）。
- llama.cpp 的 Metal 后端使用 MPS，而不是直接使用 ANE；要原生使用 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用最实用的路径：Core ML + INT4 权重 + FP16 激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。在 SoC 内与 CPU、GPU 集成，但使用独立的内存域。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 转换的能力。
- Chat template、Llama 3.2、Phi-3 都在 AI Hub 上作为一等制品提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在进步，但仍属小众。
- 最适合 Windows ARM 上的 copilot 应用；在 AMD/Intel 桌面上适合本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU compute shader 在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——同一后端下约为原生性能的 70-80%。
- WebLLM 有 17.6k GitHub star；OpenAI 兼容的 JS API；Apache 2.0。
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 正式版、Firefox Android 仍在追赶。移动端整体覆盖约 70-75%。

### NVIDIA Jetson 家族

- Orin Nano Super（8GB）：可跑 Llama 3.2 3B、Phi-3，tok/s 表现良好。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：性能是 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、分块 prefill——把数据中心的优化移植到了边缘。

### 各目标的量化选择

| 目标 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译好的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV cache + 操作系统开销 = OOM。除非接受激进的 KV 量化（Q4 KV），边缘部署应把上下文保持在 4K-8K。

### 语音是杀手级应用

语音 Agent 对延迟敏感（首个 token < 500 ms）。本地推理彻底消除了网络延迟。再结合语音转文字（Whisper Turbo 变体可在边缘运行），边缘推理就成了生产级质量的语音闭环。

### 应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 上约 41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- 数据中心与边缘的带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 滞后）。

```figure
edge-bandwidth-pipe
```

## 使用它

`code/main.py` 用带宽受限的算术计算各边缘目标的理论解码吞吐上限。它把结果与实测基准对比，并突出显示瓶颈在带宽而非算力的地方。

## 交付它

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，它选择量化格式和转换 pipeline。

## 练习

1. 运行 `code/main.py`。对 Snapdragon 8 Gen 3（带宽约 77 GB/s）上 Q4 的 7B 模型，计算解码上限。与实测 6-8 tok/s 对比——运行时高效吗？
2. Android 上的 WebGPU 要求 Chrome v121+。为旧浏览器设计降级方案——通过同样的 OpenAI 兼容 API 走服务端。
3. 你的 iOS 应用需要 4K 上下文的流式输出。哪种模型/格式组合能让 iPhone 16 上的活跃内存保持在 4 GB 以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能放下 3B。如果你的产品同时面向两者，如何统一推理栈？
5. 论证「WebLLM 在 2026 年是否已可用于生产」。引用覆盖率、性能以及 Firefox Android 的缺口。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ANE | 「Apple 神经引擎」 | M 系列和 A 系列中的端侧 NPU；统一内存 |
| Hexagon | 「Qualcomm NPU」 | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | 「浏览器 GPU」 | W3C 标准化的浏览器 GPU API；2026 年 Chrome/Safari |
| WebLLM | 「浏览器 LLM 运行时」 | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | 「NVIDIA 边缘」 | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | 「边缘 TensorRT」 | TensorRT-LLM 的 2026 边缘移植；EAGLE-3 + NVFP4 |
| 统一内存 | 「共享内存池」 | CPU 和 NPU 看到同一块 RAM；没有拷贝开销 |
| 带宽受限 | 「内存受限」 | decode 受限于读取权重的字节/秒速率 |
| Core ML | 「Apple 转换」 | Apple 用于 ANE 原生模型的框架 |
| QNN | 「Qualcomm 栈」 | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/)——格局与基准。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/)——Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)——2026 年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2)——设计与基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml)——ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)——为 Hexagon 预转换的模型。
