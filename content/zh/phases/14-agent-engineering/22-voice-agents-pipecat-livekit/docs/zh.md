# 语音智能体：Pipecat 与 LiveKit

> 2026 年，语音智能体已是正式的生产类别。Pipecat 提供基于 Python 帧的 pipeline（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 把 AI 模型连接到用户。高端技术栈的生产延迟目标落在端到端 450–600ms。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## 学习目标

- 描述 Pipecat 的帧式 pipeline：DOWNSTREAM（source→sink）和 UPSTREAM（控制）。
- 说出标准语音 pipeline 的各阶段，以及 Pipecat 支持哪些传输方式。
- 解释 LiveKit Agents 的两个语音智能体类（MultimodalAgent、VoicePipelineAgent），以及各自适用的场景。
- 总结 2026 年的生产延迟预期，以及它们如何驱动架构选择。

## 问题

语音智能体不是把 TTS 硬接到文本循环上。延迟预算极其苛刻（约 600ms），部分音频是默认状态，轮次检测是一个模型，传输方式从电话 SIP 到 WebRTC 不等。你要么构建基于帧的 pipeline（Pipecat），要么依赖平台（LiveKit）。

## 概念

### Pipecat（pipecat-ai/pipecat）

- Python 帧式 pipeline 框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — source → sink（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并有用于指标、追踪和 RTVI 的 observer。

典型 pipeline：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 把 AI 模型连接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两个语音智能体类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或同类服务直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本层控制。
- 通过 Transformer 模型做语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- 通过 LiveKit Inference 提供 50+ 个免 API key 的模型；通过插件再提供 200+ 个。

### 商业平台

Vapi（在优化的高端技术栈上约 450–600ms）和 Retell（在 180 次测试通话中端到端约 600ms）构建在这些之上。如果你想要托管语音技术栈，又没有 WebRTC 团队，就选平台。

### 这种模式容易出错的地方

- **没有打断处理。** 用户插话，智能体还在说。需要 Pipecat 的 UPSTREAM cancel 帧，LiveKit 中也有等价机制。
- **忽略 STT 置信度。** 低置信度转写被当作真理喂给 LLM。应基于置信度设门槛或请求确认。
- **TTS 句中截断。** pipeline 在语句中途取消时，TTS 需要知道并截断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。发布前把整条链加起来。

### 2026 年的典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首 token：150–400ms
- TTS 首个音频：100–200ms
- 传输 RTT：30–80ms

端到端 450–600ms 属于高端。800–1200ms 很常见。超过 1500ms 就让人觉得坏了。

```figure
voice-pipeline
```

## 构建它

`code/main.py` 是一个基于帧的玩具 pipeline，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)`。
- 一个五阶段 pipeline（VAD → STT → LLM → TTS → transport），由脚本化的 processor 组成。
- 一个 UPSTREAM cancel 帧，演示打断。

运行它：

```
python3 code/main.py
```

trace 展示正常流程，以及一次在语句中途中止 TTS 的打断取消。

## 使用它

- **Pipecat** 用于完全控制——自定义 processor、Python 优先、可插拔 provider。
- **LiveKit Agents** 用于 WebRTC 优先的部署和电话。
- **Vapi / Retell** 用于没有 WebRTC 团队的托管语音智能体。
- **OpenAI Realtime / Gemini Live** 用于直接的音频输入/音频输出（MultimodalAgent）。

## 交付它

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 风格的语音 pipeline，包含 VAD + STT + LLM + TTS + 传输，以及打断处理。

## 练习

1. 给你的玩具 pipeline 增加一个指标 observer：统计每个阶段每秒的帧数。延迟在哪里累积？
2. 实现置信度门槛的 STT：低于阈值时，请求「能再说一遍吗？」
3. 增加语义轮次检测：简单规则——如果转写以「?」结尾，就结束轮次。
4. 阅读 Pipecat 的传输文档。把标准库传输换成 SmallWebRTCTransport 配置（桩）。
5. 在同一查询上比较 OpenAI Realtime 与 STT+LLM+TTS 级联。文本层控制带来多少延迟成本？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Frame | 「事件」 | pipeline 中带类型的数据单元（audio、transcript、text、control） |
| Processor | 「pipeline 阶段」 | 带 process(frame) 的处理器 |
| DOWNSTREAM | 「正向流」 | 从 source 到 sink：音频输入，语音输出 |
| UPSTREAM | 「反馈流」 | 控制：取消、指标、打断 |
| VAD | 「语音活动检测」 | 检测用户何时在说话 |
| 语义轮次检测 | 「智能结束轮次」 | 基于模型判断用户说完了 |
| MultimodalAgent | 「直接音频智能体」 | 音频进、音频出；中间没有文本 |
| VoicePipelineAgent | 「级联智能体」 | STT + LLM + TTS；文本层控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 帧式 pipeline、processor、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，带延迟基准
