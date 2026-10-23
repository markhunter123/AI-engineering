# 项目 03 — 实时语音助手（ASR → LLM → TTS）

> 一个体验良好的语音 Agent，端到端延迟低于 800ms，知道你是否说完了，能处理打断（barge-in），还能在不卡顿的情况下调用工具。2026 年，Retell、Vapi、LiveKit Agents 和 Pipecat 都达到了这条线。它们的形态相同：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 串联，每一跳都有严格的延迟预算。构建一个，测量 WER、MOS 和误截断率，并在丢包条件下运行它。

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## 问题

2025–2026 年，语音是移动最快的 AI 用户体验类别。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让低于 800ms 的首音频输出触手可及。门槛不只是延迟，而是交互手感：不打断用户、不被用户打断、能从句子中途的插话中恢复、在对话中途调用工具而不让音频卡住、在不稳定的移动网络上存活。

靠拼接三个 REST 调用是做不到的。架构必须端到端流式 pipeline 化。动手构建后，失败模式就会显现：为电话音频调优的 VAD 被背景电视声触发、轮次检测器在等一个永远不会出现的标点、TTS 在发声前缓冲 400ms。这个项目要求你在负载下逐个修复这些问题，并发布一份延迟与质量报告。

## 概念

pipeline 有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转写）、**轮次检测**（VAD 加上一个小型轮次检测模型，读取部分转写寻找句末线索）、**LLM**（一旦判定轮次结束就开始流式输出 token）、**TTS**（在收到第一个 LLM token 后约 200ms 内开始流式输出音频）。

三个横切关注点。**Barge-in（打断）**：当 Agent 正在说话时用户开口，TTS 取消，ASR 立即接管。**工具使用**：对话中途的 function calling（天气、日历）必须在旁路信道上运行，不能阻塞音频；如果延迟超过 300ms，Agent 先发一个确认语（「稍等一秒……」）。**背压**：丢包时暂存部分转写，VAD 提高语音门限，Agent 避免在消息尚未确认时开口。

衡量门槛是量化的。在 15 dB SNR 的 Hamming VAD 基准上 WER 低于 8%。100 通实测通话的首音频输出 p50 低于 800ms。误截断率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 上 50 路并发通话。这些数字就是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首个部分转写低于 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转写的小型 Transformer）
- LLM：紧密集成用 OpenAI GPT-4o-realtime，或用 Gemini 2.5 Flash Live，或用级联的 Claude Haiku 4.5（流式补全，音频路径独立）
- TTS：Cartesia Sonic-2（首字节最低）、ElevenLabs Flash v3，或自托管开源方案 Orpheus
- 工具：天气/日历/预订用 FastMCP 旁路信道；工具耗时超过 300ms 时 Agent 先发填充语
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音 trace 支持音频回放
- 部署：自托管 Whisper + Orpheus 用单台 g5.xlarge（24GB 显存）；追求最低延迟就用托管 API

```figure
ce-voice-latency
```

## 构建它

1. **WebRTC 会话。** 起一个 LiveKit room 和一个流式传输麦克风音频的 Web 客户端。在服务端挂一个加入该 room 的 Agent worker。

2. **ASR 流式处理。** 把 20ms 的 PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分转写和最终转写。记录每个部分转写的延迟。

3. **VAD 与轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件上，用最新的部分转写触发 LiveKit 轮次检测器。只有当 VAD 判定静音 500ms 且轮次检测器的完成度评分大于 0.6 时，才确认「轮次结束」。

4. **LLM 流。** 轮次结束时，带上进行中的对话和最终转写发起 LLM 调用。流式输出 token。在第一个 token 到达时交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。第一个音频块必须在首个 LLM token 之后 200ms 内离开服务端。把音频块发到 LiveKit room；客户端通过 WebRTC 抖动缓冲播放。

6. **Barge-in。** 当 TTS 播放时 VAD 检测到新的用户语音，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启用 ASR。发布一个 `tts_canceled` span。

7. **工具旁路信道。** 把天气和日历注册为 function calling 工具。被调用时并发发起请求；如果 300ms 内没有结果，让 LLM 输出「稍等，我查一下」作为填充语；工具返回后继续。

8. **评测 harness。** 录制 100 通通话。计算 WER（对照留出转写）、误截断率（用户在句子中途时 TTS 被取消）、首音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 用合成呼叫方在单台 g5.xlarge 上驱动 50 路并发通话。测量持续负载下的首音频输出 p95。

## 使用它

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付它

交付物是 `outputs/skill-voice-agent.md`。给定一个领域（客户支持、日程安排或自助终端），它会按衡量门槛调好 ASR/VAD/LLM/TTS pipeline，搭起一个 LiveKit Agent。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 通录制通话的首音频输出 p50 低于 800ms |
| 20 | 轮次转换质量 | Hamming VAD 基准上误截断率低于 3% |
| 20 | 工具使用正确性 | 对话中途的 tool call 返回正确数据且不阻塞音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时的 WER 和轮次转换稳定性 |
| 15 | 评测 harness 完备性 | 配置公开、测量可复现 |
| **100** | | |

## 练习

1. 在 g5.xlarge 上把 Deepgram Nova-3 换成 faster-whisper v3 turbo。测量延迟和 WER 差距。找出 CPU 与 GPU 取舍真正关键的地方。

2. 增加一个插话仲裁策略：当用户在 tool call 期间插话时，Agent 该怎么做？比较三种策略（硬取消、先完成工具再停止、排队到下一轮）。

3. 运行对抗性轮次检测测试：让用户在句子中途长时间停顿。调整 VAD 静音阈值和轮次检测器评分阈值，在不超过 900ms 的前提下把误截断率降到最低。

4. 通过 Twilio 把同一个 Agent 部署到 PSTN。对比 PSTN 与 WebRTC 的首音频输出。解释抖动缓冲和编解码器上的差异。

5. 为非英语语言（日语、西班牙语）增加语音活动检测。测量 Silero VAD v5 与语言特定微调模型的误触发率对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 轮次检测 | 「话语结束」 | 在给定 VAD 静音和部分转写的条件下，判断用户是否说完的分类器 |
| Barge-in（打断） | 「插话处理」 | VAD 检测到新的用户语音时，在播放中途取消 TTS |
| 首音频输出 | 「延迟」 | 从用户停止说话到第一个音频包离开服务端的耗时 |
| VAD | 「语音门」 | 把音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲 | 「音频平滑」 | 客户端短暂缓存数据包以吸收网络波动的缓冲 |
| 填充语 | 「确认语」 | 工具较慢时 Agent 发出的短语，用来避免冷场 |
| MOS | 「平均意见分」 | 语音感知质量评分；NISQA 是自动化替代指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) —— 参考 WebRTC Agent 框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) —— 备选的 Python 优先流式 Agent 框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) —— 集成式语音模型的参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) —— 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) —— VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) —— 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) —— 生产级语音 Agent 架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) —— 另一个生产参考
