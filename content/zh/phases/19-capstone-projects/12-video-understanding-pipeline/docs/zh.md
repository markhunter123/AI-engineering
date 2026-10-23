# 项目 12 —— 视频理解流水线（场景、问答、搜索）

> Twelve Labs 把 Marengo + Pegasus 做成了产品。VideoDB 发布了面向视频的 CRUD API。AI2 的 Molmo 2 公开了 VLM checkpoint。Gemini 的长上下文原生支持数小时视频。TimeLens-100K 在规模上定义了时间定位。2026 年的流水线已经定型：场景分割、逐场景字幕 + embedding、转写对齐、多向量索引，以及用（start, end）时间戳加帧预览来回答的查询。项目要摄取 100 小时视频，跑通公开基准，并测量计数类和动作类问题上的幻觉。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 hours

## 问题

长视频问答是 2026 年规模下最吃带宽的多模态问题。Gemini 2.5 Pro 能原生读完一段 2 小时视频，但要把 100 小时视频摄取成可查询的语料，仍然需要场景级索引。生产形态把这几件事拼在一起：场景分割（TransNetV2 或 PySceneDetect）、用 VLM 做逐场景字幕（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转写对齐（带词级时间戳的 Whisper-v3-turbo），以及一个把字幕、帧 embedding 和转写并排存放的多向量索引。查询流水线用（start, end）时间戳加帧预览来回答。

基准是公开的（ActivityNet-QA、NeXT-GQA），再加上你自己 100 条查询的自定义集。计数类和动作类问题上的幻觉是已知的困难失效类别；这个项目会专门测量它。

## 概念

摄取时三条流水线并行。**场景分割**把视频切成场景。**VLM 字幕**为每个场景生成一条字幕，并从关键帧生成一个帧 embedding。**ASR 对齐**产出词级时间戳。三条流按（scene_id, time range）连接。每个场景在多向量索引（Qdrant）里有三种向量：字幕 embedding、关键帧 embedding、转写 embedding。

查询时，自然语言问题同时打向这三种向量；结果用 RRF 合并；一个时间定位适配器（TimeLens 风格）细化 top 场景内的（start, end）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）拿到查询 + top 场景 + 裁剪后的帧，给出带引用时间戳和帧预览的回答。

幻觉的测量很重要。计数类（「有多少人进了房间？」）和动作类（「厨师是先倒还是先搅拌？」）问题出了名地不可靠。要把它们的准确率与描述类问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年的最先进方案）或 PySceneDetect
- ASR：经 faster-whisper 的 Whisper-v3-turbo，带词级时间戳
- VLM 字幕与回答：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant，支持多向量（字幕 / 帧 / 转写）
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 题人工标注集
- 幻觉基准：计数类和动作类子集，带人工标注

```figure
cf-scene-index
```

## 构建它

1. **摄取遍历器。** 接受 YouTube URL 或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 跑 TransNetV2 或 PySceneDetect，产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 阶段。** 对音频跑 Whisper-v3-turbo；导出词级时间戳；切成逐场景的转写片段。

4. **VLM 字幕。** 对每个场景，用关键帧和一个简短的字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出字幕 + 帧 embedding。

5. **多向量索引。** Qdrant 集合，三个命名向量。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三路稠密查询；用倒数排名融合合并；取 top-k=5 个场景。

7. **时间定位。** 在 top 场景上跑 TimeLens 风格适配器，细化该场景内的（start, end）窗口。

8. **VLM 合成。** 用查询 + top-3 场景片段（图像或短视频）+ 转写调用 Gemini 2.5 Pro。强制要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 跑 ActivityNet-QA 和 NeXT-GQA。构建 100 条查询的自定义集。报告总体准确率 + 逐类拆分（计数、动作、描述）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付它

`outputs/skill-video-qa.md` 是交付物。给定一个 YouTube URL 或上传的视频，流水线为场景建索引，并用带时间戳引用的方式回答问题。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 留出定位集上的交并比 |
| 20 | 问答准确率 | NeXT-GQA 和自定义 100 查询集 |
| 20 | 摄取吞吐量 | 每美元花费对应的视频小时数 |
| 20 | UI 与引用体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 分别统计计数类和动作类准确率 |
| **100** | | |

## 练习

1. 在字幕阶段把 Gemini 2.5 Pro 换成 Qwen3-VL-Max。在人工评分的 50 场景样本上报告字幕质量差值。

2. 把每场景的帧 embedding 从多向量缩减为一个池化向量。测量检索性能的回退。

3. 构建「严格计数」模式：合成器提取每一个被计数的实例及其时间戳，用户点击验证。测量用户验证是否降低了幻觉。

4. 基准摄取成本：三种 VLM 选择下每美元能处理多少小时视频。挑出最优区间。

5. 加说话人分离转写：对音频跑 pyannote 说话人分离，并为每个说话人的转写建立 embedding。演示「Alice 关于 X 说了什么？」这类查询。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | 「镜头检测」 | 在镜头边界处把视频切成场景 |
| Multi-vector index | 「字幕 + 帧 + 转写」 | 带命名向量的 Qdrant 集合，每种表示一个向量 |
| Temporal grounding | 「到底在什么时候发生」 | 为查询答案细化（start, end）窗口 |
| Frame embedding | 「视觉表示」 | 关键帧的向量 embedding；用于场景视觉相似度 |
| RRF fusion | 「倒数排名融合」 | 合并多个排序列表的策略；经典的混合检索技巧 |
| Counting hallucination | 「数错」 | VLM 在「有多少个 X」问题上的已知失效模式 |
| ActivityNet-QA | 「视频问答基准」 | 长视频问答准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) —— 开放的 VLM checkpoint
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) —— 规模化的时间定位
- [Gemini 视频长上下文](https://deepmind.google/technologies/gemini) —— 托管方案的参考
- [VideoDB](https://videodb.io) —— 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) —— 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) —— 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) —— 经典的开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) —— 评估基准参考
