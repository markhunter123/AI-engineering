# 水印——SynthID、Stable Signature、C2PA

> 2026 年 AI 生成内容的来源追溯由三项技术构成。SynthID（Google DeepMind）——图像水印于 2023 年 8 月推出，2024 年 5 月扩展到文本和视频（Gemini + Veo），2024 年 10 月通过 Responsible GenAI Toolkit 开源文本水印，2025 年 11 月随 Gemini 3 Pro 推出统一的多媒体检测器。文本水印以不可察觉的方式调整下一 token 的采样概率；图像/视频水印能经受压缩、裁剪、滤镜和帧率变化。Stable Signature（Fernandez 等人，ICCV 2023，arXiv:2303.15435）——微调潜空间扩散解码器，使每个输出都包含一条固定消息；被裁剪（保留 10% 内容）的生成图像在 FPR<1e-6 下检出率超过 90%。后续工作《Stable Signature is Unstable》（arXiv:2405.07145，2024 年 5 月）——微调可以移除水印，同时保持质量。C2PA——密码学签名、防篡改的元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 互补：元数据可以被剥离，但携带更丰富的来源信息；水印能在转码后存活，但携带的信息更少。

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature，以及 2024 年击破它的移除攻击。
- 说明 C2PA 的作用，以及它为何与水印互补。
- 描述关键局限：模型特定的信号、改写下的鲁棒性，以及语义保持攻击（arXiv:2508.20228）。

## 问题

2023-2024 年，深度伪造和 AI 生成内容大规模进入政治与消费场景。水印是被提出的技术来源信号：在生成时标记，之后进行检测。2025 年的证据：没有一种水印是无条件鲁棒的，但与 C2PA 元数据叠加后，组合提供了可用的来源追溯方案。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人 2023 的机制，由 Google 产品化：

1. 在每个解码步骤，对前 K 个 token 做哈希，把词表伪随机地划分为「绿色」和「红色」集合。
2. 通过给绿色集合的 logit 加上 δ，使采样偏向绿色集合。
3. 生成结果中绿色 token 的数量高于随机情况。

检测：对每个前缀重新哈希，统计生成中的绿色 token，计算 z 分数。水印文本的 z 分数 >0，人类文本约为 0。

性质：
- 读者无法察觉（δ 足够小，质量损失轻微）。
- 在能访问词表划分函数时可检测。
- 对改写不鲁棒——重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等人 ICCV 2023。微调潜空间扩散解码器，使每张生成的图像都在潜表示中包含一条固定的二进制消息。检测时用神经解码器从潜表示中解码。被裁剪到 10% 内容的图像在 FPR<1e-6 下检出率超过 90%。

2024 年 5 月的《Stable Signature is Unstable》（arXiv:2405.07145）：微调解码器可以移除水印，同时保持图像质量。生成后的对抗性微调成本很低；水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

与 Gemini 3 Pro 一同推出：一个多媒体检测器，可在单个 API 中读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的来源追溯技术栈。

### C2PA

内容来源与真实性联盟（Coalition for Content Provenance and Authenticity）。密码学签名、防篡改的元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录来源声明（谁创建、何时创建、经历了哪些变换），并由创建者的密钥签名。

与水印互补：
- 元数据可以被剥离；水印不能（轻易地）。
- 元数据信息丰富（完整来源链）；水印只携带若干比特。
- C2PA 依赖平台采用；水印自动嵌入。

Google 在搜索、广告和「关于此图片」中集成了两者。

### 局限

- **模型特定。** SynthID 只给启用 SynthID 的模型生成的输出加水印。没有 SynthID 的模型生成的输出不带水印，所以「没有 SynthID 信号」不是真实性的证明。
- **改写。** 文本水印无法在保持语义的改写后存活。
- **变换攻击。** arXiv:2508.20228（2025）展示了能同时破坏文本水印和许多图像水印的语义保持攻击（meaning-preserving attacks）。
- **微调移除。** 根据《Stable Signature is Unstable》，生成后的微调会移除嵌入的水印。

### 欧盟《人工智能法案》第 50 条

针对 AI 生成内容标注的《透明度行为准则》（第一版草案 2025 年 12 月，第二版草案 2026 年 3 月，预计 2026 年 6 月定稿，见[欧洲委员会状态页](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至 2026 年 4 月，该准则仍为草案，时间表可能变化。这是要求技术层落地的监管层。深度伪造必须标注。

### 这在第 18 阶段中的位置

第 22-23 课讲模型输出的东西（私有数据、来源信号）。第 27 课涵盖训练数据治理。第 24 课是要求这些技术措施的监管框架。

```figure
an-watermark-greenlist
```

## 使用它

`code/main.py` 构建一个玩具文本水印。token 是整数 0..N-1；加水印的采样偏向由哈希定义的绿色集合。检测器计算绿色 token 的 z 分数。你可以观察 1000 token 生成下的检测、看重写如何破坏信号，并测量人类文本上的假阳性率。

## 交付它

本课产出 `outputs/skill-provenance-audit.md`。给定一项带来源声明的部署，它审计：水印机制（如有）、C2PA 签名链（如有）、各自的对抗鲁棒性，以及各模态的覆盖情况。

## 练习

1. 运行 `code/main.py`。报告加水印的 1000 token 生成与人类撰写文本的 z 分数。指出 95% 置信阈值下的假阳性率。

2. 实现一个把 30% 的 token 替换为同义词的改写攻击。重新测量 z 分数。

3. 阅读 Kirchenbauer 等人 2023 年第 6 节关于鲁棒性的内容。为什么文本水印在改写后会失效，而图像水印能在裁剪后存活？

4. 设计一个使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的来源链。指出每个组件的一个失效模式。

5. 2024 年《Stable Signature is Unstable》的结果表明微调会移除图像水印。设计一个限制这种攻击的部署控制——例如要求微调后的 checkpoint 以签名方式发布。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|-----------------|------------------------|
| SynthID | 「Google 的水印」 | 跨模态来源信号；文本、图像、音频、视频 |
| Token 水印 | 「Kirchenbauer 式」 | 通过绿色 token z 分数检测的偏置采样文本水印 |
| Stable Signature | 「图像水印」 | 微调解码器水印；ICCV 2023 |
| C2PA | 「元数据标准」 | 密码学签名、防篡改的来源元数据 |
| 改写鲁棒性 | 「换种说法会破坏它吗」 | 文本水印的性质；目前有限 |
| 微调移除 | 「对抗性去水印」 | 通过解码器微调移除图像水印的攻击 |
| 跨模态检测器 | 「统一 SynthID」 | 2025 年 11 月跨模态的统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
