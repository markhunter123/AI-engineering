# Qwen-VL 家族与动态 FPS 视频

> Qwen-VL 家族——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最具影响力的开放视觉语言模型谱系。每一代都做出一个决定性的架构押注，开放生态的其余部分在十二个月内跟进：通过 M-RoPE 的原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT 中的窗口注意力，以及结构化的 Agent 输出格式。到 Qwen3-VL，配方已经稳定：接受原生宽高比输入的 2D-RoPE-ViT 编码器、接入大型 Qwen3 语言基座的 MLP 投影器，以及把 OCR、定位和 Agent 行为当作一等目标的训练阶段。本课按时间顺序解读这个家族，让你理解每个旋钮为什么处在现在的位置。

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度），并解释为什么三者都需要。
- 为视频选择动态 FPS 采样策略，并权衡每秒 token 数与事件检测精度。
- 按顺序说出 Qwen-VL 的四代升级，以及每一代启用了什么。
- 接上 Qwen2.5-VL 风格的 JSON Agent 输出格式，并从 VLM 响应中解析结构化 tool call。

## 问题

Qwen-VL 于 2023 年 8 月发布，直接回应 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三方面：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 跑在 336x336。对照片够用，对中文发票或密集的电子表格截图毫无用处。Qwen-VL 的第一个创新是 448x448 和带定位的边界框输出，让模型能指向物体。

视频：Video-LLaMA 堆叠逐帧编码器再喂给 LLM。对短片段有效，但对时间轴才是信号的多分钟视频不行。Qwen 团队想要一个理解时间的单一编码器。

结构化输出：LLaVA 输出自由文本。Agent 需要 JSON。Qwen-VL 在显式 JSON 输出格式上训练，包括把边界框坐标当作文本输出。

每一代 Qwen-VL 都扩展这三条轴中的一条。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数），兼容 Llama 的 Q-Former（单步、256 个 query），Qwen-7B 基座。贡献：

- 448x448 分辨率（当时开放 VLM 的 SOTA）。
- 定位：在图文对上训练，显式输出坐标 token。「The cat is at <box>(112, 204), (280, 344)</box>」。
- 从一开始就做中英双语训练。

当时的基准：英文上与 GPT-4V 有竞争力，中文上占优。定位监督才是真正的头条。

### Qwen2-VL（2024 年 9 月）——M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器取代了固定分辨率 + Q-Former 的堆栈。关键变化：

- 原生动态分辨率。ViT 接受任何能被 28 整除的 HxW（patch 14，2x 空间合并）。1120x672 的图像（40x24 个合并 patch）产生 960 个视觉 token。不缩放、不切块、不要缩略图。
- M-RoPE（多模态 RoPE）。每个 token 携带 3D 位置 (t, h, w)，而不是 1D。图像 t=0，视频 t = 帧索引。RoPE 按每条轴的频率旋转 query/key 向量。没有位置嵌入表。
- MLP 投影器。去掉 Q-Former；对合并后的 patch token 用 2 层 MLP。
- 动态 FPS 视频。默认以 1-2 FPS 采样视频，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多项多模态基准上追平 GPT-4o，并在 DocVQA 上超过它（94.5 vs 88.4）。架构变化是决定性的举措。

### Qwen2.5-VL（2025 年 2 月）——动态 FPS + 绝对时间

Qwen2.5-VL 的大转变在视频。动态 FPS 不只是「需要时多采几帧」。论文形式化了：

- 绝对时间 token。不用位置索引（第 0、1、2 帧……），而用真实时间戳。「在 0:04，猫跳了起来。」模型看到 `<time>0.04</time>` token 与帧 token 交错。
- 动态 FPS。慢镜头以 1 FPS 采样，动作场景 4+ FPS。用户或训练者选择；M-RoPE 自适应。
- ViT 中的窗口注意力。空间注意力在块内局部进行以提升吞吐量；每隔几层加一次全局注意力。
- 显式 JSON 输出格式。在 tool call 数据上训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱即可用于 Agent。
- MRoPE-v2 缩放。位置随最大输入尺寸缩放，10 分钟视频不会耗尽频率范围。

基准：Qwen2.5-VL-72B 在多数视频基准上击败 GPT-4o，在文档上追平 Gemini 2.0，并创下开放模型的 GUI 定位 SOTA（ScreenSpot：84% 准确率，GPT-4o 为 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是增量升级，重在整合而非重新发明：更大的 LLM 主干（Qwen3-72B）、扩展的训练数据、改进的 OCR、通过 Qwen3「思考模式」获得更强推理。ViT 和 M-RoPE 保持不变。论文的重点是数据和训练改进，而不是架构。

这条谱系的要点：到 2025 年，Qwen-VL 架构已经稳定。后续版本扩展的是算力和数据，而不是基本原语。

### M-RoPE 的数学

经典 RoPE 用成对坐标，把维度为 `d` 的 query `q` 按位置 `m` 旋转：

```math
\begin{aligned}
q_{\text{rot}}[2i] &= q[2i] \cos(m \theta_i) - q[2i+1] \sin(m \theta_i) \\
q_{\text{rot}}[2i+1] &= q[2i] \sin(m \theta_i) + q[2i+1] \cos(m \theta_i) \\
\theta_i &= 10000^{-2i/d}
\end{aligned}
```

M-RoPE 把隐藏维度分成三个带。假设 `d = 96`。32 维给时间，32 维给高度，32 维给宽度。每个带按自己的轴位置旋转。位于 (t=5, h=10, w=20) 的 patch，其三个带分别施加 `R_t(5)`、`R_h(10)`、`R_w(20)` 旋转。

文本 token 用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧用 `t = frame_time, h = row, w = col`。单图用 `t = 0`。

好处：一个位置编码就处理文本、图像和视频，不需要分支代码或不同的位置表。

### 动态 FPS 采样逻辑

给定时长为 `T` 秒的视频和目标 token 预算 `B`：

1. 计算能承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选一个满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动剧烈（光流启发式或用户显式要求），选更高 FPS；运动平缓则选更低。
4. 以所选 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练这套逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列以 4 FPS、每帧 81 个 token 计 = 19440 个 token，32k 上下文可以应付。

### 结构化 Agent 输出

Qwen2.5-VL 的 Agent 训练明确以结构化 tool call 为目标：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比自由格式的「click at (1024, 512)」，后者需要正则和歧义处理。这一转变正是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跃升到 84% 的原因。

```figure
mm-mrope-axes
```

## 使用它

`code/main.py` 实现：

- 为混合文本、图像 patch 和视频帧的打包序列计算 M-RoPE 位置。
- 动态 FPS 采样器：给定 (duration, budget, motion_level)，选择 FPS 并输出帧时间戳。
- 一个玩具版 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的 tool call 响应。

运行它，然后感受一下在 5 分钟视频上把固定 FPS 换成动态 FPS 的差别。

## 交付它

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定视频任务（监控、Agent、动作识别、无障碍），它输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力开关、Agent 输出模式）和延迟估计。每当你为视频产品部署 Qwen-VL 家族模型时都可以用它。

## 练习

1. 计算隐藏维度为 48（每带 16 维，base theta 10000）时，(t=3, h=5, w=7) 处 patch 的 M-RoPE 旋转。给出每个带前三对的旋转角。

2. 10 分钟安防摄像头录像以 1 FPS 产生多少帧？384 分辨率、3x 池化下总 token 数是多少？Qwen2.5-VL 默认的 32k 上下文能处理吗？

3. 为 30 秒网球回合、30 秒菜谱演示和 30 秒 UI Agent 录屏分别选择 FPS。用动态 FPS 逻辑为每个选择辩护。

4. Qwen2.5-VL 完全去掉了 Q-Former。为什么简单的 MLP 在 2025 年可行，而 2023 年不行？（提示：数据规模和编码器质量。）

5. 把三个 Qwen2.5-VL JSON tool call 输出解析为 Python dict。畸形 JSON 会怎样失败，Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| M-RoPE | 「多模态 RoPE」 | 隐藏维度中带时间、高度、宽度三个带的 3D 旋转位置嵌入 |
| 动态 FPS | 「智能采样」 | 根据运动、时长和 token 预算为每个视频选择帧采样率 |
| 绝对时间 token | 「时间戳 token」 | `<time>t</time>` 交错在序列中，让模型看到真实秒数而非帧索引 |
| 窗口注意力 | 「局部注意力」 | 为提速把空间自注意力限制在小窗口内；周期性加入全局注意力 |
| 结构化 Agent 输出 | 「JSON 模式」 | 训练数据监督，教 VLM 输出带坐标和工具名的可解析 JSON |
| min_pixels / max_pixels | 「分辨率边界」 | Qwen2.5-VL 的每请求控制项，限制总像素数，从而限制 token 数 |
| 定位（Grounding） | 「指出来」 | 把边界框坐标作为文本 token 输出；从 Qwen-VL v1 起就在用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
