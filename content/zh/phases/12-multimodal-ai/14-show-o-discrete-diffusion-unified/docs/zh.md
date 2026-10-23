# Show-o 与离散扩散统一模型

> Transfusion 混合了连续表示与离散表示。Show-o（Xie 等，2024 年 8 月）走的是相反的方向：文本 token 用因果下一 token 预测，图像 token 用 MaskGIT 风格的掩码离散扩散。两者共处同一个 Transformer，靠混合注意力掩码区分。结果是在一个主干、每种模态一个 tokenizer、一种损失形式（把下一 token 预测扩展到掩码预测）上，统一了 VQA、文生图、图像修复和混合模态生成。本课拆解 Show-o 的设计——为什么掩码离散扩散是一种并行、少步数的图像生成器——并与 Transfusion、Emu3 对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释掩码离散扩散：先按均匀比例遮住 token，再让 Transformer 恢复它们的调度。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 用一个 checkpoint 处理的三个任务：T2I、VQA、图像修复。
- 选择一种掩码调度（余弦、线性、截断），并推理它对样本质量的影响。

## 问题

Transfusion 的双损失训练可行，但动态更棘手——连续扩散损失与离散 NTP 损失的数值尺度不同。平衡损失权重本身就是一次超参数搜索。这套架构有效，但复杂。

Show-o 的答案：让两种模态都保持离散（像 Chameleon），但用掩码离散扩散并行生成图像，而不是顺序生成。训练目标变成单一的掩码 token 预测，它自然地推广了下一 token 预测。

## 概念

### 掩码离散扩散（MaskGIT）

Chang 等（2022）最初的 MaskGIT 技巧很优雅。从一张完全被遮住的图像开始（每个 token 都是特殊的 `<MASK>` id）。每一步并行预测所有被遮住的 token，然后保留置信度最高的 top-K 个预测，把其余重新遮住。经过约 8-16 次迭代，所有 token 都被填上。每一步解开多少 token 的调度需要调——余弦调度效果很好。

训练很简单：从 [0, 1] 均匀采样一个掩码比例，作用于图像的 VQ token，训练 Transformer 恢复被遮住的部分。这正是 BERT 在文本上做的事，只不过扩展到了图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 把 MaskGIT 放进一个因果语言模型 Transformer 里。注意力掩码是：

- 文本 token：因果（标准 LLM）。
- 图像 token：在图像块内部完全双向（这样被遮住的 token 在预测时能看到其他所有图像 token）。
- 文生图：文本注意先前的图像，图像注意先前的文本。

训练在以下任务之间交替：

1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，图像 token 被遮住，用掩码 token 预测损失。
3. VQA 样本：图像 → 文本，文本 token 被遮住（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的交叉熵，它同时覆盖文本 NTP（只有最后一个 token 算「被遮住」）和图像掩码扩散（随机子集被遮住）。

### 并行采样

Show-o 约 16 步生成一张图像，而不是约 1000 步（逐 token 自回归）或约 20 步（扩散）。每一步并行预测所有被遮住的 token；确定置信度最高的 top-K 个；重复。

对比：

- Chameleon / Emu3（token 上的自回归）：需要 N_tokens 次前向传播，每张图通常 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整的 Transformer 前向。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整的 Transformer 前向。

在相近规模的模型上，Show-o 比 Chameleon 快，步数与 Transfusion 大致相当，但每步成本更低（离散词表 logits 对比连续 MSE 损失）。

### 一个 checkpoint 里的多个任务

Show-o 在推理时支持四个任务，由 prompt 格式选择：

- 文本生成：标准的自回归文本输出。
- VQA：图像进，文本出。
- T2I：文本进，经掩码离散扩散输出图像。
- 图像修复：输入部分 token 被遮住的图像，补全缺失部分。

图像修复能力来自掩码预测训练，是免费的副产品。遮住 VQ token 网格的一块区域，把其余部分和文本 prompt 一起输入，预测被遮住的 token。

### 掩码调度

每一步解开多少 token 的调度决定了质量。Show-o 推荐余弦：

```math
\text{mask\_ratio}(t) = \cos\left(\frac{\pi t}{2T}\right) \qquad t = 0, \ldots, T
```

第 0 步时所有 token 都被遮住（比例 1.0）。第 $T$ 步时没有 token 被遮住。余弦把质量集中在预测信息量最大的中段比例上。线性调度也能用，但更早进入平台期。

### Show-o2

Show-o2（2025 年的后续工作，arXiv 2506.15564）扩展了 Show-o：更大的 LLM 底座、更好的 tokenizer、改进的掩码调度。架构模式相同。

### Show-o 的定位

在 2026 年的分类中：

- 离散 token + NTP：Chameleon、Emu3。简单，但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，但仍受 tokenizer 有损压缩的限制。
- 连续 + 扩散：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- VLM 中的连续 + flow matching：JanusFlow、InternVL-U。最新。

按任务选择：想要一个开放模型同时提供 T2I、图像修复和 VQA，且速度合理，选 Show-o；质量至上、且能承受双损失管线，选 Transfusion。

```figure
masked-diffusion-unmask
```

## 使用它

`code/main.py` 模拟 Show-o 的采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个模拟的「Transformer」，根据 prompt 和当前未遮住的 token 预测 logits。
- 在余弦调度下运行 8 步并行掩码采样。
- 打印中间状态（掩码图案的演化）和最终 token。

跑一下，看着掩码一步步溶解。

## 交付它

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个同时需要理解（VQA、图像描述）和生成（T2I、图像修复）且受开放权重约束的产品，它在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3 / Chameleon 家族之间做选择，并给出具体权衡。

## 练习

1. 掩码离散扩散约 16 步完成采样。为什么不能是 1 步？如果在第 0 步解开所有 token，会出什么问题？

2. 掩码扩散让图像修复成为免费能力。提出一个产品用例（真实或假设），说明 Show-o 的修复优于专用模型。

3. 余弦调度与线性调度：对 T=8 逐步追踪解开的 token 数量。哪一个更均衡？

4. 一张 512x512 的 Show-o 图像是 1024 个 token。在词表 $K=16384$ 下，模型输出 $1024 \cdot \log_2(16384) = 14{,}336$ 位数据（约 1.75 KiB）。Stable Diffusion 输出 512*512*24 位 = 6,291,456 位（约 768 KiB）的原始像素。压缩比是多少，这个压缩比换来了什么质量？

5. 读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 掩码离散扩散 | 「MaskGIT 式」 | 训练时预测被遮住的 token；推理时迭代地解开置信度最高的预测 |
| 余弦调度 | 「解开调度」 | 掩码比例在推理步上的衰减方式；把置信度增长集中在中间区间 |
| 并行解码 | 「一次出所有 token」 | 每一步用一次前向传播预测整条被遮住的 token 序列，然后确定 top-K |
| 混合注意力 | 「因果 + 双向」 | 文本 token 上因果、图像块内部双向的掩码 |
| 图像修复 | 「填补式生成」 | 以部分 token 被遮住的图像为条件，预测缺失的 token；训练目标天然支持 |
| 提交率 | 「每步 top-K」 | 每轮迭代宣布多少个 token「完成」；控制推理速度与质量的权衡 |

## 延伸阅读

- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
