# OCR 与文档理解

> OCR 是一条三阶段流水线——检测文本框、识别字符、再排布版面。每个现代 OCR 系统都会重排这些阶段，或把它们合并。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## 学习目标

- 梳理经典 OCR 流水线（检测 → 识别 → 版面）与现代端到端替代方案（Donut、Qwen-VL-OCR）
- 为序列到序列的 OCR 训练实现 CTC（Connectionist Temporal Classification）损失
- 使用 PaddleOCR 或 EasyOCR 做生产级文档解析，无需训练
- 区分 OCR、版面解析与文档理解——并为每种任务选对工具

## 问题

满是文字的图像无处不在：收据、发票、证件、扫描书页、表单、白板、招牌、截图。从中提取结构化数据——不只是字符，而是「这是总金额」——是应用视觉中价值最高的问题之一。

这个领域分为三层能力：

1. **OCR 本体**：把像素变成文本。
2. **版面解析**：把 OCR 输出按区域分组（标题、正文、表格、页眉）。
3. **文档理解**：从版面中提取结构化字段（"invoice_total = $42.50"）。

每一层都有经典和现代的做法；从「我要图像里的文字」到「我要这张收据的总金额」，两者之间的差距比大多数团队意识到的要大。

## 概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**产出逐行或逐词的四边框。
- **识别**把每个区域裁剪到固定高度，运行 CNN + BiLSTM + CTC 产生字符序列。
- **版面**重建阅读顺序（拉丁文字从上到下、从左到右；阿拉伯文、日文则不同）。

### 一段话讲清 CTC

OCR 识别要从定长特征图产生变长序列。CTC（Graves 等，2006）让你无需字符级对齐就能训练这件事。模型在每个时间步输出（词表 + blank）上的分布；CTC 损失对所有在合并重复、去除 blank 后归约为目标文本的对齐方式进行边缘化。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年奏效的原因，也是 2026 年大多数生产 OCR 模型仍在使用的训练方式。

### 现代端到端模型

- **Donut**（Kim 等，2022）——ViT 编码器 + 文本解码器；读取图像直接输出 JSON。没有文本检测器，也没有版面模块。
- **TrOCR**——ViT + Transformer 解码器，面向行级 OCR。
- **Qwen-VL-OCR / InternVL**——为 OCR 任务微调的完整视觉-语言模型；在复杂文档上达到 2026 年最佳准确率。
- **PaddleOCR**——把经典 DB + CRNN 流水线封装在成熟的生产级软件包中；仍是开源领域的主力。

端到端模型需要更多数据和算力，但避开了多阶段流水线的误差累积。

### 版面解析

对于结构化文档，运行版面检测器（LayoutLMv3、DocLayNet），为每个区域打标签：Title、Paragraph、Figure、Table、Footnote。阅读顺序于是变成「按版面顺序遍历区域并拼接」。

对于表单，使用**键值提取**模型（视觉丰富的文档用 Donut，纯扫描件用 LayoutLMv3）。它们接收图像 + 检测到的文本 + 位置，预测结构化的键值对。

### 评估指标

- **字符错误率（CER）**——Levenshtein 距离 / 参考文本长度。越低越好。生产目标：干净扫描件上 < 2%。
- **词错误率（WER）**——在词级别上的同一指标。
- **结构化字段上的 F1**——用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 上的编辑距离**——用于端到端文档解析；Donut 论文引入了归一化树编辑距离。

```figure
cv3-ctc-collapse
```

## 构建它

### 第 1 步：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效的 CuDNN 实现。贪心解码器比 beam search 简单，CER 通常只差 1% 以内。

### 第 2 步：微型 CRNN 识别器

用于行级 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN 通过最大池化把高度压到 1）。宽度就是 CTC 的时间维度。

### 第 3 步：合成 OCR

生成白底黑字的数字串，用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的 OCR 数据集还会加入字体、噪声、旋转、模糊和颜色。上面的流水线完全相同。

### 第 4 步：训练骨架

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这份简单的合成数据上，损失应在 200 步内从约 3 降到约 0.2。

## 使用它

三条生产路径：

- **PaddleOCR**——成熟、快速、多语言。一行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**——Python 原生、多语言、PyTorch 骨干。
- **Tesseract**——经典方案；当模型处理不了老旧扫描件时仍有用武之地。

端到端文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于收据、发票和结构可重复的表单，微调 Donut。对于任意文档或需要推理的 OCR，Qwen-VL-OCR 这类 VLM 是当前默认选择。

## 交付它

本节课产出：

- `outputs/prompt-ocr-stack-picker.md` —— 一个根据文档类型、语言和结构，在 Tesseract / PaddleOCR / Donut / VLM-OCR 之间做选择的提示词。
- `outputs/skill-ctc-decoder.md` —— 一个从零编写贪心和 beam-search CTC 解码器的技能，包含长度归一化。

## 练习

1. **（简单）** 在 5 位随机数字串上训练 TinyCRNN 500 步。报告留出集上的 CER。
2. **（中等）** 把贪心解码换成 beam search（beam_width=5）。报告 CER 变化。beam search 在哪些输入上胜出？
3. **（困难）** 用 PaddleOCR 处理 20 张收据，提取行项目，并针对 {item_name, price} 对与手工标注的真值计算 F1。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| OCR | 「像素到文本」 | 把图像区域变成字符序列 |
| CTC | 「免对齐损失」 | 无需逐时间步标签即可训练序列模型的损失；对所有对齐做边缘化 |
| CRNN | 「经典 OCR 模型」 | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线，至今仍在生产中使用 |
| Donut | 「端到端 OCR」 | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| Layout parsing | 「找区域」 | 检测并标注文档中的 Title/Table/Figure/Paragraph 区域 |
| Reading order | 「文本序列」 | 把识别出的区域排序成句子；拉丁文字很简单，混合版面则不简单 |
| CER / WER | 「错误率」 | 字符或词粒度上的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | 「会阅读的 LLM」 | 为 OCR 任务训练或提示的视觉-语言模型；复杂文档上的当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) —— 原始的 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 原始论文；算法思想非常密集
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) —— 无 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源生产级 OCR 栈
