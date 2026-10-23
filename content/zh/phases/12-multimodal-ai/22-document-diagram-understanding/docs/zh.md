# 文档与图表理解

> 文档不是照片。一份 PDF、科学论文、发票或手写表格，带有版式、表格、图表、脚注、页眉和语义结构，这些是普通图像理解无法捕捉的。前 VLM 时代的栈是一条流水线：Tesseract OCR + LayoutLMv3 + 表格抽取启发式。VLM 浪潮用无 OCR 模型取代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到 2026 年，前沿做法就是「把页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7」，结构化标记输出免费得到。本课解读文档 AI 三个时代的演进弧线。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、无 OCR、VLM 原生。
- 描述 LayoutLMv3 的三路输入流：文本、布局（bbox）、图像 patch，以及统一掩码。
- 比较 Donut（无 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM 原生）。
- 为新任务选择文档模型（发票、科学论文、手写表格、中文票据）。

## 问题

「理解这份 PDF」难得出人意料。信息分布在：

- 文本内容（约 90% 的信号）。
- 版式（页眉、脚注、边栏、双栏排版）。
- 表格（行、列、合并单元格）。
- 图和图表。
- 手写批注。
- 字体与排版（标题 vs 正文）。

原始 OCR 只倒出文本，其余全丢。一个关心发票的系统需要知道「总计：$1,245」来自右下角，而不是脚注。

## 概念

### 时代 1——OCR 流水线（2021 年前）

经典栈：

1. PDF → 每页图像。
2. Tesseract（或商用 OCR）抽取文本，附带逐词边界框。
3. 布局分析器识别区块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则抽取字段。

对于干净的印刷文本有效。在手写、倾斜扫描、复杂表格、非英文文字上失效。每种失败模式都需要一条定制异常路径。

### TrOCR（2021）

TrOCR（Li 等，arXiv:2109.10282）用 Transformer 编码器-解码器取代了 Tesseract 经典的 CNN-CTC，在合成 + 真实文本图像上训练。在手写和多语种文本上是明确的胜利。仍然是流水线（检测器再接 TrOCR 再接布局），但 OCR 这一步大幅改善。

### 时代 2——无 OCR（2022-2023）

第一批无 OCR 模型提出：完全跳过检测，把图像像素直接映射到结构化输出。

Donut（Kim 等，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出是用于表单理解的 JSON、用于摘要的 markdown，或任何任务专属 schema。
- 没有 OCR，没有布局，没有检测。

Nougat（Blecher 等，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出为 LaTeX / markdown。
- 处理公式、多栏版式、图表。
- 每个 arXiv 解析器都会调用的模型。

这些是专才，不是通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR，但加入布局理解：

- 三路输入流：OCR 文本 token、逐 token 的二维边界框、图像 patch。
- 跨全部三种模态的掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游任务：分类、实体抽取、表格问答。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上表现强。需要上游 OCR。在标准化文档基准上取得 VLM 之前的最佳准确率。

### DocLLM（2023）

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。以布局 token 为条件生成自由形式答案。更适合文档问答；仍然依赖 OCR 输入。

### 时代 3——VLM 原生（2024+）

2024 年的 VLM 已经足够好，可以完全取代流水线。把整页图像以高分辨率喂给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小文档。
- Qwen2.5-VL 的动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写训练。

VLM 原生与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生在以下方面胜出：

- 场景文本（手写 + 印刷、混合文字）。
- 带合并单元格的复杂表格。
- 嵌入正文的数学公式。
- 带文字标注的图表。

OCR 流水线仍在以下方面胜出：

- 大规模纯扫描工作负载，逐页延迟很关键。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的受监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 的文档理解达到接近人类的准确率。2026 年初的基准数字：

- DocVQA：Claude 4.7 约 95.1，PaliGemma 2 约 88.4，Nougat 约 77.3，流水线 LayoutLMv3 约 83。
- ChartQA：Claude 4.7 约 92.2，GPT-4V 约 78。
- VisualMRC：Claude 4.7 约 94。

闭源模型的差距主要来自分辨率和基础 LLM 规模。7B 的开源模型落后几个点，但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要公式的精确 LaTeX 输出。Nougat 就是为此训练的。用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生模型）能产出可用的 LaTeX。没有显式 LaTeX 训练，VLM 产出的是可读但不精确的转写。

对于 2026 年的科学论文流水线：对 PDF 串接 Nougat，再对棘手页面用 VLM。

### 手写

仍然是最难的子任务。印刷 + 手写混合（医生笔记、填写过的表单）是 OCR 流水线在成本上仍胜过 VLM 的领域。纯手写 VLM 正在进步（Claude 4.7、PaliGemma 2）。

### 2026 年配方

对于一个新文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄取：数学用 Nougat，图表用 VLM。
- 监管场景：OCR 流水线 + VLM 校验器做交叉核对。

```figure
mm-doc-layout
```

## 使用它

`code/main.py`：

- 一个玩具级布局感知分词器：给定（文本，bbox）对，生成 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务 schema 生成器：表单的 JSON 模板。
- 比较 OCR 流水线、Donut、Nougat 和 VLM 原生每页的 token 预算。

## 交付它

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管），在 OCR 流水线、无 OCR 专才和 VLM 原生之间选择。

## 练习

1. 你的项目每天处理 1000 万张发票。哪种栈能在不损失准确率的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单问答上优于纯 CLIP-VLM，但在场景文本上不如？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上胜过 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，提升文档准确率的关键训练数据新增是什么？

5. 设计一个监管安全的混合方案：OCR 流水线为主，VLM 为二级交叉核对。你如何解决分歧？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OCR 流水线 | 「Tesseract 风格」 | 分阶段栈：检测 -> OCR -> 布局 -> 规则；确定性但脆弱 |
| 无 OCR | 「Donut 风格」 | 图像到输出的 Transformer，跳过显式 OCR；单一模型 |
| 布局感知 | 「LayoutLM」 | 输入包含逐 token 的 bbox 坐标；跨模态统一掩码 |
| VLM 原生 | 「前沿 VLM」 | 把页面图像以高分辨率直接喂给 Claude/GPT/Qwen VLM；没有流水线 |
| DocVQA | 「文档基准」 | 文档 VQA 标准；被引用最多的分数 |
| 标记输出 | 「LaTeX / MD」 | 结构化输出格式而非自由文本；可驱动下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
