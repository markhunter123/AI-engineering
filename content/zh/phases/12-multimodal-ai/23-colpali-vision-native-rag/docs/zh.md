# ColPali 与视觉原生文档 RAG

> 传统 RAG 把 PDF 解析成文本、切成块、嵌入块、存向量。每一步都丢失信号：OCR 丢掉图表数据，分块切断表格行，文本 embedding 忽略插图。ColPali（Faysse 等，2024 年 7 月）提出了一个更简单的问题：为什么要抽取文本？直接用 PaliGemma 嵌入页面图像，用 ColBERT 式后期交互做检索，保留文档携带的全部版式、插图、字体和格式信号。已发表的基准：在视觉丰富的文档上，端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本课解读视觉原生 RAG 的论点，并构建一个微型 ColPali 式索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释双编码器检索（每文档一个向量）与后期交互检索（每文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何把它从文本 token 推广到图像 patch。
- 构建一个微型 ColPali 式索引器：页面 → patch embedding → 查询词 embedding 上的 MaxSim → top-k 页面。
- 在发票 / 财务报告用例上比较 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4。

## 问题

对 PDF 做文本 RAG，丢掉了文档的大部分内容。财务报告第三季度的收入增长通常在一张图里；医学报告的发现标注在图像上；法律合同的签署区块是版式事实，不是文本事实。

文本 RAG pipeline：

1. PDF → 通过 OCR / pdftotext 得到文本。
2. 文本 → 300-500 token 的块。
3. 块 → 双编码器 embedding（一个向量）。
4. 用户查询 → embedding → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表没被捕捉。表格被切块切开。多栏版式被压平。图注消失。

ColPali 的修正：跳过 OCR，直接嵌入页面图像。用 ColBERT 式后期交互做检索，让模型在查询时能关注细粒度 patch。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不是每文档一个向量，而是每个 token 一个向量。查询时：

- 查询 token 得到各自的 embedding（$N_q$ 个向量）。
- 文档 token 得到 embedding（$N_d$ 个向量，通常缓存）。
- 分数 = 对每个查询 token 取与文档 token 余弦相似度的最大值，再求和：$\sum_i \max_j \cos(q_i, d_j)$。

这就是 MaxSim 操作。每个查询 token「挑选」最匹配的文档 token。最终分数是求和。

优点：召回强，能处理词级语义。缺点：每文档 $N_d$ 个向量，存储昂贵。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）把 ColBERT 的模式应用到图像。

- 每页由 PaliGemma（ViT + 语言）编码为 patch embedding：每页 $N_p$ 个向量。
- 每个用户查询（文本）编码为查询 token embedding：$N_q$ 个向量。
- 分数 = $\sum_i \max_j \cos(q_i, p_j)$，即对查询文本 token 和页面图像 patch 做 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每一页，存储全部 patch embedding。查询时：嵌入查询 token，对所有已存页面 embedding 计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富的文档上，端到端比文本 RAG 高 20-40%。每个 patch 向量捕捉局部版式和内容。

缺点：每页 $N_p$ 个 patch × 4 字节浮点 × $D$ 维向量 = 存储增长很快。可用 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）把 PaliGemma 换成 Qwen2-VL。基础编码器更好，检索更好。

ColSmol 是面向本地 / 边缘的小规模变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是另一种变体：不在 patch 上做 MaxSim，而是用 VLM 把每页池化为一个向量，再做双编码器检索。索引更快、存储更小、召回更弱。

质量与成本的权衡：要质量用 ColPali，要规模用 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）把多模态检索扩展到多页、多文档推理。跨文档检索页面，为 VLM 组合出多页上下文。

### ViDoRe——基准

ColPali 的配套基准。Visual Document Retrieval Evaluation。任务包括财务报告、科学论文、行政文档、病历、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上约 80% nDCG@5；文本 RAG 在相同文档上约 50-60%。

### 端到端 RAG pipeline

对于视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储全部 patch embedding。
2. 查询：用户文本 → 查询 token embedding → 对所有已索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程没有 OCR。图表、图形、字体、版式都会流入答案。

### 存储算术

一份 50 页的财务报告，每页 729 个 patch，128 维 embedding：

- ColPali：50 * 729 * 128 * 4 字节 = 原始约 18 MB，PQ 后约 4 MB。
- 文本 RAG：50 个块 * 768 维 * 4 字节 = 约 150 kB。

ColPali 每文档的存储约为文本 RAG 的 30 倍。规模化时，OPQ / PQ 可降到约 5-10 倍，通常可以接受。

### 文本 RAG 仍然胜出的场景

- 没有版式信号的纯文本文档（wiki 文章、聊天记录）。文本 RAG 更简单、存储更便宜。
- 存储主导成本的多百万页档案。
- 严格监管要求检索同时提供可抽取 OCR 文本的场景。

2026 年的其他一切——财务报告、科学论文、法律合同、病历、UX 文档——视觉原生 RAG 都胜出。

```figure
mm-maxsim
```

## 使用它

`code/main.py`：

- 玩具 patch 编码器：把一个「页面」（小网格特征向量）映射为 patch embedding 数组。
- MaxSim 打分器：计算查询 token embedding 集合与页面 patch 集合之间的 ColBERT 式分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 交付它

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG，并估算存储。

## 练习

1. 一份 200 页年报，每页 729 个 patch，128 维 embedding，4 字节浮点。计算原始存储和 PQ 压缩（8 倍）后的存储。

2. MaxSim 是 $\sum_i \max_j \cos(q_i, p_j)$。这个求和捕捉了什么简单平均相似度捕捉不到的东西？

3. ColPali 把页面索引为 patch 集合。如果我们改为按词级索引（如 ColBERT 所做），会有什么变化？权衡是什么？

4. 为一个 100 万页语料设计端到端 pipeline，每次查询延迟预算 500ms。选择 ColQwen2 / VisRAG 并给出理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式，以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互 | 「ColBERT 风格」 | 使用逐 token 或逐 patch embedding + MaxSim 的检索，而不是单一文档向量 |
| MaxSim | 「patch 上取最大」 | 对每个查询 token 取相似度最高的文档 token，再对查询求和 |
| 双编码器 | 「单向量」 | 每文档一个向量；更快但丢失粒度 |
| 多向量 | 「每文档多向量」 | 每文档 / 页存 $N_p$ 个向量；存储成本上升但召回改善 |
| patch embedding | 「页面特征」 | VLM 编码器为每个图像 patch 产出的一个向量，按页缓存 |
| ViDoRe | 「视觉文档基准」 | ColPali 的视觉文档检索基准套件 |
| PQ 量化 | 「乘积量化」 | 在保持向量相似度的同时把存储缩小约 8 倍的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
