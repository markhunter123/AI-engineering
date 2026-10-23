# 项目 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年，文档问答的前沿已从「先 OCR 再转文本」转向视觉优先的 late interaction。ColPali、ColQwen2.5 和 ColQwen3-omni 把每一页 PDF 当作图像，用多向量 late interaction 做 embedding，让查询直接对各个 patch 计算注意力。在财务 10-K、科学论文和手写笔记上，这种模式大幅优于 OCR 优先的方案。在 1 万页上端到端构建这条 pipeline，并发布与「先 OCR 再转文本」的并排对比。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## 问题

企业手里堆着大量被 OCR pipeline 糟蹋的 PDF：带旋转表格的扫描版 10-K、公式密集的科学论文、只有作为图像才看得懂的图表、手写批注。把它们当作文本优先处理，等于丢掉一半信号。2026 年的答案是直接在原始页面图像上做 late interaction 多向量检索。ColPali（Illuin Tech）开创了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 上，视觉优先检索的得分明显高于「先 OCR 再转文本」—— 而且在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen embedding 每页约有 2048 个 patch 向量，而不是单个 1024 维向量。原始存储会急剧膨胀。DocPruner（2026）实现了 50% 的剪枝，而准确率没有可测量的损失。你要索引 1 万页，测量 ViDoRe v3 nDCG@5，在 2 秒内给出答案，并直接与「先 OCR 再转文本」的基线对比。

## 概念

Late interaction 指的是：每个查询 token 对每个 patch token 打分，然后对每个查询 token 的最大得分求和。这样无需单个池化向量也能获得细粒度匹配。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）存储逐 patch 的 embedding，并在检索时运行 MaxSim。

作答者是一个视觉语言模型，它接收查询和检索到的 top-k 页面图像，写出带证据区域的答案（边界框或页码引用）。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记号，会接入 OCR 兜底（Nougat、dots.ocr）作为可选的文本通道。

评测是一个二维矩阵。一个轴是内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个轴是检索方式（视觉优先 late interaction、先 OCR 再转文本、混合）。每个单元格给出 nDCG@5 和答案准确率。这份报告就是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，统一为纵向
- Late-interaction 模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：带多向量字段的 Vespa，或 Qdrant multi-vector，或支持 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差 patch，压缩 50%，准确率损失低于 0.5%）
- OCR 兜底（公式 / 密集表格）：dots.ocr 或 Nougat
- VLM 作答模型：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；备选 InternVL3
- 评测：ViDoRe v3 基准，多页推理用 M3DocVQA
- 查看器 UI：Next.js 15，用 canvas 叠加证据区域

```figure
ce-late-interaction
```

## 构建它

1. **摄入。** 遍历一个包含 1 万页 PDF 的语料库，涵盖 10-K、科学论文和扫描文档。把每页渲染成 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **Embedding。** 对每页图像运行 ColQwen2.5-v0.2。输出约 2048 个 128 维的 patch embedding。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant multi-vector。

3. **查询。** 对每个到来的查询，用 query tower 做 embedding（token 级 embedding）。对索引运行 MaxSim：对每个查询 token，在页面 patch embedding 上取最大点积，再求和。返回 top-k 页面。

4. **合成。** 带上查询和 top-5 页面图像调用 Qwen3-VL-30B。Prompt：「仅使用提供的页面作答。每条论断都按 (doc_id, page) 引用，并说明区域类型（图、表、段落）。」

5. **证据区域。** 对答案做后处理，抽取被引用的区域。如果 VLM 输出边界框（Qwen3-VL 会），就在查看器中将其渲染为叠加层。

6. **OCR 兜底。** 对判定为公式密集的页面（基于图像方差的启发式规则），运行 Nougat 或 dots.ocr，把 OCR 文本作为图像之外的额外通道传入。

7. **评测。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。再用同一个合成器在同一语料库上运行「先 OCR 再转文本」pipeline。产出「内容类型 × 方法」矩阵。

8. **UI。** 先用 Streamlit 做原型；生产版查看器用 Next.js 15，逐页叠加证据区域。

## 使用它

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付它

`outputs/skill-doc-qa.md` 描述交付物：一个视觉优先的多模态文档问答系统，针对特定语料库调优，并在 ViDoRe v3 上与「先 OCR 再转文本」基线做对比评测。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR 文本基线和公开排行榜相比的基准成绩 |
| 20 | 证据区域定位 | 被引用区域中真正包含答案跨度的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、作答 p95 |
| 20 | 多页推理 | 在人工标注的 100 题多页集合上的准确率 |
| 15 | 源文件查看体验 | 查看器清晰度、叠加层保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上对比 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一个答对而另一个漏掉？给索引加「content class」标签，按类型路由。

2. 激进地剪枝 embedding（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的那一点。

3. 构建混合方案：并行运行「先 OCR 再转文本」和 ColQwen，用 RRF 融合，再用 cross-encoder 重排序。混合方案是否优于任一单独方案？它在哪些地方帮助最大？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测量每美元准确率曲线。

5. 增加手写笔记支持。渲染手写语料库，用 ColQwen 做 embedding，测量检索效果。与手写 OCR pipeline 对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | 「ColPali 式检索」 | 查询 token 独立地对页面 patch 打分；由 MaxSim 聚合 |
| 多向量 | 「逐 patch embedding」 | 每篇文档有多个向量，而不是一个池化向量 |
| MaxSim | 「Late-interaction 打分」 | 对每个查询 token，在文档向量上取最大相似度，再求和 |
| DocPruner | 「Patch 压缩」 | 2026 年的剪枝方法，保留 50% 的 patch，准确率损失可忽略 |
| ViDoRe v3 | 「文档检索基准」 | 衡量视觉文档检索的 2026 年标准 |
| 证据区域 | 「引用边界框」 | 源页面上定位答案跨度的边界框 |
| OCR 兜底 | 「公式通道」 | 对公式或表格密集的页面，与视觉通道并用的文本 pipeline |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) —— late-interaction 文档检索参考
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) —— 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) —— 生产可用 checkpoint
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) —— 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) —— 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) —— 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) —— 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) —— 支持公式的 OCR 兜底
