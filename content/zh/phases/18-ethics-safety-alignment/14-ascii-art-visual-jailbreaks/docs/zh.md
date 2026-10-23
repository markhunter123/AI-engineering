# ASCII 艺术与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，《ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs》（ACL 2024，arXiv:2402.11753）。把有害请求中与安全相关的 token 遮蔽掉，用相同字母的 ASCII 艺术渲染替换，再发送伪装后的 prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳定识别 ASCII 艺术 token。该攻击绕过了 PPL（困惑度过滤器）、复述（Paraphrase）防御和重分词（Retokenization）。相关工作：ViTC 基准测量对非语义视觉 prompt 的识别能力；StructuralSleight 将攻击推广到「不常见文本编码结构」（树、图、嵌套 JSON），构成一个编码攻击家族。

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## 学习目标

- 描述 ArtPrompt 攻击：词识别步骤、ASCII 艺术替换、最终的伪装 prompt。
- 解释标准防御（PPL、Paraphrase、Retokenization）为何在 ArtPrompt 上失效。
- 定义 ViTC 并描述它测量什么。
- 描述 StructuralSleight 作为对任意不常见文本编码结构的推广。

## 问题

通过复述与角色扮演（第 12 课）以及长上下文（第 13 课）发起的攻击，作用于文本层面的模式。ArtPrompt 作用于识别层面：模型并不解析被禁止的 token，而是解析由字符渲染成的一幅图像。安全过滤器看到的是无害的标点，模型看到的却是一个词。

## 概念

### ArtPrompt 的两个步骤

第一步：词识别。给定一个有害请求，攻击者用 LLM 找出与安全相关的词（例如「how to make a bomb」中的「bomb」）。

第二步：生成伪装 prompt。把每个识别出的词替换为它的 ASCII 艺术渲染（一个 7x5 或 7x7 的字符块，构成字母形状）。模型收到的是由标点和空格组成的网格；能力足够强的模型能把它识别为该词，而安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在其基准子集上攻击成功率超过 75%。

### 标准防御为何失效

- **PPL（困惑度过滤器）。** ASCII 艺术具有很高的困惑度——但所有新颖输入都是如此。能拦住 ArtPrompt 的阈值选择，同样会拦住合法的结构化输入。
- **复述（Paraphrase）。** 对 prompt 做复述会破坏 ASCII 艺术。但在实践中，用于复述的 LLM 常常保留或重建了这些艺术图。
- **重分词（Retokenization）。** 换一种方式切分 token，并不会改变「模型的视觉能力正在识别字母形状」这一事实。

根本问题在于：安全过滤器是 token 级或语义级的，而 ArtPrompt 作用于视觉识别层面。

### ViTC 基准

ViTC 测量对非语义视觉 prompt 的识别能力，即模型读取 ASCII 艺术、wingdings 以及其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型读取视觉文本的能力越强，ArtPrompt 在它身上就越有效。这是一种能力—安全权衡。

### StructuralSleight

StructuralSleight 把 ArtPrompt 推广到不常见文本编码结构（Uncommon Text-Encoded Structures，UTES）：树、图、嵌套 JSON、JSON 中的 CSV、diff 风格的代码块。如果某种结构在训练安全数据中很少见，但模型又能解析，它就可以用来藏匿有害内容。

防御上的含义：安全必须泛化到模型能够解析的所有结构化表示。这个集合很大，而且还在增长。

### 图像模态的类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的 ArtPrompt 式攻击比 ASCII 艺术版本更强，因为图像编码器能产生更丰富的信号。

### 这一课在 Phase 18 中的位置

第 12-14 课描述了三条正交的攻击向量：迭代精炼（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课介绍防御工具侧的应对。

```figure
al-ascii-cloak
```

## 使用它

`code/main.py` 构建了一个玩具版 ArtPrompt。你可以用 ASCII 艺术字形遮蔽有害查询中的特定词，验证伪装后的字符串能通过关键词过滤器，并（可选）用简单的识别器把伪装字符串解码回来。

## 交付它

本课产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它枚举覆盖的编码攻击家族（ASCII 艺术、base64、leet-speak、UTF-8 同形字、UTES），以及拦截每一种的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串能通过简单的关键词过滤器，报告所需的字符级改动。

2. 实现第二种编码：对同一个目标词使用 base64。比较其过滤绕过率与 ArtPrompt，以及恢复难度。

3. 阅读 Jiang 等人 2024 第 4.3 节（五模型结果）。提出一个理由，解释为何在同一基准上 Claude 对 ArtPrompt 的抵抗力高于 Gemini。

4. 设计一种生成前防御，检测 prompt 中 ASCII 艺术形状的区域。测量其在合法代码、表格和数学记号上的误报率。

5. StructuralSleight 列出 10 种编码结构。设计一种能处理全部 10 种的通用防御，并估算每防御一个 prompt 的计算成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | 「ASCII 艺术攻击」 | 用 ASCII 艺术渲染遮蔽安全词的两步越狱 |
| 伪装（Cloaking） | 「把词藏起来」 | 用模型能读取而过滤器读不到的视觉表示替换被禁止的 token |
| UTES | 「不常见结构」 | 不常见文本编码结构——树、图、嵌套 JSON 等，用于夹带内容 |
| ViTC | 「视觉文本能力」 | 衡量模型读取非语义视觉编码能力的基准 |
| 困惑度过滤器 | 「PPL 防御」 | 拒绝高困惑度的 prompt；失效原因是合法的结构化输入同样得分很高 |
| 重分词 | 「分词器切换防御」 | 用不同分词器预处理 prompt；失效原因是识别发生在视觉层面 |
| 同形字（Homoglyph） | 「形近字符」 | 看起来与拉丁字母完全相同的 Unicode 字符；可绕过子串检查 |

## 延伸阅读

- [Jiang 等——ArtPrompt（ACL 2024，arXiv:2402.11753）](https://arxiv.org/abs/2402.11753)——ASCII 艺术越狱论文
- [Li 等——StructuralSleight（arXiv:2406.08754）](https://arxiv.org/abs/2406.08754)——UTES 推广
- [Chao 等——PAIR（第 12 课，arXiv:2310.08419）](https://arxiv.org/abs/2310.08419)——互补的迭代攻击
- [Anil 等——Many-shot Jailbreaking（第 13 课）](https://www.anthropic.com/research/many-shot-jailbreaking)——互补的长度攻击
