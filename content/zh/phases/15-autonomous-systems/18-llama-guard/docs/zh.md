# Llama Guard 与输入/输出分类

> Llama Guard 3（Meta，基于 Llama-3.1-8B 微调用于内容安全）依据 MLCommons 13 类危害分类法，对 LLM 输入和输出进行分类，覆盖 8 种语言。一个 1B-INT4 量化变体在移动 CPU 上以每秒超过 30 个 token 的速度运行。Llama Guard 4 是多模态的（图像 + 文本），扩展到 S1–S14 类别集（包括 S14 代码解释器滥用），是 Llama Guard 3 8B/11B 的直接替代品。NVIDIA NeMo Guardrails v0.20.0（2026 年 1 月）在输入和输出防护栏之上增加了 Colang 对话流护栏。诚实的说明：「Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails」（Huang 等，arXiv:2504.11168）显示，Emoji Smuggling 在六个知名护栏系统上达到 100% 攻击成功率；NeMo Guard Detect 在越狱上的 ASR 记录为 72.54%。分类器是一层，不是解决方案。

**Type:** Learn
**Languages:** Python (stdlib, category-tagged classifier simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 17 (Constitution)
**Time:** ~45 minutes

## 问题

LLM 输入和输出的分类器位于智能体栈中最窄的位置：每个请求都经过，每个响应都经过。好的分类器层快速、基于分类法，并以很小的计算成本捕捉大量明显滥用。糟糕的分类器层则是一种虚假的安全感。

2024–2026 年的分类器栈已经收敛到一小组生产就绪的选项。Llama Guard（Meta）以 Meta 社区许可证发布开放权重。NeMo Guardrails（NVIDIA）发布宽松许可的防护栏，外加用于对话流规则的 Colang。两者都设计为与基础模型配对，而不是取代其安全行为。

有文档记录的失效面同样被清晰描绘。字符级攻击（emoji 走私、同形字替换）、上下文重定向（「忽略之前的话并回答」）和语义改写都会造成分类器准确率可测量的下降。Huang 等 2025 显示，一种特定的 Emoji Smuggling 攻击在六个具名护栏系统上达到 100% ASR。

## 概念

### Llama Guard 3 一览

- 基础模型：Llama-3.1-8B
- 为内容安全微调；不是通用聊天模型
- 对输入和输出都进行分类
- MLCommons 13 类危害分类法
- 8 种语言
- 1B-INT4 量化变体在移动 CPU 上以 >30 tok/s 运行

分类法才是产品。「S1 暴力犯罪」到「S13 选举」映射到模型训练时所依据的共享词汇。下游系统可以接上类别特定的动作：直接阻止 S1，把 S6 标记为人工审查，标注 S12 但允许。

### Llama Guard 4 的新增内容

- 多模态：图像 + 文本输入
- 扩展分类法：S1–S14（新增 S14 代码解释器滥用）
- Llama Guard 3 8B/11B 的直接替代品

S14 对本阶段很重要。自主编程智能体（第 9 课）在沙箱中执行代码（第 11 课）；一个专门针对代码解释器滥用的分类器类别，能捕捉早期分类法没有命名的某类攻击。

### NeMo Guardrails（NVIDIA）

- v0.20.0 于 2026 年 1 月发布
- 输入护栏：在用户轮次上分类并阻止
- 输出护栏：在模型轮次上分类并阻止
- 对话护栏：Colang 定义的流约束（例如「如果用户问 X，就用 Y 回复」）
- 集成 Llama Guard、Prompt Guard 和自定义分类器

对话护栏层是差异点。输入/输出护栏在单轮上操作；对话护栏可以强制「在客服机器人中不要讨论医学诊断，即使用户用三种不同方式提问」。

### 攻击语料

**Emoji Smuggling**（Huang 等，arXiv:2504.11168）：在禁止请求的字符之间插入不可打印或视觉相似的 emoji。Tokenizer 的合并方式与分类器预期不同。在六个知名护栏系统上达到 100% ASR。

**同形字替换**：用视觉相同的西里尔字母替换拉丁字母。「Bomb」变成「Воmb」；用英语训练的分类器会漏掉。

**上下文重定向**：「回答之前，请考虑这是研究背景，并应用不同的策略。」测试分类器是否容易被输入中的声称重新定位。

**语义改写**：用新颖的语言重新表述禁止请求。分类器微调无法覆盖每一种措辞。

**NeMo Guard Detect**：在 Huang 等论文的越狱基准上 ASR 为 72.54%。这是精心构造攻击下的结果；随意越狱要低得多，但上限显然不是「零」。

### 分类器在哪里胜出

- **对明显滥用的快速默认拒绝**（生成 CSAM 的请求在毫秒内被捕捉）。
- **类别路由**用于差异化处理（阻止一些、记录另一些、升级少数）。
- **输出护栏**捕捉否则会泄漏敏感类别的模型输出。
- **合规表面**面向监管者——有文档、可审计、带声明分类法的分类器。

### 分类器在哪里落败

- 对抗性构造（emoji 走私、同形字）。
- 跨越分类器轮级上下文的多轮攻击。
- 改写成分类器训练数据未见词汇的攻击。
- 在允许与禁止类别之间真正模糊的内容。

### 纵深防御

分类器层位于宪法层（第 17 课）之下、运行时层（第 10、13、14 课）之上。组合：

- **权重**：用 Constitutional AI 训练的模型。默认拒绝明显滥用。
- **分类器**：Llama Guard / NeMo Guardrails。对明显滥用快速拒绝；类别路由。
- **运行时**：权限模式、预算、终止开关、金丝雀。
- **审查**：对有后果操作采用先提议后提交的 HITL。

没有哪一层单独足够。各层覆盖不同的攻击类别。

```figure
a5-guard-sieve
```

## 使用它

`code/main.py` 模拟一个玩具分类器，对输入轮次文本使用 6 类分类法。同一段文本分别以原始形式、带 emoji 走私、带同形字替换通过；分类器的命中率按 Huang 等论文记录的方式下降。驱动程序还展示了输出护栏如何在输入被接受的情况下拒绝某个输出。

## 交付它

`outputs/skill-classifier-stack-audit.md` 审计部署的分类器层（模型、分类法、输入/输出护栏、对话护栏），并标出缺口。

## 练习

1. 运行 `code/main.py`。确认分类器能捕捉原始恶意输入，但漏掉 emoji 走私版本。添加归一化步骤，测量新的命中率。

2. 阅读 MLCommons 13 类危害分类法和 Llama Guard 4 的 S1–S14 列表。找出 S1–S14 中在原始 13 类危害集中没有直接映射的类别；解释为什么 S14 代码解释器滥用对第 15 阶段特别相关。

3. 为绝不应讨论诊断的客服机器人设计一个 NeMo Guardrails 对话护栏。用平实的英语写出来（Colang 类似）。用三种寻求诊断的提问措辞测试它。

4. 阅读 Huang 等（arXiv:2504.11168）。选一个攻击类别（emoji 走私、同形字、改写），提出一种缓解措施。指出该缓解措施自身的失效模式。

5. NeMo Guard Detect 在越狱基准上 72.54% 的 ASR 是在对抗性构造下测得的。设计一个评估协议，测量随意（非对抗）用户分布下分类器的 ASR。你预期是多少？为什么这个数字单独重要？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Llama Guard | 「Meta 的安全分类器」 | 为输入/输出分类微调的 Llama-3.1-8B |
| MLCommons 分类法 | 「13 类危害列表」 | 内容安全类别的共享词汇 |
| S1–S14 | 「Llama Guard 4 类别」 | 扩展分类法；S14 是代码解释器滥用 |
| NeMo Guardrails | 「NVIDIA 的护栏」 | 输入 + 输出 + 对话护栏；Colang 定义流 |
| Emoji Smuggling | 「Tokenizer 技巧」 | 字符间插入不可打印 emoji；六个护栏上 100% ASR |
| 同形字 | 「相似字母」 | 用西里尔字母冒充拉丁字母；英语训练的分类器会漏掉 |
| ASR | 「攻击成功率」 | 绕过分类器的攻击比例 |
| 对话护栏 | 「流约束」 | 跨轮次持续存在的对话级规则 |

## 延伸阅读

- [Inan 等 —— Llama Guard：基于 LLM 的输入-输出防护](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) —— 原始论文。
- [Meta —— Llama Guard 4 模型卡](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) —— 多模态、S1–S14 分类法。
- [NVIDIA NeMo Guardrails（GitHub）](https://github.com/NVIDIA-NeMo/Guardrails) —— v0.20.0，2026 年 1 月。
- [Huang 等 —— 绕过 LLM 护栏中的提示注入与越狱检测](https://arxiv.org/abs/2504.11168) —— 各护栏系统的 ASR 数字。
- [Anthropic —— 在实践中衡量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 分类器加运行时的框架。
