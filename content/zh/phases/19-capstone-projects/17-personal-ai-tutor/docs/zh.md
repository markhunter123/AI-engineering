# 项目 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 与 Synthesis Tutor 都在 2026 年大规模交付了自适应多模态辅导。它们的共同形态是：苏格拉底式策略（绝不直接抛出答案）、在每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照解题输入、课程图谱检索、间隔重复排程，以及针对适龄内容的严格安全过滤。本项目的目标是交付一个特定学科的导师（K-12 代数或 Python 入门），与 10 名学习者开展为期两周的效果研究，并通过内容安全审计。

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 hours

## 问题

自适应辅导过去只是教育科技的研究细分领域。到 2026 年，它已是消费级产品。Khanmigo 部署在美国大多数学区。Duolingo Max 达到数千万月活。Google 的 LearnLM / Gemini for Education 为 Google Classroom 中的辅导提供动力。Quizlet Q-Chat 与抽认卡并列。Synthesis Tutor 凭借面向好奇孩子的辅导功能走红。共同要素包括：多模态输入（打字、说话、拍摄方程）、苏格拉底式教学法（先提问，后解释）、在每次交互后更新的学习者模型，以及严格的适龄安全措施。

你将为一个特定群体构建其中一种。衡量标准是真实的效果研究：10 名学习者、为期两周的前测与后测分数。语音循环必须自然（项目 03 的子栈）。记忆必须尊重隐私。安全过滤器必须通过面向 K-12 的 COPPA 感知红队测试。

## 概念

共有四个组件。**导师策略**是一个苏格拉底式循环：学习者索要答案时，策略提出引导性问题；答对时，进入下一个概念；卡住时，给出支架式提示。**学习者模型**是贝叶斯知识追踪（或其简单变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱**是一个 Neo4j 概念图，带先修边；策略在图上游走来选择下一个概念。**记忆**是情节式 + 语义式存储（agentmemory 风格），保存过往交互、错误与偏好。

交互是多模态的。文本输入用于键盘作答。语音输入通过 LiveKit + Whisper（复用项目 03）。数学题的拍照输入通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全方面使用 Llama Guard 4 外加适龄过滤器（拦截成人内容、暴力、自残），以及 COPPA 感知的记忆保留策略。

效果研究本身就是交付物。10 名学习者，前测与后测，为期两周。报告学习增益变化量与置信区间。与非自适应基线（相同内容以线性方式呈现、不使用导师策略）对比。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门（选其一做深）
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（使用 prompt 缓存）
- 学习者模型：贝叶斯知识追踪（经典）或用于排程的 FSRS
- 课程图谱：Neo4j 的概念图 + 先修边 + OER 内容
- 记忆：agentmemory 风格的持久化向量 + 情节式 + 语义式存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用项目 03 的子栈）
- 拍照解题：dots.ocr 或 PaliGemma 2 用于方程识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评测：Bloom 层级题目生成、前测/后测 harness、效果研究工具

```figure
cf-tutor-loop
```

## 构建它

1. **课程图谱。** 构建一个包含 50-150 个概念节点的 Neo4j 图（例如 K-12 代数从「数轴」到「二次公式」），带先修边。为每个节点附上 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 用先验初始化贝叶斯知识追踪：猜测率、失误率、学习率。每次交互后更新每个概念的掌握度。按学习者持久化。

3. **导师策略。** 使用带以下节点的 LangGraph：`read_signal`（学习者的回答是正确的 / 部分的 / 卡住的？）、`select_concept`（在课程图上游走，选出优先级最高的概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互都写入情节式存储。错误与偏好提升到语义记忆。COPPA 感知的保留策略：1 年后自动删除，家长可访问。

5. **语音路径。** 挂在导师策略上的 LiveKit Agents worker。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持打断（barge-in，复用项目 03 的机制）。

6. **拍照解题路径。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入送入导师。

7. **安全。** 每个模型输出都经过 Llama Guard 4 + 适龄过滤器（拦截自残、成人内容、暴力）。记忆访问按学习者 ID 限定作用域；提供家长删除入口。

8. **效果研究。** 10 名学习者，前测（标准化 30 题基线），两周的导师交互（每周 3 次），后测。与同样内容下 10 名学习者的非自适应基线组对比。

9. **每周进度报告。** 为每位学习者自动生成 PDF 摘要，包含探索过的主题、掌握度轨迹与推荐的后续步骤。

## 使用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付它

`outputs/skill-ai-tutor.md` 是交付物：一个特定学科的自适应导师，具备多模态输入、学习者模型、记忆、安全，以及经过测量的效果。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 学习增益变化量 | 10 名学习者两周研究中的前测/后测差异 |
| 20 | 苏格拉底式忠实度 | 对会话记录样本的评分量表得分 |
| 20 | 多模态体验 | 语音 + 拍照 + 文本端到端的连贯性 |
| 20 | 安全与隐私姿态 | Llama Guard 4 通过率 + COPPA 感知的保留策略 |
| 15 | 课程广度与图谱质量 | 概念覆盖 + 先修图一致性 |
| **100** | | |

## 练习

1. 在有、无自适应学习者模型（随机概念顺序）的条件下分别开展效果研究。报告差异。预期自适应会取胜，但幅度才是有意思的数字。

2. 增加多模态探针：同一个概念问题分别以文本、语音、图片呈现。测量学习者是否用自己偏好的模态更快掌握。

3. 构建家长看板：练习过的主题、掌握度轨迹、即将学习的概念、安全事件（任何防护栏命中）。符合 COPPA。

4. 增加语言切换模式：导师接受西班牙语输入并用西班牙语教学。测量 X-Guard 的覆盖情况。

5. 对记忆隐私做压力测试：验证即使通过语音片段重新注入攻击，学习者 A 也无法看到学习者 B 的数据。记录尝试访问并告警。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | 「提问，不要倒答案」 | 导师提出引导性问题，而不是直接给出答案 |
| 贝叶斯知识追踪 | 「BKT」 | 用于计算每个概念掌握概率的经典学习者模型方程 |
| FSRS | 「Free Spaced Repetition Scheduler」 | 2024 年的间隔重复排程器，优于 SM-2 |
| 课程图谱 | 「概念 DAG」 | 带先修边的 Neo4j 概念图 |
| 情节式记忆 | 「逐次交互日志」 | 存储每次交互以供后续检索 |
| 语义式记忆 | 「学到的模式库」 | 从情节式记忆提升而来的精简错误与偏好 |
| COPPA | 「儿童隐私法」 | 限制收集 13 岁以下儿童数据的美国法律 |

## 延伸阅读

- [Khanmigo（Khan Academy）](https://www.khanmigo.ai) — 消费级 K-12 导师参考
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 语言学习导师参考
- [Google LearnLM / Gemini for Education](https://blog.google/products-and-platforms/products/education/google-learnlm-gemini-generative-ai/) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 另一参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复排程器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈
