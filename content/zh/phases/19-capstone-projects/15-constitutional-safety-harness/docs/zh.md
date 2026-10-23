# 项目 15 — 宪法式安全护栏 + 红队靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及覆盖多语言的 X-Guard，共同定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 与 promptfoo 成为标准的对抗评测工具。NeMo Guardrails v0.12 把它们串成生产流水线。本项目将这一切装配起来：围绕目标应用的分层安全护栏、运行 6 类以上攻击族的自主红队智能体，以及一次产生可衡量无害性变化量的宪法式自我批判运行。

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## 问题

2026 年 LLM 安全的前沿问题，不是分类器是否有效（大体上有效），而是如何围绕生产应用正确组合它们，既不过度拒答，也不留下明显漏洞。Llama Guard 4 处理英文的策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图片的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业场景的分类。Anthropic 的 Constitutional Classifiers 是另一种方法，用于训练阶段而非服务阶段。

攻击的演进同样重要。PAIR 与 TAP 把越狱发现自动化。GCG 执行基于梯度的后缀攻击。多轮攻击与语码转换（code-switch）攻击利用智能体的记忆。任何已部署的 LLM 都需要一个红队靶场——garak 与 PyRIT 是典型的驱动工具——外加有文档记录的缓解措施与按 CVSS 评分的发现。

你将加固一个目标应用（8B 指令微调模型，或其他项目中的某个 RAG 聊天机器人），对它运行 6 类以上攻击族，并给出前后对比的无害性测量。

## 概念

安全流水线分为五层。**输入清洗**：去除零宽字符，解码 base64/rot13，规范化 Unicode。**策略层**：NeMo Guardrails v0.12 的 rails（域外、毒性、PII 提取）。**分类器闸门**：对输入使用 Llama Guard 4，对非英文使用 X-Guard，对图片输入使用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：对输出使用 Llama Guard 4、Presidio PII 清洗，并在适用时强制引用。**HITL 层**：被标记为高风险的输出进入 Slack 队列。

红队靶场运行在调度器上。PAIR 与 TAP 自主发现越狱。GCG 执行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色扮演、记忆利用）。语码转换攻击（英语混搭斯瓦希里语或泰语）。每次运行都会产出一份结构化发现文件，包含 CVSS 评分与披露时间线。

宪法式自我批判运行是一种训练期干预。取 1k 条有害尝试 prompt，让模型起草回复，对照一部成文宪法（不得伤害规则）进行批判，并在这个批判循环上重新训练。在留出评测集上衡量前后无害性变化量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法式训练：Anthropic 风格的自我批判循环 + 在批判结果上做 SFT
- PII 清洗：Presidio
- 目标：一个 8B 指令微调模型，或其他项目中的某个 RAG 聊天机器人

```figure
cf-safety-stack
```

## 构建它

1. **目标搭建。** 在 vLLM 上启动一个 8B 指令微调模型（或复用另一个项目中的 RAG 聊天机器人）。这就是被测应用。

2. **安全流水线封装。** 围绕目标接入五层流水线。验证每一层都可单独观测（在 Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在一个小的标注集上分别运行，建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击器与语码转换攻击器。每个都在独立队列上运行。

5. **攻击套件。** 六个攻击族：(1) PAIR 自动越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色扮演，(6) 多语言语码转换。按攻击族报告成功率。

6. **宪法式自我批判。** 整理 1k 条有害尝试 prompt。对每条，目标模型起草回复。批评者 LLM 按照成文宪法（「不得伤害」「引用证据」「拒绝非法请求」）打分。批评者提出异议的 prompt 会被改写；目标模型在「批判—改进」配对上做微调。在留出评测集上衡量前后无害性。

7. **过度拒答测量。** 在良性 prompt 套件（如 XSTest）上跟踪误报率。目标模型在良性问题上必须保持有帮助。

8. **CVSS 评分。** 对每次成功的越狱，用 CVSS 4.0 评分（攻击向量、复杂度、影响）。产出披露时间线与缓解计划。

9. **靶场自动化。** 上述一切都在 cron 上运行；发现写入队列；过度拒答回归告警发送到 Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付它

`outputs/skill-safety-harness.md` 是交付物：一套生产级的分层安全流水线，外加一个可复现的红队靶场，并附带前后对比的无害性变化量。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 演练 6 类以上攻击族、2 种以上语言 |
| 20 | 真阳性 / 假阳性权衡 | 攻击拦截率与 XSTest 良性通过率对比 |
| 20 | 自我批判变化量 | 留出评测集上的前后无害性 |
| 20 | 文档与披露 | 带时间线的 CVSS 评分发现 |
| 15 | 自动化与可重复性 | 一切都在 cron 上运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，比较有、无输出过滤层时的攻击成功率。

2. 增加第七个攻击族：通过检索文档实施的间接提示注入。测量需要额外增加的防御。

3. 实现「带帮助的拒绝」模式：当护栏拦截时，目标模型给出更安全的相关回答，而不是生硬拒绝。测量 XSTest 变化。

4. 多语言覆盖缺口：找出 X-Guard 表现欠佳的一种语言。提出针对该语言的微调数据集。

5. 在 30B 模型上运行宪法式自我批判，测量增益是否随规模扩展。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | 「纵深防御」 | 在输入、闸门、输出、HITL 处设置多重防护栏 |
| Llama Guard 4 | 「Meta 的安全分类器」 | 2026 年参考级的输入/输出内容分类器 |
| PAIR | 「越狱智能体」 | 关于 LLM 驱动越狱发现的论文（Chao 等） |
| TAP | 「Tree-of-Attacks」 | PAIR 的树搜索变体 |
| GCG | 「贪心坐标梯度」 | 基于梯度的对抗后缀攻击 |
| 宪法式自我批判 | 「Anthropic 风格训练」 | 目标起草 -> 批评者打分 -> 改写 -> 重新训练 |
| XSTest | 「良性探针集」 | 用于过度拒答回归的基准 |
| CVSS 4.0 | 「严重性评分」 | 安全发现的标准漏洞评分体系 |

## 延伸阅读

- [Anthropic 的 Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练期参考
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard（arXiv:2504.08848）](https://arxiv.org/abs/2504.08848) — 132 种语言的多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail 框架
- [PAIR（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
