# 红队测试：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——Prompt Automatic Iterative Refinement——是经典的自动化黑盒越狱方法。一个带有红队系统提示的攻击者 LLM 迭代地为目标 LLM 提出越狱方案，并在自己的聊天历史中累积尝试与响应，作为上下文反馈。PAIR 通常在 20 次查询内成功，比 GCG（Zou 等人的 token 级梯度搜索）高效数个数量级，且不需要白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代改进、上下文反馈。
- 解释为什么当目标是黑盒时，PAIR 严格比 GCG 更高效。
- 说出另外四个自动化攻击基线（GCG、AutoDAN、TAP、PAP），并说明各自的一个显著特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议，以及各自语境下「攻击成功率」的含义。

## 问题

红队测试过去是一项人工活动。少数专家测试者构造对抗提示，记录哪些有效。这无法扩展：攻击成功率需要统计样本，而每次模型发布都会让目标变成移动靶。PAIR 把红队测试操作化为一个以黑盒为目标的优化问题。

## 概念

### PAIR 算法

输入：

- 目标 LLM T（被攻击的模型）。
- 评判 LLM J（给响应是否构成越狱打分）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G：「以 [有害指令] 回应。」
- 预算 K（通常 20 次查询）。

循环，对 k = 1..K：

1. 向 A 提供目标 G 以及到目前为止的（提示, 响应）对历史。
2. A 输出一个新提示 $p_k$。
3. 把 $p_k$ 提交给 T，得到响应 $r_k$。
4. J 就目标对 $(p_k, r_k)$ 打分。
5. 若分数 >= 阈值，停止——找到越狱。
6. 否则，把 $(p_k, r_k)$ 追加到 A 的历史；继续。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 的攻击成功率超过 50%；平均成功所需查询数在 10–20 之间。

### 为什么 PAIR 高效

GCG（Zou 等人 2023）通过梯度在对抗性 token 后缀上搜索；它需要白盒模型访问，并产生不可读的后缀。PAIR 是黑盒的，产生可跨模型迁移的自然语言攻击。PAIR 的上下文反馈让攻击者从每次拒绝中学习；GCG 没有等价机制（每次新的 token 更新都必须重新发现此前的进展）。

### 相关自动化攻击

- **GCG（Zou 等人 2023，arXiv:2307.15043）。** 针对对抗后缀的 token 级梯度搜索。白盒、可迁移、产生不可读字符串。
- **AutoDAN（Liu 等人 2023）。** 在提示上进行进化搜索，由分层目标引导。
- **TAP（Mehrotra 等人 2024）。** 带剪枝的攻击树——把多个 PAIR 式 rollout 分支化。
- **PAP（Zeng 等人 2024）。** 说服性对抗提示（Persuasive Adversarial Prompts）——把人类说服技巧编码为提示模板。

### JailbreakBench 与 HarmBench

两者（2024）都标准化了评估：

- JailbreakBench（arXiv:2404.01318）。100 个有害行为，覆盖 10 个 OpenAI 政策类别。以攻击成功率（ASR）为主要指标。需要评判器（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人 2024）。510 个行为，覆盖 7 个类别，包含语义与功能危害测试。比较 18 种攻击对 33 个模型。

ASR 通常在固定查询预算下报告。比较不同攻击时预算必须匹配；200 次查询下 90% 的 ASR 与 20 次查询下 85% 的 ASR 不可比。

### 为什么这对 2026 年的部署很重要

每个前沿实验室现在都在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹会出现在模型卡（第 26 课）和安全论证附录（第 18 课）中。这种攻击并不奇异——它已是标准基础设施。

### 本节在 Phase 18 中的位置

第 12 课是自动化攻击的基础。第 13 课（多轮越狱）是一种互补的长度利用。第 14 课（ASCII 艺术 / 视觉）是一种编码攻击。第 15 课（间接提示注入）是 2026 年的生产攻击面。第 16 课涵盖防御工具的对应物（Llama Guard、Garak、PyRIT）。

```figure
al-pair-loop
```

## 使用它

`code/main.py` 构建了一个玩具 PAIR 循环。目标是一个模拟分类器，会拒绝「明显的」有害提示（关键词过滤）。攻击者是一个基于规则的改进器，尝试改写、角色扮演框架和编码。评判器给响应打分。你可以看到攻击者在约 5–15 次迭代内攻破关键词过滤器，而在语义过滤器上失败。

## 交付它

本课产出 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），各自的预算，使用了哪个评判器，基于哪个有害行为集（JailbreakBench、HarmBench、内部）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略的平均成功查询数。解释每种策略利用了哪个目标防御假设。

2. 实现第四种攻击者策略（例如，翻译成另一种语言、base64 编码）。报告针对关键词过滤目标和语义过滤目标的新平均成功查询数。

3. 阅读 Chao 等人 2023 年图 5（PAIR 与 GCG 对比）。描述两个尽管 PAIR 有高效优势、仍更偏好 GCG 的场景。

4. JailbreakBench 在固定目标集上报告 ASR。设计一个衡量攻击多样性（成功提示的方差）的附加指标。解释为什么多样性对防御评估很重要。

5. TAP（Mehrotra 2024）用分支 + 剪枝扩展 PAIR。为 `code/main.py` 勾画一个 TAP 式扩展，并描述计算成本与成功率之间的权衡。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| PAIR | 「自动化越狱」 | Prompt Automatic Iterative Refinement；攻击者 LLM + 评判 LLM 循环 |
| GCG | 「梯度越狱」 | 针对对抗后缀的白盒 token 级梯度搜索 |
| 攻击成功率（ASR） | 「k 次查询下的越狱百分比」 | 主要指标；必须同时报告查询预算和评判器身份 |
| 评判 LLM | 「打分器」 | 评估响应是否满足有害目标的 LLM |
| JailbreakBench | 「评估集」 | 带类别标签的标准化有害行为集 |
| HarmBench | 「更广的基准」 | 510 个行为，功能 + 语义危害测试 |
| TAP | 「攻击树」 | 带分支 + 剪枝的 PAIR；更高算力下获得更好的 ASR |

## 延伸阅读

- [Chao 等人 — Jailbreaking Black Box LLMs in Twenty Queries（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou 等人 — Universal and Transferable Adversarial Attacks on Aligned LLMs（arXiv:2307.15043）](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao 等人 — JailbreakBench（arXiv:2404.01318）](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika 等人 — HarmBench（ICML 2024）](https://arxiv.org/abs/2402.04249) — 更广泛的评估
