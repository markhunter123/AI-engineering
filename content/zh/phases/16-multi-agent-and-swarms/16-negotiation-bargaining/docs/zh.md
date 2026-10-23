# 谈判与议价

> 智能体会就资源、价格、任务分配和条款进行谈判。2026 年的基准证据很清楚：NegotiationArena（arXiv:2402.05863）表明 LLM 通过人设操纵（「绝望感」）可以把收益提高约 20%；「Measuring Bargaining Abilities」（arXiv:2402.15813）表明买方比卖方更难，而且扩大规模没有帮助——他们的 **OG-Narrator**（确定性报价生成器 + LLM 叙述者）把成交率从 26.67% 提升到 88.88%；大规模自主谈判竞赛（arXiv:2503.06416）跑了约 18 万场谈判，发现**隐藏思维链（chain-of-thought-concealing）**的智能体靠不让对手看到推理过程而获胜；Bhattacharya 等人 2025 年基于哈佛谈判项目指标的排名是：Llama-3 最有效，Claude-3 最具攻击性，GPT-4 最公平。本课实现合同网协议（Contract Net Protocol，FIPA 的前身，第 02 课），接上一对 LLM 风格的买方/卖方，跑一遍 OG-Narrator 式的分解，并测量成交率如何随每个结构性选择变化。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

两个智能体需要就一个价格达成一致。如果只用纯语言 prompt 让它们自行发挥，2024-2026 年的 LLM 成交率低得惊人（在 arXiv:2402.15813 参数化严格的议价中约为 27%）。扩大规模解决不了：GPT-4 在议价上并没有结构性优于 GPT-3.5；它只是更擅长议价的*语言*。

根本问题在于 LLM 把两件事混为一谈——决定报价与叙述报价。OG-Narrator 把二者分开：确定性的报价生成器负责计算数值动作，LLM 只负责叙述。成交率跃升到约 89%。

这呼应了一个经典的多智能体发现：把机制与通信层解耦就能赢。合同网协议（Contract Net Protocol，FIPA，1996；Smith，1980）是任务市场的参考机制。把 LLM 插进叙述的位置，你就得到了一个现代的 LLM 驱动任务市场。

## 概念

### 一段话讲清合同网

Smith 1980 年的合同网协议：**管理者（manager）**广播一份**提案征集（call for proposals，cfp）**；**投标者（bidder）**用包含报价的 **propose** 消息回应；管理者选出赢家，向赢家发送 **accept-proposal**，向输家发送 **reject-proposal**。赢家执行工作。可选消息：**refuse**（投标者拒绝提案）。FIPA 将其规范化为 `fipa-contract-net` 交互协议。

### OG-Narrator 为什么有效

「Measuring Bargaining Abilities of Language Models」（arXiv:2402.15813）观察到：

- LLM 经常违反议价规则（报出荒谬的价格，忽视对方的 ZOPA）。
- 它们的锚定很差（接受糟糕的首次报价；还价用的是象征性数字而非策略性金额）。
- 仅靠扩大规模解决不了这些。更大的模型能写出更可信的语言，但策略性错误类似。

OG-Narrator 的分解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

报价生成器是经典的谈判策略：Rubinstein 议价模型、Zeuthen 策略，或简单的价格针锋相对。LLM 负责叙述。消息中包含确定性的价格和自然语言的包装。

成交率跃升的原因：
- 价格保持在议价区间内。
- 锚点是策略性的，而非情绪化的。
- LLM 做它擅长的事：写作。

### NegotiationArena 的发现

arXiv:2402.05863 提供了权威基准。主要发现：

- LLM 通过采用人设（「我必须在周五前卖掉这个」）可以把收益提高约 20%——人设操纵是真实有效的战术。
- 公平/合作的智能体会被对抗性智能体利用；防御需要明确的反制姿态。
- 在约 40% 的基准场景中，对称配对会收敛到不公平的结果。

这不是「LLM 是不好的谈判者」。而是「LLM 谈判得太像人类了，包括那些可被利用的部分」。

### 思维链隐藏

大规模自主谈判竞赛（arXiv:2503.06416）在多种 LLM 策略上跑了约 18 万场谈判。赢家都不让对手看到自己的推理：

- 如果一个智能体在公开可见的草稿纸上写下「我最多只出到 75 美元；我的保留价是 70」，对手就会读到它。
- 赢家在私密环境中计算策略；输出通道只包含报价和最低限度的必要叙述。

这是 2026 年对经典博弈论（Aumann 1976 关于理性与信息）的回响：暴露你的私人估值会损失收益。LLM 不会直觉地意识到这一点，还会乐呵呵地把保留价写进推理轨迹，而这轨迹对对手可见。

工程要点：把私密草稿纸上下文与公开消息上下文分开。这不是可选项。

### Bhattacharya 等人 2025 —— 模型排名

在哈佛谈判项目指标上（原则性谈判、尊重 BATNA、利益互惠）：

- **Llama-3** 在达成交易上最有效（成交率 + 收益）。
- **Claude-3** 是最具攻击性的谈判者（高锚点、延迟让步）。
- **GPT-4** 最公平（不同配对间收益方差最小）。

这是 2025 年的快照。重点不是 2026 年 4 月哪个模型获胜——而是不同基础模型有各自持久的谈判风格。异构集成（第 15 课）把它作为一种多样性来源。

### 用合同网 + LLM 做任务分配

合同网在 LLM 多智能体中的现代复用：

1. 管理者智能体把任务分解为工作单元。
2. 向工作者智能体广播带有任务描述的 `cfp`。
3. 每个工作者返回一个报价：`(price, eta, confidence)`，其中 price 可以是 token、算力单位或美元。
4. 管理者选出赢家（单个或多个，视任务而定）并授予任务。
5. 被拒绝的工作者可以自由投标其他任务。

这种模式在超过 100 个工作者时仍能良好扩展，因为协调是广播-响应式的，而不是同步聊天。生产中的使用：Microsoft Agent Framework 的编排模式、部分 LangGraph 实现。

### LLM-Stakeholders 交互式谈判

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入了带**秘密分数**和**最低接受阈值**的多方可评分博弈。每个利益相关方都有私人效用；LLM 必须从消息中推断它们。这是把双方议价推广到 N 方联盟形成。对于能力异构的生产任务市场很有参考价值。

### 叙述与机制的规则

纵观所有 2024-2026 年的谈判基准，一致的工程规则是：

> 让 LLM 负责叙述。不要让 LLM 计算报价。

如果报价必须是一个数字（价格、ETA、数量），就从谈判状态中确定性地生成它，再让 LLM 生产包装话术。如果报价必须是一种提案结构（任务分解、角色分配），可以让 LLM 起草，但在发送前要按 schema 校验并做约束检查。

```figure
a5-og-narrator
```

## 构建它

`code/main.py` 实现了：

- `ContractNetManager`、`ContractNetTask`、`Bid` —— 管理者 + 投标者，广播 cfp，收集提案，授予任务。
- `og_narrator_bargain(state, rng)` —— OG-Narrator 买方：确定性的 Zeuthen 式让步，朝中点靠拢。
- `seller_response(state, rng)` —— 确定性的卖方还价策略（两种风格共同的结构性基准）。
- `naive_llm_bargain(state, rng)` —— 模拟全 LLM 议价者：报价方差大，常常超出 ZOPA。
- 测量：在 1000 次试验中统计成交率，每次试验重新采样保留价。

运行：

```
python3 code/main.py
```

预期输出：朴素 LLM 成交率约 65-75%；OG-Narrator 成交率约 85-95%；这 15-25 个百分点的差距，就是把报价生成与叙述分离带来的结构性优势。另附一个合同网任务市场分配示例，含三个投标者和一个任务。

## 使用它

`outputs/skill-bargainer-designer.md` 用于设计议价协议：由谁生成报价（确定性还是 LLM）、由谁叙述、私密草稿纸如何与公开消息隔离，以及如何监控成交率。

## 交付它

生产议价检查清单：

- **隔离草稿纸。** 私密状态绝不进入对方的上下文。这一点没有商量余地。
- **确定性报价生成。** 价格、数量、ETA：计算，不要用 prompt 生成。
- **校验所有传入报价**是否符合 schema。在协议边界拒绝超出 ZOPA 的报价。
- **限制轮数。** 最多 3-5 轮；僵局时升级给调解者。
- **持续测量成交率和收益方差。** 成交率下降是一个症状——通常是 prompt 漂移或对方发起的攻击。
- **记录所有被拒绝的提案**及其确定性理由。对于合同网管理者，落标的投标者需要理解原因。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在成交率上胜过朴素 LLM。差距有多大？
2. 实现**基于人设的收益提升**（arXiv:2402.05863）——买方仅在叙述中采用「这周急着买」的人设，报价生成器不变。成交率或收益会变化吗？
3. 实现思维链**隐藏**：维护一个不传给对方的私密草稿纸字符串。如果你不小心泄露了（用交换通道来模拟），会发生什么？
4. 把合同网扩展为带保留价的 N 投标者拍卖。当所有出价都超过保留价时，管理者如何在最低价与最高质量之间决定？你选哪种授予规则，为什么？
5. 阅读 Bhattacharya 等人 2025 年基于哈佛谈判项目指标的论文。实现两个不同风格的议价者（攻击型 vs 公平型）。测量对称与不对称配对下的收益方差。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| Contract Net | "任务市场" | Smith 1980，FIPA 1996。cfp + propose + accept/reject。经典任务市场。 |
| ZOPA | "可能达成协议的区域" | 买方最高价与卖方最低价的重叠区间。区间外的报价无法成交。 |
| BATNA | "谈判协议的最佳替代方案" | 交易失败时你的退路。它决定你的保留价。 |
| OG-Narrator | "报价生成器 + 叙述者" | 分解：确定性报价，LLM 叙述。 |
| Zeuthen strategy | "风险最小化的让步" | 经典的报价生成器，基于风险限额做出让步。 |
| Rubinstein bargaining | "交替出价的均衡" | 带折扣的无限期议价的博弈论模型。 |
| CoT concealment | "藏好你的推理" | arXiv:2503.06416 的赢家保持私密草稿纸；公开通道只显示报价。 |
| Persona manipulation | "情绪化姿态" | arXiv:2402.05863：绝望/紧迫人设带来约 20% 的收益提升。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) —— 基准；人设操纵与利用行为的发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) —— OG-Narrator 与「买方比卖方更难」的结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) —— 约 18 万场谈判；隐藏思维链者获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) —— 带秘密效用的多方可评分博弈
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) —— 经典机制，IEEE Transactions on Computers
