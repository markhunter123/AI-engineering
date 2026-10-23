# Agent 经济、token 激励与声誉

> 长程自主 Agent（METR 的 1 小时到 8 小时工作曲线）需要经济代理能力。新兴的**五层栈**是：**DePIN**（物理算力）→ **身份**（W3C DID + 声誉资本）→ **认知**（RAG + MCP）→ **结算**（账户抽象）→ **治理**（Agentic DAO）。生产级 Agent 激励网络包括 **Bittensor**（TAO 子网奖励特定任务的模型）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（基于 transformer 的 PoW，把算力重新分配给有产出的 AI 任务）。学术工作：AAMAS 2025 的去中心化 LaMAS 使用 **Shapley 值贡献归因**来公平奖励有贡献的 Agent；Google Research 的「Mechanism design for large language models」提出在单调聚合下的 **token 拍卖**，采用二价支付。本课构建一个最小 Agent 市场，在多 Agent pipeline 上应用 Shapley 值贡献归因，并运行一次二价 token 拍卖，让博弈论机制落到实处。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

当 Agent 共同创造价值却需要单独获得奖励时，多 Agent 系统就变得复杂。经典机制——平均分配、最后贡献者通吃——要么不公平，要么容易被钻空子。基于联盟的 Shapley 值奖励在构造上公平，但计算昂贵。2025-2026 年的文献推动了实用的近似：Shapley 采样、单调聚合拍卖，以及根据已确认贡献累积的链上声誉。

除了贡献归因，这一领域还转向了真正的经济 Agent：Bittensor TAO 奖励挖矿算力来微调子网专属模型，Fetch.ai/ASI 用 FET token 奖励 ASI-1 Mini LLM 的使用，Gonka 把 transformer 工作量证明重新分配给有产出的 AI 任务。自主交易的 Agent 今天已经存在；问题在于如何对齐激励。

本课把 Agent 经济视为一个具体的问题族——贡献归因、机制设计和声誉——并用最小数学逐一构建，让这些想法扎根。

## 概念

### 五层 Agent 经济栈

1. **DePIN（物理算力）。** 出租 GPU、存储、带宽的去中心化基础设施。Bittensor 子网、Render Network、Akash。并非 Agent 专用；Agent 使用它。
2. **身份。** W3C 去中心化标识符（DID）给每个 Agent 一个独立于任何平台的持久 ID。声誉累积到 DID 上。Agent Network Protocol（ANP）用 DID 作为发现层。
3. **认知。** Agent 的推理循环：LLM + RAG + MCP。这是其他阶段构建的内容。
4. **结算。** 账户抽象（ERC-4337）让 Agent 用自身余额支付 gas，无需持有 ETH。Agent 可以为服务、彼此或算力付费。
5. **治理。** Agentic DAO：人类*和*Agent 共同对协议变更投票的治理结构，投票权与声誉挂钩。

并非每个生产系统都用到全部五层。Bittensor 用了 1、2，部分用 3、部分用 4，完全没用 5。OpenAI 的 Agent 除了 3 都不用。这个栈是参考地图，不是硬性要求。

### Bittensor、Fetch.ai、Gonka —— 实际运行的东西

**Bittensor（TAO）。** 子网是专门任务（语言建模、图像生成、预测）。矿工提交模型输出。验证者对其排序；按质押加权的评分分配 TAO 奖励。每个子网有自己的评估。经济学教训：为特定任务的输出质量付费，而不是为消耗的算力付费。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 运行在 Fetch.ai 的网络上；用户用 FET token 支付推理费用。这里的「Agent 即对等节点」叙事更强：Fetch 上的一个 Agent 可以就某项任务调用另一个 Agent，并用 FET 支付。

**Gonka。** Transformer 工作量证明：「工作」是 transformer 的前向传播。矿工通过运行具有已知正确输出（来自训练数据）的推理任务来赚取收益。这是资源有产出的 PoW，而非基于哈希的 PoW。

截至 2026 年 4 月，三者都已达到生产级。收益分配方式不同。Bittensor 相对子网验证者奖励质量；Fetch 以付费用户衡量的效用作为奖励；Gonka 奖励可验证的推理工作。

### Shapley 值贡献归因

三个 Agent 协作完成一项任务。输出得分为 0.8。谁的贡献是多少？

Shapley 值：满足四个公理（效率、对称、线性、零贡献）的唯一贡献分配。对于 Agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是在排序 `O` 中位于 `i` 之前的 Agent 集合。实践中：枚举所有排列，记录每个 Agent 在每个排列中的边际贡献，再取平均。

对于 N=3 个 Agent，有 6 种排列。N=10 时有 360 万种——所以实践中你会采样排序，而不是枚举。

### 用于聚合的二价拍卖

Google Research（「Mechanism design for large language models」）提出用二价 token 拍卖聚合 LLM 输出。设置：N 个 Agent 各自提出一个补全；每个 Agent 对被选中都有一个私人价值。拍卖者选择价值最高的提案，并支付*第二高*的价值。在单调聚合下（价值取决于选中哪个提案，而非出价数量），这是诚实的——Agent 会按真实价值出价。

这对 LLM 系统为何重要：你可以把补全任务外包给多个定价不同的 Agent；拍卖选出最佳并公平支付，而 Agent 没有虚报的动机。

### 声誉资本

与 DID 绑定的声誉分数从已确认的贡献中累积。一个简单的更新规则：

```math
\text{rep}(i, t+1) = \alpha \cdot \text{rep}(i, t) + (1 - \alpha) \cdot \text{contribution\_quality}(i, t)
```

衰减因子 `alpha` 接近 1。声誉：

- 读取成本低，可用于路由决策（「把困难任务交给高声誉 Agent」）。
- 伪造成本高（随时间累积，绑定到 DID）。
- 可以被削减：未通过验证的贡献会被扣减。

### AAMAS 2025 去中心化 LaMAS

LaMAS 提案（AAMAS 2025）结合了：DID 身份、Shapley 值贡献归因，以及一个简单的拍卖机制。关键主张：把贡献归因步骤去中心化，使系统可审计，并能抵御单点操纵。

### 经济学在哪里崩塌

- **价格预言机操纵。** 如果贡献函数可以被钻空子，Agent 就会去钻。每个机制都需要对抗性测试。
- **女巫攻击（Sybil attack）。** 一个运营者启动 N 个假 Agent 来抬高自己的贡献。DID 能减慢但无法阻止；声誉的伪造成本是缓解手段。
- **验证成本。** 贡献归因的公平程度取决于验证者。如果验证便宜（小 LLM），就可能被钻空子；如果昂贵（人工评审团），系统就无法扩展。
- **监管悬顶。** Agent 经济与金融监管相交。截至 2026 年，Bittensor、Fetch 和 Gonka 在某些司法管辖区都处于法律灰色地带。

### Agent 经济何时合理

- **异构运营者的开放网络。** 没有单一团队控制所有 Agent。
- **可验证的输出。** 没有验证，贡献归因只能靠猜。
- **长程工作流。** 一次性任务无法从声誉累积中获益。
- **token 化支付在你的司法管辖区法律上可行。**

在封闭的企业系统中，经济学让位于更简单的分配（经理分配工作，指标是内部的）。经济学文献主要适用于开放网络。

```figure
swarm-auction
```

## 构建它

`code/main.py` 实现了：

- `shapley(value_fn, agents)`——通过枚举计算小 N 的精确 Shapley 值。
- `second_price_auction(bids)`——诚实机制；赢家支付第二高价。
- `Reputation`——与 DID 绑定、带指数衰减和削减的声誉。
- 演示 1：三个 Agent 协作，用精确 Shapley 归因贡献。
- 演示 2：五个 Agent 竞标一个任务槽；二价拍卖选出赢家与支付额。
- 演示 3：100 轮任务分配给声誉各异的 Agent；按声誉加权路由胜过随机路由。

运行：

```
python3 code/main.py
```

预期输出：每个 Agent 的 Shapley 值；显示诚实出价均衡的拍卖结果；按声誉加权路由在预热后比随机路由获得 10-20% 的质量提升。

## 使用它

`outputs/skill-economy-designer.md` 设计一个最小 Agent 经济：身份层的选择、贡献归因机制、支付机制、声誉规则。

## 交付它

在 2026 年运营 Agent 经济：

- **从声誉开始，而不是 token。** 声誉实现成本低且单独就有价值；token 会带来法律和经济复杂性。
- **先验证，再奖励。** 绝不在没有独立验证步骤的情况下分配贡献。自报质量会招致女巫攻击。
- **用 Shapley 采样，而非精确计算。** 采样 100-1000 种排序；精确枚举无法扩展。
- **限制衰减因子并为声誉设下限。** 无界衰减会抹掉合法贡献者；衰减太慢则会奖励陈旧的高声誉 Agent。
- **以对抗方式审计机制。** 开放网络前先跑红队场景。每个机制都有博弈论；你要找到漏洞，而不是让攻击者找到。

## 练习

1. 运行 `code/main.py`。确认 Shapley 值之和等于总价值（效率公理）。改变价值函数；Shapley 分配是否朝预期方向变化？
2. 实现 Shapley *采样*（在 K 种排序上做蒙特卡洛）。K 如何影响近似精度？在 N=4 时与精确解比较。
3. 在拍卖前实现一个结盟步骤：Agent 可以合并成团队，作为一个整体竞标。会形成哪些联盟？结果是否比单独竞标帕累托更优？
4. 阅读 Google Research 的机制设计文章。找出一个假设，一旦违反就会破坏诚实性。在 LLM 场景中，这种失效模式是什么样子？
5. 阅读 AAMAS 2025 去中心化 LaMAS 论文。在一个合成任务上，对 10 个 Agent 实现他们的 Shapley 步骤。精确计算需要多久？100 次抽样能逼近到什么程度？

## 关键术语

| 术语 | 常见说法 | 真实含义 |
|------|----------------|------------------------|
| DePIN | 「去中心化物理基础设施」 | token 激励的算力/存储/带宽。Bittensor、Akash、Render。 |
| DID | 「去中心化标识符」 | 可移植 ID 的 W3C 规范。Agent 声誉绑定到 DID，而非平台。 |
| ERC-4337 | 「账户抽象」 | 可赞助 gas 的合约账户，使 Agent 支付成为可能。 |
| Shapley 值 | 「公平的贡献归因」 | 满足效率、对称、线性、零贡献公理的唯一分配。 |
| 二价拍卖 | 「Vickrey 拍卖」 | 诚实机制：赢家支付第二高出价。兼容单调聚合。 |
| 声誉资本 | 「累积的质量分数」 | 与 DID 绑定、来自已确认贡献的分数；随时间衰减。 |
| Agentic DAO | 「Agent + 人类共同治理」 | Agent 选民为一等公民的 DAO，投票权与声誉挂钩。 |
| TAO / FET / GPU credits | 「token 面额」 | Bittensor TAO、Fetch.ai FET、各种 DePIN token。 |

## 延伸阅读

- [Agent 经济](https://arxiv.org/abs/2602.14219) — 2026 年五层 Agent 经济栈综述
- [Google Research — 面向大型语言模型的机制设计](https://research.google/blog/mechanism-design-for-large-language-models/) — 带单调聚合的 token 拍卖
- [AAMAS 2025 — 去中心化 LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley 值贡献归因
- [Bittensor TAO 文档](https://docs.bittensor.com/) — 子网结构与奖励分配
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 与 FET token
- [W3C 去中心化标识符（DID）规范](https://www.w3.org/TR/did-core/) — 身份基础
