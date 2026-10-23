# 面向 Agent 的共识与拜占庭容错

> 经典分布式系统的 BFT 遇上了随机性的 LLM。2025-2026 年出现了三个研究方向：**CP-WBFT**（arXiv:2511.10400）用置信度探针为每一票加权；**DecentLLMs**（arXiv:2507.14928）采用无 leader 的方式，并行收集工作者的提案，再做几何中位数聚合；**WBFT**（arXiv:2505.05103）把加权投票与层次结构聚类（Hierarchical Structure Clustering）结合，把节点分为 Core 与 Edge 两层。「Can AI Agents Agree?」（arXiv:2603.01213）给出的诚实经验结果是：即使是标量问题上的共识，今天也依然脆弱——一个具有欺骗性的 Agent 就能破坏 Mixture-of-Agents 的共识。BFT 是必要的，但不充分。本课构建一个最小化的 BFT 协议，注入三种 Agent 特有的攻击（拜占庭谎言、谄媚性从众、相关错误单一文化），并衡量各共识变体如何应对。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题

你有 N 个 LLM Agent，各自给出一个答案。它们意见不一。多数投票选出了错误的那个，因为其中两个 Agent 是相关的（同一基础模型、同一训练数据、同样的失效模式）。第三个 Agent 恰好以一种新奇的方式错了——于是多数只是虚假的多数。

再加上一个欺骗性 Agent：它故意撒谎。或者一个谄媚性 Agent：谁最后发言它就附和谁。经典 BFT 假设拜占庭节点占比为 `f < n/3`，且行为任意。而 2026 年的现实是：LLM 节点即使诚实也是随机的，跨模型之间存在相关性，并且会互相影响输出。你不能把它们当作独立的伯努利投票者。

经典 BFT（PBFT，1999）并没有错——它只是不完整。它能处理任意的比特翻转。它处理不了「三个诚实 Agent 因为共享训练数据而共享同一个幻觉」。本课从 PBFT 的基础出发，叠加三项 2025-2026 年的适配。

## 概念

### 经典 BFT 提供了什么

实用拜占庭容错（Practical Byzantine Fault Tolerance，Castro & Liskov，OSDI 1999）可容忍 `f < n/3` 个拜占庭节点。协议包含三个阶段（pre-prepare、prepare、commit）和两个原语（签名消息、法定人数证书）。在 `n >= 3f + 1` 个诚实或恶意节点之间，对单一值达成一致。

这些保证很强，但假设：

1. **故障相互独立。** 拜占庭节点不会协同行动。
2. **诚实节点真正诚实。** 诚实输出的正确性不是问题；协议只负责对齐分歧。
3. **问题有客观正确答案。** 对一个错误事实达成共识，仍然是共识。

LLM Agent 违反了这三条。两个运行同一基础模型的 Agent 共享故障。「诚实」的 LLM 仍然会产生幻觉。而在模糊问题上，「真相」就是 Agent 们决定的东西——不存在外部神谕。

### 三种 LLM 特有的攻击

**拜占庭谎言。** 一个 Agent 故意输出错误答案。只要 `f < n/3`，经典 BFT 就能处理。

**谄媚性从众。** 一个 Agent 在投票前读取他人的答案，然后附和最后发言的人。这并非恶意，但会与声音最大者相关。经典 BFT 无法阻止，因为该 Agent 能通过所有签名检查。

**相关错误单一文化。** 三个 Agent 共享同一基础模型。它们幻觉出同一个错误答案。多数是错的。经典 BFT 帮不上忙，因为三者是「诚实地」一致。

### 2025-2026 年的应对方案

**CP-WBFT**（arXiv:2511.10400）—— Confidence-Probed Weighted BFT。每个投票者给自己的答案附上一个置信度探针（自报概率，或由独立校准模型给出的预测）。票权随置信度缩放。报告称在完全图上 BFT 提升了 85.71%。针对：谄媚性从众（从众 Agent 对自己主动表明的立场往往置信度低）。

**DecentLLMs**（arXiv:2507.14928）—— 无 leader。工作者 Agent 并行提案，评估者 Agent 为提案打分，最终答案是已打分位置的几何中位数。在 `f < n/2` 时稳健。针对：拜占庭谎言和相关错误（几何中位数对离群值稳健，会向密集簇靠拢，而不是偏向有模型偏差的均值）。

**WBFT**（arXiv:2505.05103）—— 带层次结构聚类的加权 BFT。票权由响应质量加上从历史中学到的信任分来分配。把 Agent 聚成 Core 和 Edge；Core Agent 必须先达成共识，Edge Agent 再跟随。针对：可扩展性（Core 共识规模小、速度快），部分针对单一文化（Core 可以按多样性来挑选）。

### 实证：「Can AI Agents Agree?」（arXiv:2603.01213）

该论文在多个前沿模型上测量标量一致性（LLM Agent 对单个数值达成一致）。结论令人不安：

- 即使没有对手，在许多基准上，LLM Agent 在标量问题上的分歧率也超过 30%。
- 一个采用欺骗性人设的 Agent，可以把 Mixture-of-Agents 的共识从诚实基线拉偏 40 多个百分点。
- 分歧率与模型多样性相关——异构集成比同构集成分歧更多（好事：错误不相关），但收敛也更慢（坏事：达成一致所需时间更长）。

要点：BFT 给了你对齐输出的机制，但它不告诉你对齐后的输出是否正确。要结合验证（Phase 16 · 08 角色专业化）、多样性（Phase 16 · 15 辩论变体）和评估者 Agent（Phase 16 · 24 基准）一起使用。

### 核心协议，去掉枝节

面向 LLM Agent 的最小 BFT 轮次：

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

语义聚类这一步是 LLM 特有的调整。两个答案「该研究报告了 4.2%」和「4.2% 的提升」属于同一簇。朴素的字符串相等检查会漏掉这种情况。生产中使用廉价的 embedding 模型或显式的规范化。

### 阈值调参

`threshold` 参数决定何时接受、何时重试。太低：你会接受弱多数。太高：你永远接受不了任何结果。经验范围：`n=5-7` 个 Agent 时取 0.5-0.67，`n` 更小时取更高值。低于阈值时，升级给人类或另一组 Agent。

### 共识帮不上忙的场景

- **模糊问题。** 如果问题没有客观真相，共识只是一种意见。就把它称作意见。
- **复合问题。** 「写代码并解释它」——这是两个答案。要分别投票。
- **对抗性多轮。** 如果 Agent 能观察先前轮次并模仿（Du 2023 辩论），它们会不顾真相地开始互相附和。要限制轮数（通常 2-3 轮）。

```figure
swarm-consensus-wave
```

## 构建它

`code/main.py` 实现了：

- `AgentVoter` —— 一个带（answer、confidence）的脚本化策略。
- `MajorityVote` —— 经典相对多数。
- `CPWBFT` —— 带语义聚类的置信度加权投票。
- `DecentLLMs` —— 对已打分提案做几何中位数聚合。
- `Scenario` —— 在三种攻击模式下运行每个聚合器。

实现的攻击模式：

1. `byzantine`：一个 Agent 以高置信度撒谎。
2. `sycophancy`：一个 Agent 复制它看到的第一个答案，置信度也相匹配。
3. `monoculture`：三个 Agent 共享同一个错误答案（相关错误），置信度中等。

运行：

```
python3 code/main.py
```

预期输出：一张（attack, aggregator）-> final answer 的表，正确答案高亮显示。相对多数在单一文化场景下失败。CPWBFT 的置信度加权缓解谄媚。当单一文化规模不到人群一半时，DecentLLMs 的几何中位数会向诚实簇靠拢。

## 使用它

`outputs/skill-consensus-designer.md` 为多 Agent 集成设计共识协议：聚类方法、加权方式、阈值，以及低于阈值的轮次的升级策略。

## 交付它

在交付任何共识机制之前：

- **至少用上述三种模式做攻击测试。** 你的协议应当可预测地失败，而不是悄无声息地失败。
- **记录每个少数簇**及其来源。少数簇是相关错误的早期预警系统。
- **强制有界轮数。** 不要「一直辩论到达成一致」——那会奖励谄媚。
- **把一致性与正确性分开。** 共识输出交给校验器；校验器独立于集成之外。
- **监控一致率。** 急剧上升意味着从众偏差；急剧下降意味着模型漂移。

## 练习

1. 运行 `code/main.py`。确认相对多数在单一文化攻击下失败，而当单一文化置信度低于 0.7 时 CPWBFT 能部分缓解。
2. 增加第四种攻击模式：**沉默弃权**——一个 Agent 拒绝回答（「我不知道」）。每个聚合器应如何处理弃权？实现你的选择。
3. 把语义聚类从字符串规范化换成 embedding 相似度（可使用任何开源 embedding 模型）。谄媚攻击会变成什么样？
4. 阅读 CP-WBFT（arXiv:2511.10400）。实现置信度探针校准步骤（由一个独立校准模型检查每个 Agent 自报的置信度）。测量在单一文化场景下的准确率提升。
5. 阅读「Can AI Agents Agree?」（arXiv:2603.01213）。复现一个简化的标量一致性实验：三个 Agent、一个标量问题、欺骗性人设 prompt。CPWBFT 或 DecentLLMs 能抓住它吗？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| BFT | "拜占庭容错" | Castro-Liskov 1999 年的协议，在 `f < n/3` 任意故障下达成共识。 |
| Byzantine | "任何坏行为" | 可以撒谎、丢消息、静默失败的节点——除了安全崩溃，什么都可能做。 |
| Confidence probe | "你有多确定？" | 附在投票上的自报或校准器预测的概率。 |
| Semantic clustering | "同一个答案，不同的说法" | 在计票前把等价答案分组。 |
| Geometric median | "稳健的中心" | 到样本点距离之和最小的点。与均值不同，它对离群值稳健。 |
| Monoculture | "同一模型，同样的失败" | Agent 共享训练数据或基础模型时产生的相关错误。 |
| Sycophantic conformity | "附和响亮的声音" | Agent 的投票偏向最先/最响亮发言者。 |
| Core/Edge | "层次化 BFT" | WBFT 的划分：先由小规模 Core 达成共识，Edge 节点跟随。以此约束延迟。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://www.usenix.org/conference/osdi-99/practical-byzantine-fault-tolerance) —— 基础
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) —— 按置信度加权投票
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) —— 几何中位数聚合
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) —— Core/Edge 划分以约束延迟
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) —— 标量一致性的脆弱性与欺骗性人设攻击
