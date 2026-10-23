# Tree of Thoughts 与 LATS：深思熟虑的搜索

> 单条思维链轨迹没有回退的余地。ToT（Yao 等，2023）把推理变成一棵树，在每个节点上做自我评估。LATS（Zhou 等，2024）在蒙特卡洛树搜索下统一了 ToT、ReAct 和 Reflexion。24 点游戏从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% 的 pass@1。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## 学习目标

- 把推理建模为搜索：节点是「思考」，边是「扩展」，价值是「有多大希望」。
- 实现一个 stdlib 的 ToT 风格 BFS 树搜索，带自评估打分。
- 扩展成一个玩具版 LATS MCTS 循环，包含选择 / 扩展 / 模拟 / 反向传播。
- 判断何时搜索值得付出 token 倍数代价（24 点游戏、代码生成），何时单条轨迹就够了（简单问答）。

## 问题

思维链是一次线性行走。如果第一步就错了，后面每一步都建立在错误的前提上。在 24 点游戏（用四个数字和 + − × ÷ 凑出 24）中，GPT-4 的 CoT 只有 4% 的准确率。模型很早就选错了子表达式，之后无法挽回。

推理需要的能力是：提出多个候选、评估它们、挑出有希望的，并在撞上死路时回退。这就是搜索。Tree of Thoughts 和 LATS 是两种经典表述。

## 概念

### Tree of Thoughts（Yao 等，NeurIPS 2023）

每个节点是一个连贯的中间步骤（「一个思考」）。每个节点可以扩展到 K 个子思考。LLM 用打分 prompt 对每个节点做自我评估。搜索遍历这棵树——BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是承重的那一块。论文给出三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分，以及在候选之间投票。三者都在 24 点游戏上大幅胜过 CoT（用 GPT-4 从 4% -> 74%）。

### LATS（Zhou 等，ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **策略（Policy）**：提出候选的下一步行动（ReAct 风格）。
- **价值函数（Value function）**：给部分轨迹打分（ToT 风格的自评估）。
- **自我反思器（Self-reflector）**：失败时写下自然语言反思（Reflexion 风格），并用它重新播种后续的 rollout。

环境反馈（观察结果）会混入价值函数，让搜索由真实的工具结果而非模型的主观意见来指导。论文发表时的结果：HumanEval 用 GPT-4 达到 92.7% pass@1（SOTA），WebShop 用 GPT-3.5 平均 75.9（接近基于梯度的微调）。

### MCTS 极简版

每次迭代分四个阶段：

1. **选择（Select）**——用 UCT（树的置信上界）从根走到叶。
2. **扩展（Expand）**——通过策略生成 K 个子节点。
3. **模拟（Simulate）**——从某个子节点出发用策略做 rollout，用价值函数（或环境奖励）给叶子打分。
4. **反向传播（Backpropagate）**——沿路径向上更新访问计数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。`c` 要按任务调。

### 成本现实

搜索会让 token 用量爆炸。ToT 在 24 点游戏上使用的 token 是 CoT 的 100–1000 倍。LATS 也差不多。这不是免费的；把搜索留给以下情况：

- 单条轨迹明显不够用的任务（24 点游戏、复杂代码）。
- 正确性比墙钟时间更重要的任务。
- 拥有廉价可靠价值函数的任务（代码的单元测试、数学的显式目标）。

如果你的任务只有一个正确答案，而评估器又有噪声，搜索常常让结果更糟——它会找到一个「得分很高」的错误答案。

### 2026 年的定位

大多数生产 Agent 并不跑 LATS。它们跑的是 ReAct 加上工具锚定的验证（CRITIC，第 05 课）。搜索出现在一些专门领域：

- 把测试当作价值函数的编码 Agent（HumanEval 风格）。
- 探索多条查询路径的深度研究 Agent。
- LangGraph 子图里以规划为重的工作流。

AlphaEvolve（第 11 课）是 2025 年的极端案例：对代码做进化搜索，适应度可机器验证，取得前沿进展（56 年来首次改进 4x4 矩阵乘法）。

```figure
tree-of-thoughts
```

## 构建它

`code/main.py` 实现了：

- 在一个风格化的「挑选算术运算」任务上运行一个微型 ToT BFS。
- 在同一任务上运行一个玩具版 LATS MCTS 循环（选择 / 扩展 / 模拟 / 反向传播），使用 UCT 选择。
- 一个由符号分数加自评估分数组成的价值函数。

运行它：

```
python3 code/main.py
```

轨迹展示 ToT 用 BFS 在每个节点扩展三个候选，并与 LATS 通过 MCTS 收敛到最佳 rollout 的过程作对比。两者都会打印 token 数。

## 使用它

LangGraph 以子图模式提供 ToT 风格的探索；LangChain 团队 2024 年 5 月关于 LATS 的博客是参考教程。LlamaIndex 提供了一个 `TreeOfThoughts` Agent。对大多数 2026 年的生产 Agent 而言，这个模式藏在一个 `if task_complexity > threshold: use_search()` 的开关后面——参见第 05 课中的 evaluator-optimizer 模式。

## 交付它

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间做选择。

## 练习

1. 分别用 UCT c=0.1 和 c=2.0 运行玩具版 LATS。轨迹里有什么变化？
2. 把价值函数换成一个更有噪声的打分器（加入随机抖动）。MCTS 还能找到最优叶子吗？它能容忍的最低信噪比是多少？
3. 实现 beam search 版 ToT（每层保留 top-k），并与 BFS 对比。在紧张的 token 预算下哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 的轨迹数量：需要多少次 rollout 才能达到报告中的 pass@1？
5. 阅读 LATS 论文中关于「LATS 何时帮助较小」的讨论。写一段决策规则，把任务形态映射到搜索策略。

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | 「分支式 CoT」 | Yao 等——带自我评估的思考节点树 |
| LATS | 「面向 LLM 的 MCTS」 | Zhou 等——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | 「置信上界」 | 选择公式，平衡利用（Q）与探索（ln N / n） |
| Value function | 「这个状态有多好」 | 提示 LLM 给出的分数或环境奖励；供反向传播使用 |
| Policy | 「行动提议者」 | ReAct 风格的生成器；输出候选的下一步思考 / 行动 |
| Rollout | 「模拟轨迹」 | 用策略从一个节点走到叶子，用价值函数打分 |
| Backpropagate | 「更新祖先」 | 把叶子的奖励沿路径向上传递，更新访问计数和 Q 值 |
| Search cost | 「token 爆炸」 | 在 24 点游戏上是 CoT 的 100–1000 倍；采用前先做预算 |

## 进一步阅读

- [Yao 等，Tree of Thoughts（arXiv:2305.10601）](https://arxiv.org/abs/2305.10601)——经典论文
- [Zhou 等，LATS（arXiv:2310.04406）](https://arxiv.org/abs/2310.04406)——带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——用于搜索的子图模式
- [AlphaEvolve（arXiv:2506.13131）](https://arxiv.org/abs/2506.13131)——带程序化评估器的进化搜索
