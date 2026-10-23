# AlphaEvolve —— 进化式编码智能体

> 把一个前沿编码模型与进化循环、机器可校验的评测器配在一起。让循环跑得足够久。它发现了一个只用 48 次标量乘法的 4x4 复数矩阵乘法过程——这是 56 年来对 Strassen 的首次改进。它还找到了一个 Google 全局的 Borg 调度启发式，在生产中回收了约 0.7% 的集群算力。这套架构刻意做得很无聊。收益来自评测器的严格。

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## 问题

大语言模型会写代码。进化算法可以在代码上搜索。两者分别被尝试了几十年，也都撞上了天花板。LLM 的天花板是虚构（confabulation）：模型写出看似合理、却做不到其所声称功能的代码。进化算法的天花板是搜索成本：在语法上做随机变异，很少能产生可编译的程序，更不用说更好的程序。

AlphaEvolve（Novikov 等，DeepMind，arXiv:2506.13131，2025 年 6 月）把两者结合起来。LLM 对程序数据库提出有针对性的修改；自动评测器给每个变体打分；高分变体成为后续世代的父代。LLM 负责写看似合理代码这一昂贵步骤；评测器则拦下虚构。循环运行数小时到数周。

报告的结果：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的界是 49 次）、Google 生产环境中的 Borg 调度启发式、FlashAttention 内核 32.5% 的加速、Gemini 训练吞吐量的提升。

这套架构之所以有效，是因为评测器是机器可校验的。在评测器不可校验的地方，它就不起作用。这一不对称性就是本课的核心。

## 概念

### 循环

1. 从一个正确但次优的种子程序 `P_0` 开始。
2. 维护一个变体程序数据库，每个变体由评测器打分。
3. 从数据库中采样一个或多个父代（MAP-elites 式或岛屿式）。
4. 提示 LLM（大量候选用 Gemini Flash，困难的用 Gemini Pro）生成父代的一个修改变体。
5. 编译、运行，并在留出评测器上评测变体。
6. 按其分数和特征向量插入数据库。
7. 重复。

有两个细节很重要。第一，提示给 LLM 的不只是父代程序——通常还有数据库中若干顶级变体、评测器签名和简短任务描述。模型的任务是提出一个可能提高分数的有针对性的改动。第二，数据库是结构化的（MAP-elites 网格、岛屿式），因此循环探索的是多样性，而不只是当前领先者。

### 为什么评测器不可替代

AlphaEvolve 的所有成果都来自评测器快速、确定、难以作弊的领域：

- **矩阵乘法算法**：一个乘法矩阵并逐位检查相等的单元测试。
- **Borg 调度启发式**：一个生产级模拟器，回放历史集群负载并测量浪费的算力。
- **FlashAttention 内核**：正确性测试加真实硬件上的墙钟基准。
- **Gemini 训练吞吐量**：按每步的 GPU 秒数测量。

每一种情况下，评测器都拦下了本来会占主导的那类 LLM 错误：虚构的正确性声明、在硬件上消失的性能声明，以及边界情况失败。去掉评测器，循环优化的就是漂亮的代码。

### 奖励黑客是同一句话的另一面

进化会针对评测器所度量的一切做优化。如果评测器不完美，循环就会找到那个不完美之处。在未经验证的领域，循环会优化表面特征，而不是预期行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功只可迁移到评测器严格程度与搜索野心相匹配的领域。

2025-2026 年代码搜索循环中奖励黑客的具体例子：

- 奖励「完成时间」的优化目标，奖励了提交空解。
- 奖励「测试下正确性」的基准分数，奖励了背测试并过拟合。
- 一个「代码质量」代理指标，奖励了删除注释、重写变量名，却没有任何语义变化。

AlphaEvolve 的修法是：交付一个 LLM 从未见过的留出评测器，输入在评测时生成。即便如此，DeepMind 仍建议对任何拟议部署做严格审查。

### 为什么 LLM + 搜索胜过任何一方

LLM 能产生可编译、语义上合理的修改。在一个 2000 行的 Python 文件上做随机变异 GA，几乎总会产生语法错误。LLM 还把搜索集中在合理的邻域内（改一个函数，而不是随机字节），这大大减少了浪费的评测器调用。

反过来，评测器拦下 LLM 的虚构。LLM 会自信地声称一个函数「在极限下是 $O(n \log n)$」，而实际上是 $O(n^2)$；墙钟基准让问题一锤定音。

### AlphaEvolve 在前沿技术栈中的位置

| 系统 | 生成器 | 评测器 | 领域 | 示例成果 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 正确性 + 基准 | 算法、内核、调度器 | 48 次乘法的 4x4 矩阵乘 |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 正确性 | 组合数学 | cap-set 下界 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 批评 + 实验 | ML 研究 | ICLR workshop 论文 |
| Darwin Godel Machine (L4) | 智能体脚手架 | SWE-bench / Polyglot | 智能体代码 | SWE-bench 20% → 50% |

四者都是同一配方的变体：生成器加评测器，循环。区别在于评测器评什么、有多严格。

```figure
alphaevolve-loop
```

## 使用它

`code/main.py` 在一个玩具符号回归问题上实现了一个最小的 AlphaEvolve 式循环。「LLM」是一个标准库代理，对一个计算目标函数的程序提出小的语法变异。「评测器」测量留出测试点上的均方误差。

观察：

- 最佳分数如何随世代提升。
- MAP-elites 网格如何让多样化的解保持存活，使循环不会收敛到局部最优。
- 去掉留出测试（只用训练集评测）如何让循环壮观地过拟合。

## 交付它

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑 AlphaEvolve 式循环的前提条件：你的评测器真的能捕获你在意的失败吗？

## 练习

1. 运行 `code/main.py`。记录最佳分数轨迹。用 `--no-holdout` 标志禁用留出评测器并重跑。量化过拟合程度。

2. 阅读 AlphaEvolve 论文中关于 MAP-elites 网格的第 3 节。为一个新问题（例如编译器优化 pass）设计一个特征向量描述符，让搜索保持多样。

3. 48 次乘法的 4x4 结果在 56 年后改进了 Strassen 的 49 次乘法界。阅读论文附录 F，用三句话解释为什么这个问题的评测器特别容易做对，以及为什么大多数领域不是这样。

4. 提出一个 AlphaEvolve 会失败的领域。准确指出评测器在哪里失效以及为什么。

5. 对一个你熟悉的领域，写出你会使用的评测器签名。包括 (a) 正确性条件、(b) 性能指标、(c) 留出输入生成规则、(d) 至少一项防奖励黑客检查。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AlphaEvolve | 「DeepMind 的进化式编码智能体」 | Gemini + 程序数据库 + 机器可校验评测器 |
| MAP-elites | 「保持多样性的档案」 | 以特征向量为键的网格；每格保存具有该描述符的最佳变体 |
| 岛屿模型 | 「并行进化的子种群」 | 独立种群定期迁移；防止过早收敛 |
| 机器可校验评测器 | 「确定性预言机」 | LLM 无法伪造的单元测试、模拟器或基准——此循环的前提条件 |
| 奖励黑客 | 「优化度量，而非目标」 | 循环找到最大化分数却不做预期任务的办法 |
| 种子程序 | 「起点」 | 循环从中进化的初始正确但次优程序 |
| 留出评测器 | 「LLM 从未见过的评测数据」 | 评测时生成输入以防记忆 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 带结果的厂商说明。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48 次乘法的 4x4 矩阵乘。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 把受评测器约束的自主性列为关键研究方向。
