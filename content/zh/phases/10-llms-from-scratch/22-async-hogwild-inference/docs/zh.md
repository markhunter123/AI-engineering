# 异步与 Hogwild! 推理

> 投机解码（第 10 阶段 · 第 15 课）在一个序列内部并行化 token。多智能体框架跨整个序列并行，但强制引入显式协调（投票、子任务拆分）。Hogwild! Inference（Rodionov 等，arXiv:2504.06261）做了另一件事：让同一个 LLM 的 N 个实例并行运行，共享同一份键值缓存。每个 worker 都能立即看到其他所有 worker 生成的 token。现代推理模型——QwQ、DeepSeek-R1——无需任何微调，就能通过这份共享缓存自我协调。该方法仍是实验性的，但它开辟了一条与投机解码正交的全新推理并行轴。本课用标准库 Python 实现一个双 worker 的 Hogwild! 模拟器，并解释为什么共享缓存协作会从现有模型的推理能力中涌现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!），并指出各自针对的问题。
- 陈述 Hogwild! 的核心设置：多个 worker、一份共享 KV cache、通过自我提示涌现的协调。
- 把 Hogwild! 的墙钟加速比表示为 worker 数 `N`、任务级并行度 `p` 和协调开销 `c` 的函数。
- 在玩具问题上实现双 worker 的 Hogwild! 模拟器，并观察涌现出的任务分工。

## 问题

现代 LLM 通过生成很长的推理链来解决难题——5000 个 token 的逐步逻辑很常见，深度数学问题上会达到数万个 token。在 70B 模型上以 35 token/秒解码，50k token 需要 24 分钟。这谈不上交互性。

投机解码（第 10 阶段 · 第 15 课）通过在一个序列内部并行化，带来 3-5 倍加速。再往上，自回归解码的串行依赖就是硬天花板。每个新 token 都依赖此前所有 token。

显而易见的问题：我们能否跨序列并行？让同一个模型的多个副本处理同一个问题，让它们合作、分工？

已有工作：投票集成（跑 N 个模型，选多数答案）、思维树（分支推理路径再合并）、多智能体框架（给每个智能体分配子任务，用协调者调度）。这些在特定任务领域都有帮助。它们也都引入了显式的协调机制——投票规则、分支剪枝逻辑、智能体间消息协议。

Hogwild! Inference 采取了不同做法。N 个 worker 共享一份 KV cache。每个 worker 都能立即看到其他每个 worker 生成的 token，就好像那是自己的上下文。这些 worker——不经过任何训练或微调——自己摸索出如何分工。现代推理模型（QwQ、DeepSeek-R1、Claude 家族的推理模式）能读取共享缓存，并说出「我看到 worker 2 已经处理了基本情况，那我来做归纳步骤」之类的话。

加速比取决于工作负载，截至 2026 年 4 月仍是实验性的。但这个想法值得了解，因为它开辟了一条新的推理并行轴。

## 概念

### 设置

初始化 N 个 worker 进程，都运行同一个 LLM。不使用每 worker 各自的 KV cache，而是维护一份共享缓存。当 worker `i` 生成 token `t_j` 时，该 token 被写入共享缓存的下一个位置。当 worker `k` 执行下一步时，它读取缓存的当前状态（其中包含全部 N 个 worker 到目前为止生成的所有内容）。

在每一步，worker 们竞相写入 token。没有每 worker 的位置索引——缓存就是一条不断增长的单一序列。顺序由写入到达时间决定。

### 协调为什么会涌现

这些 worker 共享同一个 prompt。通常类似「你是 N 个协作解决此问题的实例之一。每个实例都读取共享内存，可以看到其他实例写了什么。避免重复工作。」这个 prompt 加上共享缓存就够了。推理模型读取缓存，注意到问题的哪些部分已经有人尝试过，然后（通常但不总是）转向未探索的部分。

Hogwild! 论文（Rodionov 等，2025）报告了这样的观察：

- worker 制定计划，并通过缓存传达给其他 worker。
- worker 发现其他 worker 推理中的错误并指出来。
- 计划失败时，worker 会调整并给出替代方案。
- 当被提示检查冗余时，worker 能检测到并转向。

这些都不需要微调。涌现行为来自模型已经具备的推理能力。

### 名字的由来

论文的名字戏仿了 Hogwild! SGD（Recht 等，2011），一个异步更新优化器。类比是：SGD 的异步 worker 都写入同一个共享参数向量；Hogwild! Inference 的 worker 都写入同一份共享 KV cache。两者都依赖经验性收敛，而非同步保证。

### RoPE 让这变得可行

旋转位置编码（RoPE，Su 等 2021）通过在 Q 和 K 向量中施加旋转来编码位置信息。由于位置是旋转而不是固化在数据中的偏移，token 的位置可以移动而无需重新计算其 KV cache 条目。当 worker `i` 在位置 `p` 写入共享缓存时，其他读取该位置的 worker 可以直接使用缓存条目——无需重新旋转。

在学习位置或绝对位置的模型中，Hogwild! 就需要在每次并发写入时使缓存失效。RoPE 让缓存保持稳定。

### 墙钟时间计算

设 `T_serial` 为一个 worker 独自解决问题所需的时间。设 `p` 为任务级可并行比例。设 `c` 为每步协调开销（读取扩展后的缓存、决定写什么）。

单 worker 时间：`T_serial`。
N worker 的 Hogwild! 时间，假设协调免费：`T_serial * ((1 - p) + p / N)`。经典 Amdahl 定律。
加上协调开销：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker 有产出，`c` 必须相对每步解码时间很小。在生成 5k+ token 的推理模型上，worker 可以承担数百个 token 的协调开销，仍然划算。在短聊天任务上，协调占主导，Hogwild! 反而不如串行。

### 具体例子

推理问题：10k token 的思维链。假设问题有 `p = 0.7` 的可并行内容（不同证明策略、不同情况分析），每个 worker 的协调开销 `c = 200` 个 token。取 `N = 4` 个 worker：

- 串行时间：10000 个解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 个解码步。
- 加速比：10000 / 5550 = 1.8 倍。

这不算多。但在更长的推理问题（50k token）上，协调开销被摊薄，加速比可以推到 2.5-3 倍。Hogwild! 相当于这样一种语言里的线程级并行——你可以自然地写出多线程代码。

### 什么时候该用 Hogwild!

- 长推理问题（数千 token），且任务可以跨独立子目标并行化。
- 经过逐步思考训练的推理模型。非推理模型的自我协调能力很差。
- 单节点部署，VRAM 足够容纳共享缓存加上 N 个 worker 进程。缓存是共享的，但每个 worker 有自己的激活内存。

### 什么时候不该用

- 短交互式聊天。协调开销占主导。
- 不可并行的任务（单一线性证明、单次编译）。N=1 就是上限。
- 非推理模型。不会涌现协调。
- 多节点部署。共享缓存需要非常快的跨 worker 同步。节点内没问题；跨节点是延迟灾难。

### 实验状态

截至 2026 年 4 月，Hogwild! 仍是一种研究方法，有开源的 PyTorch 实现。尚未进入生产采用。三个阻碍：

1. 跨并发进程的共享 KV cache 管理是非平凡的工程问题。
2. 涌现协调依赖任务；基准测试仍在建设中。
3. 相比投机解码已经带来的收益，加速幅度有限；两者可以结合，但结合后的工程又是另一层。

值得了解。值得实验。尚不值得押注产品。

```figure
continuous-batching
```

## 构建它

`code/main.py` 实现了一个玩具 Hogwild! 模拟器：

- 两个 worker 进程，每个都是确定性的「LLM」，以已知概率产出若干 token 类别之一（work-token、observe-token、coordinate-token）。
- 一份共享缓存（就是 token 列表），两个 worker 都读、都写。
- 简单的协调逻辑：当某个 worker 看到另一个 worker 在某个类别里已经产出足够多的 work token，它就换一个类别。

模拟器运行固定的步数预算，并报告：

- 产出的 work-token 总数。
- 总墙钟时间（worker 步数）。
- 相对单 worker 的有效加速比。
- 哪个 worker 写了哪个 token 的轨迹。

### 第 1 步：共享缓存

一个两个 worker 都追加写入的列表。真实实现中会用简单锁（Python 的 `threading.Lock`）；我们用一个计数器来模拟。

### 第 2 步：worker 循环

每个 worker 在每一步：

- 读取当前共享缓存。
- 根据缓存中已有的内容，决定写入哪一类 token。
- 写入一个 token。

### 第 3 步：协调启发式

如果类别 X 在缓存中已经有 K 个 token，而 worker 本想写类别 X，就切换到类别 Y。这是对推理模型行为「注意到这点已经覆盖了，换点别的做」的玩具替身。

### 第 4 步：测量加速比

用 N=1 和 N=2 个 worker 运行模拟器，总步数预算相同。统计产出的 work-token 数量。N=2 应该多产出约 1.5-1.8 倍的 work-token，因为协调驱动了任务分工。

### 第 5 步：对协调施压

降低协调启发式的敏感度。再跑一次。观察没有良好协调时，N=2 会冗余地产出相同 token，加速比降到 1 以下。这与论文的观察一致：只有当 worker 具备自我协调的推理能力时，这个技巧才成立。

## 使用它

截至 2026 年 4 月，生产中的 Hogwild! 集成仍处于研究阶段。来自 Yandex/HSE/IST 的参考实现基于 PyTorch，面向 DeepSeek-R1 和 QwQ 模型的单节点多进程部署。

务实的采用路径：

1. 分析你的推理任务负载。测量探索性 token（多种策略、情况分析、搜索）与线性 token 的比例。
2. 如果探索占主导，做一次双 worker 的 Hogwild! 实验，测量墙钟时间改善。
3. 如果改善低于 1.3 倍，你处在协调主导的区间。退回单 worker。
4. 如果改善超过 1.5 倍，推到 N=4 再测。收益递减通常在 N=4-8 附近出现。

与投机解码结合：每个 Hogwild! worker 都可以独立使用投机解码。两种加速大致相乘，3 倍的投机解码和 1.8 倍的 Hogwild! 叠加，相对朴素单 worker 解码达到约 5.4 倍。

## 交付它

本课产出 `outputs/skill-parallel-inference-router.md`。给定推理负载概貌（token 预算、任务并行特征、模型家族、部署目标），它在投票、思维树、多智能体、Hogwild! 和投机解码策略之间做路由。

## 练习

1. 用默认设置运行 `code/main.py`。确认 N=2 的 Hogwild! 配置在相同墙钟时间内产出的 work-token 比 N=1 基线更多。

2. 降低协调启发式的强度（设置 `coordination_weight=0.1`）。重新运行。展示加速比崩塌。解释原因：worker 无法协调时会重复劳动。

3. 计算一个 50k token 推理任务在 `p=0.8, c=500`、N=4 个 worker 下的预期 Hogwild! 加速比。再对 1k token 聊天任务在 `p=0.3, c=200`、N=4 下做同样的计算。为什么一个是赢，另一个是亏？

4. 阅读 Hogwild! 论文第 4 节（初步评估）。找出作者报告的两个失败模式。描述更好的协调 prompt 如何缓解每一个。

5. 在玩具模型中把 Hogwild! 与投机解码结合：每个 worker 内部使用 2-token 的投机解码。报告乘性加速比。当两个 worker 都想扩展同一个共享缓存前缀时，会出现什么记账问题？

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | 「并行 worker、共享缓存」 | 同一个 LLM 的 N 个实例并发运行，共享一份 KV cache；通过自我提示涌现协调 |
| 共享 KV cache | 「协调媒介」 | 一份不断增长的 KV 缓冲区，所有 worker 读写；让 token 在 worker 之间即时可见 |
| 涌现协调 | 「无需训练」 | 具备推理能力的 LLM 能读取共享缓存并分工，无需任何微调或显式协议 |
| 协调开销（c） | 「用于定位的 token」 | 每个 worker 读取扩展缓存并决定做什么的成本；相对总解码时间必须很小 |
| 可并行比例（p） | 「哪些能并行跑」 | 任务级并行度：总工作中非内在串行的比例 |
| RoPE 使能 Hogwild! | 「旋转位置具有平移不变性」 | 因为位置是旋转，写入共享缓存不需要重新计算此前的 token |
| 投票集成 | 「跑 N 个，选多数」 | 最简单的并行推理拓扑；适合分类，不太适合长文本推理 |
| 思维树 | 「分支与剪枝」 | 探索多条分支并剪枝的推理策略；显式协调逻辑 |
| 多智能体框架 | 「分配子任务」 | 每个智能体一个角色；协调者负责编排；协议开销重 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261)——Hogwild! 论文，在 QwQ 和 DeepSeek-R1 上的初步评估
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730)——最初的 Hogwild!，名字的来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)——RoPE，使共享缓存推理可行的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)——与 Hogwild! 正交的思维树推理策略
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)——投机解码，Hogwild! 可与之组合的序列内并行
- [Hogwild! 的 PyTorch 参考实现](https://github.com/eqimp/hogwild_llm)——论文实验的唯一事实来源
