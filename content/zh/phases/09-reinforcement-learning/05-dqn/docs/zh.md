# 深度 Q 网络（DQN）

> 2013 年：Mnih 用原始像素训练了一个 Q-learning 网络，在七款 Atari 游戏上击败了所有经典 RL 智能体。2015 年：扩展到 49 款游戏，发表在 Nature 上，点燃了深度 RL 时代。DQN 就是 Q-learning 加上三个让函数逼近稳定下来的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## 问题

表格型 Q-learning 需要为每个 (state, action) 对维护一个 Q 值。一个棋盘约有 10⁴³ 个状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格型 RL 在数千个状态时就撑不住了，更不用说数十亿。

事后看来修复显而易见：用神经网络 $Q(s, a; \theta)$ 替代 Q 表。但「事后显然」花了数十年。朴素的函数逼近配 Q-learning 会在「致命三要素」下发散——函数逼近 + 自举 + 离策略学习。Mnih 等人（2013、2015）找出了三个稳定学习的工程技巧：

1. **经验回放** 让转移去相关。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度幅度。

Atari 上的 DQN 是第一次用单一架构、单一超参数组合，从原始像素解决数十个控制问题。此后所有「深度 RL」——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这个三技巧基础之上。

## 概念

![DQN 训练循环：环境、replay buffer、在线网络、目标网络、贝尔曼 TD 损失](../assets/dqn.svg)

**目标函数。** DQN 在神经网络 Q 函数上最小化一步 TD 损失：

```math
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 ]
```

$\theta$ = 在线网络，每步用梯度下降更新。$\theta^-$ = 目标网络，定期从 $\theta$ 复制（约每 10,000 步）。$D$ = 过去转移的 replay buffer。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量为 `~10⁶` 的环形缓冲区。每个训练步均匀随机采样一个小批量。这打破了时间相关性（相邻帧几乎相同），让网络能多次从稀有的高奖励转移中学习，并让连续的梯度更新去相关。没有它，神经网络上的同策略 TD 在 Atari 上会发散。

**目标网络。** 在贝尔曼方程两边使用同一个网络 $Q(\cdot; \theta)$，会让目标每次更新都移动——「追自己的尾巴」。修复：保留第二个权重冻结的网络 $Q(\cdot; \theta^-)$。每 $C$ 步把 $\theta \to \theta^-$ 复制一次。这让回归目标在数千个梯度步内保持稳定。软更新 $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari 的奖励幅度从 1 到 1000+ 不等。裁剪到 $\lbrace-1, 0, +1 \rbrace$ 防止单个游戏主导梯度。当奖励幅度本身有意义时这是错的；对只看符号的 Atari 则没问题。

**Double DQN。** Hasselt（2016）修复最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

```math
\text{target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
```

即插即用，一致地更好。默认就该用它。

**其他改进（Rainbow，2017）：** 优先回放（更多采样高 TD 误差的转移）、dueling 架构（分离 $V(s)$ 与优势头）、噪声网络（学出来的探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每项加几个百分点；收益大致可叠加。

```figure
f3-dqn-stability
```

## 构建它

这里的代码仅用标准库、不依赖 numpy——我们在一个微型连续 GridWorld 上用手写的单隐层 MLP，所以每个训练步只需微秒级。算法与大规模 Atari DQN 完全一致。

### 第 1 步：replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 约需 50,000 容量；我们的玩具环境 5,000 就够。

### 第 2 步：微型 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。整个网络就这些。

### 第 3 步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状就是第 04 课的 Q-learning，两处不同：(a) 我们通过可微的 $Q(\cdot; \theta)$ 反向传播，而不是查表；(b) 目标使用 $Q(\cdot; \theta^-)$。

### 第 4 步：外层循环

每个回合，对 $Q(\cdot; \theta)$ 做 ε-greedy 行动，把转移推入 buffer，采样一个小批量，走一步梯度，定期同步 $\theta^- \leftarrow \theta$。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个状态为 16 维 one-hot 的微型 GridWorld 上，智能体约 500 个回合学到接近最优的策略。Atari 上则要把这个规模放大到 2 亿帧，并加上 CNN 特征提取器。

## 常见陷阱

- **致命三要素。** 函数逼近 + 离策略 + 自举可能发散。DQN 用目标网络 + 回放缓解；两者都别去掉。
- **探索。** ε 必须衰减，通常在前 ~10% 训练中从 1.0 降到 0.01。早期探索不足，Q 网络会收敛到局部盆地。
- **高估。** 对噪声 Q 取 `max` 会向上偏置。生产环境永远用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **Replay buffer 冷启动。** buffer 里有几千条转移之前不要训练。在约 20 个样本上做早期梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过期。Atari DQN 用 10,000 个环境步。经验法则：约每训练总步数的 1/100 同步一次。
- **观测预处理。** Atari DQN 堆 4 帧让状态满足马尔可夫性。任何带速度信息的环境都需要帧堆叠或循环状态。

## 使用它

2026 年，DQN 很少是最优，但仍是离策略算法的参考：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|------------------|--------------|
| 离散动作、Atari 式 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（第 9 阶段 · 07） | DQN 没有策略网络。 |
| 同策略 / 高吞吐 | PPO（第 9 阶段 · 08） | 不需要 replay buffer；更易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，不会自举爆炸。 |
| 大离散动作空间（推荐） | 带动作 embedding 的 DQN，或 IMPALA | 可以；装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级而非步级；损失不同。 |

这些经验仍然适用。回放与目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈 buffer，以及每个离线 RL 方法中。奖励裁剪以 PPO 中优势归一化的形式延续。这套架构就是蓝图。

## 交付它

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。画出每回合回报曲线。运行平均何时超过 -10？
2. **中等。** 禁用目标网络（贝尔曼目标两边都用在线网络）。测量训练不稳定性——回报是振荡还是发散？
3. **困难。** 加入 Double DQN：用在线网络选 `argmax a'`，用目标网络评估。在噪声奖励 GridWorld 上，对比 1,000 个回合后有无 Double DQN 时 `Q(s_0, best_a)` 相对真值 $V^{\ast}(s_0)$ 的偏差。

## 关键术语

| 术语 | 常被怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | 「深度 Q-learning」 | 用神经网络 Q 函数、replay buffer 与目标网络做 Q-learning。 |
| Experience replay | 「打乱转移」 | 每个梯度步均匀采样的环形缓冲区；让数据去相关。 |
| Target network | 「冻结的自举」 | 贝尔曼目标中使用的 Q 网络定期副本；稳定训练。 |
| Deadly triad | 「RL 为什么会发散」 | 函数逼近 + 自举 + 离策略 = 没有收敛保证。 |
| Double DQN | 「最大化偏差的修复」 | 在线网络选动作，目标网络评估它。 |
| Dueling DQN | 「V 头与 A 头」 | 把 Q 分解为 V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | 「所有技巧」 | DDQN + PER + dueling + n-step + noisy + distributional 合一。 |
| PER | 「优先回放」 | 按 TD 误差幅度成比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 开启深度 RL 的 2013 年 NeurIPS workshop 论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) —— Nature 论文，49 款游戏的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 堆叠技巧的论文。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书对「致命三要素」（函数逼近 + 自举 + 离策略）的论述，DQN 的目标网络与回放正是为了驯服它。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 消融研究使用的单文件 DQN 参考实现；适合与本课从零写的版本对照阅读。
