# 基准测试：WebArena 与 OSWorld

> WebArena 在四个自托管应用上测试网页智能体的能力。OSWorld 在 Ubuntu、Windows、macOS 上测试桌面智能体的能力。发布时（2023–2024），两者都显示顶尖智能体与人类之间存在很大差距。差距正在缩小；失效模式却没有改变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## 学习目标

- 描述 WebArena 的四个自托管应用，以及为什么基于执行的评估很重要。
- 解释 OSWorld 为什么使用真实操作系统截图，而不是无障碍 API。
- 说出 OSWorld 的两类主要失效模式：GUI 定位与操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么。

## 问题

通用智能体可以调用工具。它们能在浏览器里点 20 次完成一次购物结账吗？它们能只用键盘和鼠标配置一台 Linux 机器吗？这些就是 WebArena 和 OSWorld 要回答的问题。

## 概念

### WebArena（Zhou 等，ICLR 2024）

- 812 个长程任务，分布在四个自托管网页应用上：一个购物网站、一个论坛、一个类似 GitLab 的开发工具、一个业务 CMS。
- 外加实用工具：地图、计算器、草稿纸。
- 评估通过 gym API 基于执行完成——订单是否下了、issue 是否关闭、CMS 页面是否更新？
- 发布时：最好的 GPT-4 智能体成功率 14.41%，人类为 78.24%。

自托管这一设定很重要——目标应用版本固定、可复现，因此基准不会时好时坏。

### 扩展

- **VisualWebArena** — 视觉接地的任务，成功与否取决于对图像的理解（截图是一等观测）。
- **TheAgentCompany**（2024 年 12 月）— 增加终端 + 编码；更像真实的远程工作环境。

### OSWorld（Xie 等，NeurIPS 2024）

- 369 个真实计算机任务，覆盖 Ubuntu、Windows、macOS。
- 对真实应用进行自由形式的键盘和鼠标控制。
- 以 1920×1080 截图作为观测。
- 发布时：最佳模型 12.24%，人类为 72.36%。

### 主要失效模式

1. **GUI 接地（GUI grounding）。** 像素 → 元素映射。模型难以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单里有该设置、用哪个快捷键、在哪个偏好设置面板。这是人类多年积累的长尾知识。

### 后续工作

- **OSWorld-G** — 564 个样本的定位套件 + Jedi 训练集。把定位与规划解耦，这样你可以分别度量它们。
- **OSWorld-Human** — 人工策展的黄金动作轨迹。显示顶尖智能体使用的步数比必要多 1.4–2.7 倍（轨迹效率差距）。

### 为什么重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）都在受 WebArena 和 OSWorld 塑造的工作负载上训练。基准是目标；生产模型是交付的答案。

### 基准测试常见的错误

- **仅看截图的评估。** OSWorld 由截图驱动；在 OSWorld 上评估一个使用 DOM 或无障碍 API 的智能体，会错过定位挑战。
- **忽略轨迹长度。** 只打成功率，会漏掉 OSWorld-Human 揭示的 1.4–2.7 倍步数低效。
- **自托管应用过时。** WebArena 的应用固定了特定版本；不重新策展就升级，会破坏可比性。

```figure
ae-agent-human-gap
```

## 构建它

`code/main.py` 实现一个玩具式的网页智能体框架：

- 一个最小的「购物应用」状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个脚本化的智能体，尝试完成每个任务。
- 基于执行的评估器（状态检查）和轨迹效率指标（步数对比黄金轨迹）。

运行它：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，对应 OSWorld-Human 的方法论。

## 使用它

- **WebArena Verified** 自托管在内部集群上，做持续评估。
- **OSWorld** 放在虚拟机集群里评估桌面智能体。
- **Computer-use 智能体**（第 21 课）——Claude、OpenAI CUA、Gemini——都在类似的工作负载上训练。
- **你自己的产品流程**——为你最重要的 20 个任务采集黄金轨迹；每周让智能体跑一遍。

## 交付它

`outputs/skill-web-desktop-harness.md` 构建一个网页/桌面智能体框架，带基于执行的评估和轨迹效率指标。

## 练习

1. 给玩具框架增加第二个应用（一个论坛）。编写 3 个任务及黄金轨迹。
2. 增加每个任务的轨迹效率报告。在你的玩具上，智能体是黄金轨迹的 1 倍、2 倍还是 3 倍？
3. 实现一个「干扰」工具——黄金轨迹从不使用它。脚本化智能体会被诱惑吗？
4. 阅读 OSWorld-G。你会如何在自己的评估中把定位失败与规划失败分开？
5. 阅读 WebArena 的应用 README。升级某个固定应用版本时，什么会坏掉？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| WebArena | 「网页智能体基准」 | 4 个自托管应用上的 812 个任务；gym 风格评估 |
| VisualWebArena | 「视觉 WebArena」 | 视觉接地的 WebArena；截图即观测 |
| OSWorld | 「桌面智能体基准」 | 真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI 接地 | 「像素到元素映射」 | 模型在 1920x1080 中定位 UI 元素 |
| 操作知识 | 「操作系统经验」 | 哪个菜单、哪个快捷键、哪个偏好设置面板 |
| OSWorld-G | 「定位套件」 | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | 「黄金轨迹」 | 人工专家动作序列，用于度量效率 |
| 轨迹效率 | 「相对黄金轨迹的步数」 | 智能体步数除以人类最少步数 |

## 延伸阅读

- [Zhou 等，WebArena（arXiv:2307.13854）](https://arxiv.org/abs/2307.13854) — 四应用网页基准
- [Xie 等，OSWorld（arXiv:2404.07972）](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准塑造能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 与 WebArena 数据
