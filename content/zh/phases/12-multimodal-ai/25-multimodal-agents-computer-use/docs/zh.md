# 多模态 Agent 与计算机使用（项目）

> 2026 年的前沿产品是一个多模态 Agent：它阅读截图、点击按钮、浏览网页 UI、填写表单，并端到端完成工作流。SeeClick 与 CogAgent（2024）证明了 GUI 定位这一基础能力。Ferret-UI 将其扩展到移动端。ChartAgent 引入了面向图表的视觉工具使用。VisualWebArena 和 AgentVista（2026）是前沿模型追逐的基准——即使是 Gemini 3 Pro 和 Claude Opus 4.7，在 AgentVista 的困难任务上也只拿到约 30%。这个项目串联起第 12 阶段的每一条线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、定位（坐标输出）、长程记忆以及评估。

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## 学习目标

- 设计一个多模态 Agent 循环：感知 → 推理 → 行动 → 观察 → 重复。
- 构建一个 VLM 能以 JSON 形式发出的 GUI 定位输出 schema（点击坐标、输入文本、滚动、拖动）。
- 对比纯截图 Agent、可访问性树 Agent 与混合 Agent。
- 在小型 VisualWebArena 切片上搭建多模态 Agent 基准评估。

## 问题

一个订票网站的工作流：「帮我找一张 4 月 15 日飞东京的机票，靠过道座位，价格低于 800 美元，并完成预订。」

多模态 Agent 需要：

1. 截取浏览器截图。
2. 将截图 + URL + 目标解析成一个计划。
3. 发出结构化动作：点击（x、y 坐标）、输入「东京」（在元素 E 处）、向下滚动、选择（单选按钮）。
4. 把动作施加到浏览器上。
5. 观察新状态（下一张截图）。
6. 重复，直到任务完成。

每一步都是一次多模态 VLM 调用。VLM 的输出必须是可解析的 JSON。错误会跨步骤累积，因此恢复能力很重要。

## 概念

### GUI 定位：基础能力

GUI 定位（GUI grounding）是指：给定一张截图和一条自然语言指令，输出要点击的 (x, y) 坐标（或其他动作）。

SeeClick（arXiv:2401.10935）是第一个在规模上公开的成果：在合成与真实 GUI 数据上微调 VLM，把坐标当作纯文本 token 输出。可行。

CogAgent（arXiv:2312.08914）增加了面向密集 UI 的 1120×1120 高分辨率编码。网页导航得分约 84%。

Ferret-UI（arXiv:2404.05719）聚焦移动端 UI，并与 iOS 可访问性数据集成。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于恢复：如果坐标在两次截图之间发生偏移，语义提示能让系统重新定位。

### 动作 schema

典型的动作 schema 包含 6–10 种动作类型：

- `click`：(x, y)
- `type`：(text, x?, y?)
- `scroll`：(direction, amount)
- `drag`：(x0, y0, x1, y1)
- `select`：(option_index)
- `hover`：(x, y)
- `navigate`：(url)
- `wait`：(ms)
- `done`：(success, explanation)

Agent 每步发出一个动作。浏览器包装层执行该动作并返回新状态。

### 纯截图 vs 可访问性树

两种输入模式：

- 纯截图：完整图像，没有结构信息。通用性最强，适用于任何应用。
- 可访问性树：结构化的 DOM / iOS 可访问性信息。定位可靠性高得多；在树可用的地方都能工作。
- 混合：两者兼用，用树为原子动作提供可靠定位，用截图提供语义上下文。

生产环境中的 Agent 在可能时采用混合模式。浏览器自动化（Selenium + 可访问性）始终有树；桌面应用有时才有。

### 长程记忆

一个 20 步的工作流会产生 20 张截图。VLM 的上下文很快就会填满。三种压缩策略：

- 摘要链：每 5 步总结一次已发生的事情，丢弃旧截图。
- 跳帧：保留第一张、最后一张，以及每第 3 张截图。
- 工具记录日志：执行动作，保留一份文本日志记录做过什么；不再回看旧截图。

Claude 的 computer-use API 采用日志模式。更简单，也更可靠。

### 视觉工具使用

ChartAgent（arXiv:2510.04514）为图表理解引入视觉工具使用：裁剪、放大、OCR、调用外部检测。Agent 可以输出「裁剪到区域 (100, 200, 300, 400)，然后调用 OCR」作为 tool call。工具返回文本，VLM 继续推理。

这一模式可以推广：set-of-mark prompting、区域标注和外部检测工具都符合同一套「输出工具调用、接收结构化响应」的 schema。

### 2026 年的基准

- ScreenSpot-Pro。在约 1k 张网页截图上的 GUI 定位。开源 SOTA Qwen2.5-VL-72B 约 85%。前沿模型约 90%。
- VisualWebArena。端到端网页任务（购物、论坛、分类广告）。开源 SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最难的基准。覆盖 12 个领域的真实工作流。前沿模型得分 27–40%；开源模型 10–20%。
- WebArena / WebShop。较旧的基准，已被前沿模型饱和。

### 为什么仍然很难

Agent 性能的瓶颈：

1. 细粒度视觉定位。「点击那个小 X」在移动端分辨率下经常失败。
2. 长程规划。执行 10 个动作后，Agent 就会偏离目标。
3. 错误恢复。点击失败（点错按钮）时，检测并恢复几乎没有训练数据。
4. 跨页上下文。在标签页之间跳转或填写长表单时会丢失状态。

研究方向：记忆架构、显式重新规划、多模态验证（用截图比对来判断动作是否成功）。

### 项目的构建部分

项目任务：构建一个计算机使用 Agent，它能够：

1. 读取订票网站模拟页面的 HTML + 截图。
2. 规划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 发出符合动作 schema 的 JSON 动作。
4. 在一个固定的 10 任务切片上评估。

课程提供了脚手架代码，很容易扩展成真正的浏览器。

```figure
mm-agent-loop
```

## 使用它

`code/main.py` 是项目的脚手架：

- 动作 schema 的 JSON 定义（10 种动作）。
- 以 dict 表示的模拟浏览器状态。
- Agent 循环骨架：接收状态、发出动作、应用、循环。
- 10 任务迷你基准（合成页面），用于衡量端到端成功率。
- 动作失败时的错误恢复钩子。

## 交付它

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定一个计算机使用产品（领域、动作集、评估目标），设计完整的 Agent 循环、记忆策略、定位模式和预期基准分数。

## 练习

1. 为动作 schema 扩展一个 `screenshot_region` 工具（裁剪 + 放大）。哪些任务会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难的任务类别，并解释前沿模型为什么仍然失败。

3. 长程记忆压缩：设计一个摘要链，最多保留 4 张活跃截图，但记录任意数量的截图。

4. 构建一个错误恢复钩子：动作失败（找不到按钮）时，Agent 接下来怎么做？

5. 在 10 个网页任务上对比纯截图的 Claude 4.7 与截图 + 可访问性树的混合 Qwen2.5-VL。各自在哪些任务上胜出？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GUI grounding | 「点击坐标」 | 模型为截图上的指令目标输出 (x,y) 坐标 |
| Action schema | 「工具定义」 | 对合法动作（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | 「结构化 DOM」 | 来自浏览器 / iOS API 的机器可读 UI 层级结构 |
| Hybrid agent | 「截图 + 树」 | 同时使用图像与结构化信息；比单独使用任一种都更可靠 |
| Visual tool use | 「放大/裁剪/检测」 | Agent 在规划中途调用外部视觉工具（OCR、检测） |
| Summary-chain | 「记忆压缩」 | 周期性文本摘要取代冗长的截图历史 |
| VisualWebArena | 「端到端网页基准」 | 2024 年提出的端到端网页任务基准 |
| AgentVista | 「2026 高难度基准」 | 覆盖 12 领域的真实工作流；连 Gemini 3 Pro 也只有约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
