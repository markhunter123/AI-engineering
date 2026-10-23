# 技能库与终身学习（Voyager）

> Voyager（Wang 等，TMLR 2024）把可执行代码当作技能。技能有名字、可检索、可组合，并由环境反馈精炼。这是 Claude Agent SDK 技能、skillkit 以及 2026 年技能库模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示——以及各自的职责。
- 解释 Voyager 为什么把动作空间设为代码，而不是原语命令。
- 用标准库实现一个技能库，支持注册、检索、组合和失败驱动的精炼。
- 把 Voyager 的模式映射到 2026 年的 Claude Agent SDK 技能和 skillkit 生态。

## 问题

每个会话都从零重建所有能力的智能体，会在三件事上出错：

1. **浪费 token。** 每个任务都要重新引出同样的推理。
2. **丢失进展。** 在会话 A 中学到的修正无法迁移到会话 B。
3. **长程组合失败。** 复杂任务需要能力层级；一次性 prompt 无法表达它们。

Voyager 的答案：把每项可复用能力视为一段命名代码，存入库中，可按相似度检索，可与其他技能组合，并可由执行反馈精炼。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下组件组织智能体：

1. **自动课程。** 一个好奇心驱动的提议器，根据智能体当前的技能集和环境状态挑选下一个任务。探索是自底向上的。
2. **技能库。** 每个技能都是可执行代码。任务成功时加入新技能。技能按查询到描述的相似度检索。
3. **迭代提示机制。** 失败时，智能体收到执行错误、环境反馈和自验证输出，然后精炼技能。

Minecraft 评估（Wang 等，2024）：独特物品多 3.3 倍，石制工具快 8.5 倍，铁制工具快 6.4 倍，地图遍历距离长 2.3 倍，均相对基线。这些数字只属于 Minecraft，但模式可以迁移。

### 动作空间 = 代码

多数智能体发出原语命令。Voyager 发出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和 embedding 为键存储。检索出来的是一个程序，而不是一段 prompt。

这就是 2026 年的 Claude Agent SDK 技能：一段命名、可检索的代码，加上智能体按需加载的指令。

### 技能检索

新任务「做一把钻石镐」。智能体：

1. 对任务描述做 embedding。
2. 在技能库中查询 top-k 相似技能。
3. 检索出 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索到的原语加新逻辑组合出新技能。

这正是 MCP resources（第 13 阶段）和 Agent SDK 技能所实现的模式：在知识/代码面上做检索，并限定到当前任务。

### 迭代精炼

Voyager 的反馈循环：

1. 智能体写一个技能。
2. 技能在环境中运行。
3. 三种信号之一返回：`success`、`error`（带堆栈跟踪）、`self-verification failure`。
4. 智能体以该信号为上下文重写技能。
5. 循环直到成功或达到最大轮数。

这就是 Self-Refine（第 05 课）应用于代码生成，并以环境为验证依据。CRITIC（第 05 课）是同一模式，只是把外部工具作为验证器。

### 课程与探索

Voyager 的课程模块根据智能体已有什么、还没做什么，提出「在湖边建一个庇护所」这样的任务。提议器利用环境状态 + 技能清单，挑一个刚好高于当前能力的任务——探索的最佳区间。

对生产智能体而言，这转化为一个「还缺什么」算子：给定当前技能库和某个领域，我们还没覆盖哪些技能？团队通常以课程评审的方式手动实现。

### 这个模式会在哪里出错

- **技能库腐化。** 同一个技能被加了 10 次，描述略有不同。写入时去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个已被精炼的子技能。给技能做版本管理；固定到 v1 的父技能不会神奇地拿到 v3。
- **检索质量。** 技能描述上的向量检索在库超过几百个之后会退化。辅以标签过滤和硬约束（「只要 `category=tooling` 的技能」）。

```figure
voyager-skills
```

## 构建它

`code/main.py` 实现了一个标准库技能库：

- `Skill` —— name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary` —— 注册、搜索（token 重叠）、组合（依赖拓扑排序）和精炼（更新时提升版本号）。
- 一个脚本化智能体：注册三个原语技能，组合出第四个，遇到失败，然后精炼。

运行它：

```
python3 code/main.py
```

运行轨迹展示库写入、检索、组合、一次失败的执行，以及 v2 精炼——完整走完 Voyager 的循环。

## 使用它

- **Claude Agent SDK 技能**（Anthropic）—— 2026 年的参考：每个技能都有描述、代码和指令；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）—— 面向 32+ 个 AI 编程智能体的跨智能体技能管理。
- **自建技能库** —— 领域专用（数据智能体用 SQL 技能，基础设施智能体用 Terraform 技能）。Voyager 模式可以向下缩放。
- **OpenAI Agents SDK `tools`** —— 低端选择；每个工具就是一个轻量技能。

## 交付它

`outputs/skill-skill-library.md` 为任意目标运行时生成 Voyager 形态的技能库，内置注册、检索、版本管理和精炼。

## 练习

1. 给 `compose()` 增加依赖环检测。当技能 A 依赖 B、B 又依赖 A 时会发生什么？报错还是警告？
2. 实现按技能固定版本。当父技能组合子技能 `crafting@1` 时，对 `crafting@2` 的精炼绝不能静默升级父技能。
3. 把 token 重叠检索替换为 sentence-transformers embedding（或标准库 BM25 实现）。在 50 条技能的玩具库上测量 retrieval@5。
4. 增加一个「课程」智能体：给定当前库和一段领域描述，提出 5 个缺失的技能。每周运行一次。
5. 阅读 Anthropic 的 Claude Agent SDK 技能文档。把玩具库移植到 SDK 的技能 schema。可发现性有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 技能（Skill） | 「可复用能力」 | 命名的代码块 + 描述，可按相似度检索 |
| 技能库 | 「智能体的 how-to 记忆」 | 技能的持久存储，可搜索、可组合 |
| 课程（Curriculum） | 「任务提议器」 | 由当前能力缺口驱动的自底向上目标生成器 |
| 组合（Composition） | 「技能 DAG」 | 技能调用技能；执行时做拓扑排序 |
| 迭代精炼 | 「自纠正循环」 | 环境反馈 + 错误 + 自验证折回下一个版本 |
| 动作空间即代码 | 「程序化动作」 | 发出函数而非原语命令，支持时间上延展的行为 |
| 写入去重 | 「技能合并」 | 近重复描述合并为一条规范技能 |

## 延伸阅读

- [Wang 等，Voyager（arXiv:2305.16291）](https://arxiv.org/abs/2305.16291) —— 原始技能库论文
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) —— 技能作为 2026 年的产品化形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 实践中的技能与子智能体
- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) —— Voyager 底层的精炼循环
