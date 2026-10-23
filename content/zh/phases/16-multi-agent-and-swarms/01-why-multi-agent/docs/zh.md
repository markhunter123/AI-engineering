# 为什么需要多 Agent？

> 单个 Agent 会撞上墙。聪明的做法不是造一个更大的 Agent——而是使用更多 Agent。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 识别单 Agent 的上限（上下文溢出、混合专长、串行瓶颈），并解释何时拆分出多个 Agent 才是正确选择
- 比较各种编排模式（Pipeline、并行 fan-out、监督者、层级式），并针对给定的任务结构选出合适的一种
- 设计一个具有清晰角色边界、共享状态和通信契约的多 Agent 系统
- 分析多 Agent 复杂性的取舍（延迟、成本、调试难度）与单 Agent 简单性之间的权衡

## 问题

你在第 14 阶段构建了一个单 Agent。它工作正常。它能读文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及在写代码之前先调研外部 API 的要求。

Agent 卡住了。不是因为 LLM 笨，而是因为任务超出了单个 Agent 循环能处理的范围。上下文窗口被文件内容填满。Agent 忘记了 40 次 tool call 之前读过的东西。它同时想当研究员、程序员和评审员，结果三样都做得很差。

这就是单 Agent 的天花板。每当任务需要以下之一时，你就会撞上它：

- **上下文超出单个窗口的容量**——读取 50 个文件就会突破 200k token
- **不同阶段需要不同专长**——调研所需的 prompt 与代码生成不同
- **可以并行进行的工作**——既然能同时读三个文件，为什么要串行读？

## 概念

### 单 Agent 的天花板

单个 Agent 是一个循环、一个上下文窗口、一个系统 prompt。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三样东西会崩掉：

1. **上下文饱和**——工具结果不断堆积。到第 30 轮时，Agent 已经消耗了 150k token 的文件内容、命令输出和先前推理。第 5 轮的关键细节丢失了。

2. **角色混淆**——一个写着「你是研究员、程序员、评审员和测试员」的系统 prompt，产出的 Agent 只会半调研、半写代码，评审永远做不完。

3. **串行瓶颈**——Agent 读文件 A，再读文件 B，再读文件 C。三次串行 LLM 调用，三次串行工具执行。没有任何并行。

### 多 Agent 解决方案

拆分工作。给每个 Agent 一个职责、一个上下文窗口，以及一个为该职责调优的系统 prompt：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 Agent 都有：

- 一个聚焦的系统 prompt（「你是一名代码评审员。你唯一的职责是找出 bug。」）
- 自己的上下文窗口（不会被其他 Agent 的工作污染）
- 清晰的输入/输出契约（接收调研笔记，输出代码）

### 采用这种思路的真实系统

**Claude Code 子 Agent**——当 Claude Code 用 `Task` 派生子 Agent 时，它会创建一个带限定任务的子 Agent。父 Agent 保持自己的上下文干净。子 Agent 做聚焦的工作并返回摘要。

**Devin**——运行规划 Agent、编码 Agent 和浏览器 Agent。规划者把工作拆成步骤，编码者写代码，浏览器 Agent 调研文档。每个都有自己的上下文。

**多 Agent 编码团队（SWE-bench）**——SWE-bench 上表现最好的系统使用一个阅读代码库的研究员、一个设计修复方案的规划者，以及一个实现修复的编码者。单 Agent 系统得分更低。

**ChatGPT Deep Research**——并行派生多个搜索 Agent，各自探索不同角度，然后综合结果。

### 光谱

多 Agent 不是二元的，而是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单 Agent**——一个循环、一个 prompt。适合简单任务。

**子 Agent**——父 Agent 为聚焦的子任务派生子 Agent。父 Agent 维护计划，子 Agent 回报结果。这就是 Claude Code 的做法。

**Pipeline**——Agent 按顺序运行。Agent A 的输出成为 Agent B 的输入。适合分阶段的工作流：调研 -> 编码 -> 评审 -> 测试。

**团队**——Agent 并行运行，共享消息总线。每个都有角色，由编排器协调。适合需要同时运用不同技能的场景。

**Swarm（蜂群）**——许多相同或近乎相同的 Agent 共享状态。没有固定的编排器。Agent 从队列中领取工作。适合高吞吐的并行任务。

### 四种多 Agent 模式

#### 模式 1：Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 Agent 转换数据并向下传递。易于推理。某一阶段失败会阻塞后续所有阶段。

#### 模式 2：Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作拆分到并行的 Agent，然后合并结果。适合能分解为独立子任务的任务。

#### 模式 3：Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能编排器决定做什么、把工作委派给工作者，并综合结果。编排器本身就是一个具备派生工作者工具的 Agent。

#### 模式 4：对等 Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中心编排器。Agent 点对点通信。决策从交互中涌现。更难调试，但能扩展到大量 Agent。

### 什么时候不该用多 Agent

多 Agent 增加了复杂性。Agent 之间每条消息都是一个潜在故障点。调试从「读一段对话」变成「跨五个 Agent 追踪消息」。

**以下情况保持单 Agent：**

- 任务能装进一个上下文窗口（工作数据约 100k token 以内）
- 不同阶段不需要不同的系统 prompt
- 串行执行已经足够快
- 任务足够简单，拆分带来的开销大于价值

**复杂性代价：**

- 每个 Agent 边界都是一次有损压缩：Agent A 的完整上下文被概括成一条发给 Agent B 的消息
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是 bug 的来源
- 延迟增加：N 个 Agent 意味着至少 N 次串行 LLM 调用，若需来回对话则更多
- 成本成倍增加：每个 Agent 独立消耗 token

经验法则：如果一个任务的 tool call 少于 20 次，且能装进 100k token，就保持单 Agent。

```figure
swarm-messages
```

## 构建它

### 第 1 步：超载的单个 Agent

下面是一个试图包揽一切的单个 Agent。它有一个庞大的系统 prompt，以及一个同时装着调研、代码和评审的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：

- 上下文窗口随每个阶段增长。到评审步骤时，它同时包含调研笔记、代码和先前的推理。
- 系统 prompt 很笼统，无法针对每个阶段调优。
- 没有任何东西并行运行。

### 第 2 步：专家 Agent

现在拆分它。每个 Agent 只负责一项工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有一个聚焦的 prompt，并得到只包含所需输入的干净上下文窗口。

### 第 3 步：通过消息协调

用显式的消息传递把专家们连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 Agent 只接收发给自己的消息。没有上下文污染。研究员阅读的 50k token 文档永远不会进入评审员的上下文。

### 第 4 步：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多 Agent 版本使用更多总 token（三个 Agent、三次独立的 LLM 调用），但每个 Agent 的上下文保持干净。由于系统 prompt 是专门化的，每个阶段的质量都会提升。

## 使用它

本节课产出一个可复用的 prompt，用于判断何时该采用多 Agent。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 增加第四个专家：一个「测试员」Agent，它从编码者那里接收代码、从评审员那里接收评审反馈，然后编写测试
2. 修改 pipeline，让评审员能把反馈发回给编码者，形成修订循环（最多 2 轮）
3. 把串行 pipeline 改成 fan-out：并行运行研究员和一个「需求分析员」Agent，然后把它们的输出合并后再传给编码者

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Swarm | 「AI Agent 的蜂群思维」 | 一组共享状态、没有固定领导者的对等 Agent。行为从局部交互中涌现。 |
| Orchestrator | 「老板 Agent」 | 其工具包含派生和管理其他 Agent 的 Agent。它负责规划和委派，但可能不亲自干活。 |
| Coordinator | 「交通警察」 | 一种非 Agent 组件（通常只是代码，而不是 LLM），按规则在 Agent 之间路由消息。 |
| Consensus | 「Agent 们达成一致」 | 一种协议，多个 Agent 必须先达成一致才能继续。用于需要解决冲突输出的场景。 |
| Emergent behavior | 「Agent 自己搞明白了」 | 由 Agent 交互产生、但并未被显式编程的系统级模式。可能有益，也可能有害。 |
| Fan-out / fan-in | 「Agent 的 map-reduce」 | 把任务拆分给并行 Agent（fan-out），再合并其结果（fan-in）。 |
| Message passing | 「Agent 互相交谈」 | Agent 之间的通信机制：从一方向另一方发送结构化数据，取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)——多 Agent 模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)——微软的多 Agent 对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)——Claude Code 如何用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/)——基于角色的多 Agent 框架
