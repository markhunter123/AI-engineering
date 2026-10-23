# LLM 生产环境的混沌工程

> 2026 年，LLM 的混沌工程已是一门独立学科。在生产环境运行实验前的前置条件：已定义的 SLI/SLO、trace + metric + log 可观测性、自动回滚、运行手册、值班。架构有四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（护栏 + 中止 + 流量过滤）、可观测性（指标 + trace + 日志），外加回灌到 SLO 调整的反馈回路。防护栏是强制性的：如果每日错误预算消耗 > 预期的 2 倍，燃尽率告警会暂停实验；抑制窗口 + trace-ID 关联用于对告警噪声去重。节奏：每周小规模金丝雀 + SLO 评审；每月游戏日 + 事后复盘；每季度跨团队韧性审计 + 依赖关系梳理。LLM 专属实验：内存过载、网络故障、提供商中断、畸形 prompt、KV cache 驱逐风暴。工具：Harness Chaos Engineering（由 LLM 推导的建议、爆炸半径降级、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出混沌工程的五项前置条件（SLI/SLO、可观测性、回滚、运行手册、值班），并解释为什么缺任一项都会破坏这套实践。
- 画出四个平面（控制、目标、安全、可观测性）以及回灌到 SLO 的反馈回路。
- 列举五个 LLM 专属实验（内存过载、网络故障、提供商中断、畸形 prompt、KV 驱逐风暴）。
- 根据技术栈选择工具——Harness、LitmusChaos、Chaos Mesh。

## 问题

传统技术栈中的混沌测试已经成熟。LLM 技术栈引入了新的失败模式。一个 4K token 的 prompt 里带了一个「毒」字符，就能让分词器卡住 12 秒。上游提供商返回 429；你的网关重试；重试放大的并发让你的服务 OOM。突发负载下的 KV cache 驱逐风暴引发重新预填充级联，占满算力。

这些都不会出现在单元测试里。混沌工程就是让它们先于用户被发现的方法。

## 概念

### 前置条件

在生产环境运行混沌之前，必须具备：

1. **SLI/SLO** —— 已定义的服务级指标与目标。
2. **可观测性** —— trace、metric、log，接入仪表盘。
3. **自动回滚** —— Phase 17 · 20 的策略开关回滚。
4. **运行手册** —— 结构化，Phase 17 · 23。
5. **值班** —— 有人响应。

缺任何一项，混沌实验就会变成真实事故。

### 四个平面 + 反馈

**控制平面** —— 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**目标平面** —— 服务、pod、节点、负载均衡器、数据存储。

**安全平面** —— kill switch、抑制窗口、爆炸半径限制、错误预算闸门。

**可观测性平面** —— 正常指标 + trace-ID 关联，用于区分混沌引发的故障与自然故障。

**反馈回路** —— 发现回灌到 SLO 调整、运行手册更新和代码修复。

### 防护栏是强制性的

- **燃尽率告警**：如果每日错误预算消耗超过预期的 2 倍，暂停实验。
- **抑制窗口**：实验期间，屏蔽爆炸半径内的非实验告警。
- **trace-ID 关联**：所有实验引发的错误都带标签，方便值班人员去重。

### 五个 LLM 专属实验

1. **内存过载** —— 用高并发发送长上下文请求，强制触发 KV cache 抢占风暴。观察：服务会优雅卸载还是崩溃？

2. **网络故障** —— 切断推理网关与提供商之间的连接。观察：回退是否在 SLA 内生效？（Phase 17 · 19）

3. **提供商中断模拟** —— OpenAI 返回 100% 429。观察：路由是否故障转移到 Anthropic？（Phase 17 · 16、19）

4. **畸形 prompt** —— 注入使分词器卡住的载荷（例如深层嵌套的 unicode、超大的 UTF-8 码点）。观察：单个请求会不会锁死一个 worker？

5. **KV 驱逐风暴** —— 通过占满 vLLM 的 block 预算强制驱逐。观察：LMCache 会恢复，还是服务降级？

### 节奏

- **每周** —— 在预发环境做小规模金丝雀实验，也许生产 5%。
- **每月** —— 针对特定场景的排期游戏日；跨团队参与；事后复盘。
- **每季度** —— 跨团队韧性审计；更新依赖关系图。

### 工具

- **Harness Chaos Engineering** —— 商业产品；由 AI 推导实验建议；爆炸半径降级；MCP 工具集成。
- **LitmusChaos** —— CNCF 项目；基于 Kubernetes workflow。
- **Chaos Mesh** —— CNCF sandbox；Kubernetes 原生 CRD 风格。
- **Gremlin** —— 商业产品；支持面广。
- **AWS FIS** / **Azure Chaos Studio** —— 托管云服务。

### 从小开始

第一个实验：在稳态流量下 kill 掉一个 decode 副本的 pod。观察重路由与恢复。如果这一步有效且看起来安全，再升级到网络混沌。

第一个 LLM 专属实验：让一个提供商持续返回 429 五分钟。观察回退。大多数团队会发现自己的回退路径根本没测全。

### 应该记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 燃尽率暂停：预期每日预算消耗的 2 倍。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五个 LLM 实验：内存、网络、提供商、畸形 prompt、KV 风暴。

```figure
i4-chaos-guard
```

## 使用它

`code/main.py` 模拟三个带安全平面闸门的混沌实验。报告哪些实验会触发燃尽率中止。

## 交付它

本节课产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度，选出前三个实验及工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了燃尽率闸门，为什么？
2. 为一个基于 vLLM 的 RAG 服务设计前五个混沌实验。包含成功标准。
3. 燃尽率告警暂停了一个实验。你如何判断根因——是混沌还是自然故障？
4. 论证混沌应该在生产运行还是只在预发。什么时候生产才是正确答案？
5. 说出三种通用网络混沌无法复现的 LLM 专属故障模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | 「服务目标」 | 指标 + 目标；必需的前置条件 |
| Blast radius | 「影响范围」 | 实验影响的服务 / 用户集合 |
| Burn-rate alert | 「预算闸门」 | 错误预算燃尽率 > 预期 2 倍时触发 |
| Game day | 「月度演练」 | 排期的跨团队混沌演练 |
| LitmusChaos | 「CNCF workflow」 | CNCF 毕业的 Kubernetes 混沌工具 |
| Chaos Mesh | 「CNCF CRD」 | CNCF sandbox 的 Kubernetes 原生混沌 |
| Harness CE | 「商业 AI 辅助」 | Harness 混沌工程，带 AI 建议 |
| Malformed prompt | 「分词器炸弹」 | 让分词停滞的输入 |
| KV eviction storm | 「抢占级联」 | 大规模驱逐触发重新预填充 |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for Large Language Models (Apress, 2026)](https://link.springer.com/book/10.1007/979-8-8688-2827-0)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
