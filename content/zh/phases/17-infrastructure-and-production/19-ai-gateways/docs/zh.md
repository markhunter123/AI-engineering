# AI 网关 —— LiteLLM、Portkey、Kong AI Gateway 与 Bifrost

> 网关位于应用与模型提供商之间。核心功能包括提供商路由、故障转移、重试、速率限制、密钥引用、可观测性和防护栏。2026 年的市场格局：**LiteLLM** 是 MIT 开源项目，支持 100+ 提供商，兼容 OpenAI 接口，但在约 2000 RPS 附近性能崩塌（8 GB 内存，公开基准中出现级联故障）；最适合 Python、低于 500 RPS 的开发与原型场景。**Portkey** 定位为控制平面（防护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，每请求延迟开销 20-40 ms，生产套餐 $49/月。**Kong AI Gateway** 构建在 Kong Gateway 之上——Kong 自己在相同 12 个 CPU 上的基准显示：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 层最多 5 个模型）；如果已经在用 Kong，适合企业环境。**Bifrost**（Maxim AI）——自动重试并可配置退避，在 OpenAI 返回 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateway**——托管、零运维、基础重试。数据驻留要求驱动自托管决策；Portkey 和 Kong 处于中间位置，提供开源版加可选托管版。

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## 学习目标

- 列举网关的六项核心功能（路由、故障转移、重试、速率限制、密钥、可观测性、防护栏）。
- 将 2026 年的四款网关（LiteLLM、Portkey、Kong AI、Bifrost）对应到规模上限与用例。
- 引用 Kong 的基准数据（相对 Portkey 快 228%，相对 LiteLLM 快 859%），并解释它对 >500 RPS 意味着什么。
- 在数据驻留与运维预算的约束下，选择自托管还是托管方案。

## 问题

你的产品要调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供商都有不同的 SDK、错误模型、速率限制和认证方式。你想要故障转移（OpenAI 返回 429 时改试 Anthropic）、统一的凭据存储、统一的可观测性，以及按租户划分的速率限制。

在应用层重新造这套东西，会让每个服务都耦合到每个提供商。网关层把它收敛成一个进程、一套 API（通常是 OpenAI 兼容接口），再扇出到各个提供商。

## 概念

### 六项核心功能

1. **提供商路由** —— 把 OpenAI、Anthropic、Gemini、自托管模型等放在同一套 API 之后。
2. **故障转移** —— 遇到 429、5xx 或质量失败时，换一处重试。
3. **重试** —— 指数退避，限制尝试次数。
4. **速率限制** —— 按租户、按密钥、按模型。
5. **密钥引用** —— 运行时从 vault 拉取凭据（绝不放进应用）。
6. **可观测性** —— OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **防护栏** —— PII 脱敏、越狱检测、允许主题过滤。

### LiteLLM —— MIT 开源，Python

- 支持 100+ 提供商，兼容 OpenAI，提供路由配置、故障转移和基础可观测性。
- 在 Kong 的基准中约 2000 RPS 时崩塌；8 GB 内存占用，持续负载下出现级联故障。
- 最适合：Python 应用、<500 RPS、开发/预发网关、实验性路由。
- 成本：开源版 $0；有云免费层。

### Portkey —— 控制平面定位

- 自 2026 年 3 月起采用 Apache 2.0 开源。提供防护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求延迟开销 20-40 ms。
- 生产层 $49/月，含数据保留 + SLA。
- 最适合：需要把防护栏与可观测性打包提供的受监管行业。

### Kong AI Gateway —— 规模之选

- 构建在 Kong Gateway 之上（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自己的基准，在等效 12 CPU 上：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最适合：已经使用 Kong；>1000 RPS；愿意购买许可证。

### Bifrost（Maxim AI）

- 自动重试，退避可配置。
- OpenAI 返回 429 时回退到 Anthropic，是它的标志性配方。
- 较新的入局者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零运维。提供基础重试和可观测性。
- 最适合：运行在 Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在防护栏和速率限制方面不如 Kong/Portkey 全面。

### 自托管 vs 托管

数据驻留是决定性因素。医疗和金融默认自托管（LiteLLM、Portkey 开源版或 Kong）。消费级产品默认托管（Cloudflare AI Gateway）或中间档（Portkey 托管版）。混合模式：受监管租户自托管，其他租户用托管。

### 延迟预算

- LiteLLM：通常 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：1-3 ms 开销（边缘优势）。

网关延迟会直接叠加到 TTFT 上。TTFT P99 < 100 ms 的 SLA，选 Kong 或 Cloudflare。P99 < 500 ms，任选。

### 速率限制语义很重要

简单令牌桶在中等到较大规模都能工作。多租户需要滑动窗口 + 突发额度 + 按租户分级。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分级限流。

### 网关 + 可观测性 + 路由的组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选一个能同时覆盖三者的工具，或小心地把它们组合起来：2026 年大多数部署把 Helicone（可观测性）或 Portkey（防护栏）与 Kong（规模）按分工组合。

### 应该记住的数字

- LiteLLM：约 2000 RPS 时崩塌，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

```figure
mx-gateway-fallback
```

## 使用它

`code/main.py` 模拟在 3 个提供商之间带故障转移的网关路由，并注入 429/5xx。它报告延迟、重试率和故障转移命中率。

## 交付它

本节课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规要求和延迟预算，选出一款网关。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管的故障转移。在 5% 提供商错误率下，预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线为 300 ms。哪些网关能留在预算内？
3. 一个医疗客户要求自托管 + PII 脱敏 + 审计。选 Portkey 开源版还是 Kong。
4. 比较 LiteLLM 与 Kong：团队应该在什么 RPS 上限迁移？
5. 为一个多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 网关（Gateway） | “API 中间商” | 位于应用与提供商之间的进程 |
| LiteLLM | “MIT 那个” | Python 开源，100+ 提供商，2K RPS 崩塌 |
| Portkey | “防护栏网关” | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | “规模那个” | 构建在 Kong Gateway 上，基准领先 |
| Bifrost | “Maxim 的网关” | 重试 + Anthropic 故障转移配方 |
| Cloudflare AI Gateway | “边缘托管” | 边缘部署的托管网关，零运维 |
| PII 脱敏（PII redaction） | “数据清洗” | 发给模型前用正则 + NER 做掩码 |
| 越狱检测（Jailbreak detection） | “提示注入防护” | 对用户输入跑分类器 |
| 审计追踪（Audit trail） | “合规日志” | 每次 LLM 调用的不可变记录 |
| 令牌桶（Token-bucket） | “简单限流” | 基于补充的限流器 |
| 滑动窗口（Sliding-window） | “精确限流” | 时间窗口限流；公平性更好 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
