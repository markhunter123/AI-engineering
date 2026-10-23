# 托管式 LLM 平台——Bedrock、Vertex AI、Azure OpenAI

> 三家超大规模云厂商，三种截然不同的策略。AWS Bedrock 是模型市场——Claude、Llama、Titan、Stability、Cohere 同处一个 API 之后。Azure OpenAI 是与 OpenAI 的独家合作，并提供预置吞吐单元（Provisioned Throughput Units，PTU）来获得专用容量。Vertex AI 以 Gemini 为先，拥有最好的长上下文与多模态叙事。2026 年，Artificial Analysis 测得 Azure OpenAI 的延迟中位数约为 50 ms，而 Bedrock 在等效 Llama 3.1 405B 上约为 75 ms——PTU 解释了这个差距，因为专用容量胜过分时共享的按需容量。决策规则不是「哪个最快」，而是「哪个模型目录和 FinOps 面与我的产品匹配」。本节课教你带着写下来的权衡去选择，而不是靠感觉。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## 学习目标

- 说出三种平台策略（市场型、独家型、Gemini 优先型），并把每一种匹配到产品用例。
- 解释 Azure OpenAI 中预置吞吐单元（PTU）为你买到了什么，以及为什么按需 Bedrock 在 405B 量级上通常慢约 25 ms。
- 画出每个平台的 FinOps 归因面（Bedrock Application Inference Profiles 对比 Vertex 的按团队项目对比 Azure 作用域 + PTU 预留）。
- 写下「两家供应商最低原则」政策，并解释为什么在 2026 年单厂商锁定是代价高昂的错误。

## 问题

你为产品选了 Claude 3.7 Sonnet。现在你需要把它服务化。你可以直连 Anthropic API，也可以通过 AWS Bedrock 调用，或者经过网关。直连 API 最简单；Bedrock 增加了 BAA、VPC endpoint、IAM 和 CloudWatch 归因。网关则增加了故障转移、统一账单以及跨供应商的速率限制。

更深层的问题是目录。如果你的同一个产品需要 Claude、Llama 和 Gemini，你无法从一家全部买到——除非那个「一家」同时是 Bedrock 加 Vertex 加 Azure OpenAI。超大规模云厂商之间不可互换——它们在「谁拥有模型层」上各自下了不同的注。

本节课梳理这三种押注、延迟差距、FinOps 差距和锁定风险。

## 概念

### 三种策略

**AWS Bedrock**——市场型。Claude（Anthropic）、Llama（Meta）、Titan（AWS 一方）、Stability（图像）、Cohere（embedding）、Mistral，外加图像和 embedding 子目录。一个 API、一个 IAM 面、一个 CloudWatch 导出。Bedrock 的押注是：客户想要可选项，胜过想要单一模型。

**Azure OpenAI**——独家合作型。你能拿到 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心内对 OpenAI 模型做微调。在「Azure OpenAI Service」目录里没有非 OpenAI 模型——那些走 Azure AI Foundry（另一个独立产品）。Azure 的押注是：OpenAI 仍处于前沿，而客户想要在这一特定关系上获得企业级管控。

**Vertex AI**——Gemini 优先，其它都在其次。Gemini 1.5 / 2.0 / 2.5 Flash 与 Pro，外加 Model Garden（第三方）。Vertex 的押注是多模态长上下文——100 万 token 的 Gemini 上下文就是差异化所在。

### 规模下的延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署上（共享按需），Azure OpenAI 的首 token 延迟中位数约为 50 ms；Bedrock 约为 75 ms。这个差距不是 AWS 的失败——而是容量模型的差异。Azure 售卖 PTU（Provisioned Throughput Units），为你的租户预留 GPU 容量。Bedrock 的对应产品（Provisioned Throughput）也存在，但每单位起价约 $21/小时，大多数客户仍停留在共享按需。

按需共享容量要与其它所有客户的流量竞争。专用容量不需要。如果你的产品 SLA 是 P99 TTFT < 100 ms，你要么在 Azure 上买 PTU，要么买 Bedrock Provisioned Throughput，要么接受默认的波动。

### 预置吞吐经济学

Azure PTU：一块预留的推理算力。对可预测工作负载，相比按需最高可省约 70%。成本按小时固定，与实际流量无关——即使空闲你也要为预留付费。盈亏平衡点通常在 40-60% 的持续利用率附近。

Bedrock Provisioned Throughput：每小时 $21-$50，取决于模型和区域。算法类似——盈亏平衡大约在峰值利用率的一半。需要按月承诺。

Vertex 的预置容量按 Gemini SKU 出售；价格随模型和区域而异，公开宣传较少。

### FinOps 面——真正的差异化

**Bedrock Application Inference Profiles** 是市场上最干净的归因方式。给 profile 打上 `team`、`product`、`feature` 标签；把所有模型调用路由经过它；CloudWatch 无需后处理就能按 profile 拆出成本。2025 年加入，至今仍是原生最细粒度的超大规模云方案。

**Vertex** 的归因是「按团队分项目 + 到处打标签」。你把每个团队建模为一个 GCP project，给每个资源打 label，再用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 让你能对成本数据跑任意 SQL。

**Azure** 依赖订阅/资源组作用域加 tag，PTU 预留是一等成本对象。tag 从资源组继承，而非从请求继承，因此按请求归因需要 Application Insights 自定义指标，或一个在请求上盖 header 的网关。

模式是：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 若不插桩则最不透明。

### 锁定是 2026 年的风险

当只有一个模型独大时，绑定单一超大规模云没问题。到了 2026 年，前沿每月都在移动——这一季度是 Claude 3.7，下一季度是 Gemini 2.5，再下一季度是 GPT-5。锁定一个平台，就把自己锁在了前沿的三分之二之外。

高效团队采用的模式是：任何对产品至关重要的 LLM 调用，都至少要有两家供应商。Bedrock 加 Azure OpenAI 是常见的组合——Claude 从一家来，GPT 从另一家来，两者之间故障转移，同一个网关。成本上升可以忽略，因为网关会路由到最优项；而故障期间（比如 Azure OpenAI 2025 年 1 月事故、AWS us-east-1 故障）的可用性提升是决定性的。

### 数据驻留、BAA 与受监管行业

Bedrock：多数区域提供 BAA；VPC endpoint；guardrails。金融科技的常见默认。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；受监管企业的默认选择。
Vertex：HIPAA、GDPR、按区域的数据驻留；Google Cloud 的合规栈。

三者都满足基础勾选项。差异在于数据保留策略、日志如何处理，以及滥用监控是否会读取你的流量（多数默认开启；企业可退出）。

### 应该记住的数字

- Azure OpenAI 在等效 Llama 3.1 405B 上的 TTFT 中位数：约 50 ms（使用 PTU）。
- Bedrock 按需的 TTFT 中位数：约 75 ms。
- Bedrock Provisioned Throughput：每单位 $21-$50/小时。
- Azure PTU 盈亏平衡：约 40-60% 持续利用率。
- 高利用率下 PTU 相对按需的节省：最高 70%。

```figure
i4-platform-lanes
```

## 使用它

`code/main.py` 在一个合成工作负载上比较这三个平台——它建模按需与 PTU 的经济性、TTFT 波动，以及成本归因的保真度。运行它，看看 PTU 在哪里划算，以及市场型的模型广度在哪里胜过 TTFT 差距。

## 交付它

本节课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载画像（所需模型、TTFT SLA、每日流量、合规要求），它会推荐一个主平台、一个回退平台，以及一套 FinOps 插桩计划。

## 练习

1. 运行 `code/main.py`。对于 70B 级别的模型，Azure PTU 在多少持续利用率下胜过按需？算出盈亏平衡点，并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署——哪个模型放在哪家超大规模云上，前面坐什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAA、US-East 数据驻留，以及 P99 TTFT 低于 100 ms。选一个平台，并用三个具体特性论证。
4. 你发现本月 Bedrock 账单涨了 4 倍，流量却没有变化。如果没有 Application Inference Profiles，你如何找出元凶？有了 profile 之后，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜——直连 Anthropic API、Bedrock 按需，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Bedrock | 「AWS 的 LLM 服务」 | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | 「Azure 的 ChatGPT」 | 在 Azure 数据中心内提供带企业管控的独家 OpenAI 模型 |
| Vertex AI | 「Google 的 LLM」 | 以 Gemini 为先、通过 Model Garden 提供第三方模型的平台 |
| PTU | 「专用容量」 | Provisioned Throughput Unit——预留的推理 GPU，按小时计价 |
| Application Inference Profile | 「Bedrock 打标签」 | 带标签的按产品成本/用量 profile，CloudWatch 原生 |
| Model Garden | 「Vertex 目录」 | Vertex AI 的第三方模型区，与 Gemini 分开 |
| Two-provider minimum | 「LLM 冗余」 | 每条关键 LLM 路径至少跨 2 家超大规模云运行的政策 |
| BAA | 「HIPAA 文书」 | Business Associate Agreement；处理 PHI 所需；三家都提供 |
| Abuse monitoring | 「日志监视者」 | 供应商侧对 prompt/输出的安全扫描；企业版可退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/)——权威价目表与 Provisioned Throughput 定价。
- [Azure OpenAI Service 定价](https://azure.microsoft.com/en-us/pricing/details/azure-openai/)——PTU 经济学与价目表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing)——Gemini 档位与 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/)——跨供应商的持续延迟与吞吐量基准。
- [The AI Journal——AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/)——企业决策框架。
- [Finout——Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend)——归因机制并排对比。
