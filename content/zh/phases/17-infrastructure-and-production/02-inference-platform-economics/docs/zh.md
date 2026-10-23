# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先的市场（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日把每 GPU 价格上调 $1/小时；40 亿美元估值加上每日 10 万亿+ token 的处理量，说明量驱动的模式行之有效。Baseten 于 2026 年 1 月完成 3 亿美元 E 轮融资，估值 50 亿美元。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级打磨，Modal 优化 Python 原生开发者体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本节课给你一张可以直接交给创始人的矩阵。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 minutes

## 学习目标

- 说出三个细分市场（定制芯片、GPU 平台、API 优先），并把每个供应商对应到细分市场。
- 解释为什么「按 token」的 API 定价模型会向服务引擎的成本曲线收敛，而不是向硬件的成本曲线收敛。
- 计算至少三家供应商的每次请求有效成本，并说明按分钟计费（Baseten、Modal）何时胜过按 token 计费。
- 判断给定工作负载（无服务器突发型、稳定高吞吐型、微调变体、多模态）的默认平台应该是哪个。

## 问题

你评估了托管式超大规模云平台。你决定需要一个更专注、更快的供应商——要延迟就选 Fireworks，要广度就选 Together，要微调自定义模型就选 Baseten。现在你有六个真实选项，而定价页对不上。Fireworks 展示 $/百万 token；Baseten 展示 $/分钟；Modal 展示 $/秒；Replicate 展示 $/次预测。不对工作负载建模，你无法直接横向比较。

更糟的是，每个定价页背后的商业模式都不同。Fireworks 在共享 GPU 上运行自研引擎（FireAttention）；按 token 费率反映的是他们的利用率曲线。Baseten 给你 Truss + 专用 GPU；按分钟计费反映的是独占性。Modal 是真正的 Python 无服务器——按秒计费，冷启动不到一秒。同样的产出（一次 LLM 响应），三种不同的成本函数。

本节课对这六家建模，并告诉你每家何时胜出。

## 概念

### 三个细分市场

**定制芯片**——Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在相同模型上，解码速度通常比基于 GPU 的集群快 5-10 倍。每 token 价格更高（2025 年末 Groq 在 Llama-70B 上约为 $0.99/百万 token），但对延迟敏感的场景无可匹敌。Groq 是语音智能体和实时翻译的生产首选。

**GPU 平台**——Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）上，有时也用 AMD。它们处在「裸 GPU 租赁」（RunPod、Lambda）和「超大规模云托管服务」（Bedrock）之间的经济层。

**API 优先的市场**——Replicate、DeepInfra、OpenRouter、Fal。目录广，按预测或按秒付费，强调从注册到首次调用的时间。

### Fireworks——面向延迟优化的 GPU 平台

- FireAttention 引擎（自研）；宣称在等效配置下延迟比 vLLM 低 4 倍。
- 面向非交互式工作负载的 batch 档位，价格约为无服务器费率的 50%。
- 微调模型按基础模型相同的费率提供服务——这是相对那些对你的 LoRA 收溢价的供应商的真正差异化。
- 2026 年中：自 2026 年 5 月 1 日起，按需 GPU 租赁上调 $1/小时。规模采购的价格可协商。
- 财务信号：40 亿美元估值，每日处理 10 万亿+ token。

### Together——面向广度优化

- 200+ 个模型，包括在上游发布后数天内就跟进的开源版本。
- 在等效 LLM 上比 Replicate 便宜 50-70%——「AI 原生云」的定位就是量和目录。
- 推理 + 微调 + 训练在同一个 API 中。

### Baseten——面向企业级打磨

- Truss 框架：把依赖、密钥、服务配置打包进一个模型清单。
- GPU 从 T4 到 B200。按分钟计费，冷启动缓解做得合理。
- SOC 2 Type II、HIPAA 就绪。金融科技和医疗的常见选择。
- 50 亿美元估值，2026 年 1 月 E 轮（CapitalG、IVP、NVIDIA 出资 3 亿美元）。

### Modal——Python 原生体验优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一条命令即可部署。
- 按秒计费。预热后冷启动 2-4 秒；小模型不到 1 秒。
- 8,700 万美元 B 轮，估值 11 亿美元（2025）。在独立调研中开发者体验得分最高。

### Replicate——多模态广度

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- 在 LLM 按 token 费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale——Ray 原生

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载——推理步骤只是更大计算图中的一个节点。
- 托管的 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token 与按分钟——各自何时胜出

按 token 计费适用于对延迟不敏感且突发的工作负载——你只为实际用量付费。按分钟计费适用于利用率高且可预测的场景——一旦把 GPU 跑满，你就胜过了按 token。

粗略规则：对于单张专用 GPU 上持续利用率超过约 30% 的工作负载，按分钟计费（Baseten、Modal）开始胜过按 token 计费（Fireworks、Together）。低于这个水平，按 token 胜出，因为你不必为空闲买单。

### 自研引擎才是真正的护城河

上面每一个位于 vLLM 和 SGLang 之上的平台都宣称有自研引擎。FireAttention、RayTurbo、Baseten 的推理栈。自研引擎的说法带有营销色彩——诚实的框架是：vLLM + SGLang 大约占生产开源推理的 80%，平台层的差异在于开发者体验、归因和 SLA。

### 应该记住的数字

- Fireworks GPU 租赁：自 2026 年 5 月 1 日起上调 $1/小时。
- Fireworks 宣称：在等效配置下延迟比 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元融资）。
- Modal 估值：11 亿美元（B 轮，2025）。
- 持续利用率超过约 30% 时，按分钟计费胜过按 token 计费。

```figure
cost-per-token
```

## 使用它

`code/main.py` 在合成工作负载上、跨不同定价模型比较这六家供应商。报告 $/天和有效的 $/百万 token。运行它，找出按 token 与按分钟之间的盈亏平衡点。

## 交付它

本节课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载画像、SLA 和预算，它选出主推理平台并指出次选。

## 练习

1. 运行 `code/main.py`。对于单张 H100 上的 70B 模型，Baseten（按分钟）在多少持续利用率下胜过 Fireworks（按 token）？自己推导交叉点，并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并说出统一它们的网关模式。
3. Fireworks 把你主模型的价格上调了 $1/小时。如果 40% 的流量转移到 batch 档位（五折），建模混合成本的影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上每 1,000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次时呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Custom silicon | 「非 GPU 芯片」 | Groq LPU、Cerebras WSE、SambaNova RDU——为解码优化 |
| FireAttention | 「Fireworks 引擎」 | 自研注意力 kernel；宣称延迟比 vLLM 低 4 倍 |
| Truss | 「Baseten 的格式」 | 模型打包清单；依赖 + 密钥 + 服务配置 |
| Per-token | 「API 定价」 | 按消耗的 token 收费；不为空闲付费 |
| Per-minute | 「专用定价」 | 按 GPU 墙钟时间收费；高利用率时胜出 |
| Per-prediction | 「Replicate 定价」 | 按模型调用收费；图像/视频常见 |
| RayTurbo | 「Anyscale 引擎」 | 基于 Ray 的专有推理；在 Ray 集群上与 vLLM 竞争 |
| Batch tier | 「五折」 | 非交互式队列，费率更低；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate | 「Fireworks LoRA」 | 对 LoRA 服务的请求按基础模型费率收费（差异化） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing)——按 token 费率、batch 档位、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/)——按分钟费率、承诺容量、企业档位。
- [Modal 定价](https://modal.com/pricing)——按秒 GPU 费率与免费额度。
- [Together AI 定价](https://www.together.ai/pricing)——模型目录与按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing)——RayTurbo 与托管 Ray 定价。
- [Northflank——Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference)——对比评估。
- [Infrabase——2026 年 AI 推理 API 供应商](https://infrabase.ai/blog/ai-inference-api-providers-compared)——供应商格局。
