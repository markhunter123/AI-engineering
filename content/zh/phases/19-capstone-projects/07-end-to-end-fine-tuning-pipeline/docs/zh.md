# 项目 07 —— 端到端微调流水线（从数据到 SFT 到 DPO 到服务）

> 训练一个 8B 模型：用你自己的数据训练，用你自己的偏好做 DPO 对齐，然后量化、投机解码，并以可度量的 $/1M tokens 成本对外提供服务。2026 年的开源技术栈是 Axolotl v0.8、TRL 0.15、用于快速迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF，以及带 EAGLE-3 的 vLLM 0.7 用于服务。这个项目要求以可复现的方式跑完整条流水线——输入 YAML，输出可服务的端点——并按 2026 年 Model Openness Framework 发布模型卡。

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## 问题

2026 年，每一支认真做 AI 的团队都会常备一条微调流水线。原因不是他们要发布前沿基础模型，而是可度量的收益来自下游适配：领域 SFT、针对标注偏好的 DPO、用于投机解码的蒸馏草稿模型，以及用 EAGLE-3 提供服务。Axolotl v0.8 负责多 GPU 的 SFT 配置。TRL 0.15 负责 DPO 和 GRPO。Unsloth 让你在单 GPU 上快速迭代。带 EAGLE-3 的 vLLM 0.7 能把解码吞吐量提升 2-3 倍而不损失质量。工具已经可用；真正的功夫在 YAML、数据卫生和评估纪律上。

你会让一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）在任务数据上先做 SFT 再做 DPO，然后量化用于服务，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量增益。你要产出一份符合 2026 年 Model Openness Framework 的模型卡。重点是可复现性：一条命令即可端到端重跑整条流水线。

## 概念

流水线有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗，以及针对公开基准污染的切分卫生检查。**SFT**：Axolotl YAML，8xH100 上 ZeRO-3，余弦调度，序列打包，2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对可以人工标注或由模型判定，调 beta。**量化**：GPTQ + AWQ + GGUF，保证部署灵活性。**服务**：带 EAGLE-3 投机头的 vLLM 0.7（或带 SpecForge 的 SGLang）、K8s 部署、基于队列等待的 HPA。

消融实验才是交付物：在三个任务特定基准上对比仅 SFT、SFT+DPO、SFT+GRPO。服务指标：batch 1 / 8 / 32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏差评估、可复现随机种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 做去重，Nemotron-CC 分类器做质量过滤，Presidio 做 PII 清洗
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，配合 ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15 的 DPO 或 GRPO；单 GPU 迭代用 Unsloth
- 量化：GPTQ（Marlin）、AWQ，以及经 llama.cpp 的 GGUF
- 服务：带 EAGLE-3 投机解码的 vLLM 0.7（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，基于 queue-wait 指标的 HPA
- 可观测性：训练用 W&B，推理用 Langfuse

```figure
ce-finetune-stages
```

## 构建它

1. **数据流水线。** 在原始语料上运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。Presidio 清洗 PII。用显式种子写出训练/验证切分。

2. **污染检查。** 对每个验证切分，计算它与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集之间的 MinHash 重叠。拒绝任何重叠。

3. **Axolotl SFT。** 使用带 ZeRO-3、FA3、序列打包的 YAML。在 8xH100 上训练 2-3 个 epoch。日志写入 W&B。

4. **TRL DPO / GRPO。** 拿 SFT checkpoint，在偏好对上跑一个 epoch 的 DPO（或在数学/代码任务上带可验证奖励跑 GRPO）。扫描 beta。

5. **量化。** 产出三种量化：GPTQ-INT4-Marlin、AWQ-INT4、供 llama.cpp 使用的 GGUF-Q4_K_M。记录文件大小和标称吞吐量。

6. **带投机解码的服务。** vLLM 0.7 配置，EAGLE-3 起草头经 Red Hat Speculators 训练。在 batch 1 / 8 / 32 下测接受率和尾延迟。报告同一评估下相对 Anthropic / OpenAI 的 $/1M tokens。

7. **评估矩阵。** 对 base、仅 SFT、SFT+DPO、SFT+GRPO 分别运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出一张对比表。

8. **安全评估。** 开发集上的 Llama Guard 4 通过率。ShieldGemma-2 输出过滤。

9. **模型卡。** 按 MOF 2026 模板填写：数据、训练、评估、安全、许可，以及包含 YAML 和 commit SHA 的可复现性章节。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一条命令即可跑完从数据到 SFT、DPO、量化、服务、评估的全流程，并产出模型卡和已部署的端点。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对 base 的评估增益 | 在目标任务（MMLU-Pro、MT-Bench-v2、任务特定基准）上的实测增益 |
| 20 | 流水线可复现性 | 一条命令以相同种子端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查全绿 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整度 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上对比仅 SFT、SFT+DPO、SFT+GRPO。报告哪种偏好方法胜出，以及领先多少。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在质量匹配的前提下测量 $/1M tokens。

3. 在领域数据与通用 ShareGPT 上测 EAGLE-3 接受率。报告差值，以及它对延迟预算意味着什么。

4. 注入 1% 的污染（把 MMLU-Pro 答案泄漏进训练数据）并重跑评估。观察 MMLU-Pro 准确率出现不切实际的跃升。构建一个能抓住这种问题的污染检查 CI 门禁。

5. 增加 LoRA SFT 作为全量微调的替代方案。测量内存降低 10 倍时的质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | 「SFT 训练器」 | 统一的 YAML 驱动训练器，支持 SFT、DPO 和蒸馏 |
| TRL | 「偏好调优器」 | Hugging Face 的库，支持 LLM 上的 DPO、GRPO、PPO |
| GRPO | 「组相对策略优化」 | DeepSeek R1 的强化学习配方，使用可验证奖励 |
| EAGLE-3 | 「投机解码草稿」 | 预测未来 N 个 token 的起草头；vLLM 用目标模型验证 |
| MOF | 「Model Openness Framework」 | 2026 年按数据、代码、许可给模型发布打分的标准 |
| Contamination check | 「切分卫生」 | 基于 MinHash 检测测试集泄漏进训练集 |
| Acceptance rate | 「EAGLE / MTP 指标」 | 目标模型接受的草稿 token 比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) —— SFT / DPO 训练的参考实现
- [TRL 文档](https://huggingface.co/docs/trl) —— DPO 和 GRPO 的参考实现
- [Unsloth](https://github.com/unslothai/unsloth) —— 单 GPU 迭代的参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) —— GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) —— 服务栈参考
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) —— 另一种投机解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) —— 开放发布评分标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) —— 标准评估运行器
