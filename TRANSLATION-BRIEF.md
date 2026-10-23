# 翻译规范（Translation Brief）

把课程 `docs/en.md` 翻译成简体中文 `docs/zh.md`。目标读者是在学 AI 工程的中文开发者。

## 绝对不能动的（保持逐字节一致）

1. **所有代码块**（``` 围栏内的全部内容，含语言标注行）。代码注释保留英文。
2. **`mermaid` 块** — 原样保留。
3. **`figure` 块** — 原样保留（就一行 key，如 `segmentation-flood`）。
4. **数学公式**（`$...$`、`$$...$$`、`\(...\)`）— 原样保留。
5. **四行元数据** `**Type:**` / `**Languages:**` / `**Prerequisites:**` / `**Time:**` — 原样保留英文（构建系统解析它们）。
6. **链接 URL、图片路径** — 不改。链接文字 `[...]` 里的描述可译。
7. **行内代码 `` `code` `` 里的内容** — 不译。
8. **HTML 标签、markdown 结构**（标题层级、列表、表格、引用）— 结构完全一致。
9. **锚点目标**：如果正文有 `[text](#some-heading)` 站内锚点，且目标标题被翻译，需要同步把锚点留作原英文 slug 形式是做不到的——**直接保留锚点原样**（构建器会尽量对齐，个别锚点失效可接受）。

## 要翻译的

- H1 标题（第一行 `# ...`）→ 简体中文
- blockquote 引言（`>` 开头）
- 正文所有散文段落、列表项文字、表格单元格文字
- H2/H3 标题 → 中文（"Learning Objectives" → "学习目标"；"Build It" → "构建它"；"Use It" → "使用它"；"Problem" → "问题"；"Concept" → "概念"；"Hints" → "提示"；"Summary" → "小结"；"Ship It" → "交付它"；"Verify It" → "验证它"；"Practice Lab" → "实践练习"；"Interactive Lab" → "交互实验"；"Capstone Connection" → "毕业项目关联"）
- 图片 alt 文字

## 术语表（必须遵守）

**保留英文不译**：token、embedding（首次出现可写「嵌入（embedding）」）、softmax、logit、dropout、batch、epoch、checkpoint、prompt、inference、pipeline、fine-tune（可写「微调（fine-tuning）」）、RAG、LoRA、KV cache、tool calling、function calling、MCP、A2A、RAG、CoT（思维链）、GPU、TPU、CUDA、API、SDK、CLI、YAML、JSON、WebGPU、Docker、Kubernetes、GitHub、Hugging Face、PyTorch、TensorFlow、JAX、numpy、torchvision、transformers、vLLM、SGLang、OpenTelemetry、LangChain、LlamaIndex、OpenAI、Anthropic、Claude、GPT、LLaMA、Mistral、DeepSeek、Qwen、whisper、ONNX、TensorRT、Triton、Inference（作推断讲时可译「推理」）。

**固定译法**：

| 英文 | 中文 |
|---|---|
| attention | 注意力 |
| transformer | Transformer（不译） |
| neural network | 神经网络 |
| convolution | 卷积 |
| pooling | 池化 |
| overfitting / underfitting | 过拟合 / 欠拟合 |
| gradient descent | 梯度下降 |
| backpropagation | 反向传播 |
| loss function | 损失函数 |
| learning rate | 学习率 |
| supervised / unsupervised | 监督 / 无监督 |
| reinforcement learning | 强化学习 |
| reward | 奖励 |
| agent | 智能体 |
| multi-agent | 多智能体 |
| tool use | 工具使用 |
| context window | 上下文窗口 |
| fine-tuning | 微调 |
| quantization | 量化 |
| distillation | 蒸馏 |
| retrieval-augmented | 检索增强 |
| vector database | 向量数据库 |
| chunk / chunking | 分块 |
| hallucination | 幻觉 |
| guardrail | 防护栏 |
| prompt injection | 提示注入 |
| jailbreak | 越狱 |
| red teaming | 红队测试 |
| feature | 特征 |
| label | 标签 |
| dataset | 数据集 |
| segmentation | 分割 |
| object detection | 目标检测 |
| bounding box | 边界框 |
| keypoint | 关键点 |
| frame / sampling | 帧 / 采样 |
| stream / streaming | 流 / 流式 |
| cache | 缓存 |
| throughput / latency | 吞吐量 / 延迟 |
| scale / scaling | 规模 / 扩展 |
| shard / sharding | 分片 |
| quota / rate limit | 配额 / 速率限制 |

**风格**：
- 简体中文，技术写作语气，不口语化也不生硬。
- 中英文之间加空格（如「使用 PyTorch 实现」），数字与中文之间加空格。
- 标点用全角（，。：；（）「」），代码/单位内保持半角。
- 长句拆短，一句只说一件事；专有名词首次出现给英文原文。
- 不确定的专业说法宁可保留英文，不要硬译。

## 流程

1. 读指定的 `docs/en.md`（路径为 `D:\project\ai-engineering-from-scratch\phases\<phase>\<lesson>\docs\en.md`）。
2. 整体翻译后，用 Write 工具写入 `D:\project\ai-engineer\content\zh\phases\<phase>\<lesson>\docs\zh.md`（UTF-8）。
3. 自检：zh.md 的 ``` 围栏数量与 en.md 完全一致；四行元数据原样在；H1 已翻译且无重复片段。
4. 一节课写完再写下一节。全部完成后报告：完成的课数 + 任何跳过原因。
