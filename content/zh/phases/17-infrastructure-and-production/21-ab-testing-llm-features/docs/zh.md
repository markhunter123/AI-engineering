# LLM 功能的 A/B 测试 —— GrowthBook、Statsig 与「凭感觉」问题

> 传统 A/B 测试并不是为非确定性的 LLM 设计的。关键区别在于：eval 回答「模型能不能做好这件事？」A/B 测试回答「用户在不在乎？」两者都需要；靠感觉发布已经过时了。2026 年该测什么：prompt 工程（措辞）、模型选型（GPT-4 vs GPT-3.5 vs 开源模型；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：某聊天机器人的奖励模型变体带来 +70% 对话长度和 +30% 留存；Nextdoor 的 AI 主题行实验在改进奖励函数后带来 +1% CTR；Khan Academy 的 Khanmigo 在延迟与数学准确率的权衡轴上反复迭代。平台分野：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）——序列测试、CUPED、一体化。**GrowthBook**——开源、仓库原生（warehouse-native）、贝叶斯 + 频率学派 + 序列引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。如何选择取决于你对 warehouse SQL 的偏好，以及「被 OpenAI 收购」这件事对你的组织是否重要。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## 学习目标

- 区分 eval（「模型能不能做好」）与 A/B 测试（「用户在不在乎」）。
- 列举三个可测轴（prompt、模型、参数），并为每个轴选定指标。
- 解释 CUPED、序列测试与 Benjamini-Hochberg 多重比较校正。
- 依据 warehouse SQL 的站位与对企业收购的态度，选择 Statsig 或 GrowthBook。

## 问题

你手工调了一个 system prompt。感觉更好了。你发布了。转化率的变化只是噪声。你怪指标。或者你发布了一个新模型，转化率毫无变化——是模型退化了，还是改动太小检测不出来？你不知道，因为你没有做 A/B 就发布了。

eval 回答的是：模型能否在一个带标签的数据集上完成某项任务。它不回答用户是否更喜欢输出。只有受控的线上实验能回答后者，而且前提是实验有足够的功效、控制了非确定性、并做了多重比较校正。

## 概念

### eval 与 A/B 测试

**eval** —— 离线、带标签的数据集、评判者（rubric，或 LLM-as-judge，或人工）。回答：「在这个固定分布上，输出是否正确 / 有帮助 / 安全？」

**A/B 测试** —— 线上、真实用户、随机分组。回答：「新变体是否推动了真正重要的用户级指标？」

两者都需要。eval 在暴露前抓住回归；A/B 在之后确认产品影响。

### 该测什么

1. **prompt 工程** —— 措辞、system prompt 结构、示例。指标：任务成功率、用户留存、每请求成本。
2. **模型选型** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + P99 延迟。多目标。
3. **生成参数** —— temperature、top-p、max_tokens。指标：任务相关（输出多样性 vs 确定性）。

### CUPED —— 方差缩减

Controlled-experiments Using Pre-Experiment Data（利用实验前数据的受控实验）。在比较后期数据之前，先回归掉前期方差。典型方差缩减：30-70%。有效样本量白捡式提升。

实现：Statsig 和 GrowthBook 都实现了。

### 序列测试

经典 A/B 假设固定样本量。序列测试（「边看边决策」）在反复查看数据时控制假阳性率。始终有效的序列过程（mSPRT、Howard 的置信序列）让你在胜出明显时提前停止。

### 多重比较校正

以 95% 置信度跑 20 个 A/B 测试，凭运气就会产生一次假阳性。Bonferroni 校正收紧每个测试的 α；Benjamini-Hochberg 控制错误发现率（FDR）。GrowthBook 两者都实现了。

### SRM —— 样本比例失配

Assignment hash 把用户随机分配到各变体。如果 50/50 的拆分实际得到 47/53，说明某处坏了——SRM 检查会标记它。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管 SaaS。
- 序列测试、CUPED、留出人群（held-out populations）。
- 一体化：feature flag + 实验 + 可观测性。
- 最适合：想要打包产品、不在意 OpenAI 归属的团队。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多种引擎：贝叶斯、频率学派、序列。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最适合：warehouse SQL 型团队，数据团队掌控指标层，偏好开源。

### 非确定性让功效计算更复杂

同一个 prompt 会产生不同输出。传统功效计算假设观测独立同分布（IID）。存在 LLM 非确定性时，有效样本量低于名义值。把所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度、+30% 留存。
- Nextdoor 主题行：改进奖励函数后 +1% CTR。
- Khan Academy Khanmigo：在延迟与数学准确率之间反复权衡。

### 反模式：凭感觉发布

每个资深工程师都能说出一个因为「感觉更好」而没有做 A/B 就发布的功能。其中大多数在数月内拖累了产品指标，而团队毫无察觉。A/B 是强制性的约束机制。

### 应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：MIT 开源；贝叶斯 + 频率学派 + 序列。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → 样本量增加 30-50% 缓冲。

```figure
mx-sequential-test
```

## 使用它

`code/main.py` 模拟一次同时使用固定边界与序列边界的序列 A/B 测试。展示序列方法如何让你提前停止。

## 交付它

本节课产出 `outputs/skill-ab-plan.md`。给定功能改动、工作负载和基线，选择平台、闸门与样本量。

## 练习

1. 运行 `code/main.py`。在基线转化率 3%、预期提升 5% 的情况下，达到 80% 功效需要多大样本量？
2. 为一家受医疗监管、要求本地部署的客户，选择 Statsig 还是 GrowthBook。
3. 设计一个 A/B 测试，比较 GPT-4 与 GPT-3.5 在每解决工单成本上的表现。主指标、护栏指标、次要指标分别是什么？
4. 你的金丝雀通过了，但 A/B 显示转化率 -1.2%。要发布吗？写出升级判定标准。
5. 对一个前期方差为后期方差 60% 的场景应用 CUPED。计算有效样本量的提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Eval | 「离线测试」 | 在带标签数据集上评估模型能力 |
| A/B test | 「实验」 | 在用户上进行的线上随机对比 |
| CUPED | 「方差缩减」 | 用前期数据回归以降低方差 |
| Sequential test | 「可以偷看的测试」 | 允许提前停止的始终有效过程 |
| Multiple comparison | 「族错误率」 | 跑很多测试会抬高假阳性 |
| Bonferroni | 「收紧校正」 | 用 α 除以测试数量 |
| Benjamini-Hochberg | 「BH FDR」 | 控制错误发现率，较不保守 |
| SRM | 「分流失衡」 | 样本比例失配；分配逻辑有 bug |
| Statsig | 「OpenAI 家的」 | 商业一体化平台，2025 年被收购 |
| GrowthBook | 「开源那个」 | MIT、仓库原生平台 |
| mSPRT | 「序列概率比检验」 | 经典序列过程 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
