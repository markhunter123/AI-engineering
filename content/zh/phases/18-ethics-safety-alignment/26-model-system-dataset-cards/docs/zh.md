# 模型卡、系统卡与数据集卡

> 三种文档格式支撑着 AI 透明度的结构。Model Cards（Mitchell 等，2019）——模型的营养成分标签：训练数据、定量的分组分析、伦理考量、注意事项；但 Hugging Face 上只有 0.3% 的模型卡记录了伦理考量（Oreamuno 等，2023）。Datasheets for Datasets（Gebru 等，2018，CACM）——动机、组成、采集过程、标注、分发、维护；类比电子元器件的 datasheet。Data Cards（Pushkarna 等，Google 2022）——模块化分层细节（telescopic、periscopic、microscopic），作为面向不同读者的边界对象。2024–2025 年的新进展：用 LLM 自动生成（CardGen，Liu 等，2024）；模型卡的详细程度与 Hugging Face 下载量最高提升 29% 相关（Liang 等，2024）；可验证的声明（Laminator，Duddu 等，2024）；新增碳/水等可持续性报告（Jouneaux 等，2025 年 7 月）；面向欧盟/ISO 监管的卡片正在出现。System Cards（Sidhpurwala 2024；Meta 的系统级透明度；「Blueprints of Trust」arXiv:2509.20394）——端到端的 AI 系统文档，覆盖安全能力、提示注入防护、数据外泄检测、与人类价值观的对齐。

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## 学习目标

- 描述 Mitchell 等 2019 提出的原始模型卡，以及 Gebru 等 2018 提出的 datasheet。
- 描述 Data Cards 的 telescopic/periscopic/microscopic 分层。
- 描述系统卡（System Cards）及其端到端覆盖范围。
- 说明 2024–2025 年的三项进展（自动生成、可验证声明、可持续性报告）。

## 问题

监管框架（第 24 课）和实验室安全政策（第 18 课）都要求提供文档。文档格式从面向模型（模型卡）演进到面向数据集（datasheet），再到面向系统（系统卡）。每种格式针对不同范围的透明度。2024–2025 年的自动化与可验证声明工作，则针对长期存在的采用率问题。

## 概念

### 模型卡（Mitchell 等，2019）

章节包括：
- 模型详情。
- 预期用途。
- 因素（评估时相关的群体或环境因素）。
- 指标。
- 评估数据。
- 训练数据。
- 定量分析（按因素分组）。
- 伦理考量。
- 注意事项与建议。

采用问题：Oreamuno 等 2023 年对 Hugging Face 模型卡的审计发现，只有 0.3% 记录了伦理考量。

### Datasheets for Datasets（Gebru 等，2018）

类比电子元器件的 datasheet。章节：
- 动机（为什么创建这个数据集）。
- 组成（里面有什么）。
- 采集过程（如何组装）。
- 标注（如适用）。
- 用途（预期、禁止、风险）。
- 分发。
- 维护。

2021 年发表于 CACM。datasheet 是上游文档；模型卡的准确性取决于 datasheet 是否准确。

### Data Cards（Pushkarna 等，Google 2022）

模块化分层细节。三个缩放层级：
- **Telescopic（望远镜）。** 面向非专家的高层摘要。
- **Periscopic（潜望镜）。** 面向 ML 从业者的中层概览。
- **Microscopic（显微镜）。** 面向审计者的特征级细节文档。

边界对象（boundary object）这个提法：不同的读者从同一份文档中提取不同的信息。

### 系统卡

范围：端到端 AI 系统，包括模型 + 安全栈 + 部署环境。章节通常包括：
- 安全能力。
- 提示注入防护。
- 数据外泄检测。
- 与所声明的人类价值观的对齐。
- 事件响应。

Sidhpurwala 2024 以及 Meta 的系统级透明度工作。「Blueprints of Trust」（arXiv:2509.20394）把系统卡形式化为模型卡在部署层的补充。

### 2024–2025 年的进展

- **CardGen（Liu 等，2024）。** 用 LLM 自动生成模型卡；在 Mitchell 2019 的标准化字段上，其客观性报告高于许多人工撰写的卡片。
- **下载相关性（Liang 等，2024）。** 详细的模型卡与 Hugging Face 上下载率最高提升 29% 相关——采用压力如今由市场驱动，而不只是合规驱动。
- **Laminator（Duddu 等，2024）。** 通过硬件 TEE / 密码学签名实现可验证声明——模型卡可以携带「声明的证明」，而不只是声明本身。
- **可持续性（Jouneaux 等，2025 年 7 月）。** 增加碳、水与计算能耗足迹；ISO 标准正在形成。
- **监管卡片。** 欧盟 AI 法（第 24 课）GPAI 实践准则的透明度章节要求把模型卡作为合规产物。

### 本课在 Phase 18 中的位置

第 24–25 课是监管层与 CVE 层。第 26 课是文档层。第 27 课是训练数据治理，也就是 datasheet 的上游。第 28 课是产出卡片中所引用评估的研究生态。

```figure
an-card-scopes
```

## 使用它

`code/main.py` 为一个玩具部署生成最小的模型卡、datasheet 和系统卡。每份都遵循标准章节结构。你可以检查格式并比较三者的范围。

## 交付它

本课产出 `outputs/skill-card-audit.md`。给定模型卡、datasheet 或系统卡，它审计章节覆盖度、数值分组分析，以及是否带有可验证声明。

## 练习

1. 运行 `code/main.py`。检查生成的卡片。找出内容薄弱的章节（只有占位符），并指出哪些证据可以加强它们。

2. 为模型卡增加跨两个人口群体的定量分组分析（第 20 课）。

3. 阅读 Oreamuno 等 2023 关于 0.3% 采用率的研究。对模型卡规范提出一项结构性修改，以提高伦理考量的采用率。

4. Laminator（Duddu 等，2024）使用 TEE 实现可验证声明。设计一个模型卡字段，携带某项评估结果的密码学证明，并描述验证者的角色。

5. 为你过去的一个项目或一个假想的部署写一份系统卡（是系统卡，不是模型卡）。指出对第三方审计者最有价值的章节。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Model Card | 「Mitchell 那张卡」 | Mitchell 等 2019 提出的 ML 模型标准文档 |
| Datasheet | 「Gebru 的 datasheet」 | Gebru 等 2018 提出的数据集标准文档 |
| Data Card | 「Pushkarna 的卡」 | Google 2022 的模块化分层数据文档 |
| System Card | 「部署卡」 | 包含安全栈的端到端 AI 系统文档 |
| Boundary object | 「一个文档，多种读者」 | Data Cards 的提法：同一文档服务于不同受众 |
| Verifiable attestation | 「Laminator 证明」 | 附加在文档声明上的密码学或 TEE 证明 |
| Sustainability field | 「碳 / 水足迹」 | 2025 年新增的环境核算字段 |

## 延伸阅读

- [Mitchell 等 — Model Cards for Model Reporting（arXiv:1810.03993，FAT* 2019）](https://arxiv.org/abs/1810.03993) — 经典的模型卡
- [Gebru 等 — Datasheets for Datasets（CACM 2021，arXiv:1803.09010）](https://arxiv.org/abs/1803.09010) — datasheet 论文
- [Pushkarna 等 — Data Cards（Google 2022）](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala 等 — Blueprints of Trust（arXiv:2509.20394）](https://arxiv.org/abs/2509.20394) — 系统卡的形式化
