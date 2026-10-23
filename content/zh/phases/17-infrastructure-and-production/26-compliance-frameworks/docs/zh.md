# 合规 —— SOC 2、HIPAA、GDPR、PCI-DSS、欧盟 AI 法案、ISO 42001

> 多框架覆盖是 2026 年企业交易的基本门槛。**EU AI Act（欧盟 AI 法案）**：自 2024 年 8 月 1 日起生效。大部分高风险要求于 2026 年 8 月 2 日起执行。违反高风险系统义务的罚款最高 1500 万欧元或全球年营业额的 3%（第 99(4) 条）；禁止性 AI 实践的罚款最高 3500 万欧元或 7%（第 99(3) 条）。只要服务欧盟用户，就在全球范围内适用。**Colorado AI Act（科罗拉多 AI 法案）**：2026 年 6 月 30 日生效（经 SB25B-004 从 2026 年 2 月推迟）——要求对高风险系统做影响评估，并赋予对 AI 决策的上诉权。弗吉尼亚在信贷/就业/住房/教育领域有类似规定。**SOC 2 Type II**：事实上的 B2B AI 要求（金融科技要求 Type II，而非 Type I）。**GDPR**：有记录的最大 AI 专项罚款是对 Clearview AI 的 3050 万欧元（荷兰 DPA，2024 年 9 月）；意大利 Garante 于 2024 年 12 月对 OpenAI 开出 1500 万欧元罚单（后于 2026 年 3 月在上诉中被推翻）。推理时实时 PII 脱敏才是可辩护的标准；事后清理不够。**HIPAA**：医疗行业强制——没有 BAA 就不能把 PHI 发送给外部 AI 服务。**PCI-DSS**：覆盖 AI 交互层需要配置 + 合同协议，不会自动获得。**ISO 42001**：新兴的 AI 治理标准，与 ISO 27001 一起日益成为采购要求。参考画像：OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及针对 ChatGPT 支付组件的 PCI-DSS。跨框架映射能减少审计疲劳：访问控制同时映射到 ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举与 LLM 产品相关的 2026 年七大框架，并把每个框架对应到客户细分。
- 引用 EU AI Act 的执行时间线（2024 年 8 月生效；2026 年 8 月执行高风险要求）以及两级罚款上限（高风险义务为 1500 万欧元 / 3%，禁止性实践为 3500 万欧元 / 7%）。
- 解释为什么事后 PII 清理对 GDPR 不够，并指出实时推理层脱敏才是可辩护的标准。
- 描述跨框架控制映射（例如访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR 第 32 条 + HIPAA §164.312(a)）。

## 问题

某企业客户的采购部门要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 以及「EU AI Act 合规声明」。你的团队只有 SOC 2 Type I。距离 Type II 还有六个月，GDPR 第 30 条记录还没开始做。

多框架覆盖不是 LLM 的问题——它是企业 SaaS 的问题，只是叠加了 LLM 特有的部分。2026 年的采购团队想要一张矩阵：每行一个框架，每列一个控制项，而不是一份 PDF。

## 概念

### 七大框架

| 框架 | 范围 | LLM 特有要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基线 | 对流程控制进行 6-12 个月的审计 |
| HIPAA | 美国医疗 | 必须签 BAA；没有签署协议，PHI 不得离开你的基础设施 |
| GDPR | 欧盟用户 | 实时 PII 脱敏；数据主体权利；第 30 条记录 |
| PCI-DSS | 支付数据 | 涉及支付的 AI 需要配置 + 合同 |
| EU AI Act | 服务欧盟用户 | 风险等级分类；高风险系统：合格性评估、文档、日志 |
| Colorado AI Act | 服务科罗拉多州居民 | 影响评估；上诉权 |
| ISO 42001 | AI 治理 | 新兴；与 ISO 27001 配套 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止性 AI 实践开始执行。
- 2026 年 8 月 2 日：高风险系统开始执行（合格性评估、文档、日志）。
- 2027 年 8 月：受协调立法约束的产品中的高风险系统。

风险等级：不可接受（禁止）、高风险（合格性评估 + 日志）、有限风险（透明度）、最小风险（无约束）。大多数 B2B LLM SaaS 属于有限风险；高风险适用于就业、信贷、教育、执法、移民、基本服务。

罚款（第 99 条）：违反高风险系统义务，最高 1500 万欧元或全球年营业额的 3%（第 99(4) 条）；禁止性 AI 实践，最高 3500 万欧元或 7%（第 99(3) 条）；取较高者。

### GDPR —— 实时脱敏是标准

事后清理（LLM 看过数据之后再脱敏）不是可辩护的姿态——模型已经看到了数据。实时推理层脱敏是 2026 年的标准：

- 在 LLM 调用之前做实体识别。
- 一致性 token 化（Mesh 方法）保留语义。
- 只存储脱敏后的 prompt + 经用户同意的原始数据。

近期执法：对 Clearview AI 的 3050 万欧元罚款（荷兰 DPA，2024 年 9 月）是迄今有记录的最大 AI 专项 GDPR 罚款；对 OpenAI 的 1500 万欧元罚款（意大利 Garante，2024 年 12 月）是最大的 LLM 专项罚款，不过该裁决已于 2026 年 3 月在上诉中被推翻，且仍在进一步审理中。事后处理的声明在审计中已经站不住脚。

### HIPAA —— BAA 不是可选项

没有签署业务伙伴协议（Business Associate Agreement），就不能把 PHI 发送给外部 AI 服务。三大超大规模厂商的 LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI 直连 API 提供 BAA。Anthropic 直连 API 提供 BAA。发送 PHI 之前先确认。

### SOC 2 Type II

Type I：控制项已设计并有文档。
Type II：控制项在 6-12 个月内有效运行。

2026 年的 B2B 采购默认要求 Type II。Type I 只是入场券，Type II 才是门槛。

常见审计驱动项：访问日志（谁看过什么）、变更管理（如何部署）、风险评估（每季度）、事件响应（是否演练过？）。Phase 17 · 25 的审计日志可以直接复用。

### 跨框架映射

一份访问控制策略可以满足多个框架的控制项：

| 控制项 | 框架 |
|---------|-----------|
| 访问日志 | ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS 第 6 项要求、HIPAA 违规通知范围 |
| 传输加密 | ISO 27001 A.8.24、GDPR 第 32 条、HIPAA §164.312(e) |
| 密钥管理 | ISO 27001 A.8.19、PCI DSS 第 8 项要求、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）能自动化这种映射。规模上去以后值得花这笔钱。

### ISO 42001 —— 新兴标准

2023 年底发布。与 ISO 27001 一起日益成为采购要求。它是一套 AI 治理框架，涵盖风险管理、数据质量、透明度和人类监督。

### OpenAI 的参考画像

OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及针对 ChatGPT 支付组件的 PCI-DSS。这大致就是 2026 年的企业基本门槛。

### 应该记住的数字

- EU AI Act 罚款：最高 1500 万欧元 / 3%（高风险义务，第 99(4) 条）；最高 3500 万欧元 / 7%（禁止性实践，第 99(3) 条）。
- EU AI Act 高风险执行：2026 年 8 月 2 日。
- 有记录的最大 AI 专项 GDPR 罚款：3050 万欧元，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大的 LLM 专项 GDPR 罚款：1500 万欧元，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉被推翻）。
- SOC 2 Type II 窗口：6-12 个月的控制运行。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（经 SB25B-004 从 2026 年 2 月推迟）。

```figure
i4-control-matrix
```

## 使用它

`code/main.py` 是一个用 Python 写的合规映射表——给定一个控制项，列出它满足的框架。

## 交付它

本课产出 `outputs/skill-compliance-matrix.md`。给定客户细分与地理范围，明确所需框架和控制项。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。赢得这笔交易的最低可行合规姿态是什么？
2. 把三个假想的 LLM 产品按 EU AI Act 风险等级分类。进入高风险后有什么变化？
3. 你不小心把 PHI 发送给了没有 BAA 的提供商。走一遍事件响应流程。
4. 论证 ISO 42001 对一家中型 AI 厂商来说在 2026 年是否「必要」。
5. 把你的 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架控制项。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SOC 2 Type II | 「审计过的控制项」 | 控制项在 6-12 个月内运行，由独立方鉴证 |
| HIPAA BAA | 「医疗合同」 | 业务伙伴协议；处理 PHI 必须签署 |
| GDPR | 「欧盟隐私法」 | 实时 PII 脱敏是 2026 年可辩护的标准 |
| EU AI Act | 「欧盟 AI 规则」 | 高风险 2026 年 8 月执行；1500 万欧元 / 3%（高风险义务）——3500 万欧元 / 7%（禁止性实践） |
| Colorado AI Act | 「美国州级 AI 法」 | 2026 年 6 月 30 日生效（经 SB25B-004 推迟）；影响评估 |
| ISO 42001 | 「AI 治理」 | 面向 AI 风险 + 透明度的新兴框架 |
| ISO 27001 | 「安全 ISMS」 | 信息安全管理体系基线 |
| Conformity assessment | 「欧盟 AI 文档包」 | 高风险要求：文档、测试、日志 |
| Cross-framework mapping | 「一个控制项，多个框架」 | 单份策略满足多个框架的控制项 |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) —— 参考合规画像。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) —— 一手来源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) —— 一手来源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) —— AI 管理体系标准。
