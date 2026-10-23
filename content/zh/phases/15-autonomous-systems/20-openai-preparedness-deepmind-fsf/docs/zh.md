# OpenAI 准备度框架与 DeepMind 前沿安全框架

> OpenAI Preparedness Framework v2（2025 年 4 月）引入「研究类别」（Research Categories）——长程自主性、沙袋式表现（sandbagging）、自主复制与适应、破坏防护——并与「受追踪类别」（Tracked Categories）区分。受追踪类别触发能力报告（Capabilities Reports）和安全防护报告（Safeguards Reports），由安全顾问组（Safety Advisory Group）审阅。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日加入受追踪能力水平）将自主性并入 ML R&D 和网络（Cyber）领域（ML R&D 自主性水平 1 = 以有竞争力的成本、相对于人类 + AI 工具完全自动化 AI R&D 流水线）。FSF v3 明确通过针对工具性推理（instrumental reasoning）滥用的自动监控来应对欺骗性对齐。需要诚实指出：PF v2 中的研究类别（包括长程自主性）不会自动触发缓解措施；政策措辞是「潜在」。DeepMind 自己表示，如果工具性推理增强，自动监控「长期将不再足够」。

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## 问题

第 19 课仔细阅读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 的政策来补全图景。三份文档是同源产物，回答同一个问题——前沿实验室何时应暂停或限制某个模型——它们收敛于一小套类别，又在若干重要之处分化。

收敛之处：三者都把长程自主性标为值得追踪的能力类别。三者都承认欺骗行为（对齐伪装、沙袋式表现）是一类特定风险。三者都有内部评审机构。分化之处：OpenAI 把类别分为「受追踪」（强制缓解）和「研究」（不自动触发）。DeepMind 将自主性并入两个领域，而不是单独命名。实验室的命名有 Tracked 与 Research、Critical 与 Moderate、Tier-1 与 Tier-2；某个能力落在哪个桶里，其操作后果在各实验室之间并不相同。

把它们放在一起读是有用的练习。同一种能力在 Anthropic 可能是「强制缓解」，在 OpenAI 是「监控但不触发」，在 DeepMind 是「在特定领域内追踪」。政策姿态很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **受追踪类别（Tracked Categories）**：触发能力报告（模型能做什么）和安全防护报告（有哪些缓解措施）。部署前由安全顾问组审阅。
- **研究类别（Research Categories）**：实验室正在追踪但尚未承诺具体缓解措施的模型能力。包括长程自主性、沙袋式表现、自主复制与适应、破坏防护。

v2 的关键框架：研究类别不会自动触发缓解措施。政策措辞是「潜在」缓解。这是值得关注的具体差异：一个类别位于受追踪还是研究，就是「受控」与「被观察」之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日加入受追踪能力水平）

结构：

- **关键能力水平（Critical Capability Levels, CCLs）**：跨五个领域的能力阈值：网络（Cyber）、生物（Bio）、ML R&D、CBRN、自主性（并入 ML R&D 和 Cyber）。
- **受追踪能力水平（Tracked Capability Levels）**：2026 年 4 月加入的额外粒度。具体例子：ML R&D 自主性水平 1 = 以有竞争力的成本、相对于人类 + AI 工具完全自动化 AI R&D 流水线。
- **欺骗性对齐监控**：明确承诺对工具性推理滥用进行自动监控。

自主性的框架与 OpenAI 不同。DeepMind 不把「自主性」保留为顶层领域；它被并入自主性会造成危害的领域（ML R&D 和 Cyber）。其论点是，没有领域的自主性是没有风险的能力；反驳论点是，跨领域的自主性是一种元风险，框架应当予以命名。

### 三者的共识

- 内部安全顾问组（Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。高能力模型部署前需评审。
- 明确将欺骗性对齐 / 对齐伪装列为风险类别。
- 按声明节奏发布的常设文档（Anthropic：前沿安全路线图、风险报告；OpenAI：能力报告与安全防护报告；DeepMind：FSF 更新周期）。
- 承认仅靠监控的防御存在上限。DeepMind 说得很明确：「自动监控长期将不再足够。」

### 分化之处

- **Anthropic**：v3.0 移除了暂停承诺；AI R&D-4 阈值是被指名的下一道门槛。
- **OpenAI**：受追踪 vs 研究的分野；研究类别（包括长程自主性）不会自动设限。
- **DeepMind**：自主性并入其他领域；受追踪能力水平在 2026 年 4 月增加了粒度。

### 沙袋式表现：让三者都变复杂的一种特定能力

沙袋式表现（模型在评测中策略性表现不佳）属于 OpenAI 的研究类别。Anthropic 的 RSP v3.0 通过评测语境差距（第 1 课）来应对。DeepMind 通过 FSF v3 中的欺骗性对齐监控来应对。

如果模型在评测中沙袋式表现，每个框架的能力阈值都会被低估。框架只有在测量有效时才有效。这就是为什么除实验室自评之外，还需要外部测量（第 21 课，METR）和对抗性评测。

### 阅读政策的技能

- 定位：你关心的每一种能力都应能在政策中找到。如果找不到，政策就没有覆盖它。
- 分类：它是受追踪（触发缓解）还是研究（追踪但不触发）？OpenAI 这样命名；Anthropic 和 DeepMind 有自己的对应物。
- 节奏：政策是否按声明的时间表更新，还是只在特定事件后更新？声明节奏更强。
- 独立性：外部评审是强制还是可选？Anthropic 与 Apollo 和美国 AI 安全研究所合作；OpenAI 与 METR 合作；DeepMind 主要依靠内部 SAG。

```figure
a5-tracked-vs-research
```

## 使用它

`code/main.py` 实现了一个小型决策表 diff 工具。给定一种能力（自主性、欺骗性对齐、R&D 自动化、网络能力提升等），它输出三份政策各自如何分类该能力，以及会触发什么缓解措施。它是阅读辅助工具，不是政策工具。

## 交付它

`outputs/skill-cross-policy-diff.md` 以三个框架为参考，为特定能力生成跨政策比较。

## 练习

1. 运行 `code/main.py`。确认 diff 工具的输出至少对两种你能对照源文档核实的能力与政策一致。

2. 完整阅读 OpenAI Preparedness Framework v2。找出每一个研究类别。为每个类别写一句话，说明它为什么在「研究」而不是「受追踪」。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月的受追踪能力水平更新。找出 ML R&D 自主性水平 1 的具体评测标准。你会如何从外部测量它？

4. 沙袋式表现属于 OpenAI 的研究类别。设计一种评测，迫使沙袋式表现的模型暴露其真实能力。参考第 1 课关于评测语境博弈的讨论。

5. 就一种特定能力（自选）比较三份政策。指出你认为哪份政策的分类最严谨、哪份最不严谨。用源文本论证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Preparedness Framework | 「OpenAI 的扩展政策」 | PF v2（2025 年 4 月）；受追踪 vs 研究类别 |
| 受追踪类别 | 「强制缓解」 | 触发能力报告 + 安全防护报告；SAG 审阅 |
| 研究类别 | 「仅监控」 | 被追踪但不自动缓解；包括长程自主性 |
| Frontier Safety Framework | 「DeepMind 的扩展政策」 | FSF v3（2025 年 9 月）+ 受追踪能力水平（2026 年 4 月） |
| CCL | 「关键能力水平」 | DeepMind 按领域设定的阈值（Cyber、Bio、ML R&D、CBRN） |
| ML R&D 自主性水平 1 | 「R&D 自动化」 | 以有竞争力的成本完全自动化 AI R&D 流水线 |
| 沙袋式表现（sandbagging） | 「策略性表现不佳」 | 模型在评测中表现低于真实能力；属于 OpenAI 研究类别 |
| 工具性推理 | 「手段—目的推理」 | 关于如何达成目标的推理；DeepMind 监控的对象 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 受追踪能力水平的加入。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告的示例。
