# LLM 评估——RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 捕捉不到语义等价。人工评审又无法规模化。LLM-as-judge 是生产环境的答案——前提是校准足够，让这个分数值得信任。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题

你的 RAG 系统回答：「June 29th, 2007。」
标准答案（gold reference）是：「June 29, 2007。」
精确匹配（Exact Match）得 0 分，F1 约 75%，而人工评分会给 100%。

现在把这个场景乘以 10,000 个测试用例，再乘以检索器、分块、prompt 或模型的每一次变更。你需要一个评估器：理解语义、能低成本大规模运行、不会在回归问题上撒谎，并且能暴露正确的失败模式。

2026 年有三个框架专门解决这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。提供四个 RAG 指标（faithfulness、answer-relevance、context-precision、context-recall），后端是 NLI + LLM 评判。有研究支撑，轻量。
- **DeepEval。** 面向 LLM 的 Pytest。提供 G-Eval、任务完成度、幻觉、偏差等指标，原生适配 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：带思维链的 LLM-as-judge，自定义评判标准，输出 0-1 分。

三者都依赖 LLM-as-judge。本课帮你建立对该方法的直觉，以及围绕它的信任层。

## 概念

![四个评估维度，LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge（LLM 评判）。** 用一个 LLM 按评分标准（rubric）给输出打分，替代静态指标。给定 `(query, context, answer)`，提示评判 LLM：「在 faithfulness 上打 0-1 分。」然后取回分数。

为什么有效：LLM 能以极低的成本近似人类判断。GPT-4o-mini 每个评分样本约 $0.003，1000 个样本的回归评估不到 $5。

为什么它会悄悄失效：

1. **评判偏差。** 评判模型偏好更长的答案、自己模型家族的答案，以及与 prompt 风格一致的答案。
2. **JSON 解析失败。** 坏 JSON → NaN 分数 → 被静默排除在聚合之外。用过 RAGAS 的人都懂这种痛。用 try/except 加显式失败模式来设门槛。
3. **模型版本漂移。** 升级评判模型就会改变每一项指标。冻结评判模型及其版本。

**RAG 四指标。**

| 指标 | 回答的问题 | 后端 |
|--------|----------|---------|
| Faithfulness | 答案中的每个论断都来自检索到的上下文吗？ | 基于 NLI 的蕴含判断 |
| Answer relevance | 答案回应了问题吗？ | 从答案生成假设问题，与真实问题比较 |
| Context precision | 检索到的分块中有多大比例是相关的？ | LLM 评判 |
| Context recall | 检索是否返回了所有需要的内容？ | 对照标准答案做 LLM 评判 |

**G-Eval。** 定义自定义标准：「答案是否引用了正确的来源？」框架会自动把它展开成思维链式的评估步骤，然后打出 0-1 分。适合 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在算出与人工标注的相关性之前，永远不要相信评判的原始分数。跑 100 个手工标注的样本，画出评判分数 vs 人工分数，计算 Spearman rho。如果 rho < 0.7，你的评判标准需要改进。

```figure
n5-judge-gauge
```

## 构建它

### 步骤 1：用 NLI 计算忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把答案分解成原子论断，用 NLI 逐条对照检索到的上下文检查。忠实度 = 被支持的论断占比。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案隐含的问题与用户所问不同，相关性就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式「打 0-1 分」的 prompt 更稳定。

### 步骤 4：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

以 pytest 文件形式交付，每个 PR 都运行，出现回归就阻止合并。

### 步骤 5：从零实现的玩具评估

见 `code/main.py`。仅用标准库近似实现忠实度（答案论断与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）。不能用于生产，只用来展示形态。

## 常见陷阱

- **没有校准。** 与人工标注相关性只有 0.3 的评判就是噪声。上线前必须做一轮校准。
- **自评。** 用同一个 LLM 既生成又评判，会把分数抬高 10-20%。评判要换一个模型家族。
- **成对评判中的位置偏差。** 评判模型偏好先出现的选项。务必随机化顺序，并交换双方各跑一次。
- **原始聚合隐藏失败。** 平均分 0.85 常常掩盖 5% 的灾难性失败。一定要检查最低分位。
- **黄金数据集腐化。** 没有版本、随时间漂移的评估集会破坏纵向对比。每次变更都给数据集打标签。
- **LLM 成本。** 规模上去后，评判调用主导成本。用满足校准阈值的最便宜模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它

2026 年的技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | RAGAS 的无参考答案模式 |
| 人工在环抽查 | LangSmith 或 Phoenix 加标注界面 |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 做监控，DeepEval 做 CI，G-Eval 做新维度。三者都跑；它们的分歧很有价值。

## 交付它

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 例子上使用 RAGAS，验证 faithfulness 指标能抓到每一个。
2. **中等。** 人工为 50 个 QA 答案的正确性打 0-1 分，用 G-Eval 评分，计算评判与人工之间的 Spearman rho。
3. **困难。** 用 DeepEval 搭建 pytest CI 门禁。故意让检索器退化，验证门禁失败。再加上对最低 10% 分位的阈值检查告警。

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | 提示评判模型按评分标准给输出打 0-1 分。 |
| RAGAS | RAG 指标库 | 开源评估框架，提供 4 个无参考 RAG 指标。 |
| Faithfulness | 答案有依据吗？ | 答案论断中被检索上下文蕴含的比例。 |
| Context precision | 检索到的分块相关吗？ | top-K 分块中真正有用的比例。 |
| Context recall | 检索找全了吗？ | 标准答案论断中被检索分块支持的比例。 |
| G-Eval | 自定义 LLM 评判 | 评分标准 + 思维链评估步骤 + 0-1 分。 |
| 校准 | 信任但验证 | 评判分数与人工分数的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) —— RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) —— G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) —— 开源生产级技术栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) —— 偏差、校准与局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) —— 集成 RAGAS、DeepEval、Phoenix 的统一框架。
