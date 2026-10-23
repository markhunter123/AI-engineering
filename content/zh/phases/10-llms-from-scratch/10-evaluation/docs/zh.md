# 评估：基准测试、评测套件与 LM Harness

> 古德哈特定律：当一个度量变成目标，它就不再是一个好的度量。每个前沿实验室都在刷榜。MMLU 分数一路上涨，而模型仍然数不清「strawberry」里有几个 R。唯一真正重要的评测，是你自己的评测——在你的任务上，用你的数据。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes

## 学习目标

- 构建一个自定义评测 harness，针对语言模型运行多项选择和开放式基准
- 解释为什么标准基准（MMLU、HumanEval）会饱和，并失去区分前沿模型的能力
- 用恰当的指标实现任务特定的评测：精确匹配、F1、BLEU 和 LLM-as-judge 打分
- 设计一套针对你具体用例的自定义评测套件，而不是只依赖公开排行榜

## 问题

MMLU 于 2020 年发布，包含 57 个学科的 15,908 道题。三年之内，前沿模型就把它刷饱和了。GPT-4 得分 86.4%。Claude 3 Opus 得分 86.8%。Llama 3 405B 得分 88.6%。排行榜被压缩到 3 分的区间里，其中的差异只是统计噪声，而不是真实的能力差距。

与此同时，这些模型却会在 10 岁小孩都能不假思索完成的任务上失败。Claude 3.5 Sonnet 在 MMLU 上得分 88.7%，最初却数不清「strawberry」里的字母——这个任务不需要任何世界知识，也不需要任何推理，只需要逐字符迭代。HumanEval 用 164 道题测试代码生成。模型能拿到 90% 以上的分数，却仍会写出在边界情况下崩溃的代码，而这些情况任何初级开发者都能发现。

基准表现与真实世界可靠性之间的鸿沟，是 LLM 评估的核心问题。基准只能告诉你模型在这个基准上表现如何，几乎不能告诉你该模型在你的具体任务、你的具体数据、你的具体失效模式下会表现如何。如果你在做客服机器人，MMLU 毫无意义。如果你在做代码助手，HumanEval 只覆盖函数级生成——它对调试、重构或跨文件解释代码一无所知。

你需要自定义评测。不是因为基准没用——它们在粗略选型时很有用——而是因为最终评估必须与你的部署条件精确匹配。

## 概念

### 评测版图

评估分为三类，成本和信号质量各不相同。

**基准（Benchmarks）**是标准化的测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你让模型跑基准，得到一个分数。优点是：所有人都用同一套测试，因此可以比较模型。缺点是：模型和训练数据越来越多地污染这些基准。实验室的训练数据包含基准题目。分数上去了，能力未必。

**自定义评测（Custom evals）**是你为具体用例构建的测试套件。你定义输入、期望输出和评分函数。法律文档摘要器就在法律文档上评测。SQL 生成器就在你的数据库 schema 上评测。这些评测构建成本高，但它们是唯一能预测生产表现的评估。

**人类评估（Human evals）**用付费标注员按有用性、正确性、流畅性和安全性等标准判断模型输出。这是开放式任务的黄金标准，因为自动评分在这些任务上会失效。Chatbot Arena 已在 100 多个模型上收集了超过 200 万张人类偏好投票。缺点是：成本（每次判断 0.10-2.00 美元）和速度（数小时到数天）。

```mermaid
graph TD
    subgraph Eval["Evaluation Landscape"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nCheap, standardized\nGameable, stale"]
        C["Custom Evals\nYour task, your data\nHighest signal\nExpensive to build"]
        H["Human Evals\n(Chatbot Arena)\nGold standard\nSlow, costly"]
    end

    B -->|"rough model selection"| C
    C -->|"ambiguous cases"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 基准为什么会失效

有三种机制会让基准分数不再反映真实能力。

**数据污染。** 训练语料抓取互联网。基准题目就存在于互联网上。模型在训练时看到了答案。这不算传统意义上的作弊——实验室不会故意加入基准数据。但网络规模的抓取让排除几乎不可能。

**应试训练。** 实验室会针对基准表现优化训练数据配比。如果训练混合数据中有 5% 是 MMLU 风格的多项选择，模型就学会了这种格式和答案分布。MMLU 是四选一。模型会学到答案在 A/B/C/D 上近似均匀分布——即使它不知道答案，这也能帮上忙。

**饱和。** 当每个前沿模型在某个基准上都拿到 85-90% 时，这个基准就失去了区分度。剩下的 10-15% 题目可能是歧义的、标注错误的，或者需要冷门领域知识。MMLU 从 87% 提升到 89%，可能只意味着模型多背下了两道冷门题，而不是变聪明了。

### 困惑度：快速健康检查

困惑度（perplexity）衡量模型对一段 token 序列有多惊讶。形式上，它是平均负对数似然的指数：

```math
\text{PPL} = \exp\left(-\frac{1}{N} \sum_i \log P(\text{token}_i \mid \text{context})\right)
```

困惑度为 10 意味着，模型在每个 token 位置上的平均不确定程度，相当于在 10 个选项中均匀猜测。越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30。GPT-3 约为 20。Llama 3 8B 约为 7。

困惑度适合在同一测试集上比较模型，但它有盲点。模型可以通过擅长预测常见模式来获得低困惑度，同时却对罕见但重要的模式表现糟糕。它也完全不能说明指令遵循、推理或事实准确性。把它当作健康检查，而不是最终结论。

### LLM-as-Judge

用一个强模型来评估弱模型的输出。思路很简单：让 GPT-4o 或 Claude Sonnet 按 1-5 分给回复的正确性、有用性和安全性打分。用 GPT-4o-mini 时每次判断约 0.01 美元，而且与人类判断的相关性出奇地好——多数任务上约有 80% 的一致率。

评分 prompt 比模型本身更重要。含糊的 prompt（「给这个回复打分」）会产生噪声很大的分数。带评分细则的结构化 prompt（「如果答案事实正确并引用了来源，给 5 分；正确但无来源给 4 分；部分正确给 3 分……」）会产生一致、可复现的分数。

失效模式：裁判模型会出现位置偏差（在成对比较中偏好第一个回复）、冗长偏差（偏好更长的回复）和自我偏好（GPT-4 给 GPT-4 输出的评分高于同等质量的 Claude 输出）。缓解办法：随机化顺序、按长度归一化、使用与被评估模型不同的裁判。

### 基于成对比较的 ELO 评分

这是 Chatbot Arena 的做法。把不同模型对同一 prompt 的两个回复展示出来，由人类（或 LLM 裁判）选出更好的那个。从成千上万次这样的比较中，为每个模型计算 ELO 评分——也就是国际象棋使用的同一套系统。

ELO 的优点：相对排名比绝对打分更可靠，能优雅地处理平局，而且比逐个独立打分用更少的比较就能收敛。截至 2026 年初，Chatbot Arena 的排名显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在榜首的差距在 20 ELO 分以内。

```mermaid
graph LR
    subgraph ELO["ELO Rating Pipeline"]
        direction TB
        P["Prompt"] --> MA["Model A Output"]
        P --> MB["Model B Output"]
        MA --> J["Judge\n(Human or LLM)"]
        MB --> J
        J --> W["A Wins / B Wins / Tie"]
        W --> E["ELO Update\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 评测框架

**lm-evaluation-harness**（EleutherAI）：标准的开源评测框架。支持 200 多个基准。用一条命令就能让任意 Hugging Face 模型跑 MMLU、HellaSwag、ARC 等。Open LLM Leaderboard 就使用它。

**RAGAS**：专门面向 RAG pipeline 的评测框架。它衡量忠实性（答案是否与检索到的上下文一致？）、相关性（检索到的上下文是否与问题相关？）和答案正确性。

**promptfoo**：面向 prompt 工程的配置驱动评测。在 YAML 中定义测试用例，对多个模型运行，得到通过/失败报告。适合 prompt 回归测试——确保 prompt 改动不会破坏已有的测试用例。

### 构建自定义评测

这是生产环境唯一重要的评测。流程如下：

1. **定义任务。** 模型到底应该做什么？要精确。「回答问题」太模糊。「给定一封客户投诉邮件，抽取产品名、问题类别和情绪」才是可评测的任务。

2. **创建测试用例。** 原型评测至少 50 个，生产环境 200 个以上。每个测试用例是一个（input, expected_output）对。要包含边界情况：空输入、对抗性输入、歧义输入、其他语言的输入。

3. **定义评分。** 结构化输出用精确匹配。文本相似度用 BLEU/ROUGE。开放式质量用 LLM-as-judge。抽取任务用 F1。可以按权重组合多个指标。

4. **自动化。** 每次评测用一条命令运行。没有手工步骤。以支持随时间比较的格式存储结果。

5. **长期跟踪。** 孤立的评测分数没有意义。你需要趋势线。上次 prompt 改动后分数提高了吗？切换模型后回退了吗？把评测与 prompt 一起做版本管理。

| 评测类型 | 每次判断成本 | 与人类的一致率 | 最适合 |
|-----------|------------------|----------------------|----------|
| 精确匹配 | 约 $0 | 100%（适用时） | 结构化输出、分类 |
| BLEU/ROUGE | 约 $0 | 约 60% | 翻译、摘要 |
| LLM-as-judge | 约 $0.01 | 约 80% | 开放式生成 |
| 人类评估 | $0.10-$2.00 | 不适用（它就是真值） | 歧义、高风险任务 |

```figure
perplexity-loss
```

## 构建它

### 第 1 步：最小评测框架

定义核心抽象。一个评测用例包含输入、期望输出和可选的元数据字典。评分器接收预测和参考，返回 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 第 2 步：评分函数

构建精确匹配、token F1 和一个模拟的 LLM-as-judge 评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 第 3 步：ELO 评分系统

实现带 ELO 更新的成对比较。这正是 Chatbot Arena 用来给模型排名的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 第 4 步：困惑度计算

用 token 概率计算困惑度。实际中你会从模型的 logit 得到这些概率。这里用概率分布来模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 第 5 步：汇总结果

计算一次评测运行的汇总统计：均值、中位数、阈值通过率，以及各指标的分项数据。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### 第 6 步：运行完整 Pipeline

把所有部件串起来。定义一个任务，创建测试用例，模拟两个模型，运行评测，从成对比较计算 ELO，打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

「好」模型给出精确答案。「坏」模型给出冗长的改写。精确匹配会严厉惩罚冗长模型。token F1 和 LLM-as-judge 则更宽容。这说明了为什么指标选择很重要：同一个模型，评分方式不同，看起来可能极好也可能极差。

### 第 7 步：ELO 锦标赛

在多个轮次中对模型进行成对比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### 第 8 步：困惑度比较

比较不同质量水平「模型」的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## 使用它

### lm-evaluation-harness（EleutherAI）

对任意模型运行基准的标准工具。

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

面向 prompt 工程的配置驱动评测。在 YAML 中定义测试，对多个提供商运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### 用 RAGAS 做 RAG 评估

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 衡量的是通用评测遗漏的东西：模型的答案是否有检索上下文的依据，而不只是答案在抽象意义上是否「正确」。

## 交付它

本课产出 `outputs/prompt-eval-designer.md`——一个可复用的 prompt，能为任意任务设计自定义评测套件。给它一段任务描述，它会生成测试用例、评分函数和通过/失败阈值建议。

它还产出 `outputs/skill-llm-evaluation.md`——一个决策框架，帮助你根据任务类型、预算和延迟要求选择正确的评测策略。

## 练习

1. 增加一个「一致性」评分器：把同一输入跑 5 次，测量输出一致的频率。确定性输入上答案不一致，说明 prompt 脆弱或温度设置过高。

2. 扩展 ELO 跟踪器以支持多个裁判函数（精确匹配、F1、LLM-as-judge）并加权。比较在精确匹配权重高和 F1 权重高两种情况下排行榜如何变化。

3. 为一个具体任务构建评测套件：把邮件分为 5 个类别。创建 100 个多样化测试用例，包括边界情况（可能属于多个类别的邮件、空邮件、其他语言的邮件）。测量不同「模型」（规则式、关键词匹配、模拟 LLM）的表现。

4. 实现污染检测：给定一组评测题和一份训练语料，检查有多大比例的评测题（或其近似改写）出现在训练数据中。研究者正是这样审计基准有效性的。

5. 构建一个「模型 diff」工具。给定两个模型版本的评测结果，高亮哪些具体测试用例变好了、哪些回退了、哪些保持不变。这是评测版的代码 diff——理解一次改动是帮忙还是添乱的关键。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------------|----------------------|
| MMLU | 「那个基准」 | Massive Multitask Language Understanding——57 个学科的 15,908 道多项选择题，到 2025 年已被刷到 88% 以上并饱和 |
| HumanEval | 「代码评测」 | OpenAI 的 164 道 Python 函数补全题，只测试孤立的函数生成 |
| SWE-bench | 「真实编码评测」 | 来自 12 个 Python 仓库的 2,294 个 GitHub issue，衡量包含测试生成在内的端到端缺陷修复 |
| 困惑度（perplexity） | 「模型有多困惑」 | $\exp(-\mathop{\mathrm{avg}}(\log P(\text{token}_i \text{ given context})))$——越低表示模型给实际 token 的概率越高 |
| ELO 评分 | 「模型的国际象棋排名」 | 根据成对胜负记录计算的相对技能评分，Chatbot Arena 用它给 100 多个模型排名 |
| LLM-as-judge | 「用 AI 给 AI 打分」 | 强模型依据评分细则给弱模型的输出打分，与人类裁判约有 80% 一致率，每次判断约 0.01 美元 |
| 数据污染（data contamination） | 「模型见过测试题」 | 训练数据包含基准题目，抬高了分数却没有提升真实能力 |
| 评测套件（eval suite） | 「一堆测试」 | 一组带版本管理的（input, expected_output, scorer）三元组，用来衡量某项具体能力 |
| 通过率（pass rate） | 「做对了百分之多少」 | 分数超过阈值的评测用例比例——比平均分更可操作，因为它衡量的是可靠性 |
| Chatbot Arena | 「模型排名网站」 | LMSYS 平台，拥有 200 万以上人类偏好投票，通过 ELO 评分产出最受信任的 LLM 排行榜 |

## 延伸阅读

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)——MMLU 论文，尽管已经饱和，仍是被引用最多的 LLM 基准
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374)——OpenAI 的 HumanEval 论文，奠定了代码生成评测方法论
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)——系统分析用 LLM 评估 LLM，包括位置偏差和冗长偏差的发现
- [Arena（原 LMSYS Chatbot Arena）](https://arena.ai/leaderboard)——拥有 200 万以上投票的众包模型比较平台，最受信任的真实世界 LLM 排名
