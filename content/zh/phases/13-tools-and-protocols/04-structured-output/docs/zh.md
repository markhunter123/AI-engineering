# 结构化输出——JSON Schema、Pydantic、Zod 与约束解码

> 「好好请求模型返回 JSON」即使在前沿模型上也有 5% 到 15% 的失败率。结构化输出用约束解码填上这个缺口：模型被字面意义上地禁止发出违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 以及 Zod 的 `.parse`，是同一理念的五种表层形式。本课构建 schema 校验器和严格模式契约，学员将在所有生产抽取管道中使用它们。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 为抽取目标编写 JSON Schema 2020-12，使用正确的约束（enum、min/max、required、pattern）。
- 解释为什么严格模式与约束解码提供的保证不同于「生成后再校验」。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 交付一条带类型化修复和类型化拒绝处理的抽取管道。

## 问题

一个读取采购订单邮件的智能体需要把自由文本变成 `{customer, line_items, total_usd}`。有三种做法。

**做法一：提示模型输出 JSON。** 「用 JSON 回复，字段为 customer、line_items、total_usd。」在前沿模型上有 85% 到 95% 的成功率。它以六种方式失败：缺花括号、尾逗号、类型错误、幻觉字段、在 token 上限处被截断、泄漏散文（如 “Here is your JSON:”）。

**做法二：生成后校验。** 自由生成、解析、按 schema 校验、失败重试。可靠但昂贵——每次重试都要付费，截断类 bug 每次出现都要多花一轮。

**做法三：约束解码。** 供应商在解码时强制 schema。无效 token 被从采样分布中屏蔽。输出保证可解析、保证通过校验。失败坍缩为一种模式：拒绝（模型判定输入不符合 schema）。

2026 年每家前沿供应商都提供做法三的某种形式。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，模型拒绝时响应中出现 `refusal`。
- **Anthropic。** 对 `tool_use` 输入强制执行 schema；不存在 `stop_reason: "refusal"` 这种东西，但没有工具调用的 `end_turn` 就是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 产出类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，按 Zod schema 校验供应商输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同的主线：只声明一次 schema，端到端强制执行。

## 概念

### JSON Schema 2020-12——通用语言

每个供应商都接受 JSON Schema 2020-12。最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加三项要求：每个属性都必须列在 `required` 中、所有位置都要 `additionalProperties: false`、不允许未解析的 `$ref`。如果违反这些要求，API 会在请求时返回 400。

### Pydantic：Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形状的模型生成 JSON Schema。Pydantic AI 在此基础上封装，因此你写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

然后智能体框架在边界处把 schema 翻译成 OpenAI 严格模式、Anthropic 的 `input_schema` 或 Gemini 的 `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。校验错误抛出带类型化错误路径的 `ValidationError`。

### Zod：TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，它会转换成 API 的 JSON Schema 载荷。

### 拒绝

严格模式无法强迫模型回答。如果输入无法符合 schema（「这封邮件是一首诗，不是发票」），模型会发出包含原因的 `refusal` 字段。你的代码必须把它当作一等结果处理，而不是失败。拒绝也可以作为安全信号：要求模型从受保护内容的邮件中提取信用卡号时，模型会返回附带安全原因的拒绝。

### 开源实现中的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；每一步都屏蔽会违反 FSM 的 token 的 logit。
2. **配合 JSON 解析器的 logit 屏蔽**：让流式 JSON 解析器与模型同步运行；每一步计算合法的下一 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提议 token，验证器强制 schema。

商业供应商在幕后选用其中一种。2026 年的技术水平是：对短结构化输出比普通生成更快，对长输出速度大致相当。

### 三种失败模式

1. **解析错误。** 输出不是合法 JSON。严格模式下不可能发生。非严格供应商上仍可能发生。
2. **schema 违规。** 输出能解析但违反 schema。严格模式下不可能发生。严格模式之外很常见。
3. **拒绝。** 模型拒绝回答。必须作为类型化结果处理。

### 重试策略

当你处于严格模式之外（Anthropic 工具使用、非严格 OpenAI、较旧的 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常重试一次就够。三次重试可以兜住弱模型的偶发抽风。超过三次说明 schema 有问题：模型对某些输入无法满足它，需要修改 prompt 或 schema。

### 小模型支持

约束解码在小模型上同样有效。一个带语法强制的 3B 参数开放模型，在结构化任务上胜过一个靠原始提示的 70B 参数模型。这正是结构化输出对生产重要的主要原因：它把可靠性从模型规模中解耦出来。

```figure
constrained-decoding
```

## 使用它

`code/main.py` 用标准库交付一个最小的 JSON Schema 2020-12 校验器（types、required、enum、min/max、pattern、items、additionalProperties）。它封装一个 `Invoice` schema，并把假的 LLM 输出跑过校验器，演示解析错误、schema 违规和拒绝三条路径。生产中可以把这个假输出换成任意供应商的真实响应。

值得关注的几点：

- 校验器返回带路径和消息的类型化 `[ValidationError]` 列表。这正是你希望暴露给重试 prompt 的形状。
- 拒绝分支不会重试。它记录日志并返回类型化拒绝。第 14 阶段 · 09 把拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，说明严格模式为何能关上幻觉字段的大门。

## 交付它

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、客服工单、简历等），该 skill 会产出与严格模式兼容的 JSON Schema 2020-12，以及与之对应的 Pydantic 模型，并预置类型化拒绝与重试处理的桩代码。

## 练习

1. 运行 `code/main.py`。增加第四个测试用例，其 `total_usd` 为负数。确认校验器通过 `minimum` 约束路径拒绝它。

2. 扩展校验器，支持带判别器的 `oneOf`。常见场景：`line_item` 是产品或服务，由 `kind` 标记。严格模式在这里有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 用 Pydantic BaseModel 编写同样的 Invoice schema，把 `model_json_schema()` 的输出与手写 schema 对比。找出 Pydantic 默认设置而手写版本省略的那个字段。

4. 测量拒绝率。构造十个不应被抽取的输入（一段歌词、一个数学证明、一封空白邮件），让它们通过真正的严格模式供应商运行。统计拒绝与幻觉输出的数量。这就是你实现拒绝感知重试的基准事实。

5. 从头到尾读一遍 OpenAI 的结构化输出指南。找出它在严格模式中明确禁止、而普通 JSON Schema 允许的那一个构造。然后设计一个非必要地使用该构造的 schema，并把它重构为严格模式兼容。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “schema 规范” | 每个现代供应商都使用的 IETF 草案 schema 方言 |
| Strict mode | “保证 schema” | OpenAI 通过约束解码强制执行 schema 的标志 |
| Constrained decoding | “logit 屏蔽” | 解码时屏蔽无效下一 token 的强制执行 |
| Refusal | “模型拒绝” | 输入无法符合 schema 时的类型化结果 |
| Parse error | “无效 JSON” | 输出无法按 JSON 解析；严格模式下不可能 |
| Schema violation | “形状错误” | 能解析但违反 types / required / enum / range |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI 严格模式中必需 |
| Pydantic BaseModel | “类型化输出” | 生成并校验 JSON Schema 的 Python 类 |
| Zod schema | “TypeScript 输出类型” | 用于校验供应商输出的 TS 运行时 schema |
| Grammar enforcement | “开放权重约束解码” | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝与 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月的发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 序列化到各供应商的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明与严格模式注意事项
