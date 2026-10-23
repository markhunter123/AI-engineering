# 结构化输出：JSON、Schema 校验与约束解码

> 你的 LLM 返回的是字符串，你的应用需要的是 JSON。这个落差搞垮过的生产系统，比任何模型幻觉都多。结构化输出是自然语言与类型化数据之间的桥梁。做对了，LLM 就是一个可靠的 API；做错了，你就得在凌晨 3 点用正则解析自由文本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes
**Related:** 阶段 5 · 20（结构化输出与约束解码）讲解解码器层面的理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦生产环境的 SDK 接口（OpenAI `response_format`、Anthropic 工具使用、Instructor）——如果你想理解 API 之下发生了什么，请先读阶段 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic 的 API 参数实现 JSON 模式与 Schema 约束输出
- 构建一个 Pydantic 校验层，拒绝格式错误的 LLM 输出，并带着错误反馈重试
- 解释约束解码如何在 token 层面强制输出合法 JSON，而无需后处理
- 设计稳健的抽取 prompt，可靠地把非结构化文本转换成类型化数据结构

## 问题

你让 LLM：「从这段文字中提取商品名称、价格和库存情况。」它回答：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这个回答完全正确，但对你的应用毫无用处。你的库存系统需要的是 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个带有指定键、指定类型和指定取值约束的 JSON 对象，而不是一句话。

最朴素的做法是在 prompt 里加一句「用 JSON 回答」。这能奏效 90% 的时间。剩下 10% 里，模型会把 JSON 包进 markdown 代码围栏，或者加上「这是 JSON：」这样的开场白，或者因为提前闭合括号而产出语法非法的 JSON。你的 JSON 解析器崩溃，pipeline 中断。你加 try/except 和重试循环。重试有时会产出不同的数据。于是解析问题之上又多了一致性问题。

这不是 prompt 工程问题，而是解码问题。模型从左到右生成 token。每个位置上，它都要从 10 万多个候选中挑出最可能的下一个 token。其中大多数候选在任何给定位置都会产出非法 JSON。如果模型刚输出了 `{"price":`，下一个 token 必须是数字、引号（表示字符串）、`null`、`true`、`false` 或负号，其他任何东西都会让 JSON 非法。没有约束时，模型可能挑出一个英文上完全合理、语法上却灾难性错误的词。

## 概念

### 结构化输出的谱系

结构化输出控制分为四个层级，可靠性逐级提升。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**基于 prompt**（「用合法 JSON 回答」）：没有任何强制。模型通常遵守，但并非总是如此。可靠性约 90%。失败形态：markdown 代码围栏、开场白文字、输出截断、结构错误。

**JSON 模式**：API 保证输出是合法 JSON。OpenAI 的 `response_format: { type: "json_object" }` 可以开启。输出能无错解析，但不一定符合你期望的 Schema——可能多键、类型错误、缺字段。

**Schema 模式**：API 接收一个 JSON Schema，并保证输出符合它。2026 年，每家主流提供商都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可用 `tool_choice="required"`）、Anthropic 带 `input_schema` 的工具使用，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出拥有你指定的键、类型和约束。

**约束解码**：生成过程中，解码器在每个 token 位置屏蔽掉所有会产出非法输出的 token。如果 Schema 要求一个数字，而模型即将输出一个字母，该 token 的概率会被置零。模型只能产出通向合法输出的 token。OpenAI 的结构化输出模式以及 Outlines、Guidance 等库底层实现的就是这件事。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或校验层）输出必须具备什么形状的方式。每个主流结构化输出系统都在使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 Schema 规定：输出必须是一个对象，包含字符串 `product`、非负数字 `price`、布尔值 `in_stock`，以及可选的字符串数组 `categories`。任何不符合的输出都会被拒绝。

Schema 能处理难缠的情况：嵌套对象、带类型元素的数组、枚举（把字符串限定在特定取值内）、模式匹配（对字符串用正则），以及组合器（用 oneOf、anyOf、allOf 表达多态输出）。

### Pydantic 模式

在 Python 里，你不需要手写 JSON Schema。定义一个 Pydantic 模型，它会替你生成 Schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor 库（以及 OpenAI 的 SDK）直接接受 Pydantic 模型：传入模型类，拿回校验后的实例。如果 LLM 输出不符合，Instructor 会自动重试。

### Function Calling 与工具使用

这是解决同一问题的另一种接口。你不要求模型直接产出 JSON，而是定义带类型参数的「工具」（函数）。模型输出一次 function call，参数是结构化的。OpenAI 称之为「function calling」，Anthropic 称之为「工具使用（tool use）」。结果一样：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个函数，而不仅仅是填参数时，优先使用工具使用。如果你有 10 种不同的抽取 Schema，模型必须根据输入挑出正确的一种，那么工具使用同时提供了 Schema 选择和结构化输出。

### 常见失败模式

即使有 Schema 强制，结构化输出仍可能以微妙的方式失败。

**幻觉取值**：输出符合 Schema，但包含编造的数据。原文写 $348，模型却产出 `{"price": 299.99}`。Schema 校验抓不到这种情况——类型对，值错。

**枚举混淆**：你把字段限定为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义上正确，但不在允许集合内。好的约束解码能防止这一点，基于 prompt 的方法不能。

**嵌套对象深度**：嵌套很深的 Schema（4 层以上）会产生更多错误。每多一层嵌套，模型就多一个可能跟丢结构的地方。

**数组长度**：模型可能在数组中产出过多或过少的元素。Schema 支持 `minItems` 和 `maxItems`，但并非所有提供商都会在解码层面强制执行。

**可选字段遗漏**：模型会省略技术上是可选的、但对你的用例在语义上很重要的字段。即使数据有时确实缺失，也把它们设为必填——迫使模型显式产出 `null`。

```figure
mx-schema-funnel
```

## 构建它

### 第 1 步：JSON Schema 验证器

从零构建一个验证器，检查 Python 对象是否符合 JSON Schema。这就是在输出侧运行的合规检查。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 第 2 步：从 Pydantic 风格模型生成 Schema

构建一个最小的「类到 Schema」转换器。定义一个 Python 类，自动生成它的 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 第 3 步：约束 token 过滤器

模拟约束解码。给定一个不完整的 JSON 字符串和一个 Schema，判断当前位置哪些 token 类别是合法的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 第 4 步：抽取 pipeline

把所有组件组合成一条抽取 pipeline：定义 Schema，模拟 LLM 产出结构化输出，校验输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 第 5 步：运行完整 pipeline

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用它

### OpenAI 结构化输出

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式内部使用约束解码。模型生成的每个 token 都保证产出符合 Pydantic Schema 的输出。无需重试，无需校验。约束已经烘焙进解码过程。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过工具使用实现结构化输出。模型发出一次 tool call，参数是结构化的并符合 input_schema。结果相同，API 接口不同。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 包装任意 LLM 客户端，并加上带校验的自动重试。如果第一次尝试未通过校验，它会把错误作为上下文发回模型，要求修正输出。这不仅适用于 OpenAI，任何提供商都可以。

## 交付它

本课产出 `outputs/prompt-structured-extractor.md`——一个可复用的 prompt 模板，给定 Schema 定义后，能从任意文本中抽取结构化数据。把 JSON Schema 和非结构化文本交给它，它返回校验过的 JSON。

它还产出 `outputs/skill-structured-outputs.md`——一个决策框架，帮助你根据提供商、可靠性要求和 Schema 复杂度选择正确的结构化输出策略。

## 练习

1. 扩展 Schema 验证器，支持 `oneOf`（数据必须恰好匹配多个 Schema 中的一个）。这能处理多态输出——例如某个字段既可以是 `Product`，也可以是结构不同的 `Service` 对象。

2. 构建一个「Schema diff」工具，比较两个 Schema，区分破坏性变更（删除必填字段、改变类型）和非破坏性变更（新增可选字段、放宽约束）。这对生产环境中抽取 Schema 的版本管理至关重要。

3. 实现一个更真实的约束解码模拟器。给定一个 JSON Schema 和由 100 个 token 组成的词表（字母、数字、标点、关键字），逐步走完生成过程，在每个位置屏蔽非法 token。测量每一步词表中合法 token 的比例。

4. 构建一套抽取评测。编写 50 条商品描述，并手工标注对应的 JSON 输出。在全部 50 条上运行抽取 pipeline，测量精确匹配、字段级准确率和类型合规率。找出最难正确抽取的字段。

5. 给抽取 pipeline 增加「置信度分数」。对每个抽取字段，估计模型的把握程度（可以基于 token 概率，也可以运行 3 次抽取并测量一致性）。把低置信度字段标记出来供人工复核。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|----------------------|
| JSON mode | 「返回 JSON」 | API 标志，保证输出是语法合法的 JSON，但不强制任何特定 Schema |
| Structured output | 「带类型的 JSON」 | 符合特定 JSON Schema 的输出，键、类型和约束都正确 |
| Constrained decoding | 「引导式生成」 | 在每个 token 位置屏蔽会产出非法输出的 token——保证 100% 符合 Schema |
| JSON Schema | 「JSON 模板」 | 一种声明式语言，用于描述 JSON 数据的结构、类型和约束（被 OpenAPI、JSON Forms 等使用） |
| Pydantic | 「Python 数据类增强版」 | 定义带类型校验的数据模型的 Python 库，FastAPI 和 Instructor 用它生成 JSON Schema |
| Function calling | 「工具使用」 | LLM 输出结构化的 function call（名称 + 带类型参数）而非自由文本——OpenAI 和 Anthropic 都支持 |
| Instructor | 「面向 LLM 的 Pydantic」 | 包装 LLM 客户端、返回校验后的 Pydantic 实例的 Python 库，校验失败时自动重试 |
| Token masking | 「过滤词表」 | 生成时把特定 token 的概率置零，使模型无法产出它们 |
| Schema compliance | 「符合形状」 | 输出包含所有必填字段、类型正确、值在约束内，且没有多余的禁止字段 |
| Retry loop | 「反复尝试直到成功」 | 把校验错误发回模型并要求修正输出——Instructor 会自动执行，直到可配置的最大次数 |

## 延伸阅读

- [OpenAI 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) —— OpenAI API 中基于 JSON Schema 的约束解码官方文档
- [Willard & Louf, 2023——"Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) —— Outlines 论文，描述如何把 JSON Schema 编译成有限状态机以实现 token 级约束
- [Instructor 文档](https://python.useinstructor.com/) —— 从任意 LLM 获取结构化输出并进行 Pydantic 校验与重试的标准库
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use) —— Claude 如何通过带 JSON Schema `input_schema` 的工具使用实现结构化输出
- [JSON Schema 规范](https://json-schema.org/) —— 所有主流结构化输出系统所用 Schema 语言的完整规范
- [Outlines 库](https://github.com/outlines-dev/outlines) —— 使用正则和 JSON Schema 编译为有限状态机的开源约束生成库
- [Dong 等，"XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models"（MLSys 2025）](https://arxiv.org/abs/2411.15100) —— 当前最先进的语法引擎；下推自动机编译，遮蔽 token 的开销约为每 token 100 ns。
- [Beurer-Kellner 等，"Prompting Is Programming: A Query Language for Large Language Models"（LMQL）](https://arxiv.org/abs/2212.06094) —— LMQL 论文，把约束解码表述为带类型和取值约束的查询语言。
- [Microsoft Guidance（框架文档）](https://github.com/guidance-ai/guidance) —— 模板驱动的约束生成；Outlines 和 XGrammar 的厂商无关补充。
