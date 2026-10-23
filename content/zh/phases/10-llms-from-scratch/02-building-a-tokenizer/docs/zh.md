# 从零构建分词器

> 第 01 课给你的是玩具。这一课给你的是武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## 学习目标

- 构建一个生产级 BPE 分词器，能处理 Unicode、空白归一化和特殊 token
- 实现字节级回退，使分词器能编码任何输入（包括 emoji、CJK 和代码），且不产生未知 token
- 添加预分词正则模式，在施加 BPE 合并之前按词边界切分文本
- 在语料上训练自定义分词器，并在多语言文本上将其压缩比与 tiktoken 对比评估

## 问题

你在第 01 课写的 BPE 分词器能处理英文。现在把日文扔给它。或者 emoji。或者混用制表符与空格的 Python 代码。

它崩了。

不是因为 BPE 错了——而是因为实现不完整。生产级分词器要处理任意编码的原始字节，在切分前归一化 Unicode，管理那些永远不会被合并的特殊 token，把预分词与子词切分串起来，而且这一切都要足够快，不能成为处理 15 万亿 token 的训练 pipeline 的瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约 100,000 个。这些不是玩具数字。这些词表背后的合并表是在数百 GB 文本上训练出来的，而周边的那套机制——归一化、预分词、特殊 token 注入、chat 模板格式化——才是一个只能处理「hello world」的分词器和一个能处理整个互联网的分词器之间的分水岭。

你就要把那套机制造出来。

## 概念

### 完整 pipeline

生产级分词器不是一个算法。它是一条由五个阶段组成的 pipeline，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有具体职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| 归一化 | NFKC Unicode，可选小写化，可选去除重音 |「fi」连字（U+FB01）变成「fi」（两个字符）。没有这一步，同一个词会得到不同的 token。 |
| 预分词 | 在 BPE 之前把文本切成块 | 防止 BPE 跨词边界合并。「the cat」永远不应产生 token「e c」。 |
| BPE 合并 | 对字节序列施加学到的合并规则 | 核心压缩环节。把原始字节变成子词 token。 |
| 特殊 token | 注入 [BOS]、[EOS]、[PAD] 和 chat 模板标记 | 这些 token 有固定 ID，从不参与 BPE 合并。模型需要它们来表达结构。 |
| ID 映射 | 把 token 字符串转换成整数 ID | 模型看到的是整数，不是字符串。 |

### 字节级 BPE

第 01 课的分词器在 UTF-8 字节上操作。那个决定是对的。但我们跳过了一个重要问题：当这些字节不是合法 UTF-8 时会怎样？

字节级 BPE 通过把每个可能的字节值（0-255）都视为合法 token 来解决这个问题。基础词表恰好 256 项。任何文件——文本、二进制、损坏的数据——都能被分词，而不会产生未知 token。

GPT-2 加了一个小技巧：把每个字节映射到可打印的 Unicode 字符，让词表保持人类可读。字节 0x20（空格）在它的映射里变成字符「Ġ」。这纯粹是外观上的。算法本身并不关心。

真正的威力在于：字节级 BPE 能处理地球上每一种语言。一个中文字符占 3 个 UTF-8 字节。日文可能是 3-4 个字节。阿拉伯文、天城文、emoji——全都只是字节序列。BPE 算法在这些字节序列中发现模式的方式，与它在英文 ASCII 字节中发现模式的方式完全相同。

### 预分词

在 BPE 碰你的文本之前，你需要先把它切成块。这防止合并算法创建跨越词边界的 token。

GPT-2 用一条正则表达式切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这条模式按缩写（「don't」变成「don」+「't」）、带可选前导空格的单词、数字、标点和空白切分。前导空格会保留并附着在单词上——所以「the cat」变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，它完全跳过正则。它把原始字节流当作一条长序列，让 BPE 算法自己去发现边界。这更简单，但给了 BPE 更多自由去创建跨词 token。

这个选择很重要。GPT-2 的正则防止分词器学会把前一个词末尾的「the」和下一个词开头的「the」合并。SentencePiece 允许这种合并，有时能带来更高效的压缩，但 token 的可解释性更差。

### 特殊 token

每个生产级分词器都会为结构化标记保留 token ID：

| Token | 用途 | 使用方 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 用于 batch 对齐的填充 | BERT、T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除了它） | BERT、WordPiece |
| `<\|im_start\|>` | chat 消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | chat 消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户回合标记 | Llama 3 |
| `<\|assistant\|>` | 助手回合标记 | Llama 3 |

特殊 token 永远不会被 BPE 切分。它们在合并算法运行之前被精确匹配，替换成固定 ID，周围的文本则照常分词。

### Chat 模板

这是大多数人搞不清、大多数实现出错的地方。

当你向 chat 模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON。它看到的是一条扁平的 token 序列。chat 模板用特殊 token 把消息转换成那条扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板搞错，模型就会输出垃圾。它是在一种精确格式上训练的。任何偏差——少一个换行、换错一个 token、多一个空格——都会把输入推到训练分布之外。

### 速度

Python 对生产级分词来说太慢了。

tiktoken（OpenAI）用 Rust 编写，带 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些实现比纯 Python 快 10-100 倍。

换个角度看：Llama 3 预训练要处理 15 万亿 token。以每秒 100 万 token（较快的 Python）计算，需要 174 天。以每秒 1 亿 token（Rust）计算，需要 1.7 天。

你用 Python 构建是为了理解算法。在生产中，你会使用编译好的实现，只接触 Python 包装层。

```figure
weight-tying
```

## 构建它

### 第 1 步：字节级编码

基础。把任意字符串转换成字节序列，把每个字节映射到可打印字符以便显示，再逆转这个过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，看看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

「hello」是 5 个字节。「你好」是 6 个字节（每个字符 3 个）。火焰 emoji 是 4 个字节。字节级分词器不关心是什么语言。字节就是字节。

### 第 2 步：基于正则的预分词器

使用 GPT-2 的正则模式把文本切成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们回退到 ASCII 字符类。对于生产级多语言分词器，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格留在单词上。缩写在撇号处切开。标点成为独立的块。BPE 永远不会跨越这些边界合并 token。

### 第 3 步：在字节序列上运行 BPE

核心算法与第 01 课相同，但现在对预分词后的块独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：特殊 token 处理

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整分词器类

把所有环节串起来：归一化、按特殊 token 切分、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

真正的考验。把英文、中文、emoji 和代码扔给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 个字节。emoji 产生 4 个字节。这些都不会让分词器崩溃。都不会产生未知 token。这就是字节级 BPE 的力量。

## 使用它

### 对比真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器。看看它们各自如何处理同一段多语言文字。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一文本得到不同的 token 数。Llama 3 的 128K 词表在合并常见模式上更激进。GPT-4 的 100K 处于中间。Mistral 的 32K 产生更多 token，但 embedding 层更小。

权衡始终一样：更大的词表意味着更短的序列，但更多参数。

## 交付它

本课产出用于构建和调试生产级分词器的 prompt。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它检查你最常见的合并 token 实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器，按空白和数字切分但保留前导空格。在同一语料上将其词表与 GPT-2 正则方案对比。
3. **困难：** 添加一个 chat 模板方法，接收 `{"role": ..., "content": ...}` 消息列表，产出 Llama 3 chat 格式正确的 token 序列。对照 HuggingFace 的实现测试它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE | 「在字节上工作的分词器」 | 基础词表为 256 个字节值的 BPE——能处理任何输入而不产生未知 token |
| 预分词 | 「在 BPE 之前切分」 | 基于正则或规则的切分，防止 BPE 跨词边界合并 |
| NFKC 归一化 | 「Unicode 清理」 | 先做规范分解再做兼容组合——「fi」连字变成「fi」，全角「A」变成「A」 |
| Chat 模板 | 「消息如何变成 token」 | 把 role/content 消息列表转成扁平 token 序列的精确格式——因模型而异，必须与训练格式一致 |
| 特殊 token | 「控制 token」 | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、chat 标记——在合并前精确匹配 |
| Fertility | 「每词 token 数」 | 输出 token 数与输入单词数之比——GPT-4 处理英文为 1.3，韩文为 2-3，越高说明上下文浪费越多 |
| tiktoken | 「OpenAI 分词器」 | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10-100 倍 |
| 合并表 | 「词表」 | 训练期间学到的字节对合并的有序列表——它就是分词器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken)——GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)——支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783)——128K 词表与分词器训练的细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226)——与语言无关的分词
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py)——最初的字节到 Unicode 映射
