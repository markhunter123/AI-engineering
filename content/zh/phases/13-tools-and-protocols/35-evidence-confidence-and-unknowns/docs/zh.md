# 二进制分析中的证据、置信度与未知项

> 关于一个已编译程序的结论是一条记录：它带有哈希、置信度、权威来源，以及一份它无法展示的内容清单。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~105 minutes

## 学习目标

- 构建一条仿照 rea `Evidence` 的证据记录，包含 observed、derived 和 inferred 三种置信度。
- 计算基于内容哈希的 id，使文件移动到其他目录时 id 保持不变。
- 把每个提供方无法展示的内容记录为局限性，并让它们随记录一同流转。
- 在函数的观测运行支持该结果之前，拒绝给出结论。
- 将残余未知项保存为带修订版本的记录，并用哈希链串联。
- 以精确键缓存提供方的答案，并拒绝过期的重放。

## 问题

你问一个 Agent：`ledger-x86_64-O1.o` 中的 `grade` 在分数为 55 时返回什么。Agent 调用 rea，阅读 Ghidra 伪代码，回答 0。评审者问它凭什么知道。它手上只有一段 Ghidra 打印出来的文本。在默认路径上，这段文本写着 `uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);`，而该位置上的字节写的是 `xor eax, eax`。

这个表达式和这条指令对任何输入都产生 0。Agent 并不知道这一点。它信任那段文本，因为文本是某个工具产出的。2026 年的一项研究让三个 Agent 执行 24 个逆向工程任务（[NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)），指出了四类反复出现的弱点：训练偏差、对观察结果的过度信任、上下文限制，以及计划惯性（plan persistence）。另一项 2026 年研究把文本植入二进制文件，发现模型在 40 个案例中有 35 个提出了被植入的不安全操作（[arXiv 2607.12507](https://arxiv.org/abs/2607.12507)）。

rea 6.5.0 为此给 Agent 提供了一套词汇表。每次成功的 tool call 都会产生一条 `Evidence` 记录，其中带有取值为 `observed`、`derived` 或 `inferred` 的 `confidence`，一个 `authority`，以及一组 `limitations`。随附的技能文件告诉 Agent：「每个结论都必须区分观察、推断和未知。引用 Evidence ID，保留局限性与不完整的覆盖范围，并且绝不要暗示静态分析观察到了执行。」

有三件事 rea 留给你自己完成。它的快照缓存的是精确的查询结果，而不是分析本身。它的验证只要某个比较维度没有变化，就判定一条声明通过；而它自己的注释对此写道，一次通过「does not establish global implementation equivalence」。它的义务台账只从 JavaScript 和托管代码图中获取候选，因此原生函数永远无法通过它达到 `verified`。

本课用纯 Python 补上这块缺失的拼图。你编写一个台账，把关于 fixture 的每一条事实都存为带内容哈希的记录。你针对 `percent_change` 和 `grade` 作出预测，把它们标记为推断，再用你自己编译的测试程序的运行结果加以检验。你导入两条真实的 rea 记录及其来源快照，你的台账会重新计算 rea 写下的每一个哈希。

```figure
re-evidence-ledger
```

## 置信度三个词，权威来源五个词

`confidence` 是一个只有三个取值的枚举，绝不是数字。`authority` 有五个取值。当产出方没有声明时，rea 使用 `shipped-artifact` 和 `observed`。

| 取值 | 在 fixture 上的含义 | 本课中的记录 |
|---|---|---|
| `observed` | 从字节或一次运行中读出的事实 | ELF 读取器的节与符号读取、两条 Ghidra 记录、17 次主机运行 |
| `derived` | 由观测记录计算得到的事实 | switch 表模型、除法模型、每一次未知项变更 |
| `inferred` | 尚无任何观察证实的预期 | 18 条预测 |
| `shipped-artifact` | 磁盘上的文件是事实来源 | 对 `ledger-x86_64-O1.o` 的每一次读取 |
| `controlled-replay` | 由你控制的运行产生的事实 | `harness.c` 的主机运行 |
| `analyst-inference` | 你自己的判断 | 预测与未知项变更 |
| `historical-reference` | 更早的源码树或文档 | 本课没有 |
| `external-service` | 文件和主机之外的服务 | 本课没有 |

有两种混淆最浪费时间。第一种把伪代码当成对行为的观察；它实际上只是对 Ghidra 打印内容的观察。第二种把空列表当成不存在的证据。rea 的设计说明很严格：「缺失的覆盖范围是未知，而不是不存在。」`code/main.py` 中的台账把这两条规则都写进了代码。

## 记录的结构

一条记录恰好有十六个字段，rea 6.5.0 会拒绝任何其他字段。`code/main.py` 与 rea 使用相同的字段名，因此任意一方的记录都能在另一方通过校验。第十六个字段 `relationship_coverage` 说明调用者或被调用者列表的完整程度，取值为 `complete`、`partial` 或 `unknown`；当该操作不报告调用关系时为 `null`。

| 字段 | 保存的内容 |
| --- | --- |
| `evidence_id` | `ev_` 加 64 个十六进制数字，即内容的哈希 |
| `subject` | `name`、`digest.sha256`、`format`、`architecture`、`local_path`，或 `null` |
| `provider` | `id`、`name`、`version`，或 `null` |
| `analysis_profile` | 提供方的设置及其摘要，或 `null`；该字段始终存在 |
| `predicate_type` | 一个标签，默认为 `rea.analysis` |
| `operation`、`parameters` | 工具及其输入 |
| `raw_result`、`normalized_result` | 提供方输出与规范化结果 |
| `confidence`、`authority` | 上文所述的枚举 |
| `environment` | `id`、`platform`、`architecture`、`isolation`，或 `null` |
| `limitations` | 纯文本句子 |
| `relationship_coverage` | `status` 为 `complete`、`partial` 或 `unknown` 并附诊断信息，或 `null` |
| `locations` | `address`、`address-range`、`artifact-path`、`file-offset`、`file-offset-range` |
| `evidence_links` | 该记录所依赖的 `ev_` id |

这是 rea 为 `grade` 写下的真实记录，仅保留与本课相关的字段。完整文件是 `code/fixtures/rea-evidence-grade.json`，大小 37,573 字节。

```json
{
  "evidence_id": "ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1",
  "subject": {
    "name": "ledger-x86_64-O1.o",
    "digest": { "sha256": "9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f" },
    "format": "elf",
    "architecture": "x86_64",
    "local_path": "/private/tmp/aiefs-re/phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code/fixtures/ledger-x86_64-O1.o"
  },
  "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
  "analysis_profile": {
    "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
    "parameters": { "jump_table_evidence": "typed-case-default-blocks-v1", "import_mode": "ephemeral-source-immutable" },
    "digest": "9c7f6b24be69294e61da046e4583e3856cb9e40433b149c1bf8bb534d2fe03ef"
  },
  "operation": "analyze_function",
  "parameters": { "procedure": "grade" },
  "confidence": "observed",
  "authority": "shipped-artifact",
  "environment": null,
  "limitations": [
    "Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.",
    "Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals."
  ],
  "relationship_coverage": {
    "status": "complete",
    "diagnostics": [
      "Only resolved Ghidra call references are returned; unresolved computed or indirect calls remain unknown, while function classifications distinguish thunks and externals."
    ]
  },
  "locations": [],
  "evidence_links": []
}
```

真实的 profile 中，`parameters` 对象有 25 个键，真实的 `limitations` 列表有十句话。保留它们。局限性是内容的一部分，删掉一条就会改变 id，你的台账随后会拒绝该记录。

注意这条记录没有说的内容。`environment` 为 `null`，因为 Ghidra 只是读取文件，没有运行任何东西。`locations` 为空，因为该操作把地址报告在 `normalized_result` 里。`relationship_coverage` 的状态是 `complete`，其诊断信息说明只返回已解析的 Ghidra 调用引用，未解析的计算调用或间接调用仍属未知。因此，调用者与被调用者列表只对已解析的调用是完整的。

## id 是内容哈希，路径不在其中

rea 计算 `evidence_id` 的方式是：对记录的一个投影做 RFC 8785 规范化 JSON 编码，再取 SHA-256，前面加上 `ev_`。该投影保留主体的 `digest`、`format` 和 `architecture`，丢弃 `name` 与 `local_path`。当 `relationship_coverage` 的值为 `null` 时它也被丢弃，因此一条没有调用关系的记录，其 id 与这个字段出现之前相同。除此之外的一切，包括 `raw_result` 和每一条局限性，都参与哈希。

RFC 8785 会对对象键排序、去除空白，并固定数字与字符串的表示形式。对于数字都是整数的记录，Python 标准库会产出相同的字节：

```python
def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

`code/main.py` 在哈希之前会拒绝浮点数，因为 RFC 8785 与 Python 对某些浮点数的打印方式不同。两条 rea 记录和快照都不含浮点数，因此这套配方能复现 rea 的每一个哈希：

| rea 的值 | 由 `code/main.py` 重新计算的结果 |
|---|---|
| `grade` 的 `ev_67b6d3f992eaf49a…` | 一致 |
| `percent_change` 的 `ev_e88b10dc0c6400e5…` | 一致 |
| 覆盖 `{provider, parameters}` 的 profile 摘要 `9c7f6b24be69294e…` | 一致 |
| `grade` 快照条目的 `query_26122aebe9d8abae…` | 一致 |
| `percent_change` 条目的 `query_0f95ef38038d3666…` | 一致 |

为什么要排除路径？对相同字节的同一次观察，在任何机器、任何目录下都应得到相同的 id。上面清单中的 `local_path` 指向某一台 Mac 上的目录。把 fixture 复制到 `/home/you/re` 时，id 不会改变。`rebind_subject_path()` 让你能以新路径展示记录；当 `sha256`、`format` 或 `architecture` 不同时，它会拒绝。rea 6.5.0 在为新路径上的目标重放快照记录时，用 `rebindEvidenceToTargetPath` 做同样的事。

`validate_evidence()` 会重新计算 id，并以 rea 的原句拒绝不匹配的记录：「Evidence semantic identifier does not match its record」。它还会拒绝 `analysis_profile.provider` 与 `provider` 不一致的记录。测试 `test_tampered_id_and_mismatched_profile_are_rejected` 修改真实 `grade` 记录中的一行伪代码，并观察台账如何拒绝它。

## 自己观测对象

在任何提供方介入之前，台账先读取文件。`observe_object()` 用 `struct.unpack_from` 解析 ELF64 节头、`.symtab`、`.strtab` 和 `.rela.text`。它写出一条 `inspect_object` 记录，并为每个函数符号写出一条 `read_function_bytes` 记录。这五条记录的 `confidence` 都是 `observed`，`authority` 都是 `shipped-artifact`，因为事实来源就是磁盘上的文件。

| 记录 | 内容 | Id |
|---|---|---|
| `inspect_object` | 五个节及其偏移、大小和哈希，四个函数，三条重定位 | `ev_645e0a145f28bd5c…` |
| `read_function_bytes percent_change` | `.text+0x00` 处的 16 个字节 | `ev_2572a7d254ad15cb…` |
| `read_function_bytes grade` | `.text+0x10` 处的 46 个字节，SHA-256 为 `ffcb75c2a7bb1104…` | `ev_a467c2427bfc0c56…` |
| `read_function_bytes total` | `.text+0x40` 处的 31 个字节 | `ev_b1e845cea2543073…` |
| `read_function_bytes grade_label` | `.text+0x60` 处的 63 个字节 | `ev_c50041be26e6b7f4…` |

`llvm-objdump -t` 的结果与此一致：`grade` 位于 `0000000000000010`，大小为 `2e`。`.rela.text` 中的三条重定位都是针对 `.rodata` 的 `R_X86_64_PC32`，偏移分别为 `0x36`、`0x84` 和 `0x93`，加数分别为 `0x10`、`0x24` 和 `-0x4`。每条记录都带有一个 `file-offset-range` 位置和一条局限性：「节字节是从磁盘读出的。读取器不施加重定位，也不运行任何代码。」

第 32 课完整构建了这个读取器。这里只有 60 行，因为本课关注的是你如何记录这次读取。

## 推导出表，并声明这是推导

`grade` 的 46 个字节中包含第 33 课解码的四种模式：

| 字节 | 指令 | 在 `grade` 中的偏移 |
|---|---|---|
| `48 69 c0 67 66 66 66` | `imul rax, rax, 0x66666667` | 3 |
| `48 c1 f9 22` | `sar rcx, 34` | 13 |
| `83 c1 fa` | `add ecx, -6` | 23 |
| `83 f9 04 77 0c` | `cmp ecx, 4` 然后 `ja` | 28 |

先乘以 `0x66666667` 再右移 34 位，是有符号除以 10。文件中并没有 10。`derive_grade_model()` 把这一事实写成一条局限性：「除数 10 是从乘数 0x66666667 和移位 34 推导出来的。文件中没有 10。」

`add ecx, -6` 把第一个分支设为 6。`cmp ecx, 4` 配合无符号跳转，允许 6 到 10 共五个分支。

`.text+0x33` 处的 `lea` 携带加数为 `0x10` 的重定位。PC32 重定位从 4 字节位移字段的末尾开始计算，因此表从 `.rodata+0x14` 开始。那里的五个小端字读出 `1, 2, 3, 4, 4`。`ledger.c` 中的 switch 对 60 到 69 返回 1，对七十多返回 2，对八十多返回 3，对 90 到 100 返回 4。编译器把这个 switch 变成了保存返回值的查找表。

得到的记录 `ev_605106bff2d9998f…` 的 `confidence` 为 `derived`，其 `evidence_links` 指名的两条观测记录是它的依据。从台账中移除字节记录后，这条派生记录就会出现悬空链接。关掉 `ev_a467c242…` 后，上图展示的正是这种状态。

现在与 Ghidra 对比。它对相同字节给出的伪代码是：

```c
uint grade(int param_1)
{
  uint uVar1;
  uint uVar2;

  uVar2 = param_1 / 10 - 6;
  uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  if (uVar2 < 5) {
    uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }
  return uVar1;
}
```

Ghidra 施加了重定位，并把表命名为 `DAT_001000b4`；在它把 `.text` 放到 `0x100000` 之后，这个地址就是 `.rodata+0x14`。它还原出了除以 10 和五个分支。对于默认值，它打印出一个恒为 0、却从不写明 0 的表达式。`native_value_flow` 中的 p-code 把这些操作标记为 `is_dead: true`。这条记录是对 Ghidra 的忠实观测。你从中读出的含义是推导出来的，你预期一次运行会给出的值是推断出来的。

`percent_change` 的记录展示了另一种失败。Ghidra 的签名是 `undefined1 [16] percent_change(int param_1,int param_2,ulong param_3)`，而源码写的是 `int percent_change(int before, int after)`。`idiv` 会写入 `EDX:EAX`，零值路径没有碰 `EDX`，于是反编译器推断出一个 16 字节的返回值和一个不存在的第三个参数。它对 `grade` 的 `native_api.return_type.confidence` 本身就是 `"low"`。把还原出的签名当作提供方的推断，并按推断记录它。

## 预测是推断记录

预测是一条关于行为的声明，而目前还没有任何观察陈述过它。`claim_suite()` 从派生模型和源码构建十八条预测。`record_prediction()` 把每一条存为记录，`confidence` 为 `inferred`，`authority` 为 `analyst-inference`，并链接到它依据的记录。

| 声明 | 预测值 | 依据 |
|---|---|---|
| `percent_change(12400, 15500)` | 25，来自 310000 / 12400 | 除法模型 `ev_076a9167…` |
| `percent_change(0, 5)` | 0，零值保护 | 除法模型 |
| `percent_change(100, 50)` | -50 | 除法模型 |
| `percent_change(-7, 3)` | -142，因为 `idiv` 向零截断 | 除法模型 |
| `grade(55)` | 0，索引 -1 无法通过无符号比较 | 表模型 `ev_605106bf…` |
| `grade(60)`、`grade(69)` | 1、1 | 表模型 |
| `grade(70)`、`grade(85)` | 2、3 | 表模型 |
| `grade(95)`、`grade(100)`、`grade(105)` | 4、4、4 | 表模型 |
| `grade(110)`、`grade(-5)` | 0、0 | 表模型 |
| `total(1, 2, 3)`、`total()` | 6、0 | 源码记录 `ev_2f1e6d25…` |
| `grade_label(85)` | `"B"` | 源码记录与表模型 |
| 90 到 100 的 `grade(score)` | 全部十一个输入均为 4 | 表模型 |

`predict()` 使用 C 语义，而不是 Python 语义。`c_div()` 向零截断，`wrap32()` 把每个中间结果折叠进 32 位。Python 的 `//` 向负无穷取整，因此 `(3 - -7) * 100 // -7` 得到 -143，而 C 程序得到 -142。第 36 课会把这张网格做得大得多。

每条预测都会开启一个残余未知项。该未知项要求 `observed` 证据且 `authority` 为 `controlled-replay`，并推荐一种探针：在主机构建上执行 `execute_function`。下一节将产出这类证据。第一条 grade 声明对应的问题写作：「当编译后的函数运行时，grade(95) == 4 是否成立？」

## 用运行来检验

`code/fixtures/harness.c` 包含 `ledger.c`，每次调用打印一个结果。`observe_runs()` 用 PATH 上第一个可用的 `clang`、`cc` 或 `gcc` 编译它，并把网格中的每个输入运行一次。每次运行成为一条记录：

```json
{
  "operation": "execute_function",
  "parameters": { "function": "grade", "args": [95] },
  "normalized_result": { "result": 4, "stdout": "4\n", "exit_code": 0 },
  "confidence": "observed",
  "authority": "controlled-replay",
  "environment": { "id": "host-process", "platform": "darwin", "architecture": "arm64", "isolation": "process" },
  "limitations": [
    "The run executed a host build of ledger.c, not the x86-64 object. The claim about the object rests on the shared source and on the compiler.",
    "One run observes one input. It proves nothing about other inputs."
  ]
}
```

运行记录的主体是 harness 二进制文件，在这台 Mac 上是一个 Mach-O arm64 文件，有它自己的 SHA-256。提供方是 `course-harness`，附带编译器的版本行，这里是 `Apple clang version 17.0.0 (clang-1700.6.3.2)`。`evidence_links` 指名源码记录 `ev_2f1e6d251543d017…`，因为这次运行与目标文件只共享 `ledger.c`，没有别的。

请把第一条局限性读两遍。x86-64 目标文件从未在这台机器上运行。实际运行的是同一份源码为另一种架构构建的第二个版本。这仍然是一次观察，而且是在没有模拟器的情况下能取得的最强证据，因此该声明在附上这句话后通过。省略这句话的报告，就丢失了读者需要的信息。

17 次单输入运行全部与预测一致。`code/fixtures/run-evidence.json` 保存了这 17 条记录和源码记录，于 2026-10-13 用 `--record` 导出。没有编译器时，`build_ledger()` 改为导入这个证据包，报告会在 `runs` 字段中说明这一点。

## 提升规则

一条声明只能通过观测到的运行，从推断提升为受支持。`check_claim()` 查找函数与参数都相同的 `execute_function` 记录，逐一比较结果与预测，并统计已覆盖的输入：

| 状态 | 条件 | rea 的完成度台账 |
|---|---|---|
| `pass` | 每个输入都有相符的运行，且无任何不符 | `pass` |
| `fail` | 任何一次运行不符 | `fail` |
| `unknown` | 部分输入有运行，其余没有 | `unknown` |
| `unsupported` | 没有任何输入有运行 | `unsupported` |

`promote_claim()` 是严格形式。只要任何一个 id 是 `inferred` 或 `derived`，它就抛出 `ClaimError`。当某个 id 指向另一个函数或另一组参数的运行时，它也会抛出。列表为空时抛出；运行覆盖的输入少于声明所涉输入时也抛出。`grade(55)` 既有 Ghidra 伪代码记录，也有表模型，但两者都不能提升它。在图中关掉 `ev_7be90c0c…`，就能看到该声明跌回 `unsupported`，而关于 `grade` 的两条记录仍然存在。

范围声明展示了部分覆盖的情形。它的十一个输入中只有 95 和 100 两个出现在运行网格里。`check_claim()` 返回 `unknown`，`covered` 为 11 中的 2；`promote_claim()` 以「2 of 11 inputs have observed runs」拒绝。台账不会把它四舍五入地算作通过。

`completion_ledger()` 对应 rea 的 `EvidenceCompletionLedger`。它的 id 是 `ecl_` 加上其记录的哈希，只有当 `pass` 等于 `total` 时 `complete` 才为真。由于那一条范围声明，演示会打印 `ecl_da64933ea13befed…` 且 `complete: false`。

## 未知项是带哈希链的记录

rea 把残余未知项存为一条严格记录。`code/main.py` 保留相同的字段和不变式。

id 是 `unk_` 加上对 `{domain, question, scope_digest}` 取 SHA-256，其中问题的空白会折叠为单个空格。两位分析者对同一文件提出同一个问题，会得到相同的 id。`scope_digest` 是该对象的 SHA-256。

每一次变更都是一个新修订版本。修订版 1 的 `previous_revision_digest` 为 `null`。之后每个修订版都指名前一版的 `revision_digest`，而摘要覆盖除自身之外的每个字段。`verify_unknown_chain()` 遍历列表并重新计算每个摘要，检查每条链接是否指向前一修订版，还检查每个修订版所指的变更证据是否在台账中。

每一次变更本身也是一条证据记录，就像 rea 用 `rea-unknown-registry` 提供方所做的那样。我们的记录带有 `confidence` 为 `derived`、`authority` 为 `analyst-inference`，没有主体。其局限性写着：「注册表变更记录的是分析者的决定。它不是答案的证明。」

这就是为什么演示的 81 条记录中包含 36 条派生变更记录：18 条开启和 18 条更新。

每个修订版都强制执行三条不变式：

- 只有状态为 `resolved` 的未知项才带 `resolution`，且只有修订版 1 没有前驱。
- 状态为 `contradicted` 的未知项至少指名一个矛盾 id，且任何 id 不能同时出现在两个列表中。
- 状态为 `verified` 的解决结果至少指名一个证据 id，且其置信度和权威来源符合要求。

第三条规则是台账最常拒绝的地方。rea 的工作流说明写道：「只有合格的观测证据才能验证一个解决结果。」因此 `resolve_unknown()` 拒绝表模型，拒绝 `authority` 为 `shipped-artifact` 的观测记录，而接受运行记录 `ev_f4ffbe8f922ec7fe…`。`grade(95)` 的解决修订版是修订版 2，其 `previous_revision_digest` 为 `ffca840b55a8a188…`，即修订版 1 的摘要。

`update_unknown()` 接收 `expected_revision`，当它已过期时抛出 `revision-conflict`，与 rea 的 `update_unknown` 工具一致。正确的反应正是 rea 文档给出的：重新读取当前修订版，再决定一次。

演示结束后仍有一个未知项保持开启：`unk_f2eb0d7f84c80a19…`，即「当编译后的函数运行时，90 到 100 的每个分数都满足 grade(score) == 4 吗？」。它的状态是 `investigating`，支撑 id 中包含那两次运行，而它的探针仍指名九个尚未运行的输入。

## 答案缓存不是分析数据库

rea 有三样看起来像缓存的东西。Ghidra 的目标快照是输入的私有副本，用 `-readOnly -deleteProject` 导入，关闭时删除。内存中的 `AnalysisSnapshotCache` 针对一个二进制身份最多保存 10,000 个条目。快照文件保存 `target`、`binding`、`entries`、`workflow_entries` 和一个 `evidence_bundle`。

缓存键是 `query_` 加上对 `{target, binding, operation, parameters}` 取 SHA-256。`target` 为 `{sha256, kind, format, architecture}`，`raw-binary` 目标还带有 `entry_point`。`binding` 是提供方和整个分析 profile（含摘要）。改动 25 个 profile 参数中的任何一个，比如把 `jump_table_evidence` 从 `typed-case-default-blocks-v1` 改成 `v2`，摘要就会变化，于是所有键都随之改变。

`code/fixtures/rea-snapshot.json` 是 rea 在两次运行期间写出的文件，大小 142,690 字节。rea 以仅所有者可读的权限写入，而提交到仓库的副本对所有人可读。它有两个条目，以及一个包含同样两条记录的证据包。`AnswerCache.import_snapshot()` 会重新计算每个 `query_id`，只要有任何一个不符就拒绝该文件。当快照的 `target` 或 `binding` 与当前活动值不同时，它以 rea 的措辞 `profile_mismatch` 拒绝。当快照的 `format_version` 不是 `course-answer-cache-1` 时也会拒绝。

本机上的三次测量。第一次 `grade` 查询耗时 8.6 秒，包含一次全新的 Ghidra 导入。针对同一快照的 `percent_change` 查询耗时 8.5 秒，因为不同的查询就是一次缓存未命中。第三次 `grade` 查询耗时 2.1 秒，返回了完全相同的记录和相同的 `evidence_id`。rea 中的内存重放会附加一条局限性：「Loaded from a local REA analysis snapshot; this call did not re-run the provider.」。我们的 `lookup()` 也会附加一句同类说明，让读者能分辨重放与全新答案。

任何变更都会终结缓存。在 rea 6.5.0 中，重命名或添加注释会在 rea 把调用发给提供方之前清空快照缓存。失败的回复无法证明写入已经回滚，因此写入失败同样会终止重放。一次改变了数据库的元数据导入也一样。

## Agent 为什么需要它

通过 MCP 驱动 rea 的 Agent，每次调用都能看到完整记录，并可在报告中引用 `evidence_id`。2026 年的研究指出了问题所在。对工具输出的过度信任是 NDSS BAR 研究中四类弱点之一。在文本植入研究中，仅把二进制内容作为数据展示，40 个案例中仍有 15 个提出了不安全操作；加入来源校验与工具授权后，这些提议被消除。第三项研究观察了 Agent 通过 MCP 驱动 Ghidra 的过程（[arXiv 2606.06838](https://arxiv.org/abs/2606.06838)）。在包含数百个函数的二进制文件中，只有 10% 到 15% 的函数在 Agent 开始产出桩代码之前得到了真正的分析。

台账用记录回应上述每一种情况，Agent 只报告台账中已有的内容。没有观测运行的声明是 `unsupported`；只有部分运行的声明是 `unknown`；提供方文本只是关于该提供方的 `observed`，仅此而已。把 `outputs/` 中的技能文件交给 Agent，并让台账成为它写下结论的唯一位置。无论用的是 Claude、Cursor、Codex、OpenClaw、Hermes，还是任何读取 SKILL.md 的 Agent，都应如此。

## 构建它

运行这个仅用标准库的演示：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
```

演示读取目标文件，导入两条 rea 记录，验证 rea 快照，并推导出两个模型。随后它记录 18 条预测，开启 18 个未知项，编译并运行 harness，检查每一条声明，并解决 17 个未知项。预期输出大致如下：

```json
{
  "records": 81,
  "by_confidence": { "observed": 25, "derived": 38, "inferred": 18 },
  "rea_snapshot": { "entries": 2, "replay_hit": true, "stale_profile_refused": true },
  "grade_table": [1, 2, 3, 4, 4],
  "runs": "live: Apple clang version 17.0.0 (clang-1700.6.3.2)",
  "claims": [ { "claim_id": "grade(95)", "status": "pass", "covered": "1/1" }, { "claim_id": "grade(90..100)", "status": "unknown", "covered": "2/11" } ],
  "completion_ledger_id": "ecl_da64933ea13befed9483f8b00447bb3f4ceb74324c01e2515eff03f3af3ca750",
  "complete": false,
  "unknowns": { "resolved": 17, "investigating": 1 }
}
```

没有 C 编译器时，`runs` 会显示 `imported 17 recorded run records from run-evidence.json`，其余部分相同。

按这个顺序阅读 `code/main.py`：

1. `canonical_json()` 与 `compute_evidence_id()` 承载哈希配方和语义投影。
2. `create_evidence()` 与 `validate_evidence()` 负责构建和校验记录。
3. `EvidenceLedger.record()` 与 `import_bundle()` 存储记录并拒绝冲突。
4. `open_unknown()`、`update_unknown()` 与 `resolve_unknown()` 维护修订链。
5. `observe_object()` 读取 ELF，`derive_grade_model()` 读取表。
6. `claim_suite()`、`predict()` 与 `record_prediction()` 写入推断层记录。
7. `observe_runs()` 把每次 harness 运行转成观测记录。
8. `check_claim()`、`promote_claim()` 与 `completion_ledger()` 应用提升规则。
9. `AnswerCache` 像 rea 的快照那样为答案生成键，并拒绝过期的答案。

## 使用它

每个目标、每次调查各用一个台账。把目标的 SHA-256 放进每个主体，把问题的范围摘要放进每个未知项。更换工具时，改动 `provider`，让 id 自然分叉。重建 fixture 时，主体摘要会改变，所有旧记录都变成关于另一个文件的证据。

在你知道局限性的那一刻就写下来。不施加重定位的读取器要说明这一点，主机构建的运行要说明这一点，推断除数的模型也要说明这一点。一条没有写下局限性的记录，看起来比事实允许的更强，后来的读者无法弥补。

把每一段提供方文本都当作数据。来自目标的字符串和伪代码，是目标所控制的文本。它们永远不能变成指令，也永远不能变成对行为的观察。rea 6.5.0 用 `_meta.provenance.untrusted_content: true` 标记这类结果。它的服务器指令要求模型把目标内容当作数据，并把其中任何要求运行、安装、下载或发送内容的请求报告为提示注入尝试。这个标记有帮助，但你的台账仍必须让目标文本远离指令区。

## 交互实验

### 实验 A：重新计算一个 rea id

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code
python3 -q
```

运行：

```python
from main import *
record = load_evidence_file(FIXTURES / "rea-evidence-grade.json")
record["evidence_id"] == compute_evidence_id(record)
record["subject"]["local_path"] = "/home/you/re/ledger-x86_64-O1.o"
record["evidence_id"] == compute_evidence_id(record)
record["limitations"].pop()
record["evidence_id"] == compute_evidence_id(record)
```

前两个比较为真，因为路径不在哈希中。第三个为假，因为局限性也是内容。

### 实验 B：拒绝一次提升

```python
ledger, report = build_ledger(live_runs=False)
claim = [c for c in claim_suite(ledger.find(operation="derive_switch_model")[0], ledger.find(operation="derive_arithmetic_model")[0], ledger.find(operation="read_source")[0]) if c.claim_id == "grade(55)"][0]
promote_claim(ledger, claim, [ledger.find(operation="derive_switch_model")[0]["evidence_id"]])
```

该调用抛出 `ClaimError` 并指出那个派生 id。现在传入运行 id：

```python
run = run_records_for(ledger, "grade", [55])[0]
promote_claim(ledger, claim, [run["evidence_id"]])["status"]
```

状态为 `pass`，返回的 `limitations` 带有关于主机构建的那句话。

### 实验 C：拒绝过期快照

```python
import json
snapshot = json.loads(REA_SNAPSHOT_PATH.read_text())
target, binding = rea_snapshot_binding(snapshot)
AnswerCache(target, binding).import_snapshot(snapshot)
stale = dict(binding["analysis_profile"]["parameters"], jump_table_evidence="typed-case-default-blocks-v2")
AnswerCache(target, {"provider": binding["provider"], "analysis_profile": make_profile(binding["provider"], stale)}).import_snapshot(snapshot)
```

第一次导入返回 2。第二次抛出带有 `profile_mismatch` 的 `CacheError`。只改了一个版本字符串，整个快照就被拒绝。

### 实验 D：打断哈希链

```python
unknown_id = [u for u, r in ledger.unknowns.items() if r[-1]["status"] == "investigating"][0]
ledger.verify_unknown_chain(unknown_id)
ledger.unknowns[unknown_id][0]["severity"] = "critical"
ledger.verify_unknown_chain(unknown_id)
```

第一次调用返回 2。第二次抛出 `EvidenceIntegrityError`，因为修订版 1 不再匹配自己的摘要，而修订版 2 指向一个已不存在的摘要。

## 实践练习

关闭那个仍开启的未知项，并加入第二个提供方。

要求：

- 扩展 `run_grid()`，让范围声明缺失的九个输入运行，然后确认该未知项在修订版 3 变为 `resolved`。
- 把 `llvm-objdump -d` 加为第二个提供方，给它独立的 `provider` 对象，并把它对 `grade` 的指令清单记录为 `observed`。
- 写一条 `derived` 记录，声明 Ghidra 与 llvm-objdump 的两份指令清单解码出相同的 14 条指令，并链接两者。
- 为 `grade_label(105)` 开启一个严重程度为 `high` 的未知项（因为表索引 4 读取 `labels[4]`），并用一次运行解决它。
- 用 `export_bundle()` 导出台账，导入一个全新的台账，并证明每条链仍然通过校验。

成功的标准是：报告中每一句话都引用证据包中的某个 id，且那条 `unknown` 声明消失。

## 评分实验

五个关于本课 fixture 的问题。用你构建的代码找出每个答案，然后写入 `code/lab_answers.json`。数字写成数字或十六进制字符串（如 `"0x40"`），列表写成 JSON 列表。

1. `t1`：`compute_evidence_id` 对 rea-evidence-grade.json 给出的证据 id。
2. `t2`：grade 记录中局限性的条数。
3. `t3`：grade 记录的 relationship_coverage 状态。
4. `t4`：grade 记录的 confidence 与 authority，写成两个字符串的列表。
5. `t5`：grade 记录主体的 SHA-256。

检查你的答案：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/lab.py
```

检查器会基于已提交的 fixture，用 `code/main.py` 计算每个期望值。它为每个任务打印 `PASS`、`FAIL` 或 `TODO`，并给出得分；只有五项全部通过时才以 0 退出。`python3 code/lab.py my-answers.json` 可以检查另一个文件。`code/tests/test_lab.py` 证明检查器接受正确答案、拒绝错误答案。

### 观看运行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?symbols=off) 这台机器维护一个证据台账，有四个托盘：observed、derived、inferred 和 unknown。这个链接打开的是关闭符号表的运行，因此函数名和函数末尾被归档为未知项，开头部分被归档为推断。

## 课程产物

本课产出 `outputs/skill-evidence-ledger.md`。它告诉 Agent 如何把每个结论写成带置信度、权威来源、局限性和链接的记录，如何把未知项保存为修订版本，以及一条声明在什么情况下是 `pass`、`fail`、`unknown` 或 `unsupported`。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这 28 项测试证明：

- 两个 rea `evidence_id`、profile 摘要以及两个快照 `query_id` 都能从内容重新计算
- 改变路径后 id 不变，改变局限性后 id 改变
- 被篡改的记录、profile 提供方不匹配、浮点数以及未知枚举值都会被拒绝
- 重复记录返回 `duplicate`，冲突记录抛出异常，证据包导入是原子的
- 已记录运行证据包导入 18 条记录且没有悬空链接
- 未知项从修订版 1 开启，拒绝过期的 `expected_revision`，且只能由置信度为 observed、权威来源为 `controlled-replay` 的证据解决
- 修改过的修订版会打断哈希链
- 提升会拒绝推断、派生、空以及无关证据，并在有相符运行时通过
- 部分覆盖为 `unknown`，不符的运行为 `fail`
- 缓存对精确键命中，对其他参数未命中
- 缓存拒绝被提升的 profile 版本、错误目标、错误格式版本和被篡改的键
- ELF 读取器在偏移、大小、字节和重定位上与 `llvm-objdump` 一致，表读出 `1, 2, 3, 4, 4`
- 预测使用 C 除法，已记录的运行与全部 17 条单输入声明一致
- 有编译器时，实时运行同样一致

## 生产失败模式

| 故障 | 薄弱报告的说法 | 台账要求 |
|---|---|---|
| 把伪代码读成行为 | 「grade 对不及格分数返回 0」 | 该输入对应的 `execute_function` 记录，否则为 `unsupported` |
| 信任反编译器签名 | 「percent_change 返回 16 字节」 | 把签名标记为提供方的推断，并带上其 `low` 置信度 |
| 把空列表读成不存在 | 「该函数没有调用者」 | `unknown` 状态，或工具提供覆盖字段时记录该字段 |
| 把路径算进 id | 复制后说「记录变了」 | 不含 `name` 与 `local_path` 的语义投影 |
| 副本丢了局限性 | 一条看起来比源更强的记录 | id 不匹配，导入时被拒绝 |
| 网格只覆盖一部分却按全部算 | 「90 到 100 的分数全部通过」 | `unknown`，`covered` 为 11 中的 2 |
| 把主机运行说成目标文件的运行 | 「x86-64 目标文件返回了 25」 | 声明上携带主机构建的局限性 |
| 用一句话关闭未知项 | 「已解决，见上文」 | 一条 `verified` 解决结果，指名具有所需权威来源的观测 id |
| 覆盖过期修订版 | 两位分析者的编辑发生冲突 | `revision-conflict` 并重新读取 |
| 发布后重放旧快照 | 来自已变更契约的答案 | 导入时 `profile_mismatch` |
| 重命名后仍保留缓存 | 隐藏元数据写入的重放 | 派发前失效，并让导出失败 |

## 操作规则

把你读到的、你计算出的和你预期的内容，分别记为三种记录，用三个词区分。关闭未知项要用一个 id，绝不用一句话。只在与存储时完全相同的键下重放答案；只要有一个版本字符串不同，就拒绝它。

## 延伸阅读

- [rea 6.5.0 源码：`src/domain/evidence.ts` 与 `src/domain/residualUnknown.ts`](https://github.com/morluto/rea)
- [RFC 8785，JSON 规范化方案（JSON Canonicalization Scheme）](https://www.rfc-editor.org/rfc/rfc8785)
- [Ghidra 12.1.4 发布说明](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Agent Failure Patterns in CTF Reverse Engineering, NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)
- [When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering, arXiv 2607.12507](https://arxiv.org/abs/2607.12507)
- [Agent 反编译研究，arXiv 2606.06838](https://arxiv.org/abs/2606.06838)
- [When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370](https://arxiv.org/abs/2609.05370)
- 本课描述的 rea 6.5.0 源码（标签 `3b79af2`）：`src/domain/evidence.ts`、`.agents/skills/reverse-engineer-anything/SKILL.md`、`docs/cli.md`、`src/domain/reconstructionVerification.ts`、`src/application/ReconstructionObligationCandidates.ts`、`docs/tool-design.md`、`src/domain/evidenceCompletionLedger.ts`、`src/domain/residualUnknown.ts`、`src/application/investigation/UnknownEvidence.ts`、`.agents/skills/reverse-engineer-anything/references/evidence-workflows.md`、`src/application/binary/AnalysisSnapshotCache.ts`、`src/domain/analysisSnapshot.ts`、`src/application/binary/BinarySession.ts`、`src/application/binary/BinarySessionRecords.ts`、`docs/mcp-contracts.md`。
