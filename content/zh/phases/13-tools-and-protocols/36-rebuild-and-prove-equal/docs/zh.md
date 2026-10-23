# 重建并证明等价：差分测试、字节比较与证据闭环

> 测试通过只能证明在你的测试上一致；要证明等价，需要一个覆盖每个边界的输入网格，以及在重定位掩码之后仍然匹配的字节。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 34 (lifting to pseudocode), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~110 minutes

## 学习目标

- 根据指令清单编写重建版本，并注明它所依据的证据记录。
- 构建输入网格，覆盖每个风险类别：零值保护、负值、截断除法、32 位溢出和除法陷阱。
- 在同一个程序中运行原始版本与重建版本，并记录每个单元，包括陷阱。
- 用原始编译选项重新编译重建版本，在掩蔽重定位字段后比较函数字节。
- 读取重定位指向的常量并加以比较，使错误的字面量无法蒙混过关。
- 用一个精简的 stdio 客户端驱动 rea 的 MCP 服务器，并用证据 id 记录每一步。
- 铸造一条失败即关闭的证明记录，其 id 是对其声明和证据链接的哈希。

## 问题

你向 rea 索取 fixture `ledger-x86_64-O1.o` 中 `percent_change` 的函数档案。Ghidra 12.1.4 返回了伪代码，智能体据此写出了这段 C：

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (int)(((long)after - before) * 100 / before);
}
```

四个测试通过：`(100, 150)` 得到 50，`(200, 100)` 得到 -50，`(0, 5)` 得到 0，`(50, 100)` 得到 100。智能体还把 `grade` 重建为针对 90、80、70 和 60 的四条 `if` 语句。又有六个测试通过。报告写着「verified」。

这份报告回答不了下面任何一个问题：

- 当 `(after - before) * 100` 超出 32 位时，原始版本返回什么？
- 分数为 110 时，原始版本返回什么？
- 重建版本编译后是否与原始版本相同的 16 个字节？
- 当两个编译后的函数字节相同时，它们的重定位是否指向相同的常量？
- 重建版本来自哪条证据记录？智能体读的是指令清单还是伪代码？
- 在没有 C 编译器的主机上，证明会得到什么状态？

本课中的每个目标都是我们自己编写并编译的程序。第 32 课关于授权范围与法律的规则依然适用。

```figure
re-36-prove-equal
```

## 证据怎么说

`code/main.py` 中的智能体循环通过 stdio 对 rea 6.5.0 发起了六次调用。记录下来的运行保存在 `code/fixtures/runs/agent-run.json`。服务器以协议版本 `2025-11-25` 回应 `initialize`。其中三次调用返回了 Evidence 记录：

| 步骤 | 工具 | 耗时 | 证据 id |
|---|---|---|---|
| 2 | `analyze_function percent_change` | 6.31 s | `ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd` |
| 3 | `analyze_function grade` | 0.02 s | `ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1` |
| 4 | `read_bytes 0x1000b4, 20` | 0.00 s | `ev_c26146cfedf4c1ee146d8ff00cfa9a5d3f5335bdd9c97ccf4e1af5f6ea3ed6bc` |

第一次调用承担了 Ghidra 导入和自动分析的开销。每条记录都带有 `confidence: "observed"`、`authority: "shipped-artifact"` 以及目标文件的 SHA-256，即 `9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f`。Ghidra 以 `.text` 位于 `0x100000` 的方式加载可重定位目标文件，因此 `percent_change` 位于 `0x100000`，`grade` 位于 `0x100010`。

`percent_change` 的伪代码正是 `long` 的来源：

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{
  long lVar1;
  undefined1 auVar3 [16];
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  ...
}
```

反编译器看到 `cltd` 与 `idivl` 同时写入 `edx` 和 `eax`，于是建模出 16 字节返回值和一个 64 位除法。同一条记录中的指令清单说得更直白：

```
0x100000: TEST EDI, EDI
0x100002: JZ 0x0010000d
0x100004: SUB ESI, EDI
0x100006: IMUL EAX, ESI, 0x64
0x100009: CDQ
0x10000a: IDIV EDI
0x10000c: RET
0x10000d: XOR EAX, EAX
0x10000f: RET
```

每个操作都是 32 位宽。`IMUL EAX, ESI, 0x64` 在 32 位内相乘并保留低半部分。`CDQ` 把 `eax` 符号扩展到 `edx`，`IDIV EDI` 用 32 位除数去除这个 64 位对。按照指令清单重建，就是 `int` 类型下的 `100 * (after - before) / before`。

`grade` 记录形状相同。伪代码给出一个表 `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`，以及一个由 `0x66666667` 构成的难以理解的默认表达式。指令清单把除以 10 展示为 `IMUL RAX, RAX, 0x66666667`、`SAR RCX, 0x22`、`SHR RAX, 0x3f`、`ADD ECX, EAX`。接着 `ADD ECX, -0x6`、`CMP ECX, 0x4` 与 `JA` 把十位限制在 6 到 10。比较之前的 `XOR EAX, EAX` 就是默认值 0。`LEA RCX, [0x1000b4]` 与 `MOV EAX, dword ptr [RCX + RAX*0x4]` 读取一张含五个 4 字节条目的表。

步骤 4 读取了这 20 个字节：`01000000 02000000 03000000 04000000 04000000`。对于十位 6、7、8、9、10，表是 `1, 2, 3, 4, 4`。这正是 `ledger.c` 中的 `switch`，其中 `case 10` 与 `case 9` 共享取值 4。

rea 自己的证据页面中的两个证明，都是从指令清单和对常量的 `read_bytes` 开始的。DX-Ball 最初的伪代码是 `FUN_00406400(void)`，没有栈输入，是指令视图纠正了它。先读指令清单。引用记录 id，然后写 C。

## 编写候选实现

`code/fixtures/candidates/` 目录中有八个重建实现，每个文件一个函数。其中四个遵循指令清单或编译出相同代码；另外四个看似合理但实际错误，或者错在某个手写测试能发现的地方。

| 文件 | 来源 | 行为 |
|---|---|---|
| `percent_change_mul_first.c` | `100 * (after - before) / before` | 指令清单的 C 写法 |
| `percent_change_split.c` | `(after * 100 - before * 100) / before` | 文本不同，经 clang 折叠后字节相同 |
| `percent_change_wide.c` | `(int)(((long)after - before) * 100 / before)` | 伪代码的 `long`，在回绕之前做除法 |
| `percent_change_div_first.c` | `(after - before) / before * 100` | 先做除法，因此舍入不同 |
| `grade_switch.c` | 带五个分支的 `switch (score / 10)` | 原始源码 |
| `grade_table.c` | `unsigned slot = score / 10 - 6; slot < 5 ? points[slot] : 0` | 指令清单的 C 写法 |
| `grade_thresholds.c` | `if (score >= 90) return 4; ...` | 最高分段没有上界 |
| `grade_case10_wrong.c` | `case 10: return 3;` | 一个表项写错 |

每个文件都用 fixture 命令编译：`clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c`。编译后的目标文件与源码一起提交，以便测试在没有 clang 的主机上也能运行。

## 第一条腿：执行并比较

McKeeman 在 1998 年对差分测试的定义要求「两个或更多可比较的系统」接受「机械生成的测试用例」，并把不同的结果、挂起或崩溃视为疑似缺陷。他点出了难点：「测试中最丑陋的问题，是评估测试的结果。」这里原始函数就是 oracle。我们不需要知道正确答案，只需要在覆盖每个风险的网格上，让原始版本与重建版本在任何地方都不出现分歧。

### 设计网格

`code/main.py` 中的 `input_grid()` 构建 92 个单元。对 `percent_change`，它取十个 `before` 值与七个 `after` 值的所有组合：

```
before: 0, 1, -1, 3, -3, 7, 100, -100, 2147483647, -2147483648
after:  0, 5, -5, 105, 536870911, 2147483647, -2147483648
```

对 `grade`，它取 22 个分数：每个分数段边界的两侧、负数、0、110，以及两个 `int` 极限。`classify()` 给每个单元打标签。溢出 32 个单元，负数 17 个，边界 14 个，普通 12 个，零值保护 7 个；负截断 5 个，舍入 2 个，超出 switch 2 个，陷阱 1 个。

每个类别都有存在的理由。零值保护单元检查 `TEST EDI, EDI` 分支。负数单元检查截断除法：C 和 `idiv` 向零舍入，所以 `-9 / 10` 是 0，`-10 / 10` 是 -1。舍入单元检查乘法发生在除法之前，因此 `(105 - 100) * 100 / 100` 是 5 而不是 0。溢出单元让 `(after - before) * 100` 超出 32 位。唯一的陷阱单元让回绕后的乘积等于 `INT_MIN`，同时 `before` 等于 -1。

Yang、Chen、Eide 和 Regehr 构建了 Csmith，用来生成没有未定义行为的程序。在未定义行为上出现的 oracle 分歧只是噪声，永远不能当作缺陷报告。有符号溢出在 C 中是未定义的。我们的 oracle 是一个具有固定行为的编译产物，因此溢出单元仍在范围内。这里的教训是分清哪些单元依据语言标准，哪些单元依据编译出的字节。

### 驱动程序与陷阱

`code/driver.c` 从 stdin 每行读取一个单元，并打印输入、原始结果和重建结果。`build_harness()` 用 `-Dpercent_change=cand_percent_change -Dgrade=cand_grade` 编译候选实现，使两个版本以不同名称链接进同一个程序。在 x86-64 Linux 主机上，原始版本就是 fixture 目标文件本身；在其他任何主机上，原始版本是本机编译的 `ledger.c`，报告会记录 `original_kind: fixture-source`。

驱动程序用 `sigsetjmp` 和 `siglongjmp` 安装一个 `SIGFPE` 处理程序。对于 `before = -1`、`after = 536870911`，差值为 536870912，32 位乘积回绕为 -2147483648。在 x86-64 上，用这个被除数和 -1 作为除数执行 `idiv` 会触发除法错误异常。商无法放进 32 位（Intel SDM 第 2 卷，IDIV）。处理程序把它转成记录值 `trap`。

在 AArch64 上，`sdiv` 不会陷入，同一组操作数返回 -2147483648。我们在 arm64 Mac 上的运行记录的正是这个值。oracle 依赖架构，报告会写明实际运行的是哪一种。

### 结果

`prove_behaviour(PLAUSIBLE)` 构建 `long` 版本和阈值版本。十个朴素用例通过，但网格在 28 个单元上失败：

| 函数 | 输入 | 类别 | 原始 | 重建 |
|---|---|---|---|---|
| `percent_change` | 3, 2147483647 | 溢出 | -133 | -1431655899 |
| `percent_change` | 100, 2147483647 | 溢出 | -101 | 2147483547 |
| `percent_change` | 2147483647, 5 | 溢出 | 0 | -99 |
| `grade` | 110 | 超出 switch | 0 | 4 |
| `grade` | 2147483647 | 超出 switch | 0 | 4 |

`percent_change` 的 26 个失败全部是 `|before| > 1` 的溢出单元。当 `before` 等于 1 或 -1 时，64 位商会截断回回绕后的值，因此这些单元只是碰巧一致。阈值版 `grade` 失败，是因为原始 switch 没有为十位 11 及以上的情况提供分支，而 `110 / 10` 等于 11。

`prove_behaviour(EQUAL)` 构建 `percent_change_mul_first.c` 和 `grade_switch.c`。全部 92 个单元一致。`percent_change_div_first.c` 在四个朴素用例中失败两个，这是手写测试确实能发现的那种情况。

`oracle_disagreements()` 还检查另一件事。`percent_change_model()` 与 `grade_model()` 是指令清单的 Python 模型，用 `wrap32()` 处理 32 位乘法，用 `idiv32()` 处理架构规则。报告在每个单元上比较模型与观测到的原始结果。零分歧意味着模型与二进制文件在每个单元上都一致。模型是派生证据，运行是观测证据，报告把它们保存在不同字段中。

## 第二条腿：重新编译并比较

网格上的行为是一条腿。第二条腿要问：用相同的编译器和选项编译重建版本，是否会产出相同的字节。DX-Ball 项目报告其辅助函数有「63 个匹配字节」。TH04 项目报告了 2,139 字节的匹配。它们各自使用的都是所属时代的编译器。rea 自己做不到这一点。它的 `compare_functions` 对提供方文本做行级差分，并附加局限性：「Assembly is opaque provider text; relocation normalization is unavailable.」

### 读取目标文件

`code/main.py` 中的 `ElfObject` 用 `struct.unpack_from` 读取一个 ELF64 可重定位目标文件。它检查魔数、64 位类别、小端数据、`e_machine` 62 和 `e_type` 1。然后遍历节头、字符串表、`.symtab` 以及每个 `.rela` 节。在 fixture 中，`percent_change` 从 `.text` 偏移 0 开始，大小 16；`grade` 位于 `0x10`，大小 46；`total` 从 `0x40` 开始，大小 31；`grade_label` 位于 `0x60`，大小 63。

`percent_change` 的 16 个字节是 `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`。该函数没有重定位。`grade` 有一条：函数内偏移 `0x26`，类型 `R_X86_64_PC32`，符号 `.rodata`，加数 `0x10`。它属于偏移 `0x23` 处那条 7 字节的 `leaq`，该指令的最后四个字节是位移字段。

### 掩蔽，然后比较记录

在 RELA 目标文件中，加数保存在重定位记录里，四个字段字节为零。候选文件 `grade_switch.o` 对应位置也是零。如果只做普通字节比较，即使两条重定位指向不同位置也会通过。`mask_relocations()` 仍然把两侧的字段清零，因此同一套方法也适用于 REL 目标文件和 Mach-O——后者的加数就放在字段里。

随后 `prove_bytes()` 按偏移、类型和目标节比较重定位记录。原始版本是 `.rodata+0x10`，候选版本是 `.rodata-0x4`。加数不同，是因为候选目标文件的 switch 表前面没有 `labels.rel` 表。不同的加数意味着不同的偏移，而不是不同的函数。

### 读取重定位指向的常量

对于结束指令的 PC 相对字段，CPU 会加上下一条指令的地址，因此目标为符号地址加加数再加 4。在原始版本中这是 `.rodata+0x14`，在候选版本中是 `.rodata+0x0`。`read_table()` 在那里读取五个 4 字节条目，两者都是 `1, 2, 3, 4, 4`。

数量五来自指令清单：`CMP ECX, 0x4` 配合 `JA` 允许索引 0 到 4，而 `[RCX + RAX*0x4]` 每个索引读取 4 字节。`code/main.py` 中的 `CLAIMS` 为 `grade` 记录了这一形状，并为 `percent_change` 记录零条重定位。

这一步之所以存在，是因为 TH08 项目记录过的一个缺陷。它较早的比较「把目标地址回填到指令的重定位字段，然后比较指令字节，却不检查该地址上的浮点字面量数据。因此源码中的 `0.0` 可能看起来与目标中的 `128.0` 完全一致。」修复后审计了 1,548 处浮点字面量引用，又纠正了十二处。`grade_case10_wrong.c` 就是我们对这个缺陷的复现。它的 46 个代码字节与原始版本相同，重定位记录形状相同，而表是 `1, 2, 3, 4, 3`。只有常量检查能让它失败。

### 结果

| 候选 | 大小 | 掩蔽后字节 | 重定位记录 | 表 | 状态 |
|---|---|---|---|---|---|
| `percent_change_mul_first` | 16 vs 16 | 相同 | 无 | 无 | pass |
| `percent_change_split` | 16 vs 16 | 相同 | 无 | 无 | pass |
| `percent_change_wide` | 16 vs 26 | 大小不同 | 无 | 无 | fail |
| `percent_change_div_first` | 16 vs 18 | 大小不同 | 无 | 无 | fail |
| `grade_switch` | 46 vs 46 | 相同 | 形状相同 | 两者均为 `1,2,3,4,4` | pass |
| `grade_table` | 46 vs 46 | 相同 | 形状相同 | 两者均为 `1,2,3,4,4` | pass |
| `grade_thresholds` | 46 vs 39 | 大小不同 | 1 vs 0 | 无 | fail |
| `grade_case10_wrong` | 46 vs 46 | 相同 | 形状相同 | `1,2,3,4,4` vs `1,2,3,4,3` | fail |

有两个结果值得再看一眼。`percent_change_split.c` 源码文本不同，字节却相同，因为 clang 把有符号溢出视为未定义，并把 `after * 100 - before * 100` 折叠成一次乘法。`grade_table.c` 也逐字节匹配，说明编译器从数组索引和从 `switch` 得到了同一张表。字节匹配证明：这个编译器、这组选项，从候选实现产出了相同的代码；它并不证明候选实现就是原始源码文本。

大小是诊断信息，不是验收标准。rea 的证据页面记录过一个 1,825 字节的候选，长度正确却有 467 个字节不同。`wide` 重建是 26 字节，因为 `movslq`、`imulq`、`cqto` 和 `idivq` 取代了 32 位形式。在编译之前，指令清单就已经告诉我们宽度。

## 通过 rea 运行的智能体循环

rea 的 MCP 工具中有 65 个没有对应的 CLI 命令，`verify_reconstruction` 就是其中之一。`code/main.py` 中的 `McpStdioClient` 是能访问这些工具的最小客户端。它启动 `npx -y rea-agents@6.5.0 mcp`，每行向 stdin 写一个 JSON-RPC 请求，并在线程中从 stdout 每行读取一个 JSON 对象。它先发送带协议版本 `2025-11-25` 的 `initialize`，再发送 `notifications/initialized`，然后发送 `tools/call`。每次调用的默认截止时间是 330 秒，与 rea 为 Ghidra 冷启动预留的时间相同。

`agent_loop()` 发起这六次调用并逐一记录：

```json
{
  "step": 2,
  "tool": "analyze_function",
  "arguments": {"procedure": "percent_change"},
  "seconds": 6.31,
  "is_error": false,
  "evidence_id": "ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd",
  "confidence": "observed",
  "authority": "shipped-artifact",
  "limitations": ["Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals.", "..."],
  "result_sha256": "..."
}
```

完整的 Evidence 记录以各自的 id 为键保存在同一个文件里。重建所依据的伪代码和指令清单也是 fixture 的一部分。`table_address()` 在 `grade` 的指令清单中找到 `LEA` 操作数，并把它交给 `read_bytes`。`open_binary`、`get_evidence_bundle` 和 `close_binary` 不返回 Evidence 记录，因此以 null id 记录。

当 `npx` 在 PATH 上且 `GHIDRA_INSTALL_DIR` 指向一个目录时，`run_agent(live=True)` 会驱动 rea；否则它重放 fixture，并在 stderr 上说明。测试更进一步。`code/tests/fake_rea_server.py` 是一个替身服务器，对每次 `tools/call` 用相同工具和参数的已记录结果作答。每次测试都会让客户端和循环对着它运行，无需安装 rea、Ghidra 或 Java。

### rea 的验证器能说什么、不能说什么

rea 的 `verify_reconstruction` 接收指向比较 Evidence 的声明，并对每条声明返回 pass、fail 或 unknown。它固定的局限性写着：「Pass means every declared claim passed; it does not establish global implementation equivalence.」关于函数的声明比较来自 `compare_functions` 的提供方文本；行为声明比较来自 `compare_process_captures` 的两份已记录进程捕获。当源记录的调用覆盖不完整时，关于调用的声明为 `unknown`。rea 内部没有任何东西会编译重建后的函数，或在你的输入网格上运行它。本课的两条腿，正是 rea 留给你完成的部分。

rea 的义务台账期望一条证明记录，带有 `predicate_type: "rea.reconstruction-proof"`、`operation: "verify_reconstruction_obligations"`、`confidence: "observed"` 以及 `normalized_result.passed === true`。公开的 rea 工具中没有任何一个会产出它。`code/main.py` 中的 `build_proof()` 负责铸造：

```json
{
  "predicate_type": "rea.reconstruction-proof",
  "operation": "verify_reconstruction_obligations",
  "confidence": "observed",
  "authority": "controlled-replay",
  "normalized_result": {
    "passed": true,
    "status": "pass",
    "claims": [
      {"claim_id": "percent_change.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "percent_change.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "grade.referenced-constants", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]}
    ]
  },
  "evidence_links": ["ev_e88b10dc...", "ev_67b6d3f9..."],
  "evidence_id": "ev_..."
}
```

id 是 `ev_` 加上对去掉 id 后的记录做键排序的紧凑 JSON 再取 SHA-256。`aggregate()` 是失败即关闭的：`fail` 优先，其次是 `unknown`，空的声明列表为 `unknown`。在没有 C 编译器的主机上，两条行为声明是 `unknown`，因此即使两条字节声明都通过，整条记录仍为 `unknown`。`authority` 是 `controlled-replay`，因为证明来自我们自己的 harness，而不是单独来自目标文件。

## 研究怎么说

Liu、Raff 和 Micinski 在 2026 年测量了九种配置下的八个反编译系统。通过了基准全部测试的候选实现，在模糊输入上仍然出现分歧。总体比例为 4.9%，对某个系统最高达到 13%。在 300 个 GitHub 库函数和 287 个有 CVE 依据的函数上，最强的精化模型把 Ghidra 的构建率从 75% 提升到 90%；而在同样的函数上，行为匹配率从 74% 降到 62%。他们把原因追溯到凭空捏造的字段、类型、被调用者和保护分支——这些替换掉了传统反编译器留下的可见未知项。

我们的 `percent_change_wide.c` 就是这类捏造的类型之一。伪代码给出 `long`，重建版本沿用，结果能编译、通过十个测试，却在 26 个溢出单元上失败。阈值版 grade 则是捏造的保护分支。按 LLM4Decompile 的度量，两者都算「可重新执行」——该度量只要求重新编译输出并运行基准的断言。Ghidra 自己的伪 C 在 HumanEval-Decompile 上通过了这些断言中的 20.12%。

DecompileBench 测量了来自 130 个真实程序的 23,400 个函数。它发现 LLM 方法在功能正确性上比工业反编译器低 52.2%，但可理解性得分更高。

这一领域使用的三种证明，正是你可以构建的三条腿。Phoenix 和 DREAM 重新编译输出并运行原程序的测试。McKeeman 和 Csmith 用生成的输入对原始版本进行测试，并让输入保持在已定义行为之内。Ramos 和 Engler 用有界符号执行检查两个 C 函数，证明在输入规模上限内所有被探索路径上的等价性。本课构建前两种。第三种对应第 35 课证据台账中的实践练习，适用于网格太小而不足以信任的情况。

## 构建它

运行仅用标准库的 harness：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
```

clang 在 PATH 上时，演示会编译两组候选，并各运行 92 个单元。它会重新编译八个候选，重放智能体运行，并铸造证明记录。没有 clang 时，它读取已提交的目标文件，并把行为声明报告为 `unknown`。

在 arm64 Mac 上的预期输出大致如下：

```json
{
  "host": "aarch64 Darwin",
  "originalKind": "fixture-source",
  "gridCells": 92,
  "plausible": {"naivePassed": true, "gridMismatches": 28, "oracleDisagreements": 0},
  "equal": {"naivePassed": true, "gridMismatches": 0, "oracleDisagreements": 0},
  "legTwo": {
    "percent_change_mul_first": {"status": "pass", "reasons": []},
    "percent_change_wide": {"status": "fail", "reasons": ["size 16 vs 26 bytes"]},
    "grade_case10_wrong": {"status": "fail", "reasons": ["referenced constants differ: [1, 2, 3, 4, 4] vs [1, 2, 3, 4, 3]"]}
  },
  "agentRun": {"mode": "replayed fixture", "server": {"name": "rea", "version": "6.5.0", "protocolVersion": "2025-11-25"}},
  "proof": {"status": "pass", "evidence_id": "ev_..."}
}
```

按这个顺序阅读 `code/main.py`：

1. `wrap32()`、`c_div()`、`idiv32()` 以及两个 `*_model()` 函数承载指令清单的整数语义。
2. `input_grid()` 与 `classify()` 构建并标记 92 个单元。
3. `build_harness()` 与 `execute()` 把原始版本、重命名后的候选和 `driver.c` 编译成一个程序，然后运行这些单元。
4. `prove_behaviour()` 返回第一段报告，包含不匹配项与模型分歧。
5. `ElfObject` 读取头、符号和 `.rela` 记录；`mask_relocations()` 与 `read_table()` 为比较做准备。
6. `prove_bytes()` 返回第二段报告，包含大小、掩蔽区间、重定位形状和表。
7. `McpStdioClient` 与 `agent_loop()` 驱动 rea 并记录每一步。
8. `build_proof()` 与 `aggregate()` 铸造失败即关闭的记录。

## 使用它

要证明一个新函数等价，重复以下步骤：

1. 获取该函数的 `analyze_function`，记录证据 id。先读指令清单，再读伪代码。
2. 按指令清单中的操作宽度编写候选实现。保留原始函数名。
3. 列出指令清单中的每一次乘、加、减、除。为每个操作按其宽度添加会回绕的网格输入。
4. 为清单中每个比较常量的两侧添加输入。对 `grade`，这些是减法之后的 6 和 4。
5. 添加每个在目标上可能陷入的除数，并记录运行网格的架构。
6. 运行两条腿，先读理由，再看状态。
7. 把证明记录放在候选实现旁边。它的 `evidence_links` 告诉读者该声明依据哪些记录。

任何编译器更新后都要重跑两条腿。字节匹配只是关于某一个编译器、某一组选项的陈述。

## 交互实验

### 实验 A：模型与运行

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal/code
python3 -q
```

运行：

```python
from main import *
percent_change_model(3, INT_MAX)
percent_change_model(-1, 536870911, "x86_64")
percent_change_model(-1, 536870911, "aarch64")
report = prove_behaviour(PLAUSIBLE)
report.naive_passed, len(report.mismatches), report.mismatches[0]
```

第一次调用得到 -133，即回绕后的乘积除以 3。第二次得到 `'trap'`，第三次得到 -2147483648。报告显示十个朴素用例通过、28 个网格单元失败。把 `report.oracle_disagreements` 与空列表比较。

### 实验 B：读取目标文件

```python
obj = ElfObject(ORIGINAL_OBJECT)
grade = obj.function("grade")
grade.size, grade.data[0x26:0x2a].hex(), grade.relocations
read_table(obj, grade.relocations[0], 5, 4)
```

字段字节是 `00000000`。重定位指名 `.rodata`，加数为 `0x10`，表读出 `[1, 2, 3, 4, 4]`。

### 实验 C：破坏一个常量

```python
report = prove_bytes("grade", CANDIDATES / "grade_case10_wrong.o")
report.masked_equal, report.relocations_match, report.tables_equal, report.reasons()
```

字节匹配，记录匹配，表不匹配。现在修改 `grade_case10_wrong.c`，让 `case 10` 重新返回 4，用 fixture 命令重新编译，观察状态变为 `pass`。

### 实验 D：驱动 rea

在安装好 rea 6.5.0 和 Ghidra 12.1.4 之后：

```bash
GHIDRA_INSTALL_DIR=/opt/homebrew/Cellar/ghidra/12.1.4/libexec \
JAVA_HOME=/opt/homebrew/opt/openjdk@21 \
python3 code/main.py --live
```

第一次 `analyze_function` 会花上几秒，因为 Ghidra 要导入目标文件。把新的证据 id 与已提交的运行比较。相同的字节、相同的分析 profile 会得到相同的 id，因为 id 是排除了文件路径的内容哈希。

## 实践练习

把两条腿都扩展到同一目标文件中的 `total` 和 `grade_label`。

要求：

- 对 `total`，输入是一个数组和一个计数。构建计数为 0、负计数、单个元素，以及在 32 位处回绕的求和等单元。
- 对 `grade_label`，该函数内联调用 `grade`，并读取 `labels.rel`——一张含五个相对字符串偏移的表，它们各自的重定位位于 `.rela.rodata`。
- 为 `grade_label` 添加一条 `CLAIMS` 条目，包含两条重定位和一张五个 4 字节条目的表，并比较条目指向的字符串。
- 记录模型与运行是否存在分歧的单元（如果有），并根据指令清单逐一解释。
- 为每个函数的每条腿向证明记录各添加一条声明，并让聚合保持失败即关闭。

成功的标准是：证明记录列出四个函数，每条声明都带证据 id，且当某个标签字符串不同时 `grade_label` 失败。

## 评分实验

五个关于本课 fixture 的问题。用你构建的代码找出每个答案，然后写入 `code/lab_answers.json`。数字写成数字或十六进制字符串（如 `"0x40"`），列表写成 JSON 列表。

1. `t1`：x86_64 上 `percent_change_model(3, 105)` 的结果。
2. `t2`：x86_64 上 `percent_change_model(0, 5)` 的结果。
3. `t3`：`grade_model(100)` 的结果。
4. `t4`：`input_grid()` 中单元的数量。
5. `t5`：x86_64 模型发生陷阱的网格单元数量。

检查你的答案：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/lab.py
```

检查器会基于已提交的 fixture，用 `code/main.py` 计算每个期望值。它为每个任务打印 `PASS`、`FAIL` 或 `TODO`，并给出得分；只有五项全部通过时才以 0 退出。`python3 code/lab.py my-answers.json` 可以检查另一个文件。`code/tests/test_lab.py` 证明检查器接受正确答案、拒绝错误答案。

### 观看运行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) 这台机器以伪代码收尾，并把它归档为推断：页面并没有证明它。本课正是把这一声明变成可核对结论的那一步。

## 输出技能

本课产出 `outputs/skill-rebuild-and-prove-equal.md`。用它把函数证据和候选重建变成两条腿的证明以及一条证明记录。该技能要求先读指令清单再读伪代码，网格覆盖每个风险类别，做重定位掩蔽并检查常量，以及失败即关闭的聚合。

## 验证它

运行演示和测试套件：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这 35 项测试证明：

- `c_div()` 向零截断，`wrap32()` 遵循二进制补码
- 模型对 `(3, INT_MAX)` 给出 -133，在 x86-64 上对陷阱单元给出 `trap`，在 AArch64 上给出 -2147483648
- 网格有 92 个单元并覆盖每个风险类别，而朴素用例避开了所有困难类别
- ELF 读取器在记录的偏移和大小上找到四个函数
- `percent_change` 为 16 字节且无重定位，`grade` 在 `0x26` 处有一条 `R_X86_64_PC32`
- 重定位字段字节为零，在这个 RELA 目标文件中掩蔽不会改变任何东西
- 八个已提交候选都达到预期状态，并带有记录的理由
- `grade_case10_wrong` 通过掩蔽后字节比较，只在常量检查上失败
- 已记录的第一段运行显示十个朴素用例通过、28 个不匹配、零个模型分歧
- 有编译器时，两组候选都以相同结果重建并重跑
- 智能体运行按 SHA-256 定位到已提交的目标文件，步骤有序，每个 Evidence 步骤都有 `ev_` id
- `read_bytes` 记录与 ELF 读取器找到的表一致
- 客户端驱动替身服务器，并记录与真实运行相同的证据 id
- 来自服务器的错误结果会写入运行记录
- `aggregate()` 是失败即关闭的，证明 id 可以从其内容重新计算

## 生产失败模式

| 故障 | 薄弱报告的说法 | harness 必须证明什么 |
|---|---|---|
| 重建沿用反编译器的 `long` | 「十个测试通过」 | 26 个溢出单元与原始版本不同 |
| 用阈值替换 switch | 「每个分数段都通过」 | 110 及更大的分数在原始版本中返回 0 |
| 先除后乘 | 「公式在代数上相等」 | 舍入单元不同，朴素用例失败 |
| 跳过陷阱单元 | 「没有崩溃」 | x86-64 触发 `#DE`，AArch64 返回 `INT_MIN`，报告写明主机 |
| 不做掩蔽直接比较字节 | 「字节相同」 | 重定位记录指向不同偏移 |
| 只掩蔽不读常量 | 「掩蔽后字节相同」 | `1,2,3,4,3` 与 `1,2,3,4,4` 不同 |
| 把大小当作验收 | 「长度相同」 | 相同长度内部存在不同字节 |
| 编译器或选项不同 | 「字节不同，重建有误」 | 同一源码产出不同字节，比较无效 |
| 把模型当作观测 | 「模型一致」 | 运行是观测，模型是派生，分歧会被列出 |
| 声明没有证据 id | 「已验证」 | 每条声明都链接其覆盖函数的 `ev_` 记录 |
| CI 主机没有编译器 | 「通过」 | 行为声明为 `unknown`，聚合为 `unknown` |
| 把伪代码当作源码 | 「反编译器说是 long」 | 指令清单显示 `IMUL EAX`、`CDQ`、`IDIV EDI` |

## 操作规则

依据指令清单编写重建版本，并注明它来自哪条记录。在让每个中间结果都回绕、并跨过每个比较常量的网格上，把它与原始版本一起运行。用原始选项重新编译，掩蔽重定位字段，比较记录，并读取它们指向的常量。为每条声明记录证据 id，让未经验证的声明保持未知。

## 延伸阅读

- [McKeeman, Differential Testing for Software, Digital Technical Journal 10(1), 1998](https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf)
- [Yang, Chen, Eide, Regehr, Finding and Understanding Bugs in C Compilers, PLDI 2011](https://doi.org/10.1145/1993498.1993532)
- [Ramos, Engler, Practical, Low-Effort Equivalence Verification of Real Code, CAV 2011](https://doi.org/10.1007/978-3-642-22110-1_55)
- [Schwartz, Lee, Woo, Brumley, Native x86 Decompilation Using Semantics-Preserving Structural Analysis, USENIX Security 2013](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370, 2026](https://arxiv.org/abs/2609.05370)
- [Gao et al., DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Tan et al., LLM4Decompile, arXiv 2403.05286, 2024](https://arxiv.org/abs/2403.05286)
- [System V gABI，第 4 章，重定位](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)
- [x86-64 psABI 仓库，重定位类型](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, IDIV](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm Architecture Reference Manual for A-profile, DDI 0487, SDIV](https://developer.arm.com/documentation/ddi0487/latest)
- [rea 6.5.0 MCP 契约](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/mcp-contracts.md)
- [rea 6.5.0 重建义务台账](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/reconstruction-obligation-ledgers.md)
