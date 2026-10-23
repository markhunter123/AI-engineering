# 恢复数据与类型：结构体、数组、C++ 对象与 DWARF

> 反编译器在有类型表时打印 `a->balance`，没有类型表时打印 `*(int *)(param_1 + 4)`。本课从访问模式中构建这张表，再从 DWARF 5 中读出真正的表，并展示两者在哪些地方一致。

**Type:** Build
**Languages:** Python, C, C++, x86-64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~140 minutes

## 学习目标

- 按 Intel SDM 的寻址规则，把结构体成员访问读作基址寄存器加位移，把数组访问读作基址加索引乘以步长。
- 把 System V 对齐规则手工应用到 `struct account`，并解释每一个填充字节。
- 构建一个类型恢复 pass，把反汇编清单中的访问转换为字段偏移、宽度、种类、步长和大小，并为每一项给出证据。
- 为可重定位目标文件的 `.debug_info`、`.debug_abbrev`、`.debug_str_offsets` 和 `.debug_addr` 编写 DWARF 5 读取器，并与 `llvm-dwarfdump` 对照检查。
- 比较恢复出的布局与 DWARF 布局，把每一处差异归类为证据缺失、访问合并或无法证明的种类。
- 从重定位中读取 Itanium C++ 虚表、其 typeinfo 及其修饰名，并把虚调用解析到某个槽位。

## 问题

你在不带 `-g` 的情况下编译了 `account.c`，然后问一个 Agent `bump` 做了什么。Agent 把目标文件交给 Ghidra 12.1.4，得到如下结果：

```c
void bump(long param_1,int param_2)
{
  *(int *)(param_1 + 4) = param_2 + *(int *)(param_1 + 4);
  return;
}
```

同一份源码用 `-g` 编译，同一个反编译器给出：

```c
int bump(account *a,int by)
{
  a->balance = by + a->balance;
  return a->balance;
}
```

两个目标文件里的指令完全相同。差别在于一张类型表。有了这张表，反编译器知道第一个参数指向一个 32 字节的结构体，其偏移 4 处的字段是一个名为 `balance` 的 4 字节 `int`。没有这张表，反编译器只知道有一个 8 字节宽的东西经 `%rdi` 传入，并且 `+4` 处的 4 个字节被读取和写入过。

那张表里的每样东西都有来源。偏移 4 来自 `addl 0x4(%rax), %ecx` 的位移字节。宽度 4 来自 `l` 后缀。名字来自 `.debug_info` 里的一个 `DW_TAG_member` 条目。本课构建这两半：一个恢复指令所能证明之内容的 pass，和一个解码编译器所记录内容的读取器。两者之间的差距，就是类型名的代价。

```figure
re-41-struct-recovery
```

## 成员访问就是一次位移

Intel SDM 第 1 卷第 3.7.5 节把有效地址定义为 `Offset = Base + (Index * Scale) + Displacement`，其中 scale 为 1、2、4 或 8，displacement 为 8、16 或 32 位。第 3.7.5.1 节补充了 64 位模式下的形式 `RIP + Displacement`。结构体成员访问使用第一种形式且不带索引：基址寄存器保存对象的地址，位移就是成员偏移。数组访问增加了索引项，元素大小由 scale 或一次单独的乘法承担。

以 `-O0` 下的 `bump` 为例。`llvm-objdump -d -r` 给出的清单如下：

```text
       4: 48 89 7d f8                  	movq	%rdi, -0x8(%rbp)
       e: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      12: 03 48 04                     	addl	0x4(%rax), %ecx
      15: 89 48 04                     	movl	%ecx, 0x4(%rax)
```

按第 33 课的规则解码 `03 48 04`。`03` 是 `ADD r32, r/m32`。ModRM `48` 是 `01 001 000`：`mod = 01` 表示带 8 位位移的内存操作数，`reg = 001` 是 `%ecx`，`r/m = 000` 是 `%rax`。位移字节是 `04`。该指令读取 `%rax + 4` 处的 4 个字节并把它们加到 `%ecx`。`llvm-objdump` 打印的 `l` 后缀是操作数大小，因此该字段宽 4 字节。`%rax` 来自栈帧槽位 `-0x8(%rbp)`，而该槽位接收了 `%rdi`，即 System V 调用约定下的第一个整数参数。由此得到三件事：第一个参数是指针，它指向的对象在偏移 4 处有一个 4 字节字段，且 `bump` 读写该字段。

这些字节里没有任何东西说明它是 `int`、`balance` 或 `struct account`。也没有任何东西说明对象有多大。结构体其余的 28 个字节对 `bump` 不可见。

## System V 布局规则

x86-64 psABI 第 3.1.2 节给出三条聚合类型规则。「结构体和联合体采用其对齐要求最严格的成员的对齐。每个成员被分配到满足相应对齐的最低可用偏移。任何对象的大小始终是该对象对齐的整数倍。」同一文档的图 3.1 给出标量的大小和对齐：`char` 为 1 和 1，`short` 为 2 和 2，`int` 为 4 和 4，`double` 为 8 和 8，任何指针为 8 和 8。

把这些规则应用到 `account.c` 中的结构体：

```c
struct account {
    char kind;
    int balance;
    short code;
    double rate;
    const char *label;
};
```

| 成员 | 大小 | 对齐 | 能容纳的最低偏移 | 前置填充 |
|---|---|---|---|---|
| `kind` | 1 | 1 | 0 | 0 |
| `balance` | 4 | 4 | 4 | 3（字节 1 到 3） |
| `code` | 2 | 2 | 8 | 0 |
| `rate` | 8 | 8 | 16 | 6（字节 10 到 15） |
| `label` | 8 | 8 | 24 | 0 |

最后一个成员结束于 32。最严格的对齐是 8，而 32 是 8 的倍数，因此大小为 32，没有尾部填充。32 个字节中有 9 个是填充，psABI 说「任何填充的内容都是未定义的」。

成员的顺序决定大小。同样的五个成员按 `double, pointer, int, short, char` 排列，依次占据偏移 0、8、16、20、22，结束于 23，向上取整到 24。`code/main.py` 里的 `sysv_layout()` 应用这三条规则并给出两种结果。Python 的 `ctypes` 模块通过宿主编译器的约定应用同样的规则；对同一个结构体，`ctypes.sizeof`、`ctypes.alignment` 以及每个字段的 `.offset` 给出的分别是 32、8 和 `0, 4, 8, 16, 24`。在这台机器上构建并运行时，程序本身会打印第三份结果：

```text
sizeof 32 align 8
kind 0 balance 4 code 8 rate 16 label 24
sum 315 tag -3 projected 318.75 label_length 7
```

那次本机构建是 arm64。Arm AAPCS64 在第 5.10.1 节陈述了相同的聚合类型规则，Apple 的 arm64 文档保留了这五种标量大小，因此两个目标架构上的数字一致。本课的 x86-64 目标文件用 `-target x86_64-unknown-linux-gnu` 编译，以便第 33 课的 ELF 读取器和下面的 DWARF 读取器处理它们。

## 手工读出结构体的其余部分

`bump` 只看到一个字段。`account.c` 中其余四个函数看到了其他字段，每个函数展示了不同的指令形式。下面的地址都来自 `account-x86_64-O0-g.objdump.txt`。

`sum_balances` 对结构体数组做索引：

```text
      51: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      55: 48 63 4d ec                  	movslq	-0x14(%rbp), %rcx
      59: 48 c1 e1 05                  	shlq	$0x5, %rcx
      5d: 48 01 c8                     	addq	%rcx, %rax
      60: 8b 40 04                     	movl	0x4(%rax), %eax
```

`%rax` 是 `list`，`%rcx` 是符号扩展为 64 位的 `i`，`shlq $0x5` 把它乘以 32。`addq` 之后，`%rax` 为 `list + 32 * i`，加载读取元素 `i` 偏移 4 处的 4 个字节。移位次数是这个目标文件中结构体大小唯一以数字形式出现的地方。包含这些指令的循环从 `0x49` 运行到 `0x72` 处的 `jmp 0x49`。

`tag` 读取两个窄字段：

```text
      8c: 0f be 00                     	movsbl	(%rax), %eax
      98: 0f bf 48 08                  	movswl	0x8(%rax), %ecx
```

`movsbl` 加载偏移 0 处的一个字节并做符号扩展，因此偏移 0 处的字段是有符号 1 字节整数。`movswl` 以符号扩展加载偏移 8 处的两个字节，因此偏移 8 处的字段是有符号 2 字节整数。在 `-O1` 下，第一次访问变成 `cmpb $0x6c, (%rdi)`，它仍然说明偏移 0 处是 1 字节，但对符号只字未提。

`projected` 读取一个浮点数：

```text
      cc: f2 0f 2a 40 04               	cvtsi2sdl	0x4(%rax), %xmm0
      d5: f2 0f 59 40 10               	mulsd	0x10(%rax), %xmm0
```

`cvtsi2sdl` 把内存中的有符号 32 位整数转换为 double，这确认偏移 4 处是有符号 4 字节整数。`mulsd` 乘以来自内存的 double，因此偏移 16 处是 8 字节浮点值。

`label_length` 读取一个指针，然后沿指针访问：

```text
      f3: 48 8b 40 18                  	movq	0x18(%rax), %rax
      f7: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
      fb: 80 3c 08 00                  	cmpb	$0x0, (%rax,%rcx)
```

`movq` 读取偏移 24 处的 8 个字节。单看这一点，它可能是 8 字节整数，也可能是指针。`cmpb` 把加载的值用作内存操作数的基址。被当作地址使用的值就是指针。`cmpb` 从它那里按索引 `%rcx` 读取 1 个字节；在 `-O1` 下，指针在循环内通过 `leaq 0x1(%rcx), %rcx` 前进，因此所指向的元素相隔 1 字节。

把五个函数放在一起，结构体就是 `{ i8 at 0, i32 at 4, i16 at 8, f64 at 16, pointer at 24 }`，步长为 32。1 到 3 以及 10 到 15 处的空隙，正是 psABI 规则预测的填充。指令没有说明的是：成员的名字、指针是 `const char *`、偏移 0 处有符号字节的声明类型（源码写作 `char`），以及五个函数接收的是同一类型。最后一点来自 `main`，它把一个栈帧对象传给全部五个函数。在 `-O0` 下，调用点让这一点可见：`leaq -0x70(%rbp), %rdi`，随后在 `callq bump` 之前有 `addq $0x20, %rdi`，在 `callq tag` 之前有 `addq $0x40, %rdi`。偏移 0、`0x20`、`0x40` 是同一个数组的三个元素，而填充它的 `0x60` 字节的 `memcpy` 是 96 字节，即三个 32 字节的元素。

## 分析 pass

`code/main.py` 中的 `trace_function()` 按地址顺序遍历一个函数的指令，并维护一个小型抽象状态。一个 holder 是一个 64 位寄存器或一个栈帧槽位，例如 `-0x8(%rbp)`。每个 holder 取三种值之一。`Pointer` 由一个根（root）、一个常量偏移和一组步长构成，其中每个步长记录「已加上该字节数的未知倍数」。`Index` 是值未知、且为某个 scale 倍数的整数。其余一切视为无。

初始状态下，六个 System V 参数寄存器分别持有 `Pointer("param0", 0)` 到 `Pointer("param5", 0)`。之后每条指令应用一条规则：

- 64 位寄存器与栈帧槽位之间的 `movq` 复制值。对寄存器做 32 位或更窄的写入会把它变成 scale 为 1 的 `Index`。
- 带指针基址的 `leaq` 产生一个加上位移的指针；带索引寄存器时，它把 `scale * index_scale` 加到步长集合中。基址为 `%rbp` 或 `%rsp` 的 `leaq` 命名一个栈帧对象，`main` 就是这样传递 `rows` 的。
- 对 `Index` 执行 `shlq $n` 会把它的 scale 乘以 `2^n`，`imulq $k` 则乘以 `k`。
- 指针与索引相加的 `addq` 保留指针并把索引 scale 加入步长集合。对指针执行 `addq $k` 会移动其偏移，同时也会被记录为该寄存器的一次自增。
- 一次调用会让所有调用者保存（caller-saved）寄存器变为未知，持有指针的参数寄存器会被记录为一个调用点。
- 任何基址寄存器持有指针的内存操作数都会变成一个 `Access`，带上指针的根与偏移、来自助记符的宽度与种类，以及是否为存储的标志。

宽度和种类只由助记符决定。`access_width()` 把 `b`、`w`、`l`、`q` 后缀映射为 1、2、4、8 字节，`movsbl` 与 `movswl` 映射为有符号窄加载，`movzbl` 映射为无符号，`sd` 与 `ss` 标量 SSE 指令映射为 8 字节和 4 字节浮点数，`cvtsi2sdl` 映射为有符号 4 字节整数源，16 字节的 `movaps` 系列映射为向量。`leaq` 和 `nop` 形式不算访问。

```python
if memory.base in FRAME_REGISTERS:
    return Pointer(f"frame({memory.base}{format_signed(memory.displacement) if memory.displacement else ''})", 0)
base = env.get(memory.base)
if not isinstance(base, Pointer):
    return Unknown
strides = set(base.strides)
if memory.index is not None:
    index_value = env.get(memory.index)
    strides.add((index_value.scale if isinstance(index_value, Index) else 1) * memory.scale)
return Pointer(base.root, base.offset + memory.displacement, frozenset(strides), base.origin)
```

通过指针加载 8 个字节会产生一个新指针，其根命名了它来自的字段，例如 `param0->0x18`。当这个值后来被用作基址，或用 `leaq` 前进时，它来自的字段就被标记为已证明的指针。正是这条规则把偏移 24 从「8 字节」变成「指针」。

### 步长的四类证据

`infer_layout()` 合并每个见证函数的访问，并从四个来源收集步长，每个来源都附带产生它的指令。

| 种类 | 来源 | 展示它的构建 |
|---|---|---|
| `scale` | 一个 scale 大于 1 的 `Index` 到达了指针，来自 `shlq $0x5` 或 `imulq` | `-O0`，`0x59` 处的 `shlq $0x5, %rcx` |
| `loop-step` | 循环内部对基址或索引寄存器的 64 位自增，且该循环同时包含这次访问 | `-O1`，`0x24` 处的 `addq $0x20, %rdx` |
| `offset-delta` | 同一循环内通过同一基址寄存器的同宽度访问，偏移之间相隔一个常数 | `-O2`，八个相隔 32 的加载 |
| `call-site` | 同一个栈帧对象被传给各见证函数时的偏移 | `-O0`，`main` 传递 `rows`、`rows+0x20`、`rows+0x40` |

一个循环是一次向后的 `jmp` 或条件分支；只有当循环范围同时包含该自增和该访问时，这次自增才计数。这条规则排除了 `-O0` 下推进 `i` 的 `addl $0x1, %eax`，因为它是 32 位加法，而且访问不经过 `%eax`。最终步长是所发现全部值的最大公约数。当步长不小于最后一个字段的结束位置时，大小取步长。

### -O2 下的向量化循环

在 `-O2` 下，clang 对 `sum_balances` 做向量化。`0x50` 到 `0xaa` 的循环体每轮读取八个元素：

```text
      3b: 48 8d 87 e4 00 00 00         	leaq	0xe4(%rdi), %rax
      50: 66 0f 6e 50 80               	movd	-0x80(%rax), %xmm2
      55: 66 0f 6e 98 60 ff ff ff      	movd	-0xa0(%rax), %xmm3
      61: 66 0f 6e 90 40 ff ff ff      	movd	-0xc0(%rax), %xmm2
      69: 66 0f 6e a0 20 ff ff ff      	movd	-0xe0(%rax), %xmm4
      7d: 66 0f 6e 10                  	movd	(%rax), %xmm2
      81: 66 0f 6e 58 e0               	movd	-0x20(%rax), %xmm3
      8a: 66 0f 6e 50 c0               	movd	-0x40(%rax), %xmm2
      8f: 66 0f 6e 60 a0               	movd	-0x60(%rax), %xmm4
      a0: 48 05 00 01 00 00            	addq	$0x100, %rax
      aa: 75 a4                        	jne	0x50 <sum_balances+0x40>
```

`%rax` 从 `list + 0xe4` 开始。八个位移把加载放在相对 `list` 的偏移 `0x04, 0x24, 0x44, 0x64, 0x84, 0xa4, 0xc4, 0xe4` 处，即 `4, 36, 68, 100, 132, 164, 196, 228`。自增量是 `0x100`，即 256。一个见到偏移就记录的 pass 会报告八个字段和一个至少 232 字节的结构体。清单没错，结论错了。

`fold_unrolled()` 应用一条规则。在一个循环内，取通过同一基址寄存器的同宽度访问。它们的偏移相隔 `d`，这里是 32。如果循环步长为 `k * d`，且恰好有 `k` 个不同偏移，那么循环被展开了 `k` 次，每次访问都是同一元素在偏移 `(offset - lowest) mod d` 处的访问。这里 `256 = 8 * 32`，共有八个偏移，因此八次加载都折叠到偏移 4，步长证据记录 `offset-delta 32`、`unrolled 32` 和 `loop-step 256`。`0xe0` 处的标量收尾循环从 `shlq $0x5, %rdx` 得到 `scale 32`，从 `addq $0x20, %rdx` 得到 `loop-step 32`。全部五个值的最大公约数是 32。

这张图有一个关闭折叠的按钮。关掉后，`-O2` 构建会显示字节 32 之后的七个字段，且没有大小。这就是只信位移、不看循环的读者看到的状态。

## 从字节读 DWARF 5

`account.c` 的 `-g` 构建带有八个调试节。`.debug_info` 为 493 字节，`.debug_abbrev` 179，`.debug_str` 230，`.debug_str_offsets` 124，`.debug_addr` 112，`.debug_line` 312，`.debug_line_str` 12，`.debug_frame` 264。DWARF 5 标准的引言把 `.debug_str_offsets`、`.debug_addr` 和 `.debug_line_str` 列为第 5 版新增的节。`code/main.py` 中的读取器需要前五个。

### 单元头

标准第 7.5.1.1 节给出完整的编译单元头：`unit_length`、`version`、`unit_type`、`address_size` 和 `debug_abbrev_offset`。`.debug_info` 的前十二个字节是：

```text
e9 01 00 00   unit_length = 0x1e9 = 489
05 00         version = 5
01            unit_type = DW_UT_compile
08            address_size = 8
00 00 00 00   debug_abbrev_offset = 0
```

`unit_type` 字节是第 5 版新增的，标准在同一节里说明了这一点。长度只计算自身之后的字节，因此下一个单元会从 `4 + 489 = 493` 开始，也就是该节的末尾。`llvm-dwarfdump --debug-info` 在其 `Compile Unit:` 行打印同样的五个值，第一个测试解析该行并做比较。

### 缩写表

第 2.1 节说，每个调试信息条目（DIE）「由一个标识标签和一系列属性组成」。条目本身只存储一个缩写代码。该代码在 `.debug_abbrev`（第 7.5.3 节）中选中一条声明，这条声明给出标签名、说明是否跟随子条目，并按 form 列出每个属性。这个目标文件有 14 条声明。这里涉及其中两条：

```text
[12] DW_TAG_structure_type  DW_CHILDREN_yes
     DW_AT_name              DW_FORM_strx1
     DW_AT_byte_size         DW_FORM_data1
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
[13] DW_TAG_member           DW_CHILDREN_no
     DW_AT_name              DW_FORM_strx1
     DW_AT_type              DW_FORM_ref4
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
     DW_AT_data_member_location  DW_FORM_data1
```

`parse_abbreviations()` 依次读取 ULEB128 编码的 code、ULEB128 的 tag、一个 children 字节，然后成对读取属性与 form，直到出现 `0, 0` 对为止。第 7.5.6 节给出 form 的编码：`data1` 是一个字节，`ref4` 是相对单元起始的 4 字节偏移，`strx1` 是字符串偏移表中的 1 字节索引，`sec_offset` 和 `data4` 是 4 字节，`addrx` 是地址表中的 ULEB128 索引，`exprloc` 是一个 ULEB128 长度加上相应字节数，`flag_present` 不占任何字节。

### 手工解码 balance 成员

`account` 的结构体条目从偏移 `0x195` 开始，字节为 `0c 15 20 00 03`。缩写代码是 12，即结构体。`DW_AT_name` 是字符串索引 `0x15`，`DW_AT_byte_size` 是 `0x20` = 32，`DW_AT_decl_file` 是 0，`DW_AT_decl_line` 是 3，即源码中 `struct account {` 所在的行。

第二个成员条目从 `0x1a3` 开始，字节为 `0d 10 88 01 00 00 00 05 04`：

1. `0d` 是缩写 13，一个成员。
2. `10` 是作为 `strx1` 的 `DW_AT_name`：字符串索引 16。
3. `88 01 00 00` 是作为 `ref4` 的 `DW_AT_type`：从单元起始算起的偏移 `0x188`，即名为 `int` 的 `DW_TAG_base_type` 条目。
4. `00` 是 `DW_AT_decl_file`，`05` 是 `DW_AT_decl_line`：`account.c` 的第 5 行。
5. `04` 是 `DW_AT_data_member_location`：偏移 4。

第 5.7.6 节说，成员条目「可以带有 DW_AT_data_member_location 属性或 DW_AT_data_bit_offset 属性」，并且整数常量「是从包含实体起始处算起的字节偏移」。这个字节与 `addl 0x4(%rax), %ecx` 中的位移是同一个数字，而 `compare_layouts()` 正是把两者并排放在一起的函数。

### 字符串和地址经过两张表

字符串索引 16 并不是 `.debug_str` 中的偏移。读取器从单元条目取 `DW_AT_str_offsets_base`，其值为 8，然后读取 `.debug_str_offsets` 中 `8 + 4 * 16` 处的 4 字节条目。该条目保存 `"balance"` 在 `.debug_str` 中的偏移。地址也经过同样的间接。`sum_balances` 的条目以字节 `07 07` 开头：缩写 7，一个子程序，然后是作为 `addrx` 索引 7 的 `DW_AT_low_pc`。`.debug_addr` 中 `8 + 8 * 7` 处的 8 字节条目是 `0x30`，即该函数在 `.text` 中的地址。该表的前六个条目是六个字符串字面量的地址。

在可重定位目标文件中，两张表都还不持有最终值。`.debug_str_offsets` 中的每个条目在文件里都是零，`.rela.debug_str_offsets` 持有 29 条针对 `.debug_str` 的 `R_X86_64_32` 重定位，真实偏移放在加数（addend）中。`.debug_addr` 有 13 条针对 `.text` 和 `.rodata.str1.1` 的 `R_X86_64_64` 重定位。`.debug_info` 自身有四条重定位：缩写偏移、作为 `.debug_str_offsets+0x8` 的 `DW_AT_str_offsets_base`、`DW_AT_stmt_list`，以及作为 `.debug_addr+0x8` 的 `DW_AT_addr_base`。读取器解析某个节之前，`ElfObject.relocated()` 会对 32 位和 64 位类型应用 `S + A`。节符号在目标文件中的值为 0，因此字段得到的就是加数。`llvm-dwarfdump` 做同样的解析，所以它的输出中每个索引都已解析成名字和地址。

### 类型、函数与位置

`DwarfInfo.type_name()` 沿 `DW_AT_type` 引用前进。`label` 成员指向一个 `DW_TAG_pointer_type`，后者指向一个 `DW_TAG_const_type`，后者指向 `char` 基类型，因此名字组合为 `const char *`。`type_size()` 对基类型和结构体返回 `DW_AT_byte_size`，对指针返回 8，对数组返回元素个数乘以元素大小。`type_kind()` 把基类型的 `DW_AT_encoding` 归约为带符号的 `int` 或 `float`，把指针类型归约为 `pointer`，这正是比较所用的词汇表。

`DwarfInfo.functions()` 读取六个 `DW_TAG_subprogram` 条目。`bump` 从 `0x0` 运行到 `0x21`，即 `DW_AT_low_pc` 加上 `DW_AT_high_pc` 长度。它的栈帧基址是 `DW_OP_reg6 RBP`，其参数 `a` 的位置表达式是 `91 78`：`DW_OP_fbreg` 加上 SLEB128 值 `0x78`，即 -8。这就是清单在地址 `0x4` 处把 `%rdi` 存入的栈帧槽位 `-0x8(%rbp)`。调试信息和指令从两侧描述了同一个槽位。

## 比较两种布局

`compare_layouts()` 遍历 DWARF 成员，并在相同偏移处寻找推断出的字段。没有对应字段的成员标记为 `no-evidence`。大小不同的字段标记为 `size-mismatch`。指针成员对应的字段从未被解引用，标记为 `pointer-unproven`。种类不一致，例如指令用的是整数运算而这里却是浮点数，标记为 `kind-mismatch`。起始位置落在某个成员内部或填充内部的推断字段标记为 `unexpected`。大小则用步长与 `DW_AT_byte_size` 比较。

三个构建的数字，来自 `python3 code/main.py`：

| 构建 | 经第一个参数的访问数 | 字段数 | 步长证据 | 匹配的成员 | 大小 |
|---|---|---|---|---|---|
| `-O0` | 11 | 5 | `scale 32`, `call-site 32`, `call-site 64` | 5/5 | 32，匹配 |
| `-O1` | 9 | 5 | `loop-step 32` | 5/5 | 32，匹配 |
| `-O2` | 17 | 折叠后 5，折叠前 12 | `offset-delta 32`, `unrolled 32`, `loop-step 256`, `scale 32`, `loop-step 32` | 5/5 | 32，匹配 |

在每个构建中，推断出的空隙都是字节 1 到 3 和 10 到 15，而这些正是 DWARF 布局中的填充字节。每个推断字段都位于自身大小的整数倍处，这是 psABI 规则的要求，`natural_alignment_ok()` 会记录这一点。

逐个加入函数，图景就会变化。只有 `bump` 时，布局是偏移 4 处的一个字段，大小未知，下界为 8。`sum_balances` 加入步长。`tag` 加入偏移 0 和 8。`projected` 加入偏移 16 处的浮点数。`label_length` 加入偏移 24 处的指针，并证明它是指针。`main` 在 `-O0` 下加入调用点，在 `-O1` 和 `-O2` 下什么也不加，因为编译器内联了所有调用。图会按这个顺序为每个构建逐步展示。

有三件事仍然够不着。名字不在指令里。限定符 `const` 不在指令里。`label` 的指向类型只能从字符串循环中的 `leaq 0x1(%rcx), %rcx` 得知「元素相隔 1 字节」，这与 `char`、`unsigned char` 和 `_Bool` 都相容。在 `-O1` 下，偏移 0 处字节的符号未知，因为 `cmpb $0x6c, (%rdi)` 比较时不扩展。pass 在那里报告 `int8`，而在 `movsbl` 做符号扩展的 `-O0` 下报告 `i8`。这些就是第 35 课记录为未知的事实。

## C++ 对象

`shapes.cpp` 定义了 `Shape`，它有一个虚析构函数、一个纯虚函数 `area()` 和一个虚函数 `name()`；还定义了 `Rect`，它派生自 `Shape` 并新增两个 `int` 成员。在 x86-64 Linux 上，clang 的布局由 Itanium C++ ABI 规定。

### 名字修饰

ABI 第 5.1.2 节说「名字的修饰方式是在其编码前加 `_Z`」。这个目标文件有 19 个这样的符号。`_ZNK4Rect4areaEv` 读作：`_Z`，然后 `N` 表示嵌套名，`K` 表示 const 成员函数，`4Rect` 和 `4area` 是带长度前缀的源码名字，`E` 结束嵌套名，`v` 表示空参数列表。`llvm-cxxfilt` 打印 `Rect::area() const`。`_Z8describeRK5Shapei` 是 `describe(Shape const&, int)`：`R` 是引用，`K` 是 const，`5Shape` 是类，`i` 是 `int`。第 5.1.4 节中的特殊名字是：`_ZTV` 表示虚表，`_ZTI` 表示 typeinfo，`_ZTS` 表示 typeinfo 名称字符串。构造函数和析构函数是 `C1`、`C2`、`D0`、`D1` 和 `D2`，因此 `_ZN4RectD0Ev` 与 `_ZN4RectD2Ev` 都打印为 `Rect::~Rect()`：删除析构函数和基对象析构函数共享同一个源码名字。

`code/main.py` 中的 `Demangler` 覆盖了这个子集，外加第 5.1.5.2 节的内置类型编码和第 5.1.10 节的 `S_` 替换。测试对全部 19 个名字运行它，并与捕获的 `llvm-cxxfilt --no-strip-underscore` 输出比较。这个标志在 macOS 上很重要：该工具默认会剥掉一个前导下划线，随后就无法识别 `Z8describe...`。

### 从重定位读虚表

ABI 第 2.5.2 节按顺序列出虚表的组成部分：虚调用偏移、虚基类偏移、到顶部的偏移（offset to top）、typeinfo 指针，然后是虚函数指针。地址点（address point），即对象的 vptr 所持有的值，是第一个函数指针的地址（第 2.6.2 节）。`Rect` 没有虚基类，因此它的表只有到顶部的偏移、typeinfo 和函数指针。

符号 `_ZTV4Rect` 为 0x30 字节，六个 8 字节槽位。在目标文件中每个字节都是零。内容在 `.rela.data.rel.ro._ZTV4Rect` 中：

```text
0000000000000008 R_X86_64_64              _ZTI4Rect
0000000000000010 R_X86_64_64              _ZN4RectD2Ev
0000000000000018 R_X86_64_64              _ZN4RectD0Ev
0000000000000020 R_X86_64_64              _ZNK4Rect4areaEv
0000000000000028 R_X86_64_64              _ZNK4Rect4nameEv
```

槽位 0 保存到顶部的偏移，其值为 0，不需要重定位。槽位 1 是 typeinfo。地址点在 `+16` 处。从那里开始，槽位 0 是完整对象析构函数，槽位 1 是删除析构函数，槽位 2 是 `area`，槽位 3 是 `name`。`read_vtable()` 读取符号大小、其范围内的重定位以及偏移 0 处的零。`Shape` 的表形状相同，只是槽位 2 是 `__cxa_pure_virtual`，即纯虚调用会到达的运行时函数。

### RTTI

第 2.9.4 节把 `std::type_info` 描述为一个 vptr，后跟一个指向修饰类型名的指针，并列出了派生类。`_ZTI4Rect` 有三条重定位：`_ZTVN10__cxxabiv120__si_class_type_infoE+0x10`、`_ZTS4Rect` 和 `_ZTI5Shape`。第一条是 `__si_class_type_info` 的虚表，ABI 用这个类表示只有一个位于偏移 0 的公有非虚基类的类型，`+0x10` 是该虚表的地址点。第二条指向字符串 `4Rect`，即修饰类型名，`type_info` 上的 `name()` 返回的正是这个字符串。第三条是基类。`_ZTI5Shape` 使用 `__class_type_info`，没有基类指针。`read_typeinfo()` 返回这三项事实。

### 虚调用

`-O0` 下的 `describe` 做了一次虚调用：

```text
       f: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
      13: 48 8b 07                     	movq	(%rdi), %rax
      16: ff 50 10                     	callq	*0x10(%rax)
```

第一个 `movq` 加载对象指针。第二个读取对象偏移 0 处的 8 个字节，即 vptr，因为 ABI 把它放在带有虚函数的类的首位。`callq` 经过 vptr 之后的 `0x10`，而 `0x10 / 8` 就是槽位 2。`find_virtual_calls()` 匹配这个模式并报告两个候选：来自 `_ZTV4Rect` 的 `Rect::area() const`，以及来自 `_ZTV5Shape` 的 `__cxa_pure_virtual`。实际运行哪一个取决于对象，这正是该调用是间接调用的原因。

DWARF 记录了同一个槽位。两个类中 `area` 的 `DW_TAG_subprogram` 条目都带有 `DW_AT_vtable_elem_location (DW_OP_constu 0x2)`，`name` 带有槽位 3，析构函数带有槽位 0。`Rect` 是一个 16 字节的 `DW_TAG_structure_type`，它在偏移 0 处有一个针对 `Shape` 的 `DW_TAG_inheritance` 条目，成员 `w` 在偏移 8，`h` 在偏移 12。`Shape` 为 8 字节，有一个人工成员 `_vptr$Shape`。两条独立的推导都给出 `area` 位于槽位 2：重定位表和调试条目。

在 `-O1` 下，该目标文件根本没有虚表。`main` 把 `Rect` 构造为局部对象，因此编译器在编译期解析了 `r.name()` 和 `describe(r, 2)` 并将它们内联。没有任何地方取该表的地址，因此这张表不会被生成。`describe` 仍然包含 `movq (%rdi), %rax` 和 `callq *0x10(%rax)`，因为它的参数可以是任何 `Shape`。虚表缺失只是某一个目标文件的事实，测试 `test_the_optimised_object_keeps_the_call_pattern_but_drops_the_vtables` 同时检查这两半。

Microsoft ABI 在每一个细节上都不同。修饰名以 `?` 开头并编码调用约定，因此按 C++ `__cdecl` 编译的 `void test(void)` 是 `?test@@ZAXXZ`。RTTI 定位器位于 `vfptr[-1]`，是一个 `_RTTICompleteObjectLocator`，符号 `??_7` 表示 vftable，`??_R4` 表示定位器。微软自己的文档说「编译器厂商之间没有 C++ 命名的标准」。本课的样本文件只针对 Itanium。

## 工具如何呈现恢复出的类型

下面每一行给出具体输出的，都是在本机产生并捕获到 `code/fixtures/tools/` 的。未安装的工具所在行只描述其文档，并注明这一点。

| 工具 | 版本与许可证 | 类型存放位置 | 在我们的样本上 |
|---|---|---|---|
| Ghidra | 12.1.4, 2026-09-21, Apache-2.0 | Data Type Manager，含内置、用户定义和派生类型；`.gdt` 归档；程序使用的类型会被复制进去 | 带 DWARF：`/DWARF/account.c/account size=32 align=8` 以及 `a->balance = by + a->balance;`。不带：`*(int *)(param_1 + 4)`，参数为 `long param_1`，另有 `tag(char *param_1)` 和 `*(char **)(param_1 + 0x18)` |
| IDA Pro | 9.4, 2026-07-13，商业软件；IDA Free 为非商业版，仅提供面向 x86 的云端反编译器 | Local Types 窗口、`.til` 类型库、把 DWARF 和 PDB 导入 `dwarf` 或 `pdb` 文件夹 | 未安装。仅依据文档 |
| Binary Ninja | 6.0, 2026-09-03，商业软件，另有免费版 | 类型编辑、头文件导入导出、类型归档、来自 DWARF 与 PDB 的调试信息、类型库、平台类型 | 未安装。仅依据文档 |
| radare2 | 6.2.4, 2026-10-05, LGPLv3 | `ts`、`tsc`、`afvd` | 在 `bin.dbginfo=true` 时，对可重定位目标文件，`ts` 和 `tsc account` 没有任何输出；`pdf` 显示 `mov dword [rax + 4], ecx`，并带有来自 DWARF 的源码行注释 |
| rizin | 0.9.1, 2026-06-29，LGPL 核心 | `ts`、`tsc`、`afvd` | `tsc account` 从 DWARF 打印出 `struct account { char kind; int balance; short code; double rate; const char *label; };`；不带 `-g` 的构建只列出其自带的头文件类型 |
| lldb | lldb-1703.0.236.21 | `image lookup -t` | `byte-size = 32` 以及来自 ELF 目标文件的完整 `compiler_type` 文本 |
| gdb | 此处未安装 | `ptype /o` 打印「结构体中字段的偏移和大小」，并标出空洞 | 仅依据文档 |
| pyelftools | 0.33，公有领域 | 基于 `.debug_info` 的 Python API | 一个 43 行的脚本打印同样的五个成员和 `unit version 5` |
| llvm-dwarfdump | Apple LLVM 17.0.0 | `--debug-info`、`--debug-abbrev`、`--debug-str-offsets` | 每个读取器测试的判定基准 |

Ghidra 不带 DWARF 的输出展示了同样两条规则在起作用。它的反编译器把 `char *` 赋给 `tag` 的参数，因为该函数通过它比较一个字节；把 `char **` 赋给 `0x18` 处的字段，因为 `label_length` 解引用加载的值并比较字节。这正是本课 pass 应用的两条规则：宽度来自指令，指针来自解引用。Ghidra 的反编译器文档说，p-code varnode「不一定关联着数据类型」，反编译器通过传播「最终赋予一个正式的数据类型」。「Infer constant pointers」选项把同样的传播扩展到常量。Ghidra 12.1 为反编译器增加了位域恢复，本课的样本文件没有涉及它。

手工制作类型表是故事的另一半。Ghidra 的 Data Type Manager、IDA 的 Local Types、Binary Ninja 的类型归档以及 rizin 的 `td` 都允许你根据恢复出的布局给结构体定类型，然后看反编译器重新渲染。pass 告诉你该输入什么，以及哪些字段是你在猜。

rea 6.5.0 及其 Ghidra provider 把反编译器文本作为记录返回。Ghidra 那一行的所有内容都适用于它，而记录携带类型名，却不携带产生它们的证据。对同一份清单运行 `python3 code/main.py infer`，就能得到那些证据。

## 类型推断研究

这里的 pass 是类型推断系统中负责生成约束的那一半，且仅限于从一个参数可达的一个结构体。这个领域已经走得更远。

- TIE（Lee、Avgerinos、Brumley，NDSS 2011）把类型重建表述为在二进制 IR 上生成并求解约束，度量了保守性与精确性，并在 87 个 coreutils 程序上与一个商业反编译器做了比较。
- Retypd（Noonan、Loginov、Cok，PLDI 2016）增加了子类型、多态和递归类型，并报告以 98% 的召回率重建指针上的 `const` 标注。GrammaTech 名下有一个开源实现。
- OSPREY（Zhang 等，IEEE S&P 2021）把每个内存位置视为随机变量，推断最可能的类型。
- DIRTY（Chen 等，USENIX Security 2022）和 ReSym（Xie 等，CCS 2024）用学习到的模型对反编译器输出做后处理，ReSym 报告在完整结构体布局上达到 72.9% 的精确率。
- RecStruct（Chen 等，USENIX Security 2026）无需反编译器即可恢复嵌套结构体，使用流不敏感符号执行和对栈槽位的并查集。

栈上的结构体正是本课 pass 首先跳过的东西。在 `-O1` 下，`main` 把 `rows` 放在栈上，内联代码读取 `0x24(%rsp)`、`0x40(%rsp)`、`0x48(%rsp)`、`0x30(%rsp)` 和 `0x38(%rsp)`。这些偏移对 32 取模分别是 4、0、8、16、24：同一个结构体，只是没有参数可以作种子。实践练习要求你恢复它。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
```

输出是一份包含五个部分的 JSON 报告。`dwarf` 保存 `-O0` 目标文件的单元头、结构体和函数。`oracles` 保存 psABI 规则布局、`ctypes` 布局、DWARF 布局、本机运行结果以及一个 `agree` 标志。`builds` 保存三个目标文件各自的访问、推断布局和比较结果。`cxx` 保存反修饰后的名字、两个虚表、两条 typeinfo 记录、虚调用以及 DWARF 槽位。`tools` 列出捕获的工具输出。

预期形状：

```json
{
  "dwarf": {"unit": {"unitLength": 489, "version": 5, "unitType": "DW_UT_compile", "addressSize": 8, "abbrevOffset": 0}, "...": "..."},
  "oracles": {"agree": true, "psabiRules": {"size": 32, "align": 8, "...": "..."}, "...": "..."},
  "builds": {
    "O0": {"layout": {"stride": 32, "size": 32, "gaps": [1, 2, 3, 10, 11, 12, 13, 14, 15], "...": "..."}, "comparison": {"matches": 5, "members": 5, "...": "..."}},
    "O1": {"...": "..."},
    "O2": {"...": "..."}
  },
  "cxx": {"virtualCalls": [{"slot": 2, "candidates": {"Shape": "__cxa_pure_virtual", "Rect": "_ZNK4Rect4areaEv"}, "...": "..."}], "...": "..."}
}
```

按以下顺序阅读 `code/main.py`：

1. `ElfObject` 读取节、符号和重定位，`relocated()` 对调试节应用 `S + A`。
2. `parse_abbreviations()` 和 `DwarfInfo._parse_unit()` 读取头部、缩写表和各个条目，然后通过两张表解析 `strx` 和 `addrx`。
3. `DwarfInfo.struct_layouts()` 和 `functions()` 把条目转换为 `StructLayout` 和 `FunctionInfo` 记录，配合 `type_name()`、`type_size()` 和 `type_kind()`。
4. `parse_listing()`、`parse_memory()` 和 `access_width()` 读取 `llvm-objdump` 文本。
5. `trace_function()` 是抽象状态机，`fold_unrolled()` 和 `infer_layout()` 把访问变成布局。
6. `compare_layouts()` 产生状态行，`sysv_layout()` 和 `ctypes_layout()` 是另外两个判定基准。
7. `Demangler`、`read_vtable()`、`read_typeinfo()` 和 `find_virtual_calls()` 覆盖 C++ 对象。

还有四个命令。`python3 code/main.py dwarf FILE.o` 打印任何 DWARF 5 可重定位目标文件的结构体和函数。`python3 code/main.py infer LISTING.txt [FILE.o]` 对一份清单运行 pass；若给出目标文件，则与它的 DWARF 比较。`python3 code/main.py demangle NAME...` 和 `python3 code/main.py vtable FILE.o` 覆盖 C++ 那一半。

## 使用它

面对一个你自己构建的新目标文件：

1. 捕获该目标文件的 `llvm-objdump -d -r`，并把它与目标文件放在一起。pass 读取这份文本。
2. 对清单运行 `python3 code/main.py infer`。先读 witnesses 行：它列出第一个参数被用作内存基址的那些函数。如果某个函数缺席，说明它的参数从未到达内存操作数，或者经过了一条 pass 不追踪的路径。
3. 连同访问次数一起读每个字段。一次访问就是一个见证。把只有单个见证的字段记录为推断结果。
4. 读步长证据。`call-site` 或 `scale` 条目是强证据。只有一条 `loop-step` 条目而没有任何其他种类是弱证据，因为展开的循环会报告多个元素的步长。如果出现 `unrolled`，手工检查折叠。
5. 如果存在 `-g` 构建，运行比较并读每一行不是 `match` 的结果。`no-evidence` 行命名了没有任何函数碰过的成员。`unexpected` 行命名了 pass 无法归位的访问。
6. 把布局作为结构体输入你工具的类型管理器，并记录哪些字段、名字和种类是猜测。

把三个判定基准放在手边。当你写的布局上规则、`ctypes` 和 DWARF 不一致时，你就发现了一个紧凑结构体、一个位域，或一个规则不同的目标架构。

## 交互实验

### 实验 A：解码成员条目

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types/code
python3 -q
```

运行：

```python
from main import *
elf = load_object("account-x86_64-O0-g.o")
dwarf = DwarfInfo(elf)
dwarf.header()
elf.section(".debug_info").data[0x1a3:0x1ac].hex(" ")
dwarf.dies[0x1a3].attributes
dwarf.type_name(dwarf.type_of(dwarf.dies[0x1a3]))
```

在看字典之前，先在纸上用缩写 13 解码这九个字节。然后读取未重定位的原始节 `elf.section(".debug_str_offsets").data[8:16].hex()`，以及重定位后的版本 `elf.relocated(".debug_str_offsets")[8:16].hex()`，并解释两者的差异。

### 实验 B：追踪一个函数

```python
listing = load_listing("account-x86_64-O0-g.objdump.txt")
trace = trace_function("label_length", listing["label_length"])
[(hex(a.address), a.root, a.offset, a.width, a.kind) for a in trace.accesses]
trace.pointer_uses
trace_function("sum_balances", listing["sum_balances"]).loops
```

在运行之前写下：哪一次访问证明了偏移 24 处是指针。然后在 `-O1` 清单上运行同样的行，并在 `trace.increments` 中找到 `leaq 0x1(%rcx), %rcx`。

### 实验 C：折叠向量化循环

```python
o2 = load_listing("account-x86_64-O2-g.objdump.txt")
trace = trace_function("sum_balances", o2["sum_balances"])
sorted({a.offset for a in trace.accesses})
folded, evidence = fold_unrolled(trace, trace.accesses)
sorted({a.offset for a in folded}), [(e.kind, e.value, e.text) for e in evidence]
```

然后运行 `analyze_build("O2", fold=False)["comparison"]` 并数出 `unexpected` 行。修改折叠规则，要求循环步长恰好为 `k * d` 且 `k` 至少为 2，并确认 `-O1` 的循环（一次访问、步长 32）不受影响。

### 实验 D：虚表与名字

```python
shapes = load_object("shapes-x86_64-O0-g.o")
read_vtable(shapes, "_ZTV4Rect")
read_typeinfo(shapes, "_ZTI4Rect")
demangle("_ZNK4Rect4areaEv"), demangle("_Z8describeRK5Shapei")
find_virtual_calls(load_listing("shapes-x86_64-O0-g.objdump.txt"), [read_vtable(shapes, "_ZTV5Shape"), read_vtable(shapes, "_ZTV4Rect")])
```

在纸上从 `callq *0x10(%rax)` 计算出槽位，然后与 `DwarfInfo(shapes).struct("Rect").virtual_methods` 对照确认。接着加载 `shapes-x86_64-O1-g.o` 并对它尝试 `read_vtable`。阅读 `ElfError` 并解释该符号为何缺席。

## 实践练习

恢复 `-O1` 下 `main` 中的栈上结构体。

要求：

- 扩展 `trace_function()`，使一个栈帧槽位区域可以成为根。从填充 `0x0(%rsp)` 到 `0x5f(%rsp)` 的 `movaps` 存储中为它播种，这些存储来自 `.data.rel.ro`，即 `rows` 的 96 字节初始化器。
- 把内联访问 `0x24(%rsp)`、`0x40(%rsp)`、`0x48(%rsp)`、`0x30(%rsp)` 和 `0x38(%rsp)` 归属到该根，并报告它们对候选步长取模后的偏移。
- 从 `0x120` 到 `0x12c` 的循环推导出步长 32：`%r14` 从 4 开始，以 `0x20` 递增直到 `0x64`；并说明循环上界 `0x64` 是三个元素加上字段偏移。
- 把结果与 DWARF 布局比较，达到五个匹配，并记录哪些种类仍未证明。
- 添加一个测试：当栈帧根被归属到某个参数时，该测试失败。

成功的标志是：对 `-O1` 清单运行 `python3 code/main.py infer`，`main` 作为带栈帧根的见证函数出现，且比较结果仅凭 `main` 就显示五个匹配。

## 课程产物

本课包含 `outputs/skill-recovering-data-and-types.md`。用它把一份清单和可选的调试构建转换为带逐字段证据的结构体布局、带来源的步长、与 DWARF 的比较，以及一份仍然未知事项的清单。它要求访问表、步长证据、折叠检查、存在时的 DWARF 判定基准，以及一份关于任何指令都无法提供的名字和限定符的说明。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应能证明：

- 单元头、14 条缩写和五个成员偏移与 `llvm-dwarfdump` 一致
- 名字经重定位后的 `.debug_str_offsets` 表到达，且单元的 `str_offsets_base` 和 `addr_base` 都为 8
- 六个函数带有 dump 所显示的范围、`DW_OP_reg6 RBP` 和 `DW_OP_fbreg` 位置，且它们的低地址等于清单中的函数起始地址
- 没有 `.debug_info` 的目标文件会抛出异常，LEB128 解码器把 `0x78` 当作 -8
- `pyelftools` 和 `lldb` 打印的布局与读取器相同
- psABI 规则、`ctypes`、DWARF 和本机运行都给出偏移 `0, 4, 8, 16, 24` 和大小 32，重排后的结构体为 24 字节
- 清单解析器在 `bump` 中读出 13 条指令，把 `R_X86_64_PLT32 bump-0x4` 附加到 `0x194` 处的调用上，并找到 `0x72` 处的向后跳转
- 内存操作数、宽度、种类和符号可从 AT&T 文本和助记符解码
- `-O0` 下的 `bump` 从栈帧槽位读回其指针，并在偏移 4 处产生三次访问
- 步长在 `-O0` 下来自 `shlq $0x5`，在 `-O1` 下来自 `addq $0x20, %rdx`，在 `-O2` 下来自被 256 步长折叠的八次加载
- `label_length` 通过解引用证明偏移 24 处是指针
- `main` 在 `-O0` 下以偏移 0、`0x20` 和 `0x40` 传递同一个栈帧对象，在 `-O1` 下除 `printf` 外不调用任何函数
- 每个构建都匹配全部五个成员，大小为 32，空隙等于 DWARF 填充
- 不折叠时，`-O2` 构建报告字节 32 之后的七个字段且没有大小
- 反修饰器复现全部 19 个 `llvm-cxxfilt` 字符串
- 两个虚表、两条 typeinfo 记录以及 `0x16` 处的虚调用都解析到槽位 2，DWARF 也一致
- `-O1` 的 C++ 目标文件保留调用模式但没有虚表
- Ghidra、rizin 和 radare2 的抓取结果显示：带 DWARF 时有结构体，不带时只有原始偏移
- 交互图内嵌了与分析产生的相同数据

## 生产环境失败模式

| 失败情形 | 弱工具会报告什么 | 分析必须证明什么 |
|---|---|---|
| 循环被展开 | 八个相隔 32 字节的字段和一个 232 字节的结构体 | 循环步长 256 等于 8 乘以偏移增量 32，所以只有一个偏移 4 处的字段 |
| 从一个自增读取步长 | 由 `addq $0x100` 得出 `sizeof` 为 256 | 所有步长来源的最大公约数，包括标量收尾循环的 `addq $0x20` |
| 把 8 字节字段称为指针 | 偏移 24 处是 `void *` | 后续访问把加载的值用作基址，否则种类保持 `int64` |
| 把指针称为整数 | 偏移 24 处是 `long` | `0xfb` 处的解引用，记录为指针证据 |
| 窄比较 | 带符号的 `char` | `cmpb` 不携带符号信息；只有 `movsbl` 或 `movzbl` 才携带 |
| 假定五个函数共享同一类型 | 一个没有任何别名证据的结构体 | `main` 中传递同一个栈帧对象的调用点，或分析师的明确断言 |
| 按索引从 `.debug_str` 读字符串 | 从可重定位目标文件读出乱码名字 | `str_offsets_base` 加上重定位后的 `.debug_str_offsets` 条目 |
| 把地址当数字读取 | 每个函数的 `low_pc` 都是 0 | 经重定位后的 `.debug_addr` 表解析 `addrx` |
| DWARF 4 目标文件 | 错误解析头部的读取器 | 检查头部第二个字段的版本；第 4 版没有 `unit_type` 字节 |
| 虚表缺失 | 「该类没有虚函数」 | `-O1` 目标文件内联了所有调用；`-O0` 目标文件有虚表及其重定位 |
| 从错误的基址读取虚表槽位 | 槽位编号差二 | 地址点在符号之后 16 字节处，位于 offset-to-top 和 typeinfo 之后 |
| 修饰名被剥掉 | `Z8describeRK5Shapei` 原样保留 | 在 macOS 上用 `--no-strip-underscore`，或使用会检查 `_Z` 的反修饰器 |
| 依据某个工具的猜测给结构体定类型 | 把 `char *param_1` 当作字段类型 | 宽度与解引用，写成证据，并把名字标记为猜测 |

## 运维规则

从一次访问恢复一个字段，从一个循环恢复步长，并为每一项附上对应的指令。在相信超过步长的偏移之前，先折叠被展开的循环。读 DWARF 要通过它的缩写表和重定位后的表，绝不按索引直接读字符串节。逐字段比较，把每个没有证据的成员、每个未证明的指针、每个名字都写为未知。在 C++ 目标文件中，从重定位读虚表、从调用读槽位，然后用 DWARF 确认你读到的槽位。

## 延伸阅读

- [System V 应用二进制接口，AMD64 架构处理器补充，第 3.1.2 节：数据表示](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Arm 64 位架构过程调用标准（AAPCS64），第 5.10 节：复合类型](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [为 Apple 平台编写 ARM64 代码](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [Intel 64 和 IA-32 架构软件开发人员手册，第 1 卷，第 3.7.5 节：指定偏移](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [DWARF 调试信息格式第 5 版，第 2.1、5.7.6、7.5.1.1、7.5.3、7.5.6 节](https://dwarfstd.org/doc/DWARF5.pdf)
- [Itanium C++ ABI，第 2.5 节（虚表布局）、2.9 节（运行时类型信息）、5.1 节（外部名字）](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)
- [Microsoft Learn：修饰名](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170) 与 [名字修饰](https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/name-decoration?view=msvc-170)
- [llvm-dwarfdump 命令指南](https://llvm.org/docs/CommandGuide/llvm-dwarfdump.html) 与 [llvm-cxxfilt 命令指南](https://llvm.org/docs/CommandGuide/llvm-cxxfilt.html)
- [Python ctypes：结构体布局、sizeof、对齐](https://docs.python.org/3/library/ctypes.html)
- [Ghidra Data Type Manager](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Base/src/main/help/help/topics/DataTypeManagerPlugin/data_type_manager_description.htm)、[反编译器概念](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerConcepts.html)、[反编译器选项](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerOptions.html) 与 [Ghidra 12.1.4 发布](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.1.4_build)
- [IDA 9.1 用户指南：类型](https://docs.hex-rays.com/9.1/user-guide/types)、[IDA 9.4 发布文章](https://hex-rays.com/blog/ida-9.4-release-a-new-dyld-shared-cache-swift-analysis-new-teams-add-on-and-more-) 与 [IDA Free](https://hex-rays.com/ida-free)
- [Binary Ninja 用户指南：类型](https://docs.binary.ninja/guide/types/) 与 [Binary Ninja 6.0 Krypton](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [radare2 6.2.4 发布](https://github.com/radareorg/radare2/releases/tag/6.2.4) 与 [rizin v0.9.1 发布](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
- [GDB 手册：检查符号表，ptype /o](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Symbols.html)
- [PyPI 上的 pyelftools](https://pypi.org/project/pyelftools/)
- [Lee、Avgerinos、Brumley，《TIE：对二进制程序类型的原理化逆向工程》（NDSS 2011）](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan、Loginov、Cok，《机器代码的多态类型推断》（PLDI 2016）](https://arxiv.org/abs/1603.05495) 与 [retyped 仓库](https://github.com/GrammaTech/retypd)
- [Zhang 等，《OSPREY：通过概率分析恢复剥离二进制中的变量和数据结构》（IEEE S&P 2021）](https://ieeexplore.ieee.org/document/9519451/)
- [Chen 等，《用学习到的变量名和类型增强反编译器输出》（USENIX Security 2022）](https://www.usenix.org/conference/usenixsecurity22/presentation/chen-qibin)
- [Xie 等，《ReSym：利用 LLM 从剥离二进制中恢复变量和数据结构符号》（CCS 2024）](https://dl.acm.org/doi/10.1145/3658644.3670340)
- [Chen 等，《RecStruct》（USENIX Security 2026）](https://www.usenix.org/conference/usenixsecurity26/presentation/chen-yuxin)
