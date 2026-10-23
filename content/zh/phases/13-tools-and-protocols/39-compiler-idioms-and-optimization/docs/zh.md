# 编译器惯用法：优化对你的代码做了什么，以及如何把它读回来

> 优化编译器会用更廉价的指令形态替换源代码中的构造。每一种形态都有对应的推导过程，本节课要构建一个识别器，把推导反过来执行，并用源代码来校验它。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, AArch64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode), Phase 13 · 38 (calling conventions)
**Time:** ~150 minutes

## 学习目标

- 用 Granlund-Montgomery 界推导除以任意常数的魔数与移位量，并从 clang 生成的常量中还原出除数。
- 说出 LLVM 把 `switch` 降级为跳转结构的五种方式，并从目标文件中读出条目数、表位置和条目内容。
- 把向量化循环读成 lane、累加器、每轮元素数、行程计数掩码、归约和余数循环。
- 识别尾调用、被内联的被调函数、无分支选择，以及被替换成 `memset` 或 `memcpy` 的循环。
- 在 x86-64 和 AArch64 上、覆盖 `-O0` 到 `-O3`，为每个惯用法报告其证据指令和解码出的源码形态。
- 把结果与 Ghidra、angr、radare2、rizin 和 lldb 对同一份字节的展示做比较。

## 问题

你在 `-O2` 下编译了 `idioms.c`，然后问一个智能体 `div7u` 是做什么的。智能体读到 `imulq $0x24924925, %rax, %rax`，回答说这个函数乘以 613566757、移位、再相加。每一个字都是真的。而这个函数计算的是 `x / 7`。

同一个目标文件里有一个七分支的 `switch`，完全没有跳转；一个五分支的 `switch`，是一张跳转表；一个两分支的 `switch`，是一条 `sete` 加一条 `cmovne`；还有一个遍历 `n` 个整数的循环，每轮在两个 `xmm` 寄存器里处理八个整数，剩下的部分在第二个循环里收尾。在 `-O0` 下，同样的源码生成的是 `divl`、七条目跳转表和单个标量循环。在 `-O1` 下，七分支 switch 变成了一张取值表，而向量循环还不存在。

这些形态就是惯用法（idiom）。编译器按照一条规则生成其中每一种形态，这条规则可以在编译器源码里读到；而每一种形态都能被解码回它来源的构造。解码是算术，而算术是可以校验的。第 34 课证明了其中一种：乘以 `0x66666667`。本节课为所有这些形态构建识别器，在我们自己代码的 18 个构建上运行它，并用 C 源码校验全部 373 条报告。

```figure
re-39-idioms-by-level
```

## 惯用法是带推导的形态

惯用法是编译器针对某一个源码构造发出的一串指令。它之所以可识别，有三个原因：形态固定；操作数携带推导所预测的常量；形态与程序员手写的代码不同。

`code/main.py` 中的识别器处理 `llvm-objdump -d -r` 的文本，第 33 课已经把它确立为指令边界的判定标准（oracle）。它把每个函数解析成一条条指令，记录地址、字节、助记符、操作数，以及紧随其后的重定位行。它只从目标文件里读取两样东西：`.rodata` 的字节（取值表所在之处），以及 `.rela.rodata` 条目（链接之前跳转表目标所在之处）。

每一次检测都是一条 `Idiom` 记录，包含 `kind`、解码出的 `source` 形态、`evidence` 地址列表、`detail` 字典和 `verified` 标志。该标志来自 C 源码。`parse_source()` 读取 `idioms.c`，找到每个函数，提取 `/` 和 `%` 之后的除数、`*` 之后的乘数、`switch` 的分支及其取值、`? :` 运算符、`for` 循环以及被调函数。当除出现在函数的除法列表中时，除法惯用法即被校验通过。当查找表的条目与 case 值在相差一个常数因子的意义下相等时，该表即被校验通过。当被调函数在源码中被调用时，尾调用即被校验通过。演示报告以一个 `unverified` 列表结尾，测试套件断言该列表在全部 18 个构建中都为空。

解码需要名字。识别器在每条函数的基本块上按列表顺序运行一个小型符号求值器，把源码中的参数名放进第 38 课所述 System V 与 AAPCS64 约定的参数寄存器里。栈槽按操作数文本跟踪，这样 `-O0` 代码就能穿过它的溢出存储来读取。在每个基本块的入口合并前驱状态；当状态不一致时，先前某个惯用法赋予的名字优先于常量。`digest` 中的 `cmovll %edx, %eax` 之所以能读回成 `sum % 10 < 5 ? sum % 10 : 5`，原因就在这里：循环惯用法把累加器命名为 `sum`，除法惯用法把商命名为 `sum / 10`，取模惯用法把结果命名为 `sum % 10`。源码写的是 `rem > 5 ? 5 : rem`。两个表达式都给出 `rem` 与 5 中较小的那个，并且在 `rem == 5` 时一致。

## 除以常数

Granlund 与 Montgomery 的论文《Division by Invariant Integers using Multiplication》（PLDI 1994）用一次乘法和一次移位替换 `n / d`。选定移位量 `s` 和乘数 `m = ceil(2^s / d)`。只要误差 `e = m * d - 2^s` 足够小，那么对范围内的每个 `n`，`floor(n * m / 2^s)` 都等于 `floor(n / d)`。对无符号 32 位 `n`，界为 `e <= 2^(s - 32)`。对有符号 `n`（`|n| < 2^31`），界为 `e <= 2^(s - 31)`；当 `n` 为负时，符号修正会加 1，因为 floor 对负商的舍入方向不对。

`main.py` 中的 `magic_numbers(d, bits, signed)` 从 32 开始向上搜索 `s`，返回第一对满足界的参数。这一搜索复现了 clang 在测试夹具中生成的每一个常量。

### 手算无符号除以 10

`2^35 / 10 = 3435973836.8`。向上取整得 `3435973837 = 0xCCCCCCCD`。误差是 `0xCCCCCCCD * 10 - 2^35 = 2`，而 `s = 35` 时的界是 `2^3 = 8`。在 `s = 34` 时误差会是 6，而界只有 4，所以 35 是最小的移位量。用 `n = 123` 验证：`123 * 3435973837 = 422624781951`，而 `422624781951 >> 35 = 12`。

clang 在 x86-64 上以 `-O2` 编译 `div10u` 时发出的正是这个：

```text
       0: 89 f9                        	movl	%edi, %ecx
       2: b8 cd cc cc cc               	movl	$0xcccccccd, %eax
       7: 48 0f af c1                  	imulq	%rcx, %rax
       b: 48 c1 e8 23                  	shrq	$0x23, %rax
```

`0x23` 就是 35。`movl %edi, %ecx` 把 32 位参数零扩展进 64 位寄存器，这正是无符号形式。识别器从供双操作数 `imulq` 使用的 `movl $imm` 中读出乘数，读出移位量，然后调用 `divisor_from_magic(0xCCCCCCCD, 35)`。该函数尝试 `2^35 // m = 10` 和 11，接受 10，因为 `magic_numbers(10)` 能反推出同样的乘数和移位量。

在 AArch64 上，同一个函数是 `mov w8, #0xcccd; movk w8, #0xcccc, lsl #16; umull x8, w0, w8; lsr x0, x8, #35`。常量分两个 16 位半字到达，乘法是 `umull`，移位量用十进制书写。识别器把两半拼起来，报告同一个除数。

### 手算无符号除以 7，带加法修正

`2^35 / 7 = 4908534052.57`。向上取整得 `4908534053 = 0x124924925`，需要 33 位。误差为 3，界为 8，所以移位量是 35，更小的移位都不行。32 位乘法装不下这个乘数，于是编译器改为乘以低 32 位 `0x24924925`，并按固定顺序把被除数加回来：

```text
      32: 48 69 c0 25 49 92 24         	imulq	$0x24924925, %rax, %rax
      39: 48 c1 e8 20                  	shrq	$0x20, %rax
      3d: 29 c7                        	subl	%eax, %edi
      3f: d1 ef                        	shrl	%edi
      41: 01 f8                        	addl	%edi, %eax
      43: c1 e8 02                     	shrl	$0x2, %eax
```

把它读成 `t = (n * 0x24924925) >> 32`，然后 `q = (((n - t) >> 1) + t) >> 2`。总移位量是 `32 + 1 + 2 = 35`。取 `n = 100`：`t = 14`，`(100 - 14) >> 1 = 43`，`43 + 14 = 57`，`57 >> 2 = 14`。识别器检测 32 位移位之后的 `sub`、`shr`、`add`、`shr` 尾巴，把 `2^32` 加回乘数，还原出 7。`evaluate_division()` 执行这一精确序列，测试会在 32 位边界值上把它与 `n // 7` 对照。AArch64 的形式是 `sub w9, w0, w8; add w8, w8, w9, lsr #1; lsr w0, w8, #2`，少一条指令，因为带移位的加法在这里就是一条指令。

### 手算有符号除以 3

`2^32 / 3 = 1431655765.33`，向上取整得 `1431655766 = 0x55555556`，误差为 2，而 `s = 32` 时有符号界是 `2^1 = 2`。界恰好成立，没有余量，所以移位量是 32，clang 使用 `shrq $0x20`。取 `n = -7`：乘积是 `-10021590362`，算术右移 32 位得到 `-3`，符号位被置位，于是修正加 1，结果为 `-2`，这正是 C 语言中 `-7 / 3` 的结果。有符号测试夹具 `div10s` 和 `div3s` 都把修正表现为对乘积副本做 `shrq $0x3f`，随后 `addl`。

### 这些常量不是什么

单看乘数无法确定除数。`0x66666667` 配移位量 34 是 10。同一个乘数配移位量 33 则是 5，正如第 34 课所示。因此识别器总是把乘数与总移位量成对处理，`clangMatchesAlgorithm` 字段记录这一对是否正是搜索产生的结果。2 的幂从不使用乘法。`magic_numbers()` 拒绝它们，因为编译器改用带符号偏置的移位，下一节的 `mod8s` 展示了那种形态。

### 这个惯用法属于指令选择器

在 `-O0` 下，`div10u` 是 `movl $0xa, %ecx; xorl %edx, %edx; divl %ecx`。`div10s` 是 `cltd; idivl %ecx`。而在同一个目标文件里，`digest` 用 `imulq $0x66666667` 来除以 10。一个编译选项解释了这种差别。测试夹具 `idioms-x86_64-nofastisel-O0.o` 是以 `-O0 -mllvm -fast-isel=false` 构建的，在它里面每一次除法都是乘法，包括 `div10u` 的 `movl $0xcccccccd, %ecx; imulq %rcx, %rax; shrq $0x23`。硬件除法是快速指令选择器在 `-O0` 下的选择。乘法则是 SelectionDAG 的降级结果，也就是 LLVM 的 `UnsignedDivisionByConstantInfo` 与 `SignedDivisionByConstantInfo` 产生的结果。包含该除法的 `digest` 基本块里还有一个 `switch`，而这个块交给了 SelectionDAG。识别器会同时报告两种形式：`hardware idivl` 和 `magic multiply`；图展示了 `-O0` 下混合的这种函数。

## 取模跟随商

`x % d` 等于 `x - (x / d) * d`。编译器用上面的惯用法算出商，再乘回去，然后相减。在 x86-64 上，乘回 10 的序列是 `addl %ecx, %ecx; leal (%rcx,%rcx,4), %ecx`，也就是 `2 * q * 5`。在 AArch64 上只需一条指令 `msub w0, w8, w9, w0`，当 `w9 = 10` 时它计算 `w0 - w8 * w9`。识别器沿复制指令跟踪商所在的寄存器，通过 `add`、`lea`、`shl` 和 `imul` 追踪累积的因子，当因子等于除数时接受这条 `sub`。

当同一个商还供给另一个表达式时，两个惯用法都保留在报告里。在 `-O1` 下，编译器在 `digest` 中只算一次 `sum / 10`，同时用于 `switch` 索引和取模。在 x86-64 上它甚至把商组合了两次：取模用 `leal (%rax,%rsi), %edi`，switch 用 `addl %esi, %eax`。识别器报告一次除法和一次取模，并把商标记为共享。如果报告在这里丢掉了除法，`switch` 索引就会变得无法解读。

有符号的 2 的幂取模有自己的形态，因为商根本不会被计算：

```text
      92: 8d 48 07                     	leal	0x7(%rax), %ecx
      95: 85 ff                        	testl	%edi, %edi
      97: 0f 49 cf                     	cmovnsl	%edi, %ecx
      9a: 83 e1 f8                     	andl	$-0x8, %ecx
      9d: 29 c8                        	subl	%ecx, %eax
```

对负的 `x`，代码在掩码之前先偏置 7，好让截断按 C 的要求向零舍入。掩码后的值是舍入到 8 的倍数的 `x`，相减之后剩下 `x % 8`。AArch64 用 `negs w8, w0; and w9, w0, #0x7; and w8, w8, #0x7; csneg w0, w9, w8, mi` 实现：对 `x` 和 `-x` 分别掩码，再按符号选择并取负。Ghidra 把这条规则命名为 `RuleSignMod2nOpt`，简述为「Convert INT_SREM forms: `(V + (sign >> (64-n)) & (2^n-1)) - (sign >> (64-n)) => V s% 2^n`」。

在 `-O0` 下，AArch64 把取模写成 `sdiv w9, w8, w10; mul w9, w9, w10; subs w0, w8, w9`。识别器也接受这种形式，并报告 `hardware sdiv + mul + subs`。

## 降级 switch 的五种方式

LLVM 在两个地方决定 `switch` 的形态。`SimplifyCFG::simplifySwitchLookup` 在 `-O1` 及以上运行，当一个 switch 的所有分支都产生常量、且至少有三个分支、目标平台能构建查找表时，它把该 switch 变成取值表。它的表类型有 `SingleValueKind`、`LinearMapKind`、`BitMapKind` 和 `ArrayKind`。`-O0` 流水线从不运行这一遍。其余的 switch 进入指令选择阶段的 `SwitchLowering::findJumpTables`：当簇中至少有 `min-jump-table-entries` 个条目（默认 4），并且足够稠密——`NumCases * 100 >= Range * MinDensity`，普通函数 `jump-table-density` 为 10，`-Os` 下为 40——就构建跳转表。剩下的部分变成对分支常量的二分查找，由 `splitWorkItem` 选择枢轴。测试夹具为每种结果各保留一个函数。

| 函数 | 源码 | `-O0` x86-64 | `-O2` x86-64 | `-O2` AArch64 |
|---|---|---|---|---|
| `linear_switch` | case 0 到 6 返回 10 到 70 | 跳转表，7 条目 | 线性映射 `10*k + 10`，默认 -1 | `madd` 与 `csinv` |
| `table_switch` | case 0 到 6 返回 10 到 60，再是 71 | 跳转表，7 条目 | 位于 `.rodata+0x28` 的查找表 | 位于 `.rodata+0x0` 的查找表 |
| `call_switch` | case 0 到 4 调用五个函数指针之一 | 跳转表，5 条目 | 跳转表，5 条目 | 比较树，枢轴 1 |
| `sparse_switch` | case 为 1、100、10000、1000000、100000000 | 比较链 | 比较树，枢轴 9999 | 比较树，枢轴 9999 |
| `two_cases` | case 3 和 4 返回 7 和 9 | 比较链 | 选择链 `k != 3 ? (k == 4)*9 : 7` | 两条 `csel` |

### 从目标文件读取查找表

`-O2` 下的 `table_switch`：

```text
      d0: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      d5: 83 ff 06                     	cmpl	$0x6, %edi
      d8: 77 0c                        	ja	0xe6 <table_switch+0x16>
      da: 89 f8                        	movl	%edi, %eax
      dc: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		00000000000000df:  R_X86_64_PC32	.rodata+0x24
      e3: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      e6: c3                           	retq
```

边界检查 `cmpl $6; ja` 意味着有 7 个有效条目。重定位加数 `0x24` 加上 RIP 相对字段的 4 个字节，把表定位在 `.rodata+0x28`，这正是第 33 课推导出的规则。那里的字节是 `0a 00 00 00 14 00 00 00 ...`，所以条目为 `[10, 20, 30, 40, 50, 60, 71]`，与源码中的 case 值相等。汇编器在 `.s` 文件里把这张表命名为 `.Lswitch.table.table_switch`，并附注释 `@switch.table.table_switch`。在目标文件里只剩下重定位和字节。这里没有跳转，所以控制流图只有三个基本块。

AArch64 形式用 `adrp x8` 和 `add x8, x8, #0x0` 拼出表地址，二者各带一条指向 `.rodata` 的重定位，然后用 `ldr w0, [x8, w0, uxtw #2]` 加载。识别器从 `add` 的重定位中读出加数。

### 线性映射，完全不用表

`linear_switch` 在 `k` 取 0 到 6 时返回 `10 * (k + 1)`。`simplifySwitchLookup` 检测出仿射关系，直接生成算术指令：

```text
      b0: 83 ff 07                     	cmpl	$0x7, %edi
      b3: 8d 04 bf                     	leal	(%rdi,%rdi,4), %eax
      b6: 8d 4c 00 0a                  	leal	0xa(%rax,%rax), %ecx
      ba: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      bf: 0f 42 c1                     	cmovbl	%ecx, %eax
```

求值器把 `leal (%rdi,%rdi,4)` 读成 `5*k`，把 `leal 0xa(%rax,%rax)` 读成 `10*k + 10`，把 `cmpl $7` 之后的 `cmovb` 读成 `k < 7 ? 10*k + 10 : -1`。随后识别器用每个源码 case 校验该公式，并把惯用法标记为已校验。AArch64 用 `madd w8, w0, w8, w9` 和 `csinv w0, w8, wzr, lo` 得到同一个表达式，其中 `csinv` 配 `wzr` 在默认路径上给出 `-1`。

### 从重定位读取跳转表

`-O2` 下的 `call_switch` 正是第 33 课在 `-O0` 的 `grade` 中还原出的形态：`cmpl $4; ja`、指向 `.rodata-0x4` 的 `leaq`、对 4 字节条目做 `movslq`、`addq`，以及 `jmpq *%rax`。五个条目是 `.rela.rodata` 中的重定位，`target = addend - 4k` 给出 `0x10c, 0x107, 0x10e, 0x110, 0x109`，五个不同的基本块，各自通过函数指针尾跳。`dispatch` 夹具把它扩大为 16 个分支：x86-64 上 16 条重定位条目；AArch64 上则是一张 16 个单字节的压缩表，用 `ldrb w11, [x9, x8]` 读取，用 `add x10, x10, x11, lsl #2` 放大 4 倍，基址来自第一个 case 块的 `adr`。识别器从字节表中解码出 `[0x24, 0x74, 0x54, ...]`，并报告 `compressed: true`。

### 同一个 switch 在四个优化级别下的样子

`digest` 里有一个作用于 `sum / 10` 的 switch，case 为 6 到 10，取值为 1、2、4、8、8，然后把结果乘以 16。在 x86-64 上，`-O0` 时它是跳转表；`-O1` 时是查找表，条目为 `[16, 32, 64, 128, 128]`；`-O2` 和 `-O3` 时又变回跳转表，四个 case 块分别存储 `0x10`、`0x20`、`0x40`、`0x80`。从 `-O1` 往上，乘以 16 的运算已经移进了 case。`-O0` 时 case 存储 1、2、4、8，乘法留在 switch 之后。对 `-O1` 的查找表，识别器把条目与源码取值比较，发现一个常数比例，并报告 `foldedFactor: 16`。在 AArch64 上，`-O2` 形式是比较树，枢轴为 8，对 6、7、8 做相等测试，还有一个范围测试 `sub w9, w9, #9; cmp w9, #2; b.hs`，用两条指令覆盖 case 9 和 10。识别器报告 `rangeTests: [[9, 10]]`。

有两个选项会移动这些边界。`-fno-jump-tables` 在 x86-64 上把 `table_switch`、`call_switch` 和 `digest` 的 switch 变成比较树，七分支函数的枢轴是 2 和 4。在 AArch64 上，`-mllvm -min-jump-table-entries=4` 仍让 `call_switch` 保持比较树，说明在那个目标上，条目数并不是这个五分支间接调用 switch 的决定性输入。夹具 `idioms-arm64-jt4-O2.o` 记录了那次运行。

GCC 把这两个决策都放在 `tree-switch-conversion.cc` 中，涉及 `jump_table_cluster`、`bit_test_cluster`、`switch_decision_tree` 和 `switch_conversion` 这几个类。它的 `-ftree-switch-conversion` 在 `-O2` 下开启，`-fjump-tables` 和 `-fbit-tests` 在 `-O1` 下开启；跳转表的阈值来自 `case-values-threshold`，默认值由目标机器决定。本机没有安装 GCC，所以这些名字取自 GCC 源码和手册，你可以到 Compiler Explorer 上验证。

## 循环：向量化、展开还是替换

LLVM 的循环向量化器在 `-O2`、`-O3` 和 `-Os` 下开启，在 `-O0` 和 `-O1` 下关闭，这条规则可以追溯到 2014 年 1 月的 LLVM 3.4。`-O2` 下的 `sum_n` 展示了这种形态的每一个部分：

```text
     1f6: 83 fe 07                     	cmpl	$0x7, %esi
     1f9: 77 09                        	ja	0x204 <sum_n+0x14>
     206: 81 e2 f8 ff ff 7f            	andl	$0x7ffffff8, %edx
     230: f3 0f 6f 14 37               	movdqu	(%rdi,%rsi), %xmm2
     235: 66 0f fe c2                  	paddd	%xmm2, %xmm0
     239: f3 0f 6f 54 37 10            	movdqu	0x10(%rdi,%rsi), %xmm2
     23f: 66 0f fe ca                  	paddd	%xmm2, %xmm1
     243: 48 83 c6 20                  	addq	$0x20, %rsi
     24a: 75 e4                        	jne	0x230 <sum_n+0x40>
     24c: 66 0f fe c8                  	paddd	%xmm0, %xmm1
     250: 66 0f 70 c1 ee               	pshufd	$0xee, %xmm1, %xmm0
     262: 66 0f 7e c8                  	movd	%xmm1, %eax
     270: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
     279: 75 f5                        	jne	0x270 <sum_n+0x80>
```

保护条件 `cmpl $7; ja` 把小于 8 的行程计数送到标量循环。掩码 `andl $0x7ffffff8` 把行程计数向下取整到 8 的倍数。循环体每轮加载 32 字节到两个 `xmm` 累加器，每个累加器含四个 32 位 lane，因此向量化因子是 4，交错因子是 2。循环结束后，`paddd`、两个 `pshufd`、另两个 `paddd` 和 `movd` 把八个 lane 归约成一个整数。`0x270` 处的循环处理最后 0 到 7 个元素的标量余数。识别器通过回边找到循环，看到 SIMD 寄存器，从 `paddd` 读出 lane 数，数出不同的累加器，把步长 `0x20` 除以 4，报告 `4 lanes x 2 accumulators = 8 elements per iteration`，并把掩码、保护、归约和余数循环作为字段。AArch64 用 `ldp q2, q3`、两条 `add v.4s`、`addv s0, v0.4s` 和 `fmov w8, s0` 做同样的事。

固定的行程计数会消除循环。`sum4` 是一条 `movdqu`、归约和 `movd`，没有回边。`poly8` 无法向量化，因为每一轮都依赖上一轮的累加器，于是编译器把它展开八次：从 `%rdi` 起偏移 0 到 `0x1c` 的八次加载，每次后接一次左移 5 位的移位和一次减法，共同计算 `acc * 31`。识别器统计同一基址上连续偏移的加载，报告 `loop unrolled by 8, no back edge`。在 AArch64 上，同样的加载以四对 `ldp` 出现。

用 `-fno-vectorize -fno-slp-vectorize` 关掉向量化器，`sum_n` 就变成按 8 展开的标量循环：八条 `addl k(%rdi,%rdx,4), %eax` 和一条 `addq $8`，外加一个余数循环。`sum4` 变成四次加法。识别器报告 `unrolled loop`：前者为 `copies: 8` 和 `indexStep: 8`，后者为 `copies: 4`。

第 33 课在 `-O2` 的 `total` 中找到两条回边，并称它们为两个循环。本节课给它们命名：同一个源码循环的一个向量主体和一个标量余数。只有当回边的目标能再次到达该回边时，它才构成循环。`is_loop()` 检查这一点，因为 AArch64 上的 `digest` 含有跳向共享 default 块的后向分支，它们并不是循环，而识别器的第一版把它们当成了向量循环。

### 变成库调用的循环

`LoopIdiomRecognize` 从 `-O1` 起把「向每个字节存入零」的循环变成对 `memset` 的调用。`-O2` 下的 `fill` 是 `testl %esi, %esi; jle; movl %esi, %edx; xorl %esi, %esi; jmp memset@PLT`。这次调用是带重定位 `R_X86_64_PLT32 memset-0x4` 的尾跳，函数的 `retq` 只出现在空路径上。识别器报告 `store loop recognized as memset(...)`，并对照源码循环校验它，因为源码从未提到 `memset`。

`copy16` 以常量 16 调用 `memcpy`，x86 后端把它内联展开成 `movups (%rsi), %xmm0; movups %xmm0, (%rdi)`，在其每次 `memcpy` 最多 8 次存储的限额之内。在 `-O0` 下，同一个调用是两对 8 字节 `movq`。加上 `-fno-builtin` 后，调用保持为调用：`jmp memcpy@PLT` 配 `movl $0x10, %edx`；`fill` 保留它的循环，随后向量化器把它变成每轮两次 16 字节 `movups` 存储的循环。对这种构建，识别器报告 `vectorized store loop: 2 x 16-byte stores = 32 bytes per iteration`。

## 消失或变成跳转的调用

`tail_caller` 返回 `tail_target(x + 1)`。在 `-O0` 下它是 `addl $0x1, %edi; callq tail_target@PLT; addq $0x10, %rsp; popq %rbp; retq`。在 `-O2` 下它只有 `incl %edi; jmp tail_target@PLT`，别无其他。调用者的栈帧消失了，被调函数直接返回到调用者的调用者。在 AArch64 上，跳转是 `b tail_target`，带 `R_AARCH64_JUMP26`。识别器把携带调用重定位的 `jmp` 或 `b` 视为尾调用。第 33 课的递归下降必须把它当作函数出口，因为目标在函数之外。

`twice_sum` 调用 `leaf_add`。在 `-O0` 下调用还在：`callq leaf_add@PLT`。从 `-O1` 起，`leaf_add` 的函数体 `leal (%rdi,%rsi), %eax` 出现在 `twice_sum` 内部，不再有调用。在 `-O0` 下 clang 把每个函数标记为 `optnone` 和 `noinline`，只运行 always-inliner。识别器从源码得知 `twice_sum` 调用了 `leaf_add`，却找不到对它的一次调用或尾跳，于是在调用者中寻找被调函数的函数体形态：保留助记符和源操作数、去掉目的操作数，因此 AArch64 调用者中的 `add w8, w1, w0` 匹配被调函数中的 `add w0, w1, w0`。一处匹配就是证据。

`dispatch` 把两者结合在一起：一张 16 条目的跳转表，每个 case 都是 `movl %esi, %edi; jmp fN@PLT`。识别器报告一个 switch 和 16 个尾调用，全部对照源码中的 16 次调用校验通过。

## 变成数据搬运的分支

`clamp_max` 是 `movl %esi, %eax; cmpl %esi, %edi; cmovgl %edi, %eax`。求值器把 `%edi` 命名为 `a`、`%esi` 命名为 `b`，记录比较，并把 `cmovg` 读成 `a > b ? a : b`，这正是源码文本。`abs_int` 是 `movl %edi, %eax; negl %eax; cmovsl %edi, %eax`，读作 `-x < 0 ? x : -x`；AArch64 把它写成 `cmp w0, #0; cneg w0, w0, mi`，读作 `x < 0 ? -x : x`。`is_pos` 是 `xorl %eax, %eax; testl %edi, %edi; setg %al`，读作 `(x > 0)`。用 GCC 的话说这是 `-fif-conversion`，用 LLVM 的话说是 `cmov` 选择。LLVM 的 `X86CmovConversion` 遍在认为有利可图时会把 `cmov` 变回分支，所以同一份源码在循环里可能呈现两种形态。

在 `-O0` 下这些形态一个都不存在。`clamp_max` 是一次比较、一条 `jle`、两次写入同一个栈槽和一次加载。识别器在 `-O0` 下对它什么也不报告，从 `-O1` 起报告 `branchless select`，这是图所展示的事实之一。

`mul9` 是一条 `leal (%rdi,%rdi,8), %eax`，识别器报告 `strength reduction` 并给出 `x * 9`，因为源码确实乘以 9。只有在源码中有这个因子时它才这样报告；用于放大表索引的 `lea` 不是乘法。

## 两个看起来像惯用法的加固特性

栈保护（stack protector）是编译器的特性，它留下的形态很容易被智能体误读成程序逻辑。用 `-fstack-protector-strong` 为 x86-64 Linux 编译的 `canary.c` 给出：

```text
       7: 64 48 8b 04 25 28 00 00 00   	movq	%fs:0x28, %rax
      10: 48 89 44 24 40               	movq	%rax, 0x40(%rsp)
      26: 64 48 8b 0c 25 28 00 00 00   	movq	%fs:0x28, %rcx
      2f: 48 3b 4c 24 40               	cmpq	0x40(%rsp), %rcx
      34: 75 05                        	jne	0x3b <with_canary+0x3b>
      3b: e8 00 00 00 00               	callq	0x40 <plain_leaf>
		000000000000003c:  R_X86_64_PLT32	__stack_chk_fail-0x4
```

`%fs:0x28` 是 glibc `tcbhead_t` 中的 `stack_guard`，位于字节偏移 40，LLVM 的 `getIRStackGuard` 为 glibc、bionic 和 Fuchsia 命名了这个槽。序言把 guard 复制到缓冲区上方，尾声做比较，失败路径调用 `__stack_chk_fail`。在 AArch64 Linux 上，guard 来自全局符号 `__stack_chk_guard`，经 GOT 用 `adrp` 和 `ldr` 取得，失败时执行 `bl __stack_chk_fail`。识别器报告 guard 的来源、槽位置、比较和失败调用。`plain_leaf` 没有数组，在 `strong` 下没有 canary，因为 `strong` 只保护含数组、`alloca` 或取地址局部变量的函数。研究笔记还记录了 MSVC 在 `/GS` 下的形式：一个与 `%rsp` 混合的 cookie，由 `__security_check_cookie` 检查；本机没有 MSVC 链接器，所以那种形态仅依据编译器的汇编输出描述。

`-fcf-protection=full` 会把 `endbr64`（字节 `f3 0f 1e fa`）放在每个函数的第一条指令位置。在支持控制流强制的处理器上，它是间接分支的着陆垫（landing pad）；在其他地方它作为 `nop` 执行。正因如此，Ghidra 的 x86-64 函数起始模式包含带 `endbr64` 前缀的 `push rbp; mov rbp, rsp` 变体。识别器把它报告为 `cet landing pad`，这样以四个不寻常字节开头的函数就不会被误认为数据。

## 反编译器会折叠什么

第 37 课里的每个反编译器都把其中一些推导当作重写规则来运行。Ghidra 的反编译器在 `ruleaction.cc` 中为它们命名。`RuleDivOpt` 是「Convert INT_MULT and shift forms into INT_DIV or INT_SDIV」，其形式为 `sub( (zext(V)*c), d) >> e => V / (2^n/(c-1)) where n = d*8 + e`，以及带 `sext` 和 `s>>` 的有符号变体。`RuleModOpt` 是「Simplify expressions that optimize INT_REM and INT_SREM」。`RuleDivTermAdd` 处理加法修正，`RuleSignDiv2` 把 `(V + -1*(V s>> 31)) s>> 1` 变成 `V s/ 2`，`RulePositiveDiv` 把符号位明确为零的有符号除法变成无符号除法。第 37 课在 `-O1` 的第 33 课 `grade` 上抓取了 Ghidra 12.1.4，输出读作 `uVar2 = param_1 / 10 - 6;`。除法被折叠了。

这些规则的输入不必用 Java 就能看到。虚拟环境中的 pypcode 4.0.1 使用与 Ghidra 相同的 SLEIGH 规范做提升，`code/fixtures/idioms-x86_64-O2.lifts.json` 保存了 `div10s` 的原始 P-code：七条指令对应 112 个条目，其中 7 个是开启每条指令的 `IMARK` 标记；在它们之中有 `unique[95a00:8] = sext(0x66666667)` 之后的 `RAX = RAX * unique[95a00:8]`、`RCX = RCX >> unique[d1a00:4]` 以及 `RAX = RAX s>> unique[c5c00:4]`。`shrq` 携带 41 个操作，`sarq` 携带 40 个，其中大多数在计算标志位。`RuleDivOpt` 在这个流中匹配乘法和两次移位，死代码消除会删掉没人读取的标志位。

angr 10.0.3 在我们的 `-O2` 目标文件上运行，`idioms-x86_64-O2.angr.json` 保存了它对 20 个函数的反编译输出。它折叠了无符号形式：`div10u` 是 `return a0 / 10;`。它没有折叠有符号形式：`div10s` 是 `v1 = a0 * 1717986919; return ((long long)(v1) >> 0x22) + (v1 >> 63);`，`mod10s` 是 `a0 - (...) * 2 * 5`。对 `div7u` 它打印 `(a0 / 7 + ((a0 * 6 / 7 & 0xffffffff) >> 1 & 0xffffffff) & 0xffffffff) >> 2`，修正项作为算术残留下来。它把 `mul9` 折叠为 `a0 * 9`，把 `linear_switch` 折叠为 `(a0 < 7 ? a0 * 10 + 10 : 0xffffffff)`，把 `clamp_max` 折叠为 `((int)a1 < (int)a0 ? a0 : a1)`，把 `is_pos` 折叠为 `a0 > 0`，把 `fill` 折叠为 `return memset(a0, 0, a1);`。它把 `table_switch` 打印成一次加载 `*(int *)(4 * a0 + 4195400)`，不给出取值；把 `sum_n` 打印为带 `_mm_add_epi32` 和 `_mm_shuffle_epi32` 内建函数的形式；把 `tail_caller` 打印为 `tail_target(); return;`，参数丢失。在 `digest` 中，跳转表变成了 `goto *((void *)(...))`，表运算被完整写出。

radare2 6.2.4 和 rizin 0.9.1 从源码构建进研究缓存，也在同一个目标文件上运行。radare2 的 `pdc` 把 `div10s` 打印为 `rax = rax * 0x66666667; rcx >>>= 0x3f; rax >>= 0x22; eax += ecx`，寄存器级的伪 C，没有任何折叠；把 `table_switch` 打印为 `eax = dword [rcx + rax*4]`，并在 `lea` 上附 `RELOC 32 .rodata` 注释。rizin 没有内建反编译器；它的 `aoi` 把 `imul` 的 RzIL 打印成一个 128 位乘积。lldb-1703.0.236.21 加载该 ELF 目标文件，其 `disassemble -n div10s` 打印出与 `llvm-objdump` 相同的七条指令，因为二者都用 LLVM 的反汇编器。Capstone 5.0.7 解码同样的字节，报告 `ret`、`mode64` 之类的分组；它只是解码器，而惯用法从不出现在解码器的输出里。IDA Pro 和 Binary Ninja 没有在这里运行。它们的文档描述了进行此类重写的阶段——微码优化以及 MLIL 到 HLIL 的各个遍——本节课不对它们折叠哪些惯用法做任何断言。

Compiler Explorer 是用于正方向（从源码到汇编）的工具。粘贴 `idioms.c`，选择一个编译器条目，例如 "x86-64 clang 18.1.0" 或 "ARM64 GCC 14.2.0"，并设置选项。该服务列出约 1,050 个 C 编译器条目，并提供 API：`GET /api/compilers/c` 返回标识符，`POST /api/compiler/<id>/compile` 以 JSON 返回汇编。Matt Godbolt 在 2019 年 ACM Queue 上的文章用这个工具走查了除法、population count、链式条件和求和惯用法。用它获取本机无法产生的 GCC 输出，并把 GCC 12 的跳转表阈值和 `-O2` 向量化默认行为与夹具中的 clang 输出作比较。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
```

输出是一份 JSON 报告。对 18 个构建中的每一个，它列出每个函数及其惯用法，包括 kind、源码形态、证据地址、校验结果和 detail 字段。报告以每个构建的 kind 矩阵、两种架构上 `-O2` 与 `-O3` 的对比，以及 `unverified` 列表结尾。

预期的形态：

```json
{
  "builds": {
    "idioms-x86_64-O2": {
      "arch": "x86_64", "level": "O2", "flags": "x86_64",
      "functions": {
        "div7u": [{"kind": "division", "source": "x / 7", "evidence": ["0x30", "0x32", "0x39", "0x3d", "0x3f", "0x41", "0x43"], "verified": true, "form": "magic multiply with add fix-up", "magic": "0x24924925", "effectiveMagic": "0x124924925", "shift": 35, "...": "..."}],
        "table_switch": [{"kind": "switch", "source": "switch (k) as lookup table", "table": ".rodata+0x28", "entries": 7, "values": [10, 20, 30, 40, 50, 60, 71], "...": "..."}],
        "sum_n": [{"kind": "vector loop", "lanes": 4, "accumulators": 2, "elementsPerIteration": 8, "...": "..."}]
      }
    }
  },
  "kindMatrix": {"idioms-x86_64-O2": {"division": 5, "modulo": 3, "switch": 6, "vector loop": 3, "...": "..."}},
  "sameCodeAtO2AndO3": {"x86_64": {"identical": true}, "arm64": {"identical": false, "differences": ["..."]}},
  "unverified": []
}
```

按以下顺序阅读 `code/main.py`：

1. `parse_listing()` 把 `llvm-objdump -d -r` 文本转换成两种架构上的函数、指令和附带的重定位。
2. `ElfFile` 读取节、符号和 `.rela.*` 条目，`words()` 读取表条目。
3. `magic_numbers()`、`divisor_from_magic()` 和 `evaluate_division()` 分别是 Granlund-Montgomery 推导的正向、反向和指令序列形式。
4. `parse_source()` 从 C 文件中提取校验所需的构造。
5. `Evaluator` 是逐基本块的符号遍，为寄存器和栈槽命名。
6. `find_divisions()` 和 `find_modulos()` 匹配乘法、移位、修正、乘回以及 2 的幂形式。
7. `find_switches()` 对跳转表、查找表、线性映射、选择链、比较树和比较链进行分类。
8. `find_vector_loops()`、`find_unrolled_loops()`、`find_tail_calls()`、`find_inlined_calls()`、`find_selects()`、`find_memory_builtins()`、`find_stack_protector()`、`find_cet()` 和 `find_strength_reduction()` 覆盖其余部分。
9. `analyze_function()` 按顺序运行它们，把每个惯用法得到的结果名送入下一轮求值器。

还有三个命令。`python3 code/main.py scan FILE.objdump.txt [FUNCTION]` 对单份清单运行识别器，目标文件和匹配的 C 文件需要在旁边找到。`python3 code/main.py magic DIVISOR [signed]` 打印乘数、移位量、误差，以及恒等式在边界值上是否成立。`python3 code/main.py figure-data` 打印图所嵌入的数据。

## 使用它

对自己新构建的目标文件，按以下顺序使用识别器。

1. 从目标文件捕获 `llvm-objdump -d -r`，并把精确的编译器命令保存在旁边。识别器从清单头部读取架构。
2. 在清单上运行 `scan`，先读 `unverified` 条目。每一条要么是源码解析器没看到的构造，要么是识别器误读的形态；两者都值得看一眼证据地址。
3. 对每一次除法，读取 `algorithm` 字段。当 `clangMatchesAlgorithm` 为 true 时，编译器选择了最小的有效移位量。为 false 时，编译器有一个你还不知道的原因；把常量记下来。
4. 对每个 switch，比较你所构建的各优化级别上的 `lowering`。表变成树，或树变成表，都标志着一个可以从 LLVM 源码中说出名字的阈值。
5. 对每个循环，读取 `elementsPerIteration` 和 `scalarRemainderLoop`，然后确认第 33 课报告的两条回边属于同一个源码循环。
6. 在同一个目标文件上运行一个反编译器，把它的文本引用在每个惯用法旁边，记录它折叠了哪些。这份记录是第 45 课的基线。

## 交互实验

### 实验 A：推导一个魔数，然后找到它

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization/code
python3 -q
```

运行：

```python
from main import *
magic_numbers(10), magic_numbers(10, signed=True)
magic_numbers(7), magic_numbers(7, signed=True)
divisor_from_magic(0x66666667, 34, signed=True), divisor_from_magic(0x66666667, 33, signed=True)
```

在第一次调用之前，先在纸上算出 `ceil(2^35 / 10)`。然后往 `idioms.c` 的副本里加一个函数 `unsigned div13(unsigned x) { return x / 13; }`，用 `build_fixtures.sh` 里的命令在 `-O2` 下构建，并把清单中的常量与 `magic_numbers(13)` 对照。

### 实验 B：表与树

```python
listing = load_listing("idioms-x86_64-O2.objdump.txt")
elf = load_elf("idioms-x86_64-O2.o")
sources = load_source("idioms.c")
[i.to_dict() for i in analyze_function(listing, "table_switch", elf, sources)]
[i.to_dict() for i in analyze_function(listing, "sparse_switch", elf, sources)]
elf.words(".rodata", 0x28, 7)
```

在 `table_switch` 的副本中把 `71` 改回 `70`，在 `-O2` 下重新构建，观察查找表变成线性映射。然后把一个 case 值改成 1000，观察它仍保持 `ArrayKind` 表。

### 实验 C：三个设置下的循环

```python
for name in ("idioms-x86_64-O1", "idioms-x86_64-O2", "idioms-x86_64-novec-O2"):
    print(name, [i.source for i in analyze_function(load_listing(name + ".objdump.txt"), "sum_n", load_elf(name + ".o"), sources)])
```

手绘 `-O2` 下 `sum_n` 的控制流图，标出保护、掩码、向量主体、归约和余数循环，并与 `detail` 字段比较。

### 实验 D：共享的商

```python
for name in ("idioms-x86_64-O1", "idioms-arm64-O1"):
    print([(i.kind, i.source, i.detail.get("sharedQuotient")) for i in analyze_function(load_listing(name + ".objdump.txt"), "digest", load_elf(name + ".o"), sources)])
```

阅读 `digest` 的 `-O1` 清单，找出 x86-64 上把商与其符号修正组合起来的两条指令。然后解释为什么 AArch64 构建只需要一条。

## 实践练习

为测试夹具添加位技巧，并扩展识别器。

要求：

- 用 GCC 内建函数以及移位或（shift-or）循环移位分别编写 `popcount`、`ctz` 和 `rotl` 函数，在带与不带 `-mpopcnt -mbmi` 两种情况下以 `-O2` 构建，并把目标文件和清单一起提交。
- 识别 `TargetLowering::expandCTPOP` 发出的 `0x55555555`、`0x33333333`、`0x0f0f0f0f`、`0x01010101` population count 序列，把它报告为 `popcount`，以这四个常量作为证据。
- 把 `rep bsfl` 识别为 `tzcnt` 或 `bsf` 的编码，把 `roll %cl` 识别为循环移位，并把二者都解码到对应的内建函数。
- 把新的 kind 加入 `parse_source()` 的校验，使得只有源码调用 `__builtin_popcount` 时，`popcount` 惯用法才通过校验。
- 为每种惯用法各加一个测试，从夹具清单中读取常量。

成功的标准是：`python3 code/main.py` 在两种构建中都报告三种新 kind 且 `verified: true`，测试套件报告的 `unverified` 列表仍然为空。

## 课程产物

本节课附带 `outputs/skill-compiler-idioms-and-optimization.md`。用它把一份清单、一个目标文件及其源码变成一份经过校验的惯用法报告，并附上反编译器对比。它要求为每个结论给出证据地址，为每次除法给出魔数推导，为每个 switch 给出降级类型与条目来源，并为每个惯用法引用一行反编译器输出。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 清单解析器从 `-O2` 目标文件中读出 25 个函数，保留字节，并把 `R_X86_64_PC32 .rodata+0x24` 附加到 `table_switch` 中的 `leaq` 上
- ELF 读取器从 `.rodata+0x28` 返回 `[10, 20, 30, 40, 50, 60, 71]`，并从 `.rela.rodata` 读出十条重定位
- `magic_numbers` 返回移位量 35 的 `0xcccccccd`、移位量 34 的 `0x66666667`、移位量 35 的 `0x124924925`（加法形式）、移位量 32 的 `0x55555556`，以及移位量 33 的 `0xaaaaaaab`
- 对 2 到 64 之间所有非 2 的幂的除数（有符号与无符号），恒等式在 32 位边界值上成立
- `divisor_from_magic` 从各自的参数对还原出 10、5、7 和 3，并拒绝错误的对
- 加法修正序列精确计算 `n // 7`
- 四个 x86-64 除法报告正确的乘数、移位量、除数和形式，四个 AArch64 除法则解码 `mov` 与 `movk`
- `-O0` 使用 `divl` 和 `idivl`，`-O0 -fast-isel=false` 使用乘法，`-O0` 下的 `digest` 混合了二者
- 取模被识别为乘回、`msub`、2 的幂、`sdiv` 加 `mul` 加 `subs`，以及硬件 `idivl`
- 共享的商在 `-O1` 下同时保留除法和取模
- 五种降级方式在两种架构的 `-O2` 下都被分类，包括公式、取值、目标、枢轴和选择表达式
- `-O0` 的 switch 在 x86-64 上是跳转表，在 AArch64 上是比较链
- `digest` 的 switch 在 x86-64 上先是跳转表，然后是折叠因子为 16 的查找表，再是含四个不同目标的跳转表；在 AArch64 上则是带 9 到 10 范围测试的比较树
- `-fno-jump-tables` 产生比较树；16 分支的 `dispatch` 在 x86-64 上从重定位读取目标，在 AArch64 上从压缩字节表读取目标
- 向量循环报告 4 个 lane、2 个累加器、每轮 8 个元素，以及掩码、保护、归约和余数循环
- 固定行程的循环被完全向量化或完全展开，`-fno-vectorize` 下展开 8 次
- `-O1` 没有向量化；`-O2` 与 `-O3` 在 x86-64 上完全相同，在 AArch64 上仅在 `call_switch` 上不同，`-O3` 为 17 条指令，`-O2` 为 18 条
- `-fno-builtin` 把存储循环变成向量化存储循环，并让 `memcpy` 保持为调用
- 尾调用、`memset` 替换、内联的 16 字节复制以及 `-O0` 的两次移动复制都被识别
- 从 `-O1` 起，被内联的被调函数能通过其函数体形态被找到
- 选择指令解码为 `a > b ? a : b`、`-x < 0 ? x : -x`、`(x > 0)` 和 `sum % 10 < 5 ? sum % 10 : 5`
- 栈保护报告 `%fs:0x28`、槽位置和 `__stack_chk_fail`，CET 构建的每个函数都报告 `endbr64`
- 源码解析器读取函数指针参数、switch 分支、调用和累加器名称
- 每个构建中的每种惯用法都通过校验，kind 矩阵一致
- 交互图嵌入与解析所产生相同的数据

## 生产环境中的失败模式

| 失败场景 | 弱工具的报告 | 分析必须证明什么 |
|---|---|---|
| 乘以一个大奇数常量 | 「乘以 1717986919」 | 该常量是某个 `d` 的 `ceil(2^s / d)`，误差满足界，且移位量是最小的有效值 |
| 32 位移位后的加法修正 | 五条算术指令 | 乘数有 33 位宽，尾部是同一方法的加法形式 |
| 相同乘数，不同移位量 | 同一个除数 | 移位量决定除数：34 是 10，33 是 5 |
| 2 的幂有符号取模 | 「加 7、掩码、相减」 | 偏置让舍入朝向零，结果是 `x % 8` |
| 从 `.rodata` 的索引加载 | 「一张跳转表」 | 一张取值查找表，条目从加数加 4 处的字节读取 |
| 仿射算术与一条 `cmov` | 「计算 10k + 10」 | 一个被降级为线性映射的 switch，并对照每个源码 case 校验 |
| 带有序分裂的比较链 | 「一串 if」 | 对 case 常量做二分查找，并说出枢轴 |
| 跳向 default 块的后向分支 | 「一个循环」 | 目标以无条件跳转离开该区域；它是共享的尾部 |
| 一个函数中的两条回边 | 「两个循环」 | 一个源码循环：向量主体及其标量余数 |
| 没有 `ret` 的 `jmp memset@PLT` | 「这个函数没有返回」 | 一个通过被调函数返回的尾调用，替换了存储循环 |
| 被调函数从调用图中消失 | 「从未被调用」 | 被内联了：被调函数的函数体形态出现在调用者中 |
| `%fs:0x28` 的加载与比较 | 「读取线程局部数据」 | 栈保护 canary，带有通向 `__stack_chk_fail` 的失败路径 |
| 函数开头的四个奇怪字节 | 「函数之前的数据」 | `endbr64`，CET 着陆垫 |
| 反编译器显示原始算术 | 「工具错了」 | 工具没有应用折叠；推导给出源码形态，引用的输出记录了这一差距 |

## 操作守则

匹配形态，反向执行推导，写下能证明它的常量。为每个惯用法报告证据地址和解码出的源码形态，对照你编译的源码校验每一个，并引用每个反编译器对同一份字节打印出的内容。

## 延伸阅读

- [Granlund, Montgomery, "Division by Invariant Integers using Multiplication" (PLDI 1994)](https://gmplib.org/~tege/divcnst-pldi94.pdf)
- [Warren, Hacker's Delight, 2nd edition, Chapter 10: Integer Division By Constants](https://www.informit.com/store/hackers-delight-9780321842688)
- [Lemire, Kaser, Kurz, "Faster Remainder by Direct Computation" (2019)](https://arxiv.org/abs/1902.01961)
- [libdivide, header-only division by runtime constants](https://github.com/ridiculousfish/libdivide)
- [LLVM `DivisionByConstantInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/DivisionByConstantInfo.cpp)
- [LLVM `SwitchLoweringUtils.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SwitchLoweringUtils.cpp) and [`TargetLoweringBase.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/TargetLoweringBase.cpp)
- [LLVM `SimplifyCFG.cpp`, `simplifySwitchLookup`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Utils/SimplifyCFG.cpp)
- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html) and [LLVM 3.4 release notes](https://releases.llvm.org/3.4/docs/ReleaseNotes.html)
- [LLVM `LoopIdiomRecognize.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/LoopIdiomRecognize.cpp)
- [LLVM `X86CmovConversion.cpp`](https://github.com/llvm/llvm-project/blob/release/20.x/llvm/lib/Target/X86/X86CmovConversion.cpp)
- [LLVM `StackProtector.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/StackProtector.cpp) and [`X86ISelLoweringCall.cpp`, `getIRStackGuard`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLoweringCall.cpp)
- [glibc `sysdeps/x86_64/nptl/tls.h`, `tcbhead_t`](https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/x86_64/nptl/tls.h)
- [GCC `tree-switch-conversion.cc`](https://github.com/gcc-mirror/gcc/blob/master/gcc/tree-switch-conversion.cc), [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), and [GCC 12 release notes](https://gcc.gnu.org/gcc-12/changes.html)
- [Ghidra decompiler `ruleaction.cc`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/decompile/cpp/ruleaction.cc)
- [angr decompiler documentation](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 book: decompilers](https://book.rada.re/arch/decompile.html)
- [Compiler Explorer API](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/API.md)
- [Godbolt, "Optimizations in C++ Compilers: A practical journey", ACM Queue 17(5), 2019](https://queue.acm.org/detail.cfm?id=3372264)
- [Microsoft, `/GS` (Buffer Security Check)](https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check)
