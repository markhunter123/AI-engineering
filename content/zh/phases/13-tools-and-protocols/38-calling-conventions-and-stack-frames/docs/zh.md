# 调用约定与栈帧：System V、Windows x64 与 AArch64

> 反汇编能显示一个函数读取了哪些寄存器。但只有 ABI 才能说明这些寄存器意味着什么，而 ABI 由目标文件的头部决定。因此本课先读文件头，再按把每个寄存器放进去的规则逐一核对。

**Type:** Build
**Languages:** Python, C, x86-64 and AArch64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~140 minutes

## 学习目标

- 依据目标文件头选择调用约定：ELF x86-64 使用 System V psABI，COFF AMD64 使用 Microsoft x64，ELF AArch64 使用 AAPCS64，Mach-O arm64 使用 Apple 的变体。
- 对每种 ABI 逐字节解码一个函数的序言（prologue），并把每个被保存的寄存器、返回地址和帧指针放到它们相对入口栈指针的偏移上。
- 从函数先读后写的寄存器推断参数寄存器，从入口栈指针之上的加载推断栈参数槽位。
- 区分 System V 的两条独立寄存器序列、Microsoft x64 的位置槽，以及 AAPCS64 的 NGRN 与 NSRN 计数器。
- 把每条推断与源码签名、规范中的规则，以及编译器写在 `.pdata`、`.xdata` 和 `.eh_frame` 中的展开数据逐一核对。
- 比较 Ghidra、IDA、Binary Ninja、radare2、rizin 和 angr 如何命名并恢复同样的事实，记录它们的输出与源码不一致之处。

## 问题

你用 `-O2` 为 Linux x86-64 编译了 `convention.c`，然后向 radare2 6.2.4 询问 `eight_args` 的参数。它给出了七个：`rdi`、`rsi`、`rdx`、`r8`、`r9`、`rsp+0x8` 和 `rsp+0x10`。而 C 源码里有八个参数。指令字节中只读取了一次 `%rcx`，就在 `leaq (%rdx,%rcx), %rax` 中，而工具的变量恢复没有把一个只出现在地址里的寄存器算进去。

你又在 rizin 0.9.1 中打开了同一个文件的 Windows 构建。它把调用约定报告为 `amd64`，并找到了 `r8`、`rcx`、`rdx` 和三个栈槽。这个对象是 COFF，所以约定应当是 Microsoft x64，第四个寄存器参数是 `r9`，栈槽有四个。两个工具在同一份源码上漏掉了不同的寄存器。

一份清单（listing）就是一串指令。调用约定则是存在于清单之外的契约：哪些寄存器携带参数、被调用者必须保留哪些寄存器、返回地址和栈参数位于何处、调用者必须在栈指针之上留下什么。有四份文档定义了这里关键的契约：System V AMD64 psABI（Version 1.0，构建日期 2025 年 3 月 12 日）第 3.2.1 至 3.2.3 节；Microsoft x64 调用约定、栈使用以及 prolog 与 epilog 页面（页面日期从 2018-12-17 到 2025-03-19）；Arm AAPCS64 2025Q4 版本（2026 年 1 月 23 日）第 6.1 至 6.9 节；以及 Apple 的 arm64 平台说明。本课阅读这些规则，把它们应用到一个以三种方式编译的 C 文件上，并构建一个分析器，把清单变成一份经过核对的数据：参数位置、被保存寄存器和帧偏移。

```figure
re-38-three-frames
```

## 一个文件头决定用哪本规则书

`convention.c` 有十二个函数：`leaf_add`、`eight_args`、`mixed`、`by_value`、`big_ret`、`sum_varargs`、`keeps`，以及五个调用形状相同的 `extern` 函数的调用者。结构体使用 `long long`，因为 `long` 在 MSVC 目标上是 4 字节，在其他目标上是 8 字节，而结构体大小一变，约定就变。Apple clang 17.0.0 在这台机器上把它编译了七次：`-target x86_64-unknown-linux-gnu`、`x86_64-pc-windows-msvc` 和 `aarch64-unknown-linux-gnu` 各在 `-O0` 与 `-O2` 下编译，再加上 `arm64-apple-macos` 的 `-O2`。机器上没有安装 Linux 或 Windows 头文件，所以该文件自行声明所需内容，只包含 clang 自带的 `stdarg.h` 和 `stddef.h`。

`code/main.py` 中的 `describe_object()` 读取 4 字节魔数和一个机器字段，然后说出契约的名字：

| 文件头 | 字段 | 值 | 规则书 |
|---|---|---|---|
| `7f 45 4c 46` ELF | 偏移 0x12 处的 `e_machine` | 62 | System V AMD64 psABI |
| `64 86` COFF | 偏移 0 处的 `Machine` | 0x8664 | Microsoft x64 |
| `7f 45 4c 46` ELF | `e_machine` | 183 | AAPCS64 |
| `cf fa ed fe` Mach-O 64 | 偏移 4 处的 `cputype` | 0x0100000c | 带 Apple 差异的 AAPCS64 |

三份契约在接下来这张表的几乎每一行上都不一致。每个值都引自表头行所指的文档，每一项都通过与固定样本（fixture）对照的测试来检查。

| 规则 | System V psABI | Microsoft x64 | AAPCS64 |
|---|---|---|---|
| 整数参数 | `%rdi %rsi %rdx %rcx %r8 %r9`，取下一个空闲寄存器 | `RCX RDX R8 R9`，按位置 | `x0` 至 `x7`，按 NGRN |
| 浮点参数 | `%xmm0` 至 `%xmm7`，取下一个空闲寄存器 | `XMM0` 至 `XMM3`，按位置 | `v0` 至 `v7`，按 NSRN |
| 整数返回值 | `%rax`，然后 `%rdx` | `RAX` | `x0`，然后 `x1` |
| 浮点返回值 | `%xmm0`，然后 `%xmm1` | `XMM0` | `v0` |
| 结果过大无法放入寄存器 | 地址按「视同第一个参数」的方式传入 `%rdi`，结果从 `%rax` 返回 | 指针放在 `RCX`，参数整体右移，结果从 `RAX` 返回 | 地址放在 `x8`，`x0` 至 `x7` 不变 |
| 被调用者保存 | `%rbp %rbx %r12` 至 `%r15` | `RBX RBP RDI RSI RSP R12` 至 `R15`，`XMM6` 至 `XMM15` | `r19` 至 `r29`、`SP`、`v8` 至 `v15` 的低 64 位 |
| 入口处第一个栈参数 | `8(%rsp)` | `40(%rsp)`，位于 32 字节影子存储之上 | `0(sp)` |
| 栈指针以下 | 128 字节红区 | 「视为 volatile」 | 无红区（Apple：128 字节） |
| 调用时的对齐 | 入口处 `(%rsp + 8)` 是 16 的倍数 | prolog 与 epilog 之外 16 字节对齐 | `SP mod 16 = 0` |
| 可变参数标记 | `%al` 保存所用向量寄存器的上界 | 浮点值在对应整数寄存器中再复制一份 | 寄存器中无标记；Apple 使用 8 字节栈槽 |

## System V：两条独立寄存器序列与一个红区

psABI 3.2.3 把每个参数的每个八字节（eightbyte）分类为 INTEGER、SSE 或 MEMORY，然后从两个互不影响的列表中分配寄存器。「INTEGER」取「序列 %rdi、%rsi、%rdx、%rcx、%r8 和 %r9 中的下一个可用寄存器」。SSE 取 `%xmm0` 至 `%xmm7`。「如果某个参数的任一八字节都没有可用寄存器，整个参数就通过栈传递」，而内存参数「按相反（从右到左）的顺序压入栈」。

`eight_args(a, b, c, d, e, f, g, h)` 在 `-O2` 下是 27 字节：

```text
      10: 48 01 fe                     	addq	%rdi, %rsi
      13: 48 8d 04 0a                  	leaq	(%rdx,%rcx), %rax
      17: 48 01 f0                     	addq	%rsi, %rax
      1a: 4c 01 c0                     	addq	%r8, %rax
      1d: 4c 01 c8                     	addq	%r9, %rax
      20: 48 03 44 24 08               	addq	0x8(%rsp), %rax
      25: 48 03 44 24 10               	addq	0x10(%rsp), %rax
      2a: c3                           	retq
```

六个寄存器，然后在 `8(%rsp)` 和 `16(%rsp)` 处两次加载。psABI 的图 3.3 把「内存参数 eightbyte 0」放在 `16(%rbp)`（当 `%rbp` 是帧基址时），在任何 push 之前这就是入口处的 `8(%rsp)`。`g` 在 8，`h` 在 16。

`mixed(int i, double d, long long l, float f)` 展示了这两条列表。整数取 `%edi` 和 `%rsi`，即第一个和第二个 INTEGER 寄存器。double 和 float 取 `%xmm0` 和 `%xmm1`，即第一个和第二个 SSE 寄存器。第一条指令是 `cvtsi2sd %edi, %xmm2`，而 `%xmm2` 是一个函数先写后读的临时寄存器。把 `cvtsi2sd` 的目的操作数当成读取的分析器会凭空造出第三个浮点参数。radare2 正是这么做的，它把 `mixed` 报告为 `(double arg1, double arg2, double arg3)`，分别占 `xmm0`、`xmm1`、`xmm2`。

`by_value(struct pair p)` 把一个 16 字节结构体作为两个 INTEGER 八字节放在 `%rdi` 和 `%rsi` 中传入，并以同样方式返回：`leaq 0x1(%rdi), %rax; leaq -0x1(%rsi), %rdx; retq`。`big_ret(long long seed)` 返回 32 字节，属于 MEMORY 类，psABI 规定「调用者为返回值提供空间，并把该存储的地址放在 %rdi 中传入，如同它就是第一个参数」。`seed` 移到 `%rsi`，函数以 `movq %rdi, %rax` 开头，因为「返回时 %rax 将包含调用者经 %rdi 传入的地址」。

红区是 System V 的另一个签名。psABI 3.2.2：「%rsp 所指位置之外的 128 字节区域视为保留，不得被信号或中断处理程序修改。」在 `-O0` 下，`eight_args` 是 `pushq %rbp; movq %rsp, %rbp`，然后向 `-0x8(%rbp)` 到 `-0x30(%rbp)` 做六次存储，没有 `subq`。这些存储位于栈指针以下 48 字节。`sum_varargs` 的 `-O2` 构建用 `subq $0x58, %rsp` 分配 88 字节，然后把寄存器保存区写到 `-0x58(%rsp)` 直到 `-0x80(%rsp)`：栈指针以下 128 字节，整整一个红区。用 `-mno-red-zone` 编译同一文件，`leaf_add` 会多出一条 `pushq %rax` 和一条 `addq $0x8, %rsp`，把它的 8 字节局部变量放到栈指针之上。

可变参数调用要多用一个寄存器。「%al 用作隐藏参数，指定所使用的向量寄存器数量」，取值是 0 到 8 的上界。`call_varargs` 在 `-O2` 下用 `xorl %eax, %eax` 设置它，在 `-O0` 下用 `movb $0x0, %al`。`sum_varargs` 先读它：`testb %al, %al; je` 在没有使用向量寄存器时跳过八条 `movaps %xmmN` 保存。

## Microsoft x64：按位置、影子存储与无红区

Microsoft 的规则是按位置的。「整数参数在寄存器 RCX、RDX、R8 和 R9 中传递。浮点参数在 XMM0L、XMM1L、XMM2L 和 XMM3L 中传递。」并且「前四个参数中的任何浮点和双精度参数都按位置在 XMM0 至 XMM3 中传递。」同一个 `mixed` 变成 `%ecx`、`%xmm1`、`%r8`、`%xmm3`：槽 0 是 int，所以是 `RCX`；槽 1 是 double，所以是 `XMM1`；槽 3 是 float，所以是 `XMM3`。`XMM0` 和 `XMM2` 保持为空。

调用者还必须留出空间。「调用者必须始终分配足够空间存放四个寄存器参数，即使被调用者并没有那么多参数。」相关页面称之为影子存储（shadow store）或 home 地址，四个槽「实际上归被调用函数所有」。入口时 `0(%rsp)` 是返回地址，`8(%rsp)` 到 `39(%rsp)` 是四个 home，第一个栈参数在 `40(%rsp)`。`eight_args` 在 `-O2` 下读取 `0x28(%rsp)`、`0x30(%rsp)`、`0x38(%rsp)`、`0x40(%rsp)`：40、48、56、64。

调用者一侧能看出这套算术。`call8` 在 `-O2` 下是 `subq $0x48, %rsp`，然后在 `movl $1, %ecx` 等指令之前把 5、6、7、8 存到 `0x20(%rsp)` 至 `0x38(%rsp)`。72 字节中，32 字节给影子存储，32 字节给四个栈参数，8 字节让返回地址落在 16 字节边界上。`call_varargs` 只有寄存器参数，仍然分配 `subq $0x28, %rsp`：32 字节给影子存储，8 字节给对齐。`keeps` 压入三个非 volatile 寄存器（24 字节），然后 `subq $0x20, %rsp`，好让 `ext` 在返回地址之下找到它的 32 字节影子存储。

结构体遵循自己的规则。「任何放不进 8 字节的参数，或大小不是 1、2、4、8 字节的参数，必须按引用传递。」以及「16 字节参数按引用传递。」因此 `by_value` 在 `%rcx` 的隐藏结果指针之后，于 `%rdx` 收到一个指向副本的指针：「调用者必须为返回值分配内存，并把指向它的指针作为第一个参数传入。其余参数于是整体右移一个参数位置。」`call_pair` 用 `movaps` 在 `0x20(%rsp)` 构造副本，用 `leaq 0x20(%rsp), %rdx` 传入，用 `leaq 0x38(%rsp), %rcx` 传入结果缓冲区，并从 `0x40(%rsp)` 读回 `.b`。

没有红区。「RSP 当前地址之外的所有内存都视为 volatile：操作系统或调试器可能在用户调试会话中覆盖这块内存，中断处理程序也可能如此。」`-O0` 下的 `leaf_add` 需要 8 字节局部变量，它在入口用 `pushq %rax`、在 `retq` 前用 `popq %rcx` 取得，两条单字节指令代替了一条 `subq` 和一条 `addq`。栈使用页面定义了允许这么做的叶子函数：「叶子函数是不需要函数表项的函数。它不能修改任何非 volatile 寄存器，包括 RSP。」

可变参数被调用者会把寄存器 home 到栈上。「对于 vararg 或未原型函数，任何浮点值都必须在对应的通用寄存器中再复制一份。」`sum_varargs` 在 `-O2` 下一次 `pushq %rax` 之后，把 `%rdx`、`%r8`、`%r9` 存到 `0x18(%rsp)`、`0x20(%rsp)`、`0x28(%rsp)`，即入口偏移 16、24、32：第二个、第三个和第四个寄存器参数的 home。它的 `va_list` 是 `leaq 0x18(%rsp), %rax`，即 `%rdx` 的 home，也就是第一个未命名参数。

## AAPCS64，以及 Apple 改变了什么

AAPCS64 6.8.2 用两个计数器走一遍规则列表。C.9：整数参数「复制到 x[NGRN] 的最低有效位」，然后 NGRN 递增。C.1 对浮点参数用 NSRN 做同样的事。C.12：最多 `8 - NGRN` 个双字的复合类型「复制到连续通用寄存器中」。C.13 规定一旦某个参数上了栈，「NGRN 设为 8」，因此之后没有东西再回到寄存器。`eight_args` 使用 `x0` 至 `x7`，从不碰栈。`mixed` 是 `w0`、`d0`、`x1`、`s1`。`by_value` 的输入和输出都是 `x0` 和 `x1`。

6.9 规定，本应在寄存器中传递的结果「在同样的寄存器中返回」，否则「内存块的地址应作为额外参数在 x8 中传给函数」。`big_ret` 写 `str x0, [x8]`、`stur q0, [x8, #0x8]`、`str x9, [x8, #0x18]`，从不写 `x0`。`call_big` 设置 `mov x8, sp`，而 `seed` 留在 `x0`，因为 6.9 补充「不要求被调用者保留 x8 中存储的值」，且 `x8` 不是参数寄存器之一。

6.1.1：「子程序调用必须保留寄存器 r19-r29 和 SP 的内容。」6.4.5.2：在公共接口处「SP mod 16 = 0」。栈是满递减的，且「任何线程都不得访问（读或写）其下方的非活动区域」，因此 Linux `-O0` 的 `leaf_add` 为 8 字节局部变量发出 `sub sp, sp, #0x10`，而 System V 用的是红区。

6.4.6 定义帧记录：「栈上一个由两个 64 位值组成的帧记录……地址最低的双字应指向前一个帧记录，地址最高的双字应包含进入当前函数时经 LR 传入的值。」以及「帧记录在栈帧中的位置未作规定。」clang 在这同一个文件里展示了两种位置。Linux 上 `-O2` 的 `keeps` 把记录放在底部：`stp x29, x30, [sp, #-0x30]!; mov x29, sp`。同一函数在 `-O0` 下把它放在顶部：`sub sp, sp, #0x20; stp x29, x30, [sp, #0x10]; add x29, sp, #0x10`。

Apple 的 arm64 说明保留了寄存器和对齐，改变了帧、红区和可变参数调用。「帧指针寄存器（x29）必须始终指向有效的帧记录。某些函数，例如叶子函数或尾调用，可以选择不在此列表中创建条目。」Apple 构建的 `keeps` 先保存 `x22, x21, x20, x19`，再把记录放在顶部：`stp x29, x30, [sp, #0x20]; add x29, sp, #0x20`。「平台保留寄存器 x18。不要使用该寄存器。」「ARM64 红区由紧邻栈指针下方的 128 字节组成」，而 clang 对 `-O0` 叶子函数仍然发出 `sub sp, sp, #0x10`，说明编译器默认不使用它。可变参数被分配到「适当数量的 8 字节栈槽」，并且「类型 va_list 是 char* 的别名」。Apple 上的 `call_varargs` 把 10、20、30 存到 `[sp]`、`[sp, #8]`、`[sp, #0x10]`，只传 `w0`；Linux 构建传 `w1`、`w2`、`w3`，并用 `b extva` 做尾调用。Apple 的 `sum_varargs` 读取 `add x8, sp, #0x10`，即调用者的栈指针，作为它的 `va_list`；Linux 的 `sum_varargs` 把 `x1` 至 `x7` 和 `q0` 至 `q7` 保存进一个 224 字节的帧，并用一条 `mov x8, #-0x38; movk x8, #0xff80, lsl #32` 初始化结构体字段 `__gr_offs` 为 -56、`__vr_offs` 为 -128。

符号也不同。Mach-O 给每个 C 符号加 `_` 前缀，所以清单里显示 `_keeps`，第一个函数则以私有标签 `ltmp0` 出现。分析器通过符号表重命名这些标签。

## 手工解码同一段序言，三次

`keeps(int n)` 在循环中调用 `ext(i)`，并在调用前后保留 `n`、`i` 和 `acc`，所以每种 ABI 都必须保存寄存器。阅读每个目标 `-O2` 构建的头几个字节。

System V，4 字节：

1. `55` 是 `push r64`，寄存器号在低三位：`50 + 5`，即 `%rbp`。深度 8，`%rbp` 保存在入口偏移 -8。
2. `41 56`：`41` 是设置了 `B` 的 REX，`56` 是 `50 + 6`。REX.B 把 6 扩展到 14：`pushq %r14` 于 -16。
3. `53` 是 `50 + 3`：`pushq %rbx` 于 -24。

三次 push 共 24 字节。返回地址再加 8 字节，32 是 16 的倍数，所以循环里的 `callq ext` 无需 `subq` 就已对齐。这里没有 `movq %rsp, %rbp`，因此这个构建没有帧指针。函数体把 `%rbp` 当作保存 `acc` 的被调用者保存寄存器（`xorl %ebp, %ebp`、`addl %eax, %ebp`）。尾声与之镜像：`5b 41 5e 5d c3`。

Microsoft x64，7 字节：

1. `56` 是 `pushq %rsi` 于 -8。`57` 是 `pushq %rdi` 于 -16。`53` 是 `pushq %rbx` 于 -24。这里三者都是非 volatile 的，而 `RDI` 和 `RSI` 在 System V 下是参数寄存器。
2. `48 83 ec 20`：REX.W，操作码 `83 /5` 带 8 位立即数，ModRM `ec` 是 `11 101 100`：`subq $0x20, %rsp`。深度 56。

56 加上返回地址是 64，已对齐。这 32 字节就是欠 `ext` 的影子存储。epilog 页面只允许「一条 add RSP,constant 或 lea RSP,constant[FPReg]，随后是零条或多条 8 字节寄存器弹出，再加一条 return 或 jmp」，而字节是 `48 83 c4 20 5b 5f 5e c3`。

AArch64，16 字节，四个字，小端：

1. `a9bd7bfd` 是带前索引的 `stp`。位 31:30 的 `10` 选择 64 位寄存器，位 25:23 的 `011` 选择前索引，位 22 为 0 表示存储。位 21:15 的 imm7 是 `1111010`，即 -6，按 8 缩放为 -48。位 14:10 的 Rt2 是 30，位 9:5 的 Rn 是 31（`sp`），位 4:0 的 Rt 是 29。`stp x29, x30, [sp, #-0x30]!`：栈指针下降 48，`x29` 到 -48，`x30` 到 -40。
2. `f9000bf5` 是带无符号 12 位偏移并按 8 缩放的 `str`：位 21:10 的 imm12 是 2，Rn 是 `sp`，Rt 是 21。`str x21, [sp, #0x10]`，入口偏移 -32。
3. `910003fd` 是立即数 `add`：imm12 是 0，Rn 是 `sp`，Rd 是 29。`mov x29, sp`：帧指针指向 -48 处的记录。
4. `a9024ff4` 是带带符号偏移的 `stp`：imm7 是 4，缩放为 32。Rt2 是 19，Rt 是 20。`stp x20, x19, [sp, #0x20]`：`x20` 在 -16，`x19` 在 -8。

帧共 48 字节：`x29` 在 0，`x30` 在 8，`x21` 在 16，24 处是 8 字节填充，`x20` 在 32，`x19` 在 40。记录中地址最低的字指向上一记录，下一个字保存 LR，符合 6.4.6 的要求。`cmp w0, #0x1` 位于第一次和第二次存储之间；调度器把它移进了序言，而分析器容忍栈操作之间出现一条比较或算术指令。

## 编译器把同样的事实写了两遍

解析序言是一次推导。编译器为展开器（unwinder）再写一份，两者必须一致。

在 Windows 上，第二份推导是 `.pdata` 和 `.xdata`。不带 `-fno-asynchronous-unwind-tables` 构建 `-O2` 对象，`llvm-objdump -h` 显示 0x34 字节的 `.xdata` 和 0x48 字节的 `.pdata`。0x48 是 72，即六个 12 字节的 `RUNTIME_FUNCTION` 项，每项带三个 `IMAGE_REL_AMD64_ADDR32NB` 重定位。这六项是 `sum_varargs`、`keeps`、`call8`、`call_varargs`、`call_pair` 和 `call_big`。另外六个函数是叶子函数，异常处理页面说：「如果搜索没有找到函数表项，就假定代码属于叶子函数，RSP 直接指向返回指针。」

`keeps` 的 `.xdata` 字节是 `01 07 04 00 07 32 03 30 02 70 01 60`。`UNWIND_INFO` 以 3 位的 Version 和 5 位的 Flags 开头：版本 1，无标志。SizeOfProlog 为 7，CountOfCodes 为 4，FrameRegister 和 FrameOffset 为 0。每个 `UNWIND_CODE` 是序言中的一个偏移加一个操作码与信息字节。`07 32`：偏移 7，操作 2 是 `UWOP_ALLOC_SMALL`，信息 3 表示 `3 * 8 + 8 = 32` 字节。`03 30`：操作 0 是 `UWOP_PUSH_NONVOL`，寄存器 3 是 `rbx`。`02 70`：`rdi`。`01 60`：`rsi`。这些编码按执行顺序的逆序存储，它们指明的正是字节 `56 57 53 48 83 ec 20` 中同样的三次 push 和同样的 32 字节分配。`sum_varargs` 的项是 `01 01 01 00 01 02 00 00`：1 字节序言、一个编码、8 字节的 `UWOP_ALLOC_SMALL`。展开器就是这样看 `pushq %rax` 的：一次分配，因为 `%rax` 是 volatile 的，并没有保存任何东西。

在 Linux 上，第二份推导是 `.eh_frame`，它由 psABI 4.2.4 定义、由 DWARF 5 第 6.4 节编码。对可重定位对象运行 `llvm-dwarfdump --eh-frame`，每个函数打印一个 FDE，并解析 `R_X86_64_PC32` 重定位，因此区间可直接使用。对 `keeps` 而言：

```text
  DW_CFA_def_cfa_offset: +16
  DW_CFA_def_cfa_offset: +24
  DW_CFA_def_cfa_offset: +32
  DW_CFA_offset: RBX -32
  DW_CFA_offset: R14 -24
  DW_CFA_offset: RBP -16
```

CFA 是调用前的栈指针，比入口栈指针高 8，所以 `RBP -16` 是入口偏移 -8，`R14 -24` 是 -16，`RBX -32` 是 -24，而最深的 CFA 偏移 32 是 24 字节帧加上返回地址。`keeps` 的 AArch64 FDE 写着 `DW_CFA_def_cfa: W29 +48`，并带 `W29 -48`、`W30 -40`、`W21 -32`、`W20 -16`、`W19 -8`。在这里 CFA 就是入口栈指针本身，所以这些偏移就是未经换算的帧偏移，并且 CFA 由 `x29` 定义，此时帧记录仍然有效。`eh_frame_cross_check()` 对两个对象的全部 12 个 FDE 运行这一比较。有一项需要小心：System V 下的 `call8` 在函数体中途压入两个传出参数，所以它最深的 CFA 偏移是 32，而它的序言帧是 8，检查比较的是函数体到达的最深栈指针 24，再加 8。

## 各工具如何称呼这些概念

这一领域的每个工具都为调用约定准备了一个数据结构，以及一个把约定分配给函数的 pass。名字不同，缺口也不同。闭源工具按其文档描述；开源工具在固定样本上实际运行，输出捕获在对象旁边。

**Ghidra 12.1.4（Apache-2.0）。** 规则存放在 SLEIGH 语言旁边的编译器规范文件 `.cspec` 中。`x86-64-gcc.cspec` 声明一个名为 `__stdcall` 的 `<default_proto>`，「源自 System V Application Binary Interface AMD64 Architecture Processor Supplement April 2016」，其 `<pentry>` 列表包含 `XMM0_Qa` 至 `XMM7_Qa` 和 `RDI, RSI, RDX, RCX, R8, R9`，偏移 8 处有一个栈项，针对不超过 16 字节的类型有 `join_dual_class` 规则，输出为 `XMM0_Qa, XMM1_Qa, RAX, RDX`，其余情况用 `hidden_return`。`x86-64-win.cspec` 声明 `__fastcall`，有四个 `<group>` 元素，把 `XMM0_Qa` 与 `RCX`、`XMM1_Qa` 与 `RDX` 等配对，这就是位置规则；偏移 40 处有一个栈项；对 3、5、6 或 7 字节的结构体使用 `convert_to_ptr`，注释为「只有 8、16、32、64 位的结构体才能打包进寄存器」；`unaffected` 列出 `RBX RBP RDI RSI RSP R12` 至 `R15` 和 `XMM6` 至 `XMM15`；`localrange` 为栈 8 到 39，即影子存储。`AARCH64.cspec` 声明 `__cdecl`，第一项是带 `storage="hiddenret"` 的 `x8`，然后是 `q0` 至 `q7` 和 `x0` 至 `x7`，对 17 字节及以上的结构体使用 `convert_to_ptr`，`killedbycall` 包含 `x8`，注释为「间接结果位置寄存器，未反映在 pentry 列表中」。反编译器的最后阶段名为「Establish Function's Prototype」；分析器是「Call Convention ID」和「Decompiler Parameter ID」；程序选项「Prototype Evaluation」决定默认模型。

**IDA Pro 与 IDA Free（Hex-Rays，商业软件；当前版本 9.4，构建 9.4.260714）。** 函数类型是通过 SetType 设置的 C 声明，并且「IDA 会使用类型信息在反汇编中注释函数参数的信息」。标准约定不会产生的位置用 IDA 自己的约定书写：`int __usercall func@<ebx>(int x, int y@<esi>)` 把结果放在 `EBX`、把 `y` 放在 `ESI`，而 `@<^12.4>` 按偏移和大小命名一个栈位置。`__userpurge` 是同样的形式，但由被调用者清理栈。同一页面还列出 `__spoils` 用于被破坏寄存器集合，`__hidden`、`__return_ptr` 和 `__struct_ptr` 用于本课手工恢复的那些隐式参数。反编译器的「Decompile as call」动作可以为一个指令接受 `__usercall` 原型。IDA Free 只支持 x86 和 x86-64。

**Binary Ninja 6.0「Krypton」（Vector 35，商业软件，另有可反编译 x86、x86_64、ARMv7 和 AArch64 的免费版）。** `Platform`「包含与二进制执行环境相关的所有信息，主要是所用的调用约定、操作系统和架构」，列表包括 `linux-x86_64`、`windows-x86_64`、`linux-aarch64` 和 `mac-aarch64`，每个都有一个 `default_calling_convention`。`CallingConvention` 有 `int_arg_regs`、`float_arg_regs`、`callee_saved_regs`、`caller_saved_regs`、`int_return_reg`、`high_int_return_reg`、`float_return_reg`，还有两个标志直接编码了本课中的差异：`arg_regs_share_index`，「为 True 时，第 N 个参数无论类型如何都占用整数与浮点寄存器列表的第 N 个槽」，以及 `stack_reserved_for_arg_regs`，「Windows x64 调用约定使用的影子／home 空间」。`eligible_for_heuristics`、`required_arg_regs` 和 `required_clobbered_regs` 驱动它的启发式约定检测。平台类型是 `types/platform/` 下的 C 头文件，每个平台名一个。

**radare2 6.2.4（LGPL-3.0，此处实际运行）。** `afcl` 列出架构插件所知道的约定：x86-64 为 `amd64 amd64syscall dlang golang golang_abi0 ms p9 reg swift`，AArch64 为 `arm64 dlang golang p9 reg swift`。`afc` 报告当前函数的约定，`afcr` 打印 `cc.amd64.arg0: rdi` 至 `arg5: r9` 以及 `ret0: rax`，这些配置是名为 `cc-ARCH-BITS.sdb` 的 SDB 文件。radare2 为 ELF 对象选了 `amd64`，为 COFF 对象选了 `ms`，这个区分是正确的。`afv` 列出参数与变量，`afs` 打印签名。在 Linux `-O2` 对象上，`afv` 为 `eight_args` 找到七个参数并漏掉 `rcx`；在 Windows 对象上，它找到 `rcx, rdx, r8` 和四个栈槽，漏掉 `r9`。两个缺失的寄存器都只出现在一条 `lea` 内部。对 `mixed`，它在 Linux 上报告 `double arg1 @ xmm0, arg2 @ xmm1, arg3 @ xmm2`，在 Windows 上则完全没有参数。在 AArch64 上，它为 `eight_args` 找到 `x0` 至 `x7`；对 `big_ret`，它列出 `x0` 和 `double arg2 @ d1`，而该函数读取的是 `x8` 并往 `q1` 装载一个常量。每个 `afs` 签名都以 `void` 开头，因为返回类型没有被推断。在 `-O0` 下，每个参数都存进帧，`afv` 找到了 `eight_args` 的全部八个参数，栈上的两个是 `rbp+0x10` 和 `rbp+0x18`。

**rizin 0.9.1（LGPL-3.0，此处实际运行）与 Cutter 2.5.0。** `afcl` 列出 `amd64 amd64syscall ms reg swift`。对 COFF 对象，`afc` 返回 `amd64`，说明 Microsoft 配置存在，但没有根据格式被选中。`afi` 在 Linux 上为 `eight_args` 列出七个参数，同样没有 `rcx`；在 Windows 上列出 `r8, rcx, rdx` 和三个栈槽，没有 `r9`，也没有 `0x40` 处的槽。`afv` 不存在；列举命令是 `afvl`。Cutter 是 rizin 之上的图形前端。

**angr 10.0.3（BSD-2，此处实际运行）。** 先 `CFGFast`，再 `CompleteCallingConventions(recover_variables=True)`，会按二进制格式把 `function.calling_convention` 设为 `SimCCSystemVAMD64`、`SimCCMicrosoftAMD64` 或 `SimCCAArch64`，并把 `function.prototype` 设为一个签名。它在三个目标上都恢复了 `eight_args` 的全部八个参数；把 x86-64 上 `big_ret` 的隐藏指针恢复为 `(BOT*, long long) -> BOT*`；把 Windows `by_value` 的按引用结构体恢复为 `(BOT*, BOT*) -> BOT*`；把 System V 的可变参数 `sum_varargs` 恢复为 `(long long, ...)`。它的缺口与 radare2 不同：`mixed` 在 System V 和 AArch64 上返回 `(long long, long long, double, float)`，寄存器对但顺序错，因为整数列表先于浮点列表打印；AArch64 的 `big_ret` 返回 `(long long) -> double`，即漏掉了 `x8` 约定，并把向量存储当成了返回值；AArch64 的 `sum_varargs` 返回 16 个参数，即整个寄存器保存区。

**其他。** Hopper 6.6.0 是面向 macOS 和 Linux 的商业反汇编器与反编译器；本课没有运行或记录它的约定处理。lldb 1703 和 gdb 用 `frame info` 与 `register read` 显示活动帧，这是第 42 课对同一批偏移的动态视角。Compiler Explorer 从源码并排显示三份清单，是在字节中寻找某条规则之前确认它的最快方法。

| 概念 | Ghidra | IDA | Binary Ninja | radare2 / rizin | angr | 本课 |
|---|---|---|---|---|---|---|
| 约定对象 | `.cspec` 中的 `prototype` | 函数类型中的调用约定关键字 | `Platform` 上的 `CallingConvention` | `afc` 名称、`cc-ARCH-BITS.sdb` | `SimCC` 子类 | `expected_locations(abi, name)` |
| 每个目标的默认值 | 每个 cspec 的 `<default_proto>` | 编译器设置 | `Platform.default_calling_convention` | `afc=`、`anal.cc` | 按格式 | `describe_object()` |
| 参数寄存器 | `<pentry>` 列表，按位置用 `<group>` | `__usercall` 中的 `@<reg>` | `int_arg_regs`、`float_arg_regs`、`arg_regs_share_index` | `cc.NAME.argN` | `ARG_REGS`、`FP_ARG_REGS` | `INT_ARG_REGS`、`FLOAT_ARG_REGS` |
| 影子存储 | 栈 8 到 39 的 `localrange` | 隐式 | `stack_reserved_for_arg_regs` | `cc.ms.argn=stack` | `STACKARG_SP_DIFF` | `STACK_ARG_BASE[WIN64] = 40` |
| 隐藏结果指针 | `hidden_return`、`storage="hiddenret"` | `__return_ptr`、`__hidden` | 返回类型对 `int_return_reg` 过大 | 未建模 | 第一个参数 `BOT*` | `HIDDEN_RETURN_REG` |
| 被调用者保存 | `<unaffected>` | `__spoils` 列出相反集合 | `callee_saved_regs` | profile | `CALLEE_CLEANUP` 与架构 | `CALLEE_SAVED` |
| 为函数指定约定 | 「Call Convention ID」分析器、「Establish Function's Prototype」 | 用户或类型库 | 用 `required_arg_regs` 的启发式检测 | `afc`、`afva`、`aaft` | `CallingConventionAnalysis` | 先读后写加规则书 |

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
```

输出是一份 JSON 报告。对七个对象中的每一个，它给出格式、机器和 ABI；对十二个函数中的每一个，它给出序言字节、被保存寄存器及其入口偏移、帧大小、帧指针、参数寄存器、栈参数偏移、溢出存储、返回寄存器或隐藏指针、调用深度，以及调用者的传出寄存器与存储。每个函数都以它的检查收尾：规则、由签名得到的期望值、由字节得到的观测值，以及二者是否一致。报告最后是两项展开数据的交叉核对和工具捕获。在 84 次函数分析中，共运行 489 项检查，全部 489 项一致。

预期形状：

```json
{
  "objects": {
    "convention-x86_64-windows-O2": {
      "format": "COFF", "machine": "AMD64", "abi": "Microsoft x64", "checks": "73/73 agree",
      "functions": {
        "keeps": {
          "prologue": ["56  pushq %rsi", "57  pushq %rdi", "53  pushq %rbx", "48 83 ec 20  subq $0x20, %rsp"],
          "frameBytes": 56,
          "savedRegisters": [{"register": "rsi", "entryOffset": -8}, {"register": "rdi", "entryOffset": -16}, {"register": "rbx", "entryOffset": -24}],
          "parameterRegisters": ["rcx"],
          "returnRegisters": ["rax"],
          "calls": [{"callee": "ext", "depth": 56}],
          "checks": [{"rule": "Microsoft x64: the caller must always allocate ...", "expected": "at least 32 bytes ...", "observed": [56], "agree": true}, "..."]
        }
      }
    }
  },
  "sehCrossCheck": ["..."],
  "ehFrameCrossCheck": {"...": "..."}
}
```

按以下顺序阅读 `code/main.py`：

1. `describe_object()` 读取 ELF、COFF 或 Mach-O 头部和符号表，返回 ABI 与函数列表。
2. `parse_listing()` 把 `llvm-objdump -d -r` 的文本拆成函数，保留每次调用和跳转的重定位目标，并按正确语法去掉注释：AT&T 用 `#`，AArch64 用 `//` 和 `;`，在 AArch64 中 `#` 引导立即数。
3. `effect_x86()` 和 `effect_arm()` 把一条指令转成读取、地址读取、写入、加载、存储和一个栈指针增量。AT&T 的最后一个操作数是目的地；AArch64 的第一个操作数是目的地，但存储、比较和分支除外。`lea` 读取一个地址，不加载任何东西。`mov`、`cvtsi2sd`、`setcc` 以及零值惯用法 `xor r, r` 的目的地只是写入。一次调用写入整个 volatile 集合。前索引存储先移动 `sp` 再访问，后索引加载在访问之后移动 `sp`。
4. `analyze_function()` 把函数体走到第一个 `ret` 或尾跳转为止。它跟踪栈指针低于入口值的深度，因此内存操作数 `disp(%rsp)` 位于入口偏移 `disp - depth`，而 `disp(%rbp)` 位于 `frame_pointer + disp`。它记录先读后写的参数类寄存器、处于或高于该 ABI 第一个栈参数偏移的加载、参数寄存器到帧的存储、当前栈指针以下的访问，以及每次调用时的深度。
5. `_return_registers()` 只要结果寄存器在退出前的最后访问是写入，就保留它。作为数据操作数的读取会淘汰候选；作为地址基址的读取不会，这正是 Microsoft x64 下 `big_ret` 中 `rax` 得以保留的原因——那里 `movq %rcx, %rax` 之后跟着通过 `(%rax)` 的存储。
6. `expected_locations()` 把三本规则书应用到 C 签名上：System V 的两条列表加八字节分类，Microsoft 的位置槽加按引用结构体，AAPCS64 的计数器加超过 16 字节结果用的 `x8`。
7. `run_checks()` 比较两者，并加入帧规则：被写入的被调用者保存寄存器必须被保存；每次调用时 `(rsp + 8) % 16 == 0` 或 `sp % 16 == 0`；每次 Microsoft 调用要留 32 字节影子存储；Microsoft 和 AAPCS64 下不得访问栈指针以下，System V 下至多 128 字节；尾声必须镜像序言，且栈指针回到入口值。
8. `seh_cross_check()` 和 `eh_frame_cross_check()` 读取未用 `-fno-asynchronous-unwind-tables` 构建的对象的展开数据，并与序言解析结果比较。

还有三条命令。`python3 code/main.py analyze convention-x86_64-windows-O2 keeps` 打印单个函数。`python3 code/main.py object FILE` 打印头部的判定。`python3 code/main.py seh` 打印每个有展开信息的函数的解码 `UNWIND_INFO`。

## 使用它

对你自己构建的新对象，按这个顺序使用分析器。

1. 运行 `python3 code/main.py object FILE.o`，写下它给出的 ABI。如果头部与你的构建命令不一致，停下；其中之一是错的。
2. 在与对象相同的目录捕获 `llvm-objdump -d -r FILE.o`，然后运行分析。先读序言字节和被保存寄存器，再把帧大小与 push 和 `sub` 对照。
3. 对照你的源码签名阅读参数寄存器和栈偏移。签名解释不了的寄存器，要么是隐藏指针、可变参数标记，要么是分析器误当成参数的临时寄存器。最后一种情况是效果表的 bug。用该指令的一般规则修好它。
4. 阅读任何不一致的检查。每项检查都注明它所依据的规则，所以先查规则，再查字节，最后查分析器。
5. 对象有展开数据时，运行交叉核对。两份彼此一致的推导是本课能提供的最强证据。
6. 安装了其他工具时，用准确的命令捕获其输出并比较。一致之处你有三份推导；不一致之处你有一个具体寄存器和一条具体指令可看，这就是第 35 课的一个发现。

## 交互实验

### 实验 A：一条指令，三种效果

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames/code
python3 -q
```

运行：

```python
from main import *
effect(Instruction(0, "48 8d 04 0a", "leaq", "(%rdx,%rcx), %rax"), SYSV)
effect(Instruction(0, "f2 0f 2a d7", "cvtsi2sd", "%edi, %xmm2"), SYSV)
effect(Instruction(0, "a9bd7bfd", "stp", "x29, x30, [sp, #-0x30]!"), AAPCS64)
```

在每次调用之前，写下你预期的读取、地址读取、写入和栈增量。然后把 `[sp, #-0x30]!` 改成 `ldp` 的 `[sp], #0x30`，观察增量从 `sp_delta` 变为 `post_sp_delta`。

### 实验 B：keeps 的帧

```python
info, listing = load_fixture("convention-aarch64-linux-O2")
keeps = analyze_function("keeps", listing["keeps"], info.abi)
keeps.saved
keeps.frame_pointer
[c.to_dict() for c in frame_cells(keeps)]
```

把偏移与上面的手工解码比较。然后加载 `convention-arm64-macos-O2`，运行同样三行；记录移到 -16，四个寄存器位于其下。再加载 `convention-x86_64-linux-O2-fp`，读取 `keeps` 的 `saved`：帧指针回来了，还多了一条 `pushq %rax`，它唯一的工作就是让调用时 `(rsp + 8)` 是 16 的倍数。

### 实验 C：在 mixed 上先读后写

```python
info, listing = load_fixture("convention-x86_64-linux-O2")
mixed = analyze_function("mixed", listing["mixed"], info.abi)
mixed.param_registers
mixed.return_registers
expected_locations(SYSV, "mixed")["parameters"]
expected_locations(WIN64, "mixed")["parameters"]
```

System V 列表是 `rdi, xmm0, rsi, xmm1`，Microsoft 列表是 `rcx, xmm1, r8, xmm3`。现在加载 Windows 对象，再运行前三行。然后打开 Linux 对象的 radare2 捕获，找到 `[afv mixed]`：三个 double。找出解释第三个的那条指令。

### 实验 D：展开器的第二意见

```python
for row in seh_cross_check():
    print(row["function"], row["sizeOfProlog"], row["unwindPushes"], row["unwindAllocBytes"], row["agree"])
for row in eh_frame_cross_check("convention-x86_64-linux-O2-eh"):
    print(row["function"], row["cfiSaved"], row["cfiMaxOffset"], row["deepestStack"], row["agree"])
```

在 SEH 表中找到 `sum_varargs` 的 `pushq %rax`，解释展开编码为何把它记为一次分配。再在 `.eh_frame` 表中找到 `call8`，解释那个 32。

## 实践练习

添加第四个目标和第四本规则书。

要求：

- 用 `-target x86_64-apple-macos -O2` 编译 `convention.c` 并捕获其清单。扩展 `describe_object()`，读取 Mach-O 的 `cputype` 0x01000007，并把它映射到 System V psABI（Apple 在 x86-64 上遵循它），同时处理 `_` 符号前缀。
- 检查每个函数是否与 System V 的检查一致。记录你在序言中发现的唯一差异：Darwin 在每个优化级别都保留帧指针。
- 为 Linux x86-64 对象增加一个 `-fstack-protector-strong` 构建。扩展序言扫描，使从 `%fs:0x28` 的 canary 加载和局部变量之下的存储被记为 `canary` 单元，并让 `ret` 之前的比较成为保护尾声的检查。
- 增加一个测试，当 canary 存储被算作溢出的参数时失败。
- 在运行分析器之前手动画出 `with_canary` 风格代码的帧，并把这张图与测试放在一起。

成功意味着 `python3 code/main.py` 报告新对象且所有检查一致，受保护函数的帧单元显示 canary 位于局部变量与保存寄存器之间。

## 课程产物

本课包含 `outputs/skill-calling-conventions-and-stack-frames.md`。用它把你构建的对象、它的清单和 C 签名转换成参数位置、帧布局、返回寄存器，并把每条推断与它所依据的规则以及编译器的展开数据核对。它要求先给出头部的 ABI 判定，禁止把 `cvtsi2sd` 或 `xor r, r` 的目的地当作读取，并以第一个分歧收尾。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 七个对象的 ELF、COFF 和 Mach-O 头部给出 System V、Microsoft x64、AAPCS64 与 Apple arm64
- 每个对象的符号表列出相同的十二个函数，Mach-O 的 `_` 前缀和 `ltmp0` 标签都已解析
- 清单解析器把 `ext` 保留为调用目标，并完整保留 `#-0x30` 这类 AArch64 立即数
- `leaq (%rdx,%rcx), %rax` 读取两个地址寄存器，不加载任何东西；`xorl %eax, %eax` 和 `cvtsi2sd` 只写；前索引 `stp` 在存储前移动 `sp`，后索引 `ldp` 在加载后移动
- `keeps` 在 `-O2` 下：System V 是 `55 41 56 53`，24 字节帧且无帧指针；Microsoft x64 是 `56 57 53 48 83 ec 20`，56 字节帧；AArch64 以 `a9bd7bfd` 开头，记录在 -48
- Apple 构建把记录放在 -16，位于四个被保存寄存器之上
- 在 Linux 上每个函数在 `-O0` 都有帧指针，在 `-O2` 都没有，而 `-fno-omit-frame-pointer` 会把它带回来，并附一条对齐 push
- `-O0` 叶子函数在 System V 下使用 8 字节红区，在 Microsoft x64 下使用 `pushq %rax`，在 AAPCS64 下使用 `sub sp, sp, #0x10`；`-mno-red-zone` 会消除红区的使用
- 每个尾声镜像其序言，栈指针回到入口值
- `eight_args` 在两种优化级别下都读取六个寄存器加槽 8 和 16、四个寄存器加槽 40 至 64，或 `x0` 至 `x7`
- `mixed` 在 System V 下是 `rdi, xmm0, rsi, xmm1`，在 Microsoft x64 下是 `rcx, xmm1, r8, xmm3`，两者结果都在 `xmm0`
- 16 字节结构体在 System V 和 AAPCS64 下走两个寄存器，在 Microsoft x64 下按引用经由隐藏指针传递
- 32 字节结构体经 `rdi`、`rcx` 或 `x8` 返回
- 可变参数被调用者读取 `%al`，把 `rdx, r8, r9` home 到 16、24、32，保存 `x1` 至 `x7` 和 `q0` 至 `q7`，或在 Apple 上读取调用者的栈指针
- 调用者设置预期的寄存器，把 7 和 8 存到 0 和 8，或把 5 到 8 存到 32 到 56，为可变参数调用把 `%al` 清零，并把结果缓冲区地址放在 `rdi`、`rcx` 或 `x8`
- 全部 489 项检查与其规则一致
- 六个 `UNWIND_INFO` 项与序言解析相符，叶子函数没有
- 24 个 FDE 与被保存寄存器偏移和最深的栈指针相符
- 工具捕获记录 radare2 的 `amd64` 与 `ms`、rizin 在 COFF 对象上的 `amd64`、缺失的 `rcx`、三个 double，以及 angr 的原型
- 站点模块中嵌入的图数据等于分析器的输出

## 生产环境故障模式

| 故障 | 弱工具的报告 | 分析必须证明什么 |
|---|---|---|
| 把 COFF 对象按 System V 读取 | `rdi` 和 `rsi` 被当作函数从不读取的参数 | 头部说是 0x8664；约定是 Microsoft x64 |
| 寄存器只在 `lea` 内被读取 | 八参数函数只报七个参数 | 基址和索引寄存器都是读取 |
| `cvtsi2sd` 的目的地被算作读取 | `xmm2` 里多出第三个 double 参数 | 对 `xmm2` 的第一次访问是写入；`mixed` 有两个 SSE 参数 |
| 把位置槽当成序列 | Windows 上把 `xmm0` 命名为 double 参数 | 槽 1 是 `XMM1`，槽 3 是 `XMM3` |
| 把 `push rbp` 当作帧指针 | 在一个把 `rbp` 当累加器的 `-O2` 函数里报出帧指针 | 只有 push 之后的 `mov %rsp, %rbp` 才建立帧指针 |
| 从入口栈指针量红区 | 「栈指针以下」216 字节 | 88 字节 `sub` 之后，当前栈指针以下 128 字节 |
| 把影子存储当成局部变量 | 栈参数在 8 到 32 | Microsoft 栈参数从 40 开始 |
| 把 `lea` 当成加载 | 可变参数被调用者中 `0x60(%rsp)` 处有一个栈参数 | `lea` 计算 `overflow_arg_area`；它不加载任何东西 |
| 把隐藏指针读成普通参数 | `big_ret(void *, long long)` | 所有存储都通过该指针进行，随后它被复制到 `rax`、`rax`，或留在 `x8` |
| 最深的 CFA 与序言帧比较 | 对压入传出参数的调用者报不匹配 | 与函数体到达的最深栈指针比较 |
| 从 ISA 断言约定 | COFF 对象报 `amd64` | 格式先选规则书，ISA 再选寄存器 |
| 相信单个工具的原型 | 三个 double，或十六个可变参数 | 必须与源码签名或第二份推导一致 |

## 操作准则

在读第一条指令之前先读文件头，并说出规则书的名字。把每个被保存寄存器和每个参数放在相对入口栈指针的偏移上，并用放它进去的规则以及编译器为同一帧写下的展开数据来核对。在字节或源码确认之前，把工具给出的原型当作一种主张。

## 延伸阅读

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Version 1.0 (sections 3.2.1 to 3.2.3, 4.2.4)](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [Microsoft x64 stack usage](https://learn.microsoft.com/en-us/cpp/build/stack-usage)
- [Microsoft x64 prolog and epilog](https://learn.microsoft.com/en-us/cpp/build/prolog-and-epilog)
- [Microsoft x64 exception handling: RUNTIME_FUNCTION, UNWIND_INFO, UNWIND_CODE](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64)
- [Arm Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), release 2025Q4](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Apple: Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [DWARF Debugging Information Format Version 5, section 6.4 Call Frame Information](https://dwarfstd.org/doc/DWARF5.pdf)
- [Clang driver frame-pointer defaults, `useFramePointerForTargetByDefault` in `CommonArgs.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/CommonArgs.cpp)
- [Clang command line reference: `-fomit-frame-pointer`, `-momit-leaf-frame-pointer`](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Fedora 38 change: frame pointers by default](https://fedoraproject.org/wiki/Changes/fno-omit-frame-pointer)
- [Ubuntu 24.04: performance engineering with frame pointers by default](https://ubuntu.com/blog/ubuntu-performance-engineering-with-frame-pointers-by-default)
- [Ghidra compiler specification: `x86-64-gcc.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-gcc.cspec)
- [Ghidra compiler specification: `x86-64-win.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-win.cspec)
- [Ghidra compiler specification: `AARCH64.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/languages/AARCH64.cspec)
- [Hex-Rays: SetType and the `__usercall` convention](https://docs.hex-rays.com/ida-actions/settype)
- [Hex-Rays: Decompile as call](https://docs.hex-rays.com/ida-actions/hx_userdefinedcall)
- [Binary Ninja API 6.0: `callingconvention` module](https://api.binary.ninja/binaryninja.callingconvention-module.html)
- [Binary Ninja API 6.0: `platform` module](https://api.binary.ninja/binaryninja.platform-module.html)
- [Binary Ninja user documentation: platform types](https://docs.binary.ninja/guide/types/platformtypes.html)
- [The Official Radare2 Book: calling conventions](https://book.rada.re/analysis/calling_conventions.html)
- [The Official Radare2 Book: code analysis](https://book.rada.re/analysis/code_analysis.html)
- [angr documentation: decompiler analysis passes, including calling convention recovery](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [rizin 0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
