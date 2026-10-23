# 反汇编与控制流：从字节到可检验的图

> 反汇编清单是在断言哪些字节构成指令，本课构建解码器和图，用来检验这一断言。

**Type:** Build
**Languages:** Python, C, x86-64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch)
**Time:** ~150 minutes

## 学习目标

- 依据 Intel SDM 第 2 卷、第 2 章的规则，解码 ledger 测试样本中的 x86-64 指令子集。
- 在每个测试样本的每次构建中，用 `llvm-objdump -d` 逐条验证解码器。
- 对同一段字节分别运行线性扫描与递归下降，并解释两者各自错在哪里。
- 把指令切分为基本块，并构建带类型边的控制流图。
- 使用 Cooper、Harvey 和 Kennedy 的迭代算法计算直接支配者，并从回边找出循环。
- 根据指向间接跳转表的各条重定位记录，恢复该跳转的目标。

## 问题

你在 `-O0` 下编译了 `ledger.c`，然后问 Agent `grade` 做了什么。Agent 调用了反汇编器，读了清单，回答说该函数以 `jmpq *%rax` 结束，目标未知。随后它写出的伪代码只有一个 `return 0`。

你用 `llvm-objdump -d` 打开同一个目标文件，看到跳转之后有四个块，分别存入 4、3、2、1。清单里一直都有这四个块。清单没有说明执行如何到达它们，于是工具把它们当成了死代码。

清单只回答一个问题：哪些字节构成哪条指令。它不说明哪些指令会执行、以什么顺序执行、从何处进入。要回答这些问题，还需要三种结构：控制流能够到达的字节、基本块，以及基本块之间的边。

rea 6.5.0 本身没有反汇编器。要做深入分析，它的 README 要求你自带 Ghidra、Hopper 或 IDA。做离线检查时，rea 也可以调用你安装的 GNU objdump 或 Rizin。它的调用图在未解析的目标处会停下：「调用者与被调用者只包含已解析的函数……没有目标的动态计算流仍然未知。」本课构建的正是产生这些事实的那部分，因此当工具报告它们时，你能判断其可信度。

```figure
re-33-sweep-vs-descent
```

## 指令格式：每个字段一条规则

Intel SDM 第 2 卷图 2-1 给出了每条指令的字节顺序。首先是一组可选前缀，然后是一到三个操作码字节，接着是可选的 ModRM 字节和可选的 SIB 字节。之后是 0、1、2 或 4 字节的位移，最后是 0、1、2 或 4 字节的立即数。整条指令最多 15 字节（SDM 2.3.11）。AMD APM 第 3 卷第 1 章第 1.1 至 1.7 节用更少的篇幅描述了相同格式，适合交叉核对。

每个字段决定下一个字段的含义。因此，解码器是一连串小决策，而每个决策背后都有手册中的一句话作为依据。

### 前缀分为四组

SDM 2.1.1 列出了传统前缀。第 1 组包含 `F0` lock 以及重复前缀 `F2` 和 `F3`。第 2 组包含六个段覆盖前缀（`2E` 表示 `%cs`）以及分支提示。第 3 组是 `66`，即操作数大小覆盖。第 4 组是 `67`，即地址大小覆盖。

手册规定每组只能使用一个前缀，但编译器会发出更多。`ledger-x86_64-O0.o` 中偏移 `0xf2` 处的对齐填充字节是 `66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00`，连续五个 `66`。`llvm-objdump` 把它打印为 `nopw %cs:(%rax,%rax)`，你构建的解码器也会这样打印。解码器收集遇到的每个前缀字节，事后再判断哪些起了作用。

`66` 还有第二个用途。对于 SSE 指令，它是用于选择操作码的强制前缀，因此 `66 0f ef c0` 是 `pxor %xmm0, %xmm0`，而不是任何 16 位操作。`F3 0f 6f` 是 `movdqu`。解码器先查找强制前缀与第二个操作码字节组成的对。只有当这个对没有条目时，`66` 才表示 16 位操作数。

### REX 增加四个位，并且必须紧邻操作码

SDM 2.2.1 将 REX 定义为一个从 `40` 到 `4F` 的字节，形式为 `0100WRXB`。`W` 选择 64 位操作数。`R` 扩展 ModRM 的 reg 字段。`X` 扩展 SIB 的 index 字段。`B` 扩展 ModRM 的 r/m 字段、SIB 的 base 字段，或操作码字节中的寄存器（表 2-4）。REX 必须紧邻操作码或转义字节之前；如果 REX 后面跟着传统前缀，则该 REX 会被忽略。

在 `48 89 e5` 中，字节 `48` 是设置了 `W` 的 REX。操作码 `89` 是 `MOV r/m, r`，ModRM `e5` 选择把 `%rsp` 放入 `%rbp`。去掉 REX 后，同样的字节变成 `movl %esp, %ebp`。

REX 还会改变字节寄存器的名称。只要存在任何 REX，编码 4 到 7 就会选择 `%spl`、`%bpl`、`%sil` 和 `%dil`，而不是 `%ah`、`%ch`、`%dh` 和 `%bh`。因此解码器保留两张表。

### 操作码分布在两张映射表中

单字节操作码是常见情况。`0F` 转义到双字节映射表，`0F 38` 或 `0F 3A` 转义到三字节映射表，测试样本从未使用后者（SDM 2.1.2）。有些操作码把 ModRM 的 reg 字段用作另外三个操作码位，手册中写作 `/digit`。`83 /7` 是带 8 位立即数的 `cmp`，`83 /5` 是 `sub`，`FF /4` 是间接 `jmp`。

`code/main.py` 中的解码器覆盖四个测试样本目标文件中的操作码，外加少量相邻操作码。列表包括八种算术形式、`mov`、`lea`、`movslq`、`movzbl`、`imul`、移位指令、`test` 和 `idiv`。它还覆盖 `push`、`pop`、`call`、`ret`、十六条条件跳转、多字节 `nop`，以及五个 SSE2 操作码。其他任何操作码都会抛出带地址和原因的 `DecodeError`。会猜测的解码器比会停下来的解码器更糟。

### ModRM 与 SIB 选择操作数

SDM 2.1.3 把 ModRM 拆成 `mod`（第 7:6 位）、`reg`（第 5:3 位）和 `r/m`（第 2:0 位）。`mod = 11` 表示 r/m 是寄存器。`mod = 00`、`01` 和 `10` 表示内存操作数，分别没有位移、带 8 位位移、带 32 位位移。`r/m = 100` 表示后面跟着一个 SIB 字节。SIB 拆成 `scale`（第 7:6 位）、`index`（第 5:3 位）和 `base`（第 2:0 位）；index 为 `100` 表示没有索引寄存器。

表 2-2 中有两行特殊情况在 64 位模式下很重要。`mod = 00` 且 `r/m = 101` 并不表示 `(%rbp)`，而是相对于下一条指令的 32 位位移（SDM 2.2.1.6）。`mod = 00` 且 SIB base 为 `101` 表示没有基址寄存器的 32 位位移。

以 `-O0` 下 `grade` 中的 `8d 44 08 fa` 为例。操作码 `8D` 是 `lea`。ModRM `44` 是 `01 000 100`：带 8 位位移，reg 为 `%eax`，后面跟着 SIB。SIB `08` 是 `00 001 000`：scale 为 1，index 为 `%rcx`，base 为 `%rax`。位移字节 `fa` 是 -6。AT&T 文本是 `leal -0x6(%rax,%rcx), %eax`，它计算 `score / 10 - 6`，也就是 switch 索引。

### 位移字段是链接编辑器写入的位置

`48 8d 0d 00 00 00 00` 是 `leaq (%rip), %rcx`。四个零字节是 RIP 相对操作数的位移字段。在可重定位目标文件中，它们只是占位符。`.rela.text` 中偏移 `0x69` 的条目写着 `R_X86_64_PC32.rodata-0x4`。psABI（表 4.9）把 `R_X86_64_PC32` 定义为 `S + A - P`：符号地址加上加数，再减去字段地址。

处理器把该字段加到下一条指令的地址上，因此最终目标是 `S + A + (end - P)`。对这个 `lea` 而言，字段是最后四个字节，所以 `end - P` 为 4，目标是 `.rodata + (-4) + 4 =.rodata + 0x0`。加数 -4 预先补偿了这四个字节。测试样本中每个 RIP 相对操作数的加数都包含这个 -4；因此 `-O1` 下 `lea` 的 `.rodata+0x10` 实际表示 `.rodata+0x14`。

解码器记录每条指令内部位移字段的偏移。`resolve_reference()` 在重定位表中查找该偏移，并报告链接后操作数将指向何处。正是这个函数把 `leaq (%rip), %rcx` 变成「位于 `.rodata+0x0` 的表」。

### 立即数最多 32 位，只有一个例外

SDM 2.2.1.5 指出，在 64 位模式下立即数最多为 32 位，并且会符号扩展。例外是 `MOV r64, imm64`（带 `REX.W` 的 `B8+r`），AT&T 语法称之为 `movabsq`。对于算术运算，解码器把 8 位立即数按有符号数读取（`83 c1 fa` 是 `addl $-0x6, %ecx`）；对于移位计数和 shuffle 选择器（`pshufd $0xee`），则按无符号数读取。这种区分与 `llvm-objdump` 打印的结果一致。

近分支、`push` 和 `pop` 默认使用 64 位操作数，并忽略 `66`（SDM 2.2.1.7）。因此，解码器无需读取操作数大小前缀，就把它们命名为 `pushq`、`popq`、`callq` 和 `retq`。

## 手工解码一条指令

在信任代码之前，先在纸上解码 `grade` 中的 `48 69 c0 67 66 66 66`。

1. `48` 位于 `40` 到 `4F` 之间，因此是设置了 `W = 1` 的 REX。操作数大小为 64 位。
2. `69` 是 `IMUL r, r/m, imm32`。
3. `c0` 是 ModRM `11 000 000`：寄存器形式，reg 为 `%rax`，r/m 为 `%rax`。
4. 接下来的四个字节是小端立即数：`0x66666667`。
5. AT&T 的顺序是立即数、源、目标：`imulq $0x66666667, %rax, %rax`。

这是七个字节，`llvm-objdump` 打印出相同的文本。该常量是一个倒数，使 clang 能用一次乘法再加上后面的 `sarq $0x22`（右移 34 位）来实现除以 10。你会在第 34 课再次遇到这对指令，那里提升器会把它还原成 `score / 10`。

解码器在 `Decoder.decode()`、`_State.modrm()` 和 `_State.finish()` 中执行这五个步骤。按此顺序阅读，并把每个分支与文档字符串中点名的 SDM 章节对应起来。

## 线性扫描

Schwarz、Debray 和 Andrews 在《Disassembly of Executable Code Revisited》（WCRE 2002）第 3.1 节中描述了线性扫描。从代码节的第一个字节开始，解码一条指令，在下一个字节处继续，直到节末尾停止。`objdump` 就是这样工作的，`llvm-objdump` 也一样。

```python
def linear_sweep(code: bytes, start: int = 0, end: int | None = None) -> SweepResult:
    position = start
    while position < end:
        try:
            instruction = decode(code[:end], position)
        except DecodeError:
            undecodable.append((position, code[position]))
            position += 1
            continue
        instructions.append(instruction)
        position = instruction.end
```

线性扫描只有一个失败模式，论文的图 1 展示了它。本属于数据的字节被解码成指令，随后解码器在一段时间内错过真正的指令边界。在 x86 上，解码器通常会在几个字节内重新同步，这正是该错误难以察觉的原因。只有遇到无效操作码时才能检测到错误，论文的结论是这些算法会「静默失败」。

扫描也是本课的基准（oracle）。对于每个测试样本的每条指令，解码器都必须与 `llvm-objdump -d` 一致，而 `llvm-objdump` 本身就是一种线性扫描。捕获的文本存放在 `code/fixtures/*.objdump.txt`。测试把每一行解析为地址、字节和文本，去掉 `#` 后的注释，然后进行比较。四个目标文件分别包含 86、56、84 和 11 条指令，共 237 条，解码器全部匹配。

## 递归下降

同一篇论文的第 3.2 节把第二种方法称为递归遍历（recursive traversal）。从已知入口点开始解码，直到遇到控制转移。对于直接跳转，在目标处继续；对于条件分支，在目标和下一条指令两处继续；对于调用，在调用之后继续，因为被调用者会返回；对于返回，停止。维护一个已访问地址集合，使遍历能够结束。

```python
if instruction.flow == "branch":
    worklist.append(instruction.target)
    address = instruction.end
elif instruction.flow == "indirect-jump":
    table = resolver(instruction, instructions) if resolver else None
    if table is None:
        unresolved.append(instruction)
        break
```

失败模式正好相反。间接跳转的字节中没有目标。遍历在那里停止，所有只能通过该跳转到达的块都保持未知。论文的图 2 展示了第二个风险：根据重定位推导出的目标集合指向某条指令的中间。`main.py` 中的递归下降会把这种情况记录为冲突并停止该路径，因此重叠会显现出来，而不是被静默地解码两次。

Andriesse 等人在 981 个编译器生成的二进制文件上测量了九种反汇编器（USENIX Security 2016）。真实编译器输出中的内联数据和重叠代码比文献假设的更少见，真正困难的基本操作是函数边界恢复。请记住这两个发现。下一节的陷阱在编译器输出中很少见，而 `grade` 中的间接跳转很常见。

## 一个线性扫描出错而递归下降正确的测试样本

`code/fixtures/sweep-trap.s` 是 24 字节的手写汇编，函数体内包含一个四字节表。函数用 `movzbl (%rcx,%rax), %eax` 索引该表，然后跳过它。

```asm
lookup:
        movslq  %edi, %rax
        leaq    .Ltable(%rip), %rcx
        movzbl  (%rcx,%rax), %eax
        jmp     .Ldone
.Ltable:
        .byte   0x31, 0xc0, 0x90, 0x48
.Ldone:
        addl    $1, %eax
        retq
```

用 `clang -target x86_64-unknown-linux-gnu -c sweep-trap.s` 构建它。选择这四个表字节，是为了让它们被解码成看似合理的指令。`31 c0` 是 `xorl %eax, %eax`。`90` 是 `nop`。`48` 是 REX 前缀，因此线性扫描会把它与真正的 `addl $1, %eax` 的三个字节连在一起，打印成 `addq $0x1, %rax`。捕获的 `llvm-objdump` 输出正是如此：

```text
       e: eb 04                        	jmp	0x14 <lookup+0x14>
      10: 31 c0                        	xorl	%eax, %eax
      12: 90                           	nop
      13: 48 83 c0 01                  	addq	$0x1, %rax
      17: c3                           	retq
```

这份清单看起来毫无问题。每个字节都完成了解码，每个助记符都真实存在，`retq` 也在。但清单仍在三处出错。`0xe` 处的跳转目标是 `0x14`，而清单中没有任何指令从 `0x14` 开始。真正在 `0x14` 运行的指令是 32 位的 `addl`，清单却在早一个字节处显示 64 位的 `addq`。四个表字节被当作代码出现。

从 `lookup` 符号开始递归下降，会解码六条指令，沿跳转到达 `0x14`，并报告 `addl $0x1, %eax`。它不声称四个表字节中的任何一个，也不声称 `plus_one` 之前的八个填充字节。`main.py` 中的 `trap_report()` 打印如下比较：

```json
"linearSweep": {
  "instructions": 9,
  "atJumpTarget": {"starts": "0x13", "text": "addq\t$0x1, %rax", "startsAtTarget": false},
  "claimedDataBytes": 4,
  "claimedPaddingBytes": 8
},
"recursiveDescent": {
  "instructions": 6,
  "atJumpTarget": "addl\t$0x1, %eax",
  "claimedDataBytes": 0,
  "claimedPaddingBytes": 0
}
```

测试 `test_sweep_and_objdump_share_the_same_mistake` 断言 `llvm-objdump` 在 `0x13` 处打印了 `addq $0x1, %rax`。基准是线性扫描，因此继承了它的盲点。与基准一致可以检查解码器，而递归下降检查实际执行哪些指令。

## 基本块

Allen 在《Control Flow Analysis》（1970）中定义了基本块。基本块是一段指令序列，只能从第一条指令进入，也只能从最后一条指令离开。Yakdan 等人在《No More Gotos》（NDSS 2015）第 II.A.2 节中为反编译重述了这一定义，并把头节点作为函数入口。

构造过程是对按地址排序的已解码指令做两遍扫描。第一遍收集首领（leader）。首领是入口、分支或跳转目标、表目标，或任何控制转移之后的那条指令。第二遍在每个首领处以及地址序列的每个间隙处开始一个新块。

```python
for instruction in self.instructions:
    if instruction.address in leaders or current is None or current.end != instruction.address:
        current = Block(instruction.address, [])
        self.blocks[instruction.address] = current
    current.instructions.append(instruction)
```

边由每个块的最后一条指令决定。返回没有后继。跳转有一个后继，标记为 `jump`。条件分支有两个后继，分别标记为 `taken` 和 `fallthrough`。块末尾的普通指令顺序执行到下一个块，这发生在下一条指令是分支目标的时候。间接跳转拥有其表中的目标，标记为 `table`；如果没有目标，该块会记录在 `unresolved` 中。

同一次扫描还能得出另外两个事实。入口无法到达的块是不可达块；对 `-O0` 下 `grade` 的线性扫描会产生五个这样的块。若某个块的起始地址落在另一种方法清单中某条指令的内部，那就是陷阱中的重叠情况。

## total 的控制流图

`-O1` 下的 `total` 有 31 字节、13 条指令：

```text
      40: 85 f6                        	testl	%esi, %esi
      42: 7e 18                        	jle	0x5c <total+0x1c>
      44: 89 f1                        	movl	%esi, %ecx
      46: 31 d2                        	xorl	%edx, %edx
      48: 31 c0                        	xorl	%eax, %eax
      4a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
      50: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
      53: 48 ff c2                     	incq	%rdx
      56: 48 39 d1                     	cmpq	%rdx, %rcx
      59: 75 f5                        	jne	0x50 <total+0x10>
      5b: c3                           	retq
      5c: 31 c0                        	xorl	%eax, %eax
      5e: c3                           	retq
```

首领是 `0x40`（入口）、`0x44`（`jle` 之后）、`0x50`（`jne` 的目标）、`0x5b`（`jne` 之后）和 `0x5c`（`jle` 的目标）。五个块，五条边：

| From | To | Kind | Why |
|---|---|---|---|
| `0x40` | `0x44` | fallthrough | `jle` 未跳转，计数为正 |
| `0x40` | `0x5c` | taken | 计数为零或负数，返回 0 |
| `0x44` | `0x50` | fallthrough | `nopw` 是块内的对齐填充，执行会经过它 |
| `0x50` | `0x50` | taken | `jne 0x50`，即循环 |
| `0x50` | `0x5b` | fallthrough | 循环退出，返回累加和 |

`0x4a` 处六字节的 `nopw` 是填充，使循环头 `0x50` 落在 16 字节边界上。它可达，因此属于块 `0x44`。最后一个 `retq` 之后 `0x5f` 处的一字节 `nop` 也是填充，没有路径到达它。对该函数的线性扫描会为它产生第六个块，`unreachable()` 会列出该块。

在 `-O2` 下，同一函数为 140 字节，包含两个循环。`0x80` 处的 SSE2 主体用 `paddd` 每轮累加八个值，标量余数部分在 `0xc0` 处运行。CFG 有两条回边，`0x80 -> 0x80` 和 `0xc0 -> 0xc0`，测试套件对两者都做了检查。

## 用迭代算法求支配者

当从入口到块 `b` 的每条路径都经过块 `d` 时，称 `d` 支配 `b`。`b` 的直接支配者，是支配 `b` 的块中距离 `b` 最近的那一个。这个概念可追溯到 Prosser（1959），并经 Allen（1970）发展；第 34 课需要这一结构来正确放置 `if` 和 `while`。

Cooper、Harvey 和 Kennedy 写了《A Simple, Fast Dominance Algorithm》（2001）。他们论证：在节点数少于约一千的图上，以逆后序运行、使用紧凑表示的数据流公式优于 Lengauer-Tarjan（1979）。论文说其目的「不是提出一种新算法」。它表明简单算法在实践中可以获胜，也是很好的教学工具。函数的 CFG 几乎不会有一千个节点。

```python
def intersect(first: int, second: int) -> int:
    while first != second:
        while number[first] > number[second]:
            first = idom[first]
        while number[second] > number[first]:
            second = idom[second]
    return first

changed = True
while changed:
    changed = False
    for start in order[1:]:
        processed = [p for p in self.predecessors(start) if p in idom]
        candidate = processed[0]
        for other in processed[1:]:
            candidate = intersect(other, candidate)
        if idom.get(start) != candidate:
            idom[start] = candidate
            changed = True
```

`number` 保存每个块的逆后序位置，入口为 0。`intersect` 沿着当前支配树上行两个指针，直到它们相遇。论文用后序编号和 `<` 进行比较。这段代码使用逆后序编号，因此比较符号是 `>`。其余部分遵循论文的伪代码。

对于 `-O1` 下的 `total`，结果是 `idom(0x44) = 0x40`、`idom(0x50) = 0x44`、`idom(0x5b) = 0x50`、`idom(0x5c) = 0x40`。边 `0x50 -> 0x50` 是回边，因为其目标支配其源，而回边标记一个循环（Allen 1970）。`back_edges()` 会返回它。测试 `test_dominators_on_a_hand_built_diamond_with_a_loop` 在带汇合点的 16 字节序列上运行同一段代码，因此 `intersect` 会看到两个前驱。

## -O0 下 grade 的跳转表

在 `-O0` 下，clang 把 `grade` 中的 `switch` 编译成位置无关的跳转表惯用法：

```text
      5d: 83 e8 04                     	subl	$0x4, %eax
      60: 77 38                        	ja	0x9a <grade+0x6a>
      62: 48 8b 45 f0                  	movq	-0x10(%rbp), %rax
      66: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000069:  R_X86_64_PC32	.rodata-0x4
      6d: 48 63 04 81                  	movslq	(%rcx,%rax,4), %rax
      71: 48 01 c8                     	addq	%rcx, %rax
      74: ff e0                        	jmpq	*%rax
```

可以把它读成四个事实。`subl $0x4` 后接 `ja` 是一次边界检查：索引大于 4 时转到 `0x9a` 的默认分支，因此表中有五项。带重定位的 `leaq` 把表地址放入 `%rcx`，上一节已经说明该地址是 `.rodata+0x0`。`movslq` 加载一个按 4 缩放的带符号 32 位表项。`addq` 加上表基址，因此每个表项都是该表到对应 case 块的距离。

在目标文件中，表的每个字节都是零。目标只以 `.rela.rodata` 中的重定位形式存在：

```text
RELOCATION RECORDS FOR [.rodata]:
OFFSET           TYPE                     VALUE
0000000000000000 R_X86_64_PC32            .text+0x91
0000000000000004 R_X86_64_PC32            .text+0x8c
0000000000000008 R_X86_64_PC32            .text+0x87
000000000000000c R_X86_64_PC32            .text+0x82
0000000000000010 R_X86_64_PC32            .text+0x86
```

对偏移 `4k` 处的第 `k` 项应用 `S + A - P`。存储值是 `.text + A - (.rodata + 4k)`，代码又加上 `.rodata + 0`，因此目标是 `.text + A - 4k`。第 0 项是 `0x91 - 0 = 0x91`，第 1 项是 `0x8c - 4 = 0x88`，第 2 项是 `0x87 - 8 = 0x7f`。第 3 项和第 4 项分别是 `0x82 - 12 = 0x76` 和 `0x86 - 16 = 0x76`。`score / 10` 的 case 6、7、8、9、10 分别转到存入 1、2、3、4、4 的块。这与 C 源码逐行对应，包括 9 和 10 共享同一个 case。

`recover_table()` 完成这项算术。`jump_table_resolver()` 从 `jmpq` 开始反向匹配该惯用法。它找到生成该寄存器的 `addq`、加载该寄存器的 `movslq`、生成基址的 `leaq`，以及负责边界检查的 `ja`。随后递归下降压入四个不同的目标并继续。已链接的可执行文件中，表内是最终值，没有重定位。同一个函数此时读取带符号的 32 位表项并加上表地址，`evidence` 记录目标是由哪条路径产生的。

`grade` 在 `-O0` 下的数据，来自 `python3 code/main.py cfg code/fixtures/ledger-x86_64-O0.o grade`：

| 方法 | 指令数 | 基本块 | 边 | 字节 | 问题所在 |
|---|---|---|---|---|---|
| 线性扫描 | 34 | 9 | 7 | 118 代码字节 + 10 填充字节 | 5 个不可达块，四个 case 块没有入边，填充被解码为 `nopw` |
| 递归下降 | 25 | 3 | 2 | 82 代码字节 | 36 个代码字节从未到达，`0x74` 处的 `jmpq *%rax` 未解析 |
| 带表的递归下降 | 33 | 8 | 11 | 118 代码字节 | 无 |

有了表之后，支配树就完整了。每个 case 块的 `idom` 都是 `0x62`，即包含跳转的块。`0xa1` 的 `idom` 是入口 `0x30`，因为从 `0x9a` 出发的默认路径不经过 `0x62` 就能到达 `0xa1`。

## 优化器对 -O1 下 grade 做了什么

你可能会看到有人把 `grade` 描述为一个编译成跳转表的 switch。字节展示的其实更具体。在 `-O1` 和 `-O2` 下，`grade` 中根本没有间接跳转：

```text
      2c: 83 f9 04                     	cmpl	$0x4, %ecx
      2f: 77 0c                        	ja	0x3d <grade+0x2d>
      31: 89 c8                        	movl	%ecx, %eax
      33: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000036:  R_X86_64_PC32	.rodata+0x10
      3a: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      3d: c3                           	retq
```

重定位加数 `0x10` 加上通常的 4，把表放在 `.rodata+0x14`。加载指令是普通的 `movl`，那里的字节是 `01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 04 00 00 00`。编译器用五个返回值的表替换了五个 case 块。这里有重定位、表和边界检查。控制流是三个块、三条边，带一个条件分支。`find_value_tables()` 将其报告为 `value-table`，表项为 `[1, 2, 3, 4, 4]`，`evidence: "bytes"`。

`-O2` 下的 `grade_label` 又向前走了一步。由五个字符串指针组成的 `labels` 数组变成了位于 `.rodata+0x0` 的五个 32 位偏移表。每个表项都带有一条针对字符串节的 `R_X86_64_PC32` 重定位，函数会把表地址加到所加载的表项上。`recover_table()` 读取这五条重定位，返回 `.rodata.str1.1` 中的偏移 `0, 2, 4, 6, 8`，分别对应 `F`、`D`、`C`、`B` 和 `A`。恢复跳转表的同一套算术也能恢复相对数据表，因为两者都是在已知偏移处的 `S + A - P`。

两次构建给出的教训是：要说出字节实际展示了什么。跳转表改变控制流图。值表改变一次加载。把两者都称为「switch」，会掩盖第 34 课必须重建的差异。

## rea 与 Ghidra 在这里提供了什么

rea 6.5.0 配合 Ghidra 12.1.4 provider，会把函数的 Listing 文本从 `procedure_assembly` 作为一个字符串返回。`inspect_native_instruction` 返回一条指令的字节、长度、解码文本、操作数 token、类型化引用和流，并根据 p-code 对返回进行分类。`xrefs` 返回来自 Ghidra `ReferenceManager.getReferencesTo` 的已排序唯一源地址。`analyze_function` 携带函数的 CFG 和 `body`。`procedure_info` 把 body 表示为包含区间，并警告「间隙和共享尾部不得被视为拥有的字节」。

这些事实中的每一项，都是 Ghidra 的递归下降及其分析器序列化后的结果。rea 的分析配置为 `instruction_flow_evidence`、`function_references` 和 `jump_table_evidence` 固定了契约版本。因此结果会指明产生它的规则。它无法做的是展示推导过程。当 Ghidra 找不到表时，流就是「unknown」，case 块也会从调用者、被调用者和 CFG 中缺失。

现在该你动手了。在同一个目标文件上运行 `python3 code/main.py cfg`，比较块起点、边数以及恢复出的表与 provider 返回的内容。一致之处，你有两条独立的推导路径。不一致之处，你有具体的地址可以查看，第 35 课会把它记录为未知。

关于解码器作为程序的一点说明。Capstone 6.0.0-Alpha10（2026-07-21）修复了 CVE-2026-55893 和 CVE-2026-55894 下的越界读写，5.0.9（2026-05-28）向后移植了 CVE 修复。解码器读取的是并非由它写出的字节。15 字节上限、`_Cursor.byte()` 中的截断检查，以及拒绝猜测未知操作码的行为，保护 `main.py` 免受它并非为之构建的输入影响。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
```

输出是一份 JSON 报告。对每个测试样本目标文件，它给出与基准的一致情况。对每个函数，它给出三种方法下的指令数、块数和边数。它还列出不可达块、未解析的间接跳转、回边、直接支配者以及任何表。最后以陷阱比较结束。

预期结构：

```json
{
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "oracle": "86/86 instructions agree with llvm-objdump",
      "functions": {
        "grade": {
          "range": "0x30-0xa6",
          "paddingBytes": 10,
          "linearSweep": {"instructions": 34, "blocks": 9, "edges": 7, "unreachableBlocks": ["0x76", "0x7f", "0x88", "0x91", "0xa6"], "...": "..."},
          "recursiveDescent": {"instructions": 25, "blocks": 3, "edges": 2, "unresolvedIndirectJumps": ["0x74"], "...": "..."},
          "recursiveDescentWithTables": {"instructions": 33, "blocks": 8, "edges": 11, "...": "..."},
          "tables": [{"kind": "jump-table", "table": ".rodata+0x0", "count": 5, "entries": ["0x91", "0x88", "0x7f", "0x76", "0x76"], "evidence": "relocation", "...": "..."}]
        }
      }
    }
  },
  "sweepTrap": {"...": "..."}
}
```

按以下顺序阅读 `code/main.py`：

1. `ElfObject` 读取节头、符号和 `.rela.*` 条目，`symbolize()` 按 `llvm-objdump` 的方式为地址命名。
2. `Decoder.decode()` 读取前缀和 REX，然后根据单字节或双字节操作码映射表分派。
3. `_State.modrm()` 读取 ModRM、SIB 和位移，并记录位移字段的偏移。
4. `resolve_reference()` 对位移或立即数字段上的重定位应用 `S + A - P`。
5. `linear_sweep()` 和 `recursive_descent()` 是两种方法，各自记录其失败。
6. `recover_table()` 和 `jump_table_resolver()` 通过重定位或字节读取表。
7. `ControlFlowGraph` 构建首领、块和带类型的边，`dominators()` 是迭代算法。
8. `analyze_function()` 在一个函数上运行所有方法，`compare_with_oracle()` 检查解码器。

还有两个命令。`python3 code/main.py disasm FILE.o` 打印清单，并在每条带重定位的指令后附上重定位目标。`python3 code/main.py cfg FILE.o FUNCTION` 打印单个函数的分析。

## 使用它

在新目标文件上按以下顺序使用三种方法。

1. 先运行基准比较。如果解码器在一条普通指令上与 `llvm-objdump` 不一致，说明它有 bug，之后的所有结构都会继承这个错误。
2. 从每个函数符号运行递归下降，并阅读 `unresolved`。每个条目都是一个目标未知的间接跳转或调用。
3. 对每个未解析的跳转，寻找表惯用法以及指向它的重定位。如果惯用法不匹配，读取跳转前的字节并写下你看到的内容。这条记录会成为第 35 课中的第一个未知。
4. 逐字节比较线性扫描和递归下降。只有线性扫描声称的字节是填充或数据。两者都不声称的字节是不可达字节，或者属于你尚未找到的函数。
5. 用恢复出的表从递归下降构建 CFG，并阅读回边。每条回边都是一个循环，第 34 课会把它变成 `while` 或 `for`。

把基准文本文件与目标文件放在一起。升级到新的编译器版本时，重新捕获它们，因为清单变化就意味着测试样本变化。

## 交互实验

### 实验 A：在纸上解码，然后核对

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow/code
python3 -q
```

运行：

```python
from main import *
decode(bytes.fromhex("48 8d 0d 00 00 00 00"), 0).text
decode(bytes.fromhex("8d 44 08 fa"), 0).text
decode(bytes.fromhex("66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00"), 0).length
```

在每次调用前写下你的预期。然后去掉第一个调用中的 `48`，观察操作数大小的变化。再尝试 `decode(bytes.fromhex("0f 0a"), 0)`，并阅读 `DecodeError`。

### 实验 B：lea 上的重定位

```python
elf = load_fixture("ledger-x86_64-O0.o")
code = elf.section(".text").data
lea = decode(code, 0x66)
lea.displacement_offset
resolve_reference(elf, lea)
```

该引用显示 `.rodata+0x0`。把地址改为 `grade_label` 中的 `0x116`，读取指向 `.data` 的引用。然后解码 `0x10e` 处的 `callq` 并解析它。立即数字段带有 `R_X86_64_PLT32 grade-0x4`，解析后的目标是 `0x30`，即 `grade`。

### 实验 C：观察两种方法的分歧

```python
trap = load_fixture("sweep-trap.o")
text = trap.section(".text").data
[i.text for i in linear_sweep(text, 0, 0x20).instructions]
[i.text for i in recursive_descent(text, [0], 0x20).sorted_instructions()]
```

数一数每个列表中的指令，找出从 `0x13` 开始的那一条。然后把 `sweep-trap.s` 中最后一个表字节从 `0x48` 改成 `0xe8`，重新构建，再次运行两者。现在线性扫描会把 `addl` 和 `retq` 吞进一个目标位于节外的 `callq`。函数看起来没有返回。

### 实验 D：手工计算支配者

```python
analysis = analyze_function(load_fixture("ledger-x86_64-O1.o"), "total")
analysis.cfg.edges
analysis.cfg.reverse_postorder()
analysis.cfg.dominators()
analysis.cfg.back_edges()
```

先在纸上画出五个块并计算支配者，再看结果。然后对 `-O0` 下的 `grade` 运行同样的四行代码，解释为什么 `idom(0xa1)` 是 `0x30` 而不是 `0x62`。

## 实践练习

向测试样本中添加一个间接调用点，并扩展分析。

要求：

- 编写一个 C 函数，通过从小数组中选出的函数指针进行调用。在 `-O0` 和 `-O1` 下构建它，并把目标文件连同其 `llvm-objdump` 文本一起提交。
- 让解码器在两个目标文件上都与基准一致。只有当测试样本中的某条指令需要时，才添加操作码。
- 扩展 `recursive_descent()`，使间接调用把调用点记录到 `calls` 中，目标为 `None`，并在调用之后继续。
- 编写一个解析器，通过重定位读取函数指针数组，并报告可能的被调用者。
- 把被调用者集合作为 `indirectCalls` 添加到函数报告中，并为每个目标附上证据类型。
- 编写一个测试：当解析器返回的目标不是函数符号时，测试失败。

成功标准是：`python3 code/main.py cfg` 会为每个可能的被调用者命名，并给出证明它的重定位；同时，基准比较在每个目标文件上仍然通过。

## 计分实验

关于本课测试样本的五个问题。用你构建的代码找到每个答案，然后写入 `code/lab_answers.json`。数字写成数字或十六进制字符串，例如 `"0x40"`，列表写成 JSON 列表。

1. `t1`：在 `-O1` 的 `grade` 中，递归下降解码出的指令数。
2. `t2`：在 `-O1` 的 `total` 中，基本块数。
3. `t3`：在 `-O1` 的 `total` 中，循环头的地址，用十六进制表示。
4. `t4`：在 `sweep-trap.o` 中，覆盖 `0x14` 的线性扫描指令的起始地址，用十六进制表示。
5. `t5`：在 `-O0` 的 `grade` 中，跳转表的表项数。

核对答案：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/lab.py
```

检查器会用 `code/main.py` 在已提交的测试样本上计算每个预期值。它针对每个任务打印 `PASS`、`FAIL` 或 `TODO` 以及得分，只有五个全部通过时才以 0 退出。`python3 code/lab.py my-answers.json` 会检查另一个文件。`code/tests/test_lab.py` 证明检查器接受正确答案并拒绝错误答案。

### 看它运行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?program=trap&sweep=linear) 这台机器把本课作为它的解码台和基本块面板。该链接以线性扫描打开 `sweep-trap.o`：台面上显示四个数据字节被解码成代码，一个标记钉在 `0x13` 的 `addq` 内、指向 `0x14` 的跳转目标上。把解码器切换为递归下降，就能看到跳转跳过了这些数据。

## 本课产物

本课包含 `outputs/skill-disassembly-and-control-flow.md`。用它把目标文件和函数名变成经过核对的清单、基本块、带类型的边、支配者，以及每一个未解析的控制转移。它要求基准比较、两种反汇编方法、字节核算、为每个恢复出的表提供重定位证据，并说明哪些内容仍然未知。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
python3 code/tests/test_main.py
```

验证应证明：

- 解码器在四个测试样本目标文件的全部 237 条指令上与 `llvm-objdump -d` 一致
- REX.W 选择 64 位操作数，REX.R 和 REX.B 扩展寄存器字段
- ModRM 与 SIB 渲染 AT&T 内存操作数，包括 scale、index 和带符号位移
- RIP 相对位移会记录其在指令内部的字段偏移
- 重复的传统前缀和段覆盖前缀解码结果与基准一致
- 强制的 `66` 前缀选择 SSE 操作码，并让寄存器保持 32 位
- 8 位立即数在算术运算中符号扩展，在移位计数中无符号
- 超过 15 字节的指令、被截断的指令和未知操作码都会抛出 `DecodeError`
- `lea` 上的 `R_X86_64_PC32` 解析到表，`call` 上的 `R_X86_64_PLT32` 解析到被调用者
- 线性扫描把陷阱中的表解码为代码，并且跳转目标处没有指令
- 递归下降从不声称表或填充字节，并在目标处读取 `addl $0x1, %eax`
- `llvm-objdump` 在陷阱上与线性扫描犯同样的错误
- 落在指令内部的表项会被记录为冲突
- 未解析的间接跳转会停止递归下降并被列出
- `-O1` 下的 `total` 有五个块、五条带类型的边、一条回边，以及预期的直接支配者
- 手工构建的带循环菱形图给出算法所定义的支配者
- `-O2` 下的 `total` 有两个循环
- `-O0` 下的 `grade` 从 `.rela.rodata` 得到五项跳转表，目标为 `0x91, 0x88, 0x7f, 0x76, 0x76`
- 没有表时，`-O0` 下的 `grade` 有三个块和一个未解析跳转
- `-O1` 和 `-O2` 下的 `grade` 是一个值表 `[1, 2, 3, 4, 4]`，没有间接跳转
- `-O2` 下的 `grade_label` 通过五条重定位读取一个相对字符串表
- 交互图嵌入分析所生成的相同数据

## 生产环境中的失败模式

| 失败 | 弱工具报告的内容 | 分析必须证明的内容 |
|---|---|---|
| `.text` 内的数据 | 一份每个字节都解码的貌似合理清单 | 递归下降不声称任何数据字节，且跳转目标是一条指令的起点 |
| 最后一个返回之后的填充 | 函数末尾的一条 `nopw` 指令 | 这些字节从入口不可达 |
| 通过表的间接跳转 | 「jmp rax」和四个死块 | 表地址来自 `lea` 的重定位，表项数来自边界检查，目标来自 `.rela.rodata` |
| switch 被转换为值表 | 「一个带重定位的跳转表」 | 从 `.rodata+0x14` 执行 `movl` 加载，三个块，没有间接转移 |
| 位移为零的调用 | `callq 0x113`，即下一条指令 | `R_X86_64_PLT32 grade-0x4` 解析到 `0x30` |
| 解码器猜测未知操作码 | 一个看起来正确的助记符 | 带地址的 `DecodeError`，且线性扫描记录该字节 |
| 指令长度超过 15 字节 | 一条非常长的「指令」 | 游标在读取第 16 个字节之前抛出错误 |
| 分支进入指令中间 | 两份重叠的清单 | 带两个地址的冲突记录 |
| 循环被报告为直线代码 | 伪代码中没有 `while` | 一条目标支配其源的回边 |
| 基准来自不同的编译器 | 测试在过时文本上通过 | 测试样本 README 指明确切的编译器和命令 |

## 操作规则

对照基准进行解码，从已知入口遍历控制流，并把每个字节归类为已声称、填充、数据或不可达。从重定位或你能指出的表中恢复每个间接目标。其余内容连同地址一起记录为未知。

## 延伸阅读

- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, Chapter 2: Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD64 Architecture Programmer's Manual Volume 3, Chapter 1: Instruction Encoding](https://docs.amd.com/v/u/en-US/24594_3.38_APM_Vol3_PUB)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Schwarz, Debray, Andrews, "Disassembly of Executable Code Revisited" (WCRE 2002)](https://www2.cs.arizona.edu/~debray/Publications/disasm.pdf)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Allen, "Control Flow Analysis" (1970)](https://doi.org/10.1145/800028.808479)
- [Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001)](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Lengauer, Tarjan, "A fast algorithm for finding dominators in a flowgraph" (1979)](https://doi.org/10.1145/357062.357071)
- [Capstone 6.0.0-Alpha10 release notes](https://github.com/capstone-engine/capstone/releases/tag/6.0.0-Alpha10)
- 本课描述的 rea 6.5.0 源码位于标签 `3b79af2`：`README.md`、`docs/ghidra-provider.md`、`docs/ghidra-dos.md`、`docs/reverse-engineering-tools.md`、`bridge/ghidra/ReaGhidraBridge.java`、`src/contracts/officialToolContracts.ts`、`src/ghidra/GhidraAnalysisProfile.ts`。
