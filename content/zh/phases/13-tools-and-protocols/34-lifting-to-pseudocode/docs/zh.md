# 提升为伪代码：IR、SSA 与从零恢复结构

> 反编译器的输出是一个带有隐藏推导过程的断言。当你亲手把这份推导建出来，输出中的每一行就都成为可以检验的对象。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 minutes

## 学习目标

- 把解码后的 x86-64 指令提升为一种小型 IR，显式表示标志位、加载、存储、分支、调用和返回。
- 把栈帧槽转换为局部变量，并把 System V 参数寄存器转换为函数参数。
- 在支配边界上插入 phi 函数，构建最小 SSA，然后做替换、化简并删除死代码。
- 借助后支配节点和回边，从控制流图恢复 `if`、带 `else` 的 `if`、`while` 和 `do while`。
- 解释乘以 `0x66666667` 为何等价于除以 10，并证明该恒等式。
- 逐行阅读同一批函数的 Ghidra 伪代码，并为每处差异指出其背后的指令，或者指出它只是猜测。

## 问题

你让 rea 给出 `ledger-x86_64-O1.o` 中 `percent_change` 的伪代码。Ghidra 12.1.4 在大约十秒内给出了答案：

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)

{
  long lVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  auVar2._8_8_ = 0;
  auVar2._0_8_ = param_3;
  return auVar2 << 0x40;
}
```

源码只有四行。它接收两个 `int` 值，返回一个 `int`。Ghidra 却报告了一个 16 字节的返回值，以及一个源码中并不存在的第三个参数。这十六字节机器码本身是正确的，Ghidra 也确实把其中每一个字节都正确解码了。函数签名是猜出来的，而文本里没有任何标记说明这一点。

rea 把这段文本作为 `procedure_pseudo_code` 结果返回，并在记录上附带一条限制说明。限制说明写道：「Pseudocode is Ghidra decompiler output, not original source and not text-equivalent to Hopper output」。`compare_functions` 工具按文本行对伪代码做 diff。它自己的维度记录写着「Assembly is opaque provider text」，并注明重定位归一化不可用。rea 看不到这段文本的内部，只会阅读文本的 Agent 同样看不到。

rea 的 DX-Ball 记录说明了代价。第一个反编译结果是 `FUN_00406400(void)` 调用 `__ftol()`。指令视图显示了一个栈输入和一个 x87 表达式。Agent 之所以继续追查，仅仅是因为两个视图互相矛盾。

本节课要构建反编译器所隐藏的那份推导。你从第 33 课解码出的 56 条指令出发，把它们提升为 IR，重命名为 SSA，再化简并恢复结构。然后，你把生成的伪代码与 Ghidra 为 `percent_change`、`grade` 和 `total` 生成的伪代码并排摆放，逐条解释每一处差异。测试会用 clang 编译你的输出，并在一个输入网格上对照源码运行。

```figure
lifting-stages-percent-change
```

## 测试夹具与判定基准

`code/fixtures/ledger.c` 是课程这一部分共用的测试夹具：

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (after - before) * 100 / before;
}

int grade(int score) {
    switch (score / 10) {
    case 10:
    case 9: return 4;
    case 8: return 3;
    case 7: return 2;
    case 6: return 1;
    default: return 0;
    }
}

int total(const int *values, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) sum += values[i];
    return sum;
}
```

`grade_label` 也在该文件和目标文件中，演示程序同样会提升它。本课测试的是上面三个函数。

目标文件由 `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c` 构建。`-O0` 目标文件放在它旁边，供栈槽一节使用。`code/fixtures/README.md` 列出了两条构建命令、每个目标文件的 SHA-256，以及生成 Ghidra 夹具所用的 rea 命令。在固定版本的工具链上，构建结果可以逐字节复现。

解码器的判定基准是 `llvm-objdump -d -r`。两个目标文件的输出已提交为 `ledger-x86_64-O1.objdump.txt` 和 `ledger-x86_64-O0.objdump.txt`。有一个测试会解析这些文件，按地址、字节和 AT&T 文本逐条比较所有指令。这样覆盖了 `-O1` 版全部 56 条指令，以及 `-O0` 版的每一条指令。当 `llvm-objdump` 安装在固定路径上时，第二个测试会实时运行它并再次比较。

下面是判定基准打印的 `percent_change`：

```text
0000000000000000 <percent_change>:
       0: 85 ff                        	testl	%edi, %edi
       2: 74 09                        	je	0xd <percent_change+0xd>
       4: 29 fe                        	subl	%edi, %esi
       6: 6b c6 64                     	imull	$0x64, %esi, %eax
       9: 99                           	cltd
       a: f7 ff                        	idivl	%edi
       c: c3                           	retq
       d: 31 c0                        	xorl	%eax, %eax
       f: c3                           	retq
```

## 阶段 1：从字节到指令

第 33 课构建了解码器。本课保留一份精简副本，让代码可以独立运行。该副本只覆盖 clang 为 `ledger.c` 在 `-O0` 和 `-O1` 下发出的操作码。遇到其它字节时，它会带着地址和操作码字节抛出 `DecodeError`。

解码器遵循 Intel SDM 第 2 卷第 2.1 节的字节顺序。先是前缀，然后是操作码，接着是 ModR/M、SIB、位移，最后是立即数。第 2.2.1 节补充了 REX 前缀：一个从 `0x40` 到 `0x4F` 的字节，必须紧挨在操作码之前。REX.W 选择 64 位操作数，REX.R 扩展 `reg` 字段，REX.X 扩展 SIB 索引，REX.B 扩展 `r/m` 或 SIB 基址。第 2.1.3 节定义了 ModR/M 的三个字段 `mod`、`reg`、`r/m`，以及 SIB 的三个字段 `scale`、`index`、`base`。在 64 位模式下，`mod = 00` 且 `r/m = 101` 表示相对于下一条指令的 32 位位移（第 2.2.1.6 节）。

| 操作码字节 | 指令形式 | 夹具示例 |
|---|---|---|
| `01 29 31 39 85 89` | `op Ev, Gv` | `29 fe` `subl %edi, %esi` |
| `03 2b 3b 8b 8d` | `op Gv, Ev` | `03 04 97` `addl (%rdi,%rdx,4), %eax` |
| `63` 配合 REX.W | `movsxd Gv, Ev` | `48 63 c7` `movslq %edi, %rax` |
| `69` 与 `6b` | `imul Gv, Ev, imm` | `48 69 c0 67 66 66 66` |
| `83 /0 /5 /7` | `add sub cmp Ev, imm8` | `83 c1 fa` `addl $-0x6, %ecx` |
| `c1 /5 /7` | `shr sar Ev, imm8` | `48 c1 f9 22` `sarq $0x22, %rcx` |
| `f7 /7` | `idiv Ev` | `f7 ff` `idivl %edi` |
| `ff /0 /4` | `inc Ev`, `jmp Ev` | `48 ff c2` `incq %rdx` |
| `70` 到 `7f`、`eb`、`e8` | `jcc rel8`, `jmp rel8`, `call rel32` | `74 09` `je 0xd` |
| `99`、`c3`、`90`、`0f 1f` | `cltd`, `retq`, `nop` | `66 0f 1f 44 00 00` `nopw (%rax,%rax)` |

`grade+0x23` 处的 `leaq (%rip), %rcx` 长七字节，位移为零。目标文件在偏移 `0x36`（即位移字段）处带有重定位 `R_X86_64_PC32.rodata+0x10`。提升器会在阶段 3 解析它。

## 阶段 2：基本块与支配节点

基本块是一串语句，顶部只有一个入口，底部只有一个出口。这是 Allen（1970）给出的定义，此后每一篇反编译论文都沿用。`build_blocks()` 收集 leader：函数入口、函数内每个跳转目标，以及每个跳转或返回之后的那条地址。每个块的后继由它的最后一条指令决定。条件跳转有两个后继，`jmp` 有一个。`ret` 没有后继；如果块以普通指令结尾，则顺序落到下一个 leader。

`-O1` 下的 `total` 有五个块：

| 块 | 指令 | 后继 |
|---|---|---|
| `0x40` | `testl %esi, %esi` 之后 `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | 出口 |
| `0x5c` | `xorl %eax, %eax` 之后 `retq` | 出口 |

`immediate_dominators()` 采用 Cooper、Harvey 和 Kennedy（2001）的迭代算法。它按逆后序遍历各块，对已处理前驱的支配集求交，直到不再变化。在这样小的图上，两趟就收敛。对 `total`，它得出 `0x44` 支配 `0x50`；同时，它也得出 `0x50` 通过回边支配自身，阶段 8 正是靠这一点找到循环。

同一个函数还会再跑两次。`dominance_frontiers()` 在阶段 5 用它处理 SSA。`post_dominators()` 在反向图上运行它，并加入一个虚拟出口节点，用于在阶段 8 找出汇合点。

## 阶段 3：提升到小型 IR

IR 有六种语句：`Assign`、`Store`、`Branch`、`Jump`、`IndirectJump` 和 `Return`。表达式包括常量、变量、带位宽的二元运算、类型转换（`sext`、`zext`、`trunc`）以及加载。另有五种形态只是暂时存在，之后会被化简掉：`Flags`、`Cond`、`Concat`、`SignFill` 和 `Symbolic`。

寄存器是按 64 位家族命名的变量。对 `%eax` 的写入是对 `rax` 的一次 32 位宽 `Assign`；读取 `%eax` 则是以 32 位宽读取 `rax`。在 64 位模式下，32 位写入会零扩展到整个寄存器，因此之后以 64 位读取同一值时就是一个 `zext`。提升器不知道到达定值，所以它记录读取宽度，并让 SSA 在阶段 5 插入转换。

| 指令 | IR 语句 |
|---|---|
| `subl %edi, %esi` | `flags = cmp(esi, edi)`，然后 `esi = esi - edi` |
| `testl %edi, %edi` | `flags = test(edi, edi)` |
| `je 0xd` | `if je(flags) goto 0xd else goto 0x4` |
| `xorl %eax, %eax` | `eax = 0` |
| `cltd` | `edx = signfill(eax)` |
| `idivl %edi` | `dividend = edx:eax`, `eax = dividend / edi`, `edx = dividend % edi` |
| `movslq %edi, %rax` | `rax = sext64(edi)` |
| `sarq $0x22, %rcx` | `rcx = rcx >>s 34` |
| `shrq $0x3f, %rax` | `rax = rax >>u 63` |
| `addl (%rdi,%rdx,4), %eax` | `flags = add(eax, load32(rdi + rdx * 4))`，然后 `eax = eax + load32(rdi + rdx * 4)` |
| `leaq (%rip), %rcx` 及其重定位 | `rcx = &.rodata+0x14` |
| `pushq %rbp`, `movq %rsp, %rbp`, `subq $0x10, %rsp`, `popq %rbp` | 无输出，属于栈帧维护 |
| `retq` | `return rax` |

### 标志位是一个变量

每条设置条件码的指令都会写一个名为 `flags` 的变量，每个条件跳转都会读取它。`cmp a, b` 写入 `cmp(a, b)`，`test a, b` 写入 `test(a, b)`，而 `sub` 会顺带写入同样形状的 `cmp`。`jcc` 变成 `Branch(Cond(cc, flags), target, fallthrough)`。两者在阶段 6 相遇时，由条件码决定比较运算符：

| 条件码 | 在 `cmp(a, b)` 之后 | 在 `test(a, a)` 之后 |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | 有符号 `<`, `>=`, `<=`, `>` | 与 0 的有符号比较 |
| `jb`, `jae`, `jbe`, `ja` | 无符号 `<`, `>=`, `<=`, `>` | 与 0 的无符号比较 |

`xorl %eax, %eax` 会直接提升为 `eax = 0`。它设置的标志位按 `add` 风格写入，夹具中没有任何跳转读取它们。`xor` 这个惯用法在 `grade` 中很重要，Ghidra 在那里把它弄丢了。

### cltd 与 idiv

`cltd` 用 `%eax` 的符号位填充 `%edx`。`idivl %edi` 用 `%edi` 除 64 位寄存器对 `%edx:%eax`，商放入 `%eax`，余数放入 `%edx`。提升器先把这对寄存器写入名为 `dividend` 的临时变量，这样余数语句读到的是旧的 `%eax`，而不是新的商。在 SSA 中，高半部分表现为低半部分的 `signfill`。化简器把这对寄存器归约为低半部分，除法随之变成 32 位的 `/`。

### -O0 下的栈槽

在 `-O0` 下，clang 把所有局部变量都留在栈帧里。`percent_change` 把 `%edi` 存到 `-0x8(%rbp)`，把 `%esi` 存到 `-0xc(%rbp)`，之后每次使用都重新读回。提升器会寻找 `pushq %rbp` 后跟 `movq %rsp, %rbp` 的序列。出现这个序言之后，任何基址为 `%rbp`、无索引、带位移的内存操作数都会变成名为 `local_<hex>` 的变量。`-0x8(%rbp)` 是 `local_8`，`-0xc(%rbp)` 是 `local_c`。

栈帧相关指令本身不产生 IR，对 `%rsp` 的 `subq` 和 `addq` 也不产生。其它内存操作数仍保持为 `Load` 或 `Store`。

### 重定位

`grade` 中的 `leaq (%rip), %rcx` 在文件中的位移为零。偏移 `0x36` 处的重定位写着 `R_X86_64_PC32.rodata+0x10`。链接器会把 `S + A - P` 写入该字段：`S` 是 `.rodata` 的地址，`A` 是加数 `0x10`，`P` 是该字段的地址。于是指令计算 `next_ip + disp = (P + 4) + S + A - P = S + A + 4`。

提升器不做链接，直接执行同样的算术。目标偏移是加数加上字段到指令末尾的距离：`0x10 + 4 = 0x14`。IR 写成 `rcx = &.rodata+0x14`，那里的表存放五个 `int` 值 `1, 2, 3, 4, 4`。

## 阶段 4：从调用约定得到参数

System V AMD64 psABI 第 3.2.3 节规定，INTEGER 类参数依次通过 `%rdi`、`%rsi`、`%rdx`、`%rcx`、`%r8`、`%r9` 传递。返回值放在 `%rax`，必要时再用 `%rdx`。第 3.2.1 节指出，`%rbp`、`%rbx` 以及 `%r12` 到 `%r15` 归调用者所有，必须被保留；其余寄存器归被调用者所有。

`collect_params()` 只应用一条规则：若某个参数寄存器在函数内先被读取、后才有写入，它就是一个参数。最高的这类寄存器决定参数个数，因此一个读取 `%rsi` 却从不读取 `%rdi` 的函数仍有两个参数。`percent_change` 读取 `%edi` 和 `%esi`，所以它有 `param_1` 和 `param_2`。`grade` 只读取 `%edi`。`total` 把 `%rdi` 读作 64 位基址，把 `%esi` 读作 32 位计数。

现在 Ghidra 的 `param_3` 有了解释。`idivl` 会写 `%rdx`，而 psABI 允许 128 位 INTEGER 返回值放在 `%rax:%rdx`。在目标文件中没有调用者能说明返回值有多宽，Ghidra 于是把两个寄存器都保留为返回值。

在零路径上，`xorl %eax, %eax` 只写 `%eax`。Ghidra 把那里的 `%rdx` 读作函数的一个输入，这个输入就成了 `param_3`。指令本身没错；签名是对指令未说明部分的一种猜测。

## 阶段 5：SSA，每个定义只有一个名字

提升之后，`grade` 会四次写 `rax`、三次写 `rcx`。Cytron、Ferrante、Rosen、Wegman 和 Zadeck（1991）定义了静态单赋值形式：每个变量只被赋值一次，汇合点的 phi 函数在到达该点的各个定义之间做选择。构造分两步：第 5.1 节用支配边界放置 phi 函数，第 5.2 节沿支配树遍历，重命名每个定义和使用。`insert_phis()` 和 `rename()` 分别对应这两节。

`total` 的 SSA 列表展示构造结果：

```text
block 0x44 -> 0x50
  ecx_1 = esi_0
  edx_1 = 0
  eax_1 = 0
block 0x50 -> 0x50, 0x5b
  eax_2 = phi(0x44: eax_1, 0x50: eax_3)
  rdx_2 = phi(0x44: edx_1, 0x50: rdx_3)
  flags_2 = phi(0x44: flags_1, 0x50: flags_5)
  flags_3 = add(eax_2, load32(rdi_0 + rdx_2 * 4))
  eax_3 = eax_2 + load32(rdi_0 + rdx_2 * 4)
  flags_4 = inc(rdx_2, 1)
  rdx_3 = rdx_2 + 1
  flags_5 = cmp(zext64(ecx_1), rdx_3)
  if jne(flags_5) goto 0x50 else goto 0x5b
```

寄存器的版本 0 是它在函数入口处的值。`esi_0` 和 `rdi_0` 是参数。循环头 `0x50` 有两个前驱：初始化块和它自身，因此循环内每个有定义的变量都在这里得到 phi。`flags_2` 这个 phi 一出生就是死的。最小 SSA 仍会放置它，因为 `flags` 在两个前驱中都有定义；阶段 6 再把它删掉，因为没有地方读取它。

重命名步骤还会修正位宽。`edx_1 = 0` 是 32 位定义，而 `rdx_3 = rdx_2 + 1` 是 64 位，所以 phi `rdx_2` 取更宽的位宽。`cmpq %rdx, %rcx` 中对 `%rcx` 的读取是 64 位，而它的定义 `ecx_1 = esi_0` 是 32 位，因此该使用变成 `zext64(ecx_1)`。最终伪代码中的每一个类型转换都来自这一条规则。

所谓「lite」指的是退出 SSA 的方式。`coalesce_webs()` 把每个 phi 及其参数合并为一个变量，称为 web，并给这个 web 一个名字。这对编译器为这些函数生成的形态有效。遇到平行复制时可能失效，例如一个汇合点的两个 phi 交换取值；本课不处理这种情况。Cytron 的第 7 节描述了通过着色完成的一般化退出方式。

## 阶段 6：简化、替换、删除死代码

`optimize()` 反复执行三趟，直到不再变化。化简器自底向上重写表达式，规则如下：

| 改写前 | 改写后 |
|---|---|
| `Cond(je, cmp(a, b))` | `a == b`，其余九个条件码同理 |
| `Cond(jle, test(a, a))` | `a <= 0` |
| `x + (-6)` | `x - 6` |
| `trunc32(zext64(x))`，其中 `x` 为 32 位 | `x` |
| `trunc32(a + b)`，作用于 64 位操作数 | `trunc32(a) + trunc32(b)` |
| `(signfill(x): x) / d` | 以 32 位宽计算的 `x / d` |
| `load32(base + idx * 4)` | `base[idx]` |
| `trunc32(p >>s 34) + trunc32(p >>u 63)`，其中 `p = sext64(x) * 0x66666667` | `x / 10` |

前向替换把定义搬到它唯一的使用处。规则有三个条件。使用与定义在同一块中。使用是一条语句而不是 phi。若定义包含加载，则定义与使用之间不能有存储或调用。常量或节地址可以替换到除 phi 以外的任何位置，因为 phi 的参数必须保持为变量，才能做 web 合并。

死代码删除会删掉目标无使用且表达式无调用的赋值或 phi。各趟的顺序很关键。在 `percent_change` 中，余数 `edx_2 = dividend_1 % edi_0` 是死的，因此先删它。接着 `edx_1 = signfill(eax_1)` 只剩一个使用，于是被替换进 `dividend_1 = edx_1:eax_1`。

随后，配对规则把 `signfill(eax_1):eax_1` 归约为 `eax_1`，除法变成 `eax_1 / edi_0`。现在 `eax_1 = esi_1 * 100` 只剩一个使用，`esi_1 = esi_0 - edi_0` 也只剩一个使用，两者都被搬进 return 语句。`flags` 的各次定义已被替换进分支条件并消失。四条指令变成一行：

```c
return (param_2 - param_1) * 100 / param_1;
```

### 0x66666667 乘法

`grade` 要把分数除以 10，clang 却没有发出 `idiv`。它发出 `imulq $0x66666667, %rax, %rax`，再对副本执行 `sarq $0x22, %rcx`，然后对 `%rax` 执行 `shrq $0x3f`，最后把两者相加。Granlund 和 Montgomery 的论文「Division by Invariant Integers using Multiplication」（PLDI 1994）描述了这种方法。编译器选取乘数 `m` 和移位量 `s`，使得对范围内的每个 `x`，`(x * m) >> s` 都等于 `x / d`。

这里 `m = 0x66666667 = 1717986919`，`s = 34`。算术上可以验证：`2^34 / 10 = 1717986918.4`，向上取整为 `1717986919`。误差是 `m * 10 - 2^34 = 6`。要让恒等式对每个 32 位 `x` 成立，误差乘以 `2^31` 后必须小于 `2^34`，因此误差至多为 8。6 不超过 8，所以对每个非负 `int`，`(x * m) >> 34` 等于 `floor(x / 10)`。

对负数 `x`，floor 向远离零的方向取整，而 C 向零取整。`shrq $0x3f` 提取乘积的符号位，当且仅当 `x` 为负时该位为 1。加上它便修正了取整方向。

`magic_divisor(m, s)` 测试一个候选组合。它对 `2^s` 除以 `m` 取整数商得到 `d`，然后尝试 `d` 和 `d + 1`。只有当 `m` 等于 `2^s / d` 的向上取整且误差界成立时，才接受该 `d`。同一个乘数配 `s = 33` 得到 `d = 5`，这也是正确的，因为 `0x66666667` 同样是 `2^33 / 5` 的向上取整。一个测试在包括 `int` 范围两端的 2,406 个值上，用 `c_div(x, 10)` 检查该恒等式。

开启识别时，`grade` 读作 `ecx_1 = param_1 / 10 - 6;`。关闭识别时，演示程序打印原始形式：

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

两者都能编译，计算结果也相同。请保留这个开关。当你阅读一个自己没有研究过的编译器生成的代码时，原始形式才是诚实的。

## 阶段 7：从使用方式推断类型

机器码只有位宽，没有类型。`infer_types()` 把三类证据变成 C 类型。定义均为 32 位宽的 web 是 `int`，64 位 web 是 `long`。若某个 web 是元素访问 `base[idx]` 的基址，且元素为 4 字节，则它是 `int *`。指针类型沿复制和 phi 传播，因此在 `-O0` 下 `local_8 = param_1` 会让两者都成为 `int *`。返回类型取自 `return` 语句中值的位宽。

`total` 的结果是 `int total(int *param_1, int param_2)`。Ghidra 写成 `int total(long param_1,uint param_2)`。两者都与指令一致。提升器把带缩放的加载当作指针证据；Ghidra 保留寄存器位宽，并把访问写成 `*(int *)(param_1 + uVar2 * 4)`。两者都看不到 `const`，也无法区分 `int *` 和 `unsigned *`。

Lee、Avgerinos 和 Brumley（TIE，NDSS 2011）把类型恢复表述为格上的约束求解，结果是一个上界和一个下界，而不是单一猜测。Noonan、Loginov 和 Cok（Retypd，PLDI 2016）加入多态和递归，并报告称，对编译器留下的惯用法，`const` 指针标注的召回率达到 98%。本课采用这条研究路线中最廉价的两条规则，并明确指出它们看不到什么。

无符号算术保留在表达式里，而不进入变量类型。`cmpl $0x4, %ecx` 之后的 `ja` 打印为 `(unsigned int)ecx_1 > 4`，`shrq` 打印为 `(unsigned long)rax_1 >> 63`。变量本身保持有符号，这样读者既能看到寄存器值，也能看到指令把它当作无符号数的唯一位置。

## 阶段 8：从图恢复结构

Cifuentes（1994）第 6 章把图的各个区域与高级语言的形态做匹配：先处理循环（6.6.1），再处理二路条件（6.6.2），最后处理多路条件（6.6.3）。Yakdan、Eschweiler、Gerhards-Padilla 和 Smith（NDSS 2015）走的是另一条路。他们推导每个节点执行所满足的条件，再依据这些条件构建结构，不依赖固定形态。其动机来自一次测量：Hex-Rays 在一个样本的 49,514 行中产生 1,571 条 `goto`，平均每 32 行一条；而他们的结构化方法在 1,530 个 coreutils 函数上没有产生任何 `goto`。

`structure()` 属于第一类方法中的一个小成员。它需要阶段 2 的两个事实。以 `Branch` 结尾的块，其直接后支配节点就是该条件语句的汇合点。回边是指向一个支配其源块的目标的边。自然循环则是所有不经过循环头就能到达回边源的块组成的集合。

对于汇合点为 `j` 的 `Branch`：

- 若 `j` 是跳转目标，输出 `if (!cond) { fallthrough region }`，然后从 `j` 继续。
- 若 `j` 是顺序落下的分支，输出 `if (cond) { target region }`，然后从 `j` 继续。
- 若 `j` 是函数出口，则两个分支都会返回。把较小的分支输出为 `if (cond) {... }`，另一个分支紧随其后，不加 `else`。这就是 `percent_change` 和 `total` 中的提前返回。
- 否则输出 `if (cond) { target } else { fallthrough }`，然后从 `j` 继续。这是 `-O0` 下的 `percent_change`：两个分支都存储 `local_4`，并在尾声汇合。

对于循环头：

- 若循环头以 `Branch` 结尾，且只有一个后继在循环外，输出 `while (cond) { body }`。`-O0` 下的 `total` 就是这种形态：循环头比较 `local_14` 与 `local_c`，并在 `jge` 时离开。
- 若循环头本身也是回边所在的块，或退出改由 latch 负责，则输出 `do { body } while (cond);`。`-O1` 下的 `total` 是这种形态：块 `0x50` 执行加法、递增、比较，然后跳回自身。

`IndirectJump` 会终止一个没有后继的块。`-O0` 下的 `grade` 是真正的跳转表，通过 `.rodata+0x0` 处的五个相对偏移执行 `jmpq *%rax`，第 33 课解析了这些目标。本课提升器只打印一条说明。位于 `0x76`、`0x7f`、`0x88` 和 `0x91` 的四个块只能通过间接跳转到达，因此没有被提升。实践练习会补上 `switch`。

### -O1 对 switch 做了什么

在 `-O1` 下，clang 没有为 `grade` 保留跳转表。它先计算 `score / 10 - 6`，再把结果按无符号数与 4 比较。然后从 `.rodata+0x14` 处一张包含五个 `int` 的表里加载返回值。这个无符号比较用一条指令同时覆盖 `score / 10 < 6` 和 `score / 10 > 10`：负差值会变成很大的无符号值，因此同样通不过检查。

表中为商 6 到 10 存放 `1, 2, 3, 4, 4`。带有五个 `return` 语句的 `switch` 变成了一次 `if` 加一次数组读取。提升器正是这样输出的，并按节的偏移把表命名为 `lookup_14`。

表的长度是一种启发式判断。`table_values()` 从表的起点读到下一个被任何 `.text` 重定位引用的 `.rodata` 偏移，或读到节的末尾。在这个目标文件里，下一个被引用的偏移是 `0x28`，即 `grade_label` 的表，所以 `lookup_14` 有 `(0x28 - 0x14) / 4 = 5` 项。与 4 的无符号比较也与此吻合。Ghidra 同样不知道表长，它写成 `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`，即一个字节地址加偏移，把那五个值留给读者去查。

## 与 Ghidra 并排对比

演示程序会打印每一对结果。Ghidra 文本来自 `code/fixtures/ghidra/*.json`，即 rea 在 2026-10-14 返回的 Evidence 记录。所用命令是 `npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`。每条记录都带有 `confidence: observed` 和 `authority: shipped-artifact`。文本是被观察到的，但文本内部的断言是推导出来的，而记录并没有说明是哪些。

### percent_change

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int percent_change(int param_1, int param_2)   | undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{                                              | {
  if (param_1 == 0) {                          |   long lVar1;
    return 0;                                  |   undefined1 auVar2 [16];
  }                                            |   undefined1 auVar3 [16];
  return (param_2 - param_1) * 100 / param_1;  |   if (param_1 != 0) {
}                                              |     lVar1 = (long)((param_2 - param_1) * 100);
                                               |     auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
                                               |     auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
                                               |     return auVar3;
                                               |   }
                                               |   auVar2._8_8_ = 0;
                                               |   auVar2._0_8_ = param_3;
                                               |   return auVar2 << 0x40;
                                               | }
```

| 差异 | 对应的指令或猜测 |
|---|---|
| 返回类型 `undefined1 [16]` 与 `int` | 猜测。`idivl` 同时写 `%rax` 和 `%rdx`，且没有调用者能确定返回宽度。两个寄存器都被保留。 |
| `param_3` | 由上一条推导出的猜测。零路径上只写 `%eax`，于是 `%rdx` 被当作输入读取。 |
| `lVar1 / (long)param_1 & 0xffffffff` 与 `/ param_1` | 指令。`cltd` 加 `idivl` 是对符号扩展后的 32 位值做 64 位除法，再截断到 32 位。两份文本描述的是同一件事；我们用 `signfill` 规则归约了寄存器对。 |
| 保留了 `lVar1 % (long)param_1` | 由返回类型推导而来。`%rdx` 中的余数在我们的分析中是死代码，在 Ghidra 的分析中则存活。 |
| `if (param_1 != 0)` 且零分支放在后面 | 指令顺序。`je 0xd` 跳转到零返回。Ghidra 把计算分支放在前面，我们把较小的分支放在前面。两者都正确。 |
| `return auVar2 << 0x40` | 猜测。一个 16 字节值，低半部分放 `param_3`，再移位到高半部分，这是反编译器表达 `%rax = 0` 且 `%rdx` 未知的方式。 |

`code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c` 中的修复版只改动了这个函数，测试在 81 组输入上把它与源码比较。

### grade

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
static const int lookup_14[5] = {1, 2, 3, 4, 4}; | uint grade(int param_1)
int grade(int param_1)                         | {
{                                              |   uint uVar1;
  int eax_1;                                   |   uint uVar2;
  int ecx_1;                                   |   
                                               |   uVar2 = param_1 / 10 - 6;
  ecx_1 = param_1 / 10 - 6;                    |   uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  eax_1 = 0;                                   |   if (uVar2 < 5) {
  if ((unsigned int)ecx_1 <= 4) {              |     uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
    eax_1 = lookup_14[(unsigned long)(unsigned int)ecx_1]; |   }
  }                                            |   return uVar1;
  return eax_1;                                | }
}                                              |
```

| 差异 | 对应的指令或猜测 |
|---|---|
| 两边都有 `param_1 / 10 - 6` | 指令。两边都识别出了 `0x66666667` 惯用法。 |
| `uVar1 = -(param_1 >> 0x1f) ^ (uint)(... >> 0x3f)` 与 `eax_1 = 0` | 指令 `xorl %eax, %eax` 位于 `grade+0x1a`。该寄存器被清零。Ghidra 把 `xor` 表达为 `param_1` 的符号与乘积符号位之间的异或，这实际上是同一个位的两种写法，而它没有把两者抵消。对每个 `int`，该值都是 0，测试在 141 个分数上证明了这一点。相信这段文本的读者会以为默认值取决于分数。 |
| `uVar2 < 5` 与 `(unsigned int)ecx_1 <= 4` | 指令 `cmpl $0x4, %ecx` 后接 `ja`。条件相同。Ghidra 把 `uVar2` 定为 `uint`，我们在使用处转换。 |
| `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)` 与 `lookup_14[...]` | 指令加重定位。Ghidra 加载目标文件时把 `.rodata` 放在 `0x001000a0`，所以 `0x001000b4` 就是 `.rodata + 0x14`。它应用了重定位，保留了一个字节地址。我们给表起了名字，并读出其五个值。 |
| `uint` 返回值与 `int` | 两边都是猜测。指令只说明它是 32 位。 |

### total

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int total(int *param_1, int param_2)           | int total(long param_1,uint param_2)
{                                              | {
  int eax_1;                                   |   int iVar1;
  int ecx_1;                                   |   ulong uVar2;
  long rdx_1;                                  |   
                                               |   if (0 < (int)param_2) {
  if (param_2 <= 0) {                          |     uVar2 = 0;
    return 0;                                  |     iVar1 = 0;
  }                                            |     do {
  ecx_1 = param_2;                             |       iVar1 = iVar1 + *(int *)(param_1 + uVar2 * 4);
  rdx_1 = 0;                                   |       uVar2 = uVar2 + 1;
  eax_1 = 0;                                   |     } while (param_2 != uVar2);
  do {                                         |     return iVar1;
    eax_1 = eax_1 + param_1[rdx_1];            |   }
    rdx_1 = rdx_1 + 1;                         |   return 0;
  } while ((unsigned long)(unsigned int)ecx_1 != rdx_1); | }
  return eax_1;                                |
}                                              |
```

| 差异 | 对应的指令或猜测 |
|---|---|
| `int *param_1` 与 `long param_1` | 我们在 `addl (%rdi,%rdx,4), %eax` 的基础上推断出的结果。Ghidra 保留了寄存器位宽。 |
| `uint param_2` 和 `(int)param_2` 与 `int param_2` | Ghidra 之所以选择无符号，是因为 `cmpq %rdx, %rcx` 比较的是零扩展后的计数。我们保留 `int`，在比较处转换。 |
| 两边都是 `do { } while` | 块跳回自身的指令 `jne 0x50`。编译器把循环转了型。`-O0` 目标文件保留 `while` 形态。 |
| 保留了 `ecx_1 = param_2`，而 Ghidra 用 `param_2 != uVar2` | 我们的替换规则不跨块。复制 `movl %esi, %ecx` 在初始化块中，比较在循环中，所以变量保留下来。Ghidra 把它传播了。 |
| `(unsigned long)(unsigned int)ecx_1` | 指令。`movl %esi, %ecx` 做零扩展，64 位比较读取整个寄存器。 |

行为测试把 Ghidra 为 `grade` 和 `total` 生成的文本原样编译。它加上两行 `typedef`，并根据 `.rodata+0x14` 处的二十个字节定义 `DAT_001000b4`。两个函数在整个网格上都与源码一致。

## 关于如何阅读伪代码的研究结论

LLM4Decompile（EMNLP 2024）测得，在 HumanEval-Decompile 上仅用 Ghidra 的可重新执行率为 20.12%。大约五分之一的函数重新编译后通过了基准自带的断言。用他们的 6.7B 模型精化 Ghidra 的输出后，该比例升到 52.74%。DecompileBench（Findings of ACL 2025）在来自 130 个程序的 23,400 个函数上运行了六种工业反编译器和六种 LLM 方法。LLM 的输出更易读，但功能正确性低 52.2%。

Liu、Raff 和 Micinski（arXiv 2609.05370，2026 年 9 月）解释了这一差距。他们使用 300 个 GitHub 库函数和 287 个有 CVE 依据的函数。最强的精化模型把 Ghidra 的构建率从 75% 提高到 90%，同时行为匹配率从 74% 降到 62%。即便是在基准中通过全部测试的候选实现，面对 fuzz 输入时仍会出现分歧。在八套系统上，这种情况占 4.9%；其中一套系统高达 13%。已披露漏洞中最多有十分之一在精化后的代码里消失了。

他们追溯到的原因很简单：模型凭空补出的字段、类型、被调用者和守卫条件，替换掉了反编译器留在文本中的可见未知。

这也是提升器要打印 `(unsigned long)(unsigned int)ecx_1` 而不是更漂亮名字的原因。对于无法解析的间接跳转，它打印一条说明而不是 `switch`；表名用 `lookup_14`，而不是模型可能编出的标签。每一行都在 `code/main.py` 里有可读的推导。Ghidra 的 `undefined1 [16]` 虽然难看，却诚实。精化模型给出的干净签名 `int percent_change(int, int)` 结论对、理由错；下一个函数上，同一个模型还会给出干净但错误的签名。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

演示程序解码 `-O1` 目标文件的全部四个函数，并把每条指令与已提交的 `llvm-objdump` 文本比较。对 `percent_change`、`grade` 和 `total`，它依次打印 SSA 之前的 IR、带 phi 函数的 SSA 列表，以及伪代码与 Ghidra 文本的并排结果。随后打印关闭除法识别后的 `grade`，以及 `percent_change`、`total`、`grade_label` 的 `-O0` 版本，最后打印摘要：

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

按以下顺序阅读 `code/main.py`：

1. `read_object()` 读取 ELF64 可重定位目标文件的节头、符号表和 `.rela` 节。
2. `decode_one()` 和 `render_att()` 解码一条指令，并按 `llvm-objdump` 的风格打印。
3. `build_blocks()`、`immediate_dominators()` 和 `dominance_frontiers()` 构建图相关事实。
4. `lift_insn()` 把一条指令转换成 IR 语句并解析重定位。
5. `insert_phis()` 和 `rename()` 构建 SSA 并插入位宽转换。
6. `make_simplifier()`、`substitute_once()` 和 `remove_dead_code()` 在 `optimize()` 内运行。
7. `collect_params()`、`coalesce_webs()` 和 `infer_types()` 为各个 web 命名并推断类型。
8. `structure()` 配合 `post_dominators()` 和 `natural_loops()` 输出 C 函数体。
9. `lift_function()` 把各阶段串起来，`lift_all()` 利用被调用者签名提升整个目标文件。

## 使用它

要提升你自己目标文件中的函数，请用 `clang -target x86_64-unknown-linux-gnu -c` 构建，然后调用 `lift_all(read_object(path))`。当解码器抛出 `DecodeError` 时，从消息中读出操作码字节，在 SDM 操作码表中找到它，然后往文件顶部的表中加一条记录。当提升器无法处理某个助记符时，在 `lift_insn()` 中加一条规则。当结构恢复器输出 `goto` 时，用 `IrPrinter` 打印图，判断缺少哪条形态规则。每次补充都很小，而且都会被现有测试对照 `llvm-objdump` 或 clang 检查。

安装 Ghidra 后，对同一个函数运行 rea，把两份文本并排摆放。在相信任何一份文本之前，先把差异表填完。有指令支撑的差异是呈现方式的选择；没有指令支撑的差异是猜测，而猜测可能出现在任何一边。

## 交互实验

### Lab A：一个函数走完五个阶段

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

运行：

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

沿着 `%esi` 从 `subl %edi, %esi` 到 `esi_1 = esi_0 - edi_0`，再到 `(param_2 - param_1)`。字节 `fe` 是 `subl` 的 ModR/M，其中 `mod = 11`、`reg = edi`、`r/m = esi`。

### Lab B：除法惯用法

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

第一行打印原始的乘法和移位。第二行打印 `10 5 None`。请推演为什么同一个乘数少一次移位就变成除以 5，以及为什么移位 31 无法满足误差界。

### Lab C：新目标文件让解码器停下

用 `-O2` 构建夹具并提升它：

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

解码器在 `0x56` 处停下，报 `opcode 81 is outside the fixture subset`。操作码 `81` 是带 32 位立即数的 group 1 形式，指令为 `andl $0x7ffffff8, %edx`，clang 在 `total` 的向量化循环中使用了它。在 `decode_one()` 中把它加到 `0x83` 旁边，重新运行，看看下一个让你停下的操作码是什么。子集一次扩充一条记录，每条记录都对照 `llvm-objdump` 检查。

### Lab D：grade 中 Ghidra 的默认值

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

每一行都打印 0。Ghidra 的 `uVar1` 表达式是正确的，但难以阅读；而指令 `xorl %eax, %eax` 用两个字节就说明了同一件事。

## 实践练习

为 `-O0` 下的 `grade` 补上 `switch` 恢复。

要求：

- 读取 `ledger-x86_64-O0.o` 的 `.rela.rodata` 中五条重定位（每条都是 `R_X86_64_PC32.text+offset`），并根据 `.rodata+0x0` 处的表计算五个 case 目标。
- 向 IR 增加 `Switch` 终结符，包含索引表达式、目标列表，以及表加载之前 `ja` 给出的默认分支。
- 让 switch 块对每个目标都有一个后继，以保证支配节点、后支配节点和 SSA 继续工作。
- 输出 `switch (local_10) { case 0:... }`，每个目标一个 `case`，并在汇合点之后输出共享尾声。
- 对于无法确定表范围的间接跳转，保留说明输出。
- 用 clang 编译结果，并在测试中的分数网格上与源码比较。

成功标准是：`-O0` 下的 `grade` 与 `-O1` 下的 `grade` 在整个网格上输出相同，且只有一个 `switch`，没有 `goto`。

## 评分实验

五个关于本课夹具的问题。用你构建的代码找出每个答案，然后写入 `code/lab_answers.json`。数字写成数字或 `"0x40"` 这样的十六进制字符串，列表写成 JSON 列表。

1. `t1`：调用约定在 -O1 下给 percent_change 的参数。
2. `t2`：-O1 下 total 的 SSA 形式中 phi 函数的数量。
3. `t3`：magic_divisor 对 0x66666667 和移位 34 找到的除数。
4. `t4`：-O1 下 percent_change 伪代码的最后一条语句，写成一行文本。
5. `t5`：-O1 下 total 的 param_1 恢复出的 C 类型。

检查答案：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/lab.py
```

检查器会在已提交的夹具上用 `code/main.py` 计算每个期望值。它为每个任务打印 `PASS`、`FAIL` 或 `TODO` 以及分数，并且只有五项全部通过时才以 0 退出。`python3 code/lab.py my-answers.json` 可以检查另一个文件。`code/tests/test_lab.py` 证明检查器会接受正确答案并拒绝错误答案。

### 看它运行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?opt=O0) 这台机器把本课表现为它的提升压机、类型销、结构夹具和打印器，并展示 `code/main.py` 打印的 IR、SSA 和伪代码。这个链接打开 `-O0` 目标文件：三个栈槽变成局部变量，夹具为 if/else 合上两颚。

## 产出物

本课产出 `outputs/skill-lifting-to-pseudocode.md`。用它提升一个函数，为每一行给出推导，并对照反编译器文本填写差异表。该产出物要求提供参考反汇编、重定位、调用约定、编译检查、行为比较，以及每一处差异行的判定。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

测试套件包含 25 个测试。验证内容包括：

- 解码器在两个目标文件的每条指令上都与已提交的 `llvm-objdump` 文本一致；若安装了 `llvm-objdump`，还与实时运行结果一致
- REX、ModR/M、SIB 以及 `leaq (%rip), %rcx` 的位移字段偏移都按 SDM 的描述解码
- 子集之外的操作码会抛出 `DecodeError`，而不是猜测
- ELF 读取器列出四个函数和三条 `.text` 重定位
- `total` 有五个块，`0x44` 支配 `0x50`，`0x50` 是循环头且循环体是它自己
- `%rdi` 和 `%rsi` 变成 `param_1` 和 `param_2`，`total` 的 `param_1` 是 `int *`
- `test` 和 `cmp` 配合 `je`、`jle`、`ja`、`jge` 分别变成 `==`、`<=`、无符号 `>` 和 `<`
- SSA 在 `total` 的循环头放置两个存活的 phi 函数，在 `percent_change` 中一个也没有
- `percent_change`、`grade`、`total` 的伪代码与预期文本一致
- 开启识别时 `grade` 读作 `param_1 / 10 - 6`，关闭识别时读作原始乘法
- `0x66666667` 配移位 34 时除以 10，配移位 33 时除以 5，且恒等式在整个网格上成立
- `-O0` 下的栈槽变成 `local_4`、`local_8`、`local_c`，帧指针从不出现
- `.rodata+0x14` 处的表经重定位读出 `1, 2, 3, 4, 4`
- `grade` 在 `-O0` 下的间接跳转会被报告，并列出四个未提升的块
- Ghidra 夹具包含 16 字节返回值、`param_3`、未化简的 `xor` 和 `long param_1`
- 六个生成的编译单元都能用 `clang -std=c11 -Wall -Werror` 为 `x86_64-unknown-linux-gnu` 编译
- 提升出的函数、修复后的 Ghidra 文本和源码在输入网格上打印相同的 232 行

## 生产环境中的失败模式

| 失败模式 | 薄弱解读的报告 | 推导必须表明的内容 |
|---|---|---|
| 反编译器拓宽返回值 | 「返回一个 16 字节结构体」 | `idivl` 会写 `%rdx`，且没有调用者能确定宽度 |
| 反编译器增加参数 | 「接收三个参数」 | 该寄存器只在一条路径上被读取，而这只是因为返回宽度是猜的 |
| 惯用法未被识别 | 「乘以一个魔数」 | 乘数是 `2^s / d` 的向上取整，且误差界成立 |
| 惯用法被识别但 `xor` 丢失 | 「默认值取决于输入」 | `xorl %eax, %eax` 写入 0；同一个位的两种写法相互抵消 |
| 指针被推断为整数 | 「接收一个 long」 | 从该寄存器出发的 4 字节缩放加载就是指针证据 |
| 表长未知 | 「读取越过了表尾」 | 下一个被引用的偏移或节末尾给出了边界，且范围检查吻合 |
| 循环形态误判 | 「while 循环」 | 循环头没有出口，latch 跳回，所以是 `do while` |
| 间接跳转靠猜 | 「有五个 case 的 switch」 | 目标来自 `.rela.rodata`，否则就报告为未解析 |
| 跨越存储做替换 | 「一个表达式」 | 被加载的值越过存储或调用后，含义会改变 |
| 精化模型改写文本 | 「能编译、看起来对」 | 构建率上升、行为匹配下降，编造的守卫替换了可见的未知 |
| 解码器猜测未知操作码 | 「反汇编看起来没问题」 | 带地址和字节抛出 `DecodeError`，然后添加一条并对照 `llvm-objdump` 检查 |

## 操作准则

把伪代码当成一份你尚未见过的推导。用你能读懂源码的提升器，从指令把它重新推导出来。将两份文本并排摆放，为每一处不同的行写下其背后的指令或猜测。两版都编译，并在能拿到源码时都对照源码运行。没有指令支撑的行，无论读起来多么干净，都仍然是猜测。

## 延伸阅读

- [Intel SDM Volume 2, chapter 2, Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 psABI, section 3.2, Function Calling Sequence](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Cytron et al., Efficiently Computing Static Single Assignment Form and the Control Dependence Graph, TOPLAS 1991](https://doi.org/10.1145/115372.115320)
- [Cooper, Harvey, Kennedy, A Simple, Fast Dominance Algorithm, 2001](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Cifuentes, Reverse Compilation Techniques, PhD thesis, QUT 1994](https://eprints.qut.edu.au/36820/)
- [Yakdan et al., No More Gotos, NDSS 2015](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Granlund, Montgomery, Division by Invariant Integers using Multiplication, PLDI 1994](https://doi.org/10.1145/178243.178249)
- [Lee, Avgerinos, Brumley, TIE, NDSS 2011](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, Polymorphic Type Inference for Machine Code, PLDI 2016](https://arxiv.org/abs/1603.05495)
- [LLM4Decompile, EMNLP 2024](https://arxiv.org/abs/2403.05286)
- [DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, 2026](https://arxiv.org/abs/2609.05370)
- [rea 6.5.0 Ghidra provider notes](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
- rea 6.5.0 中本课所描述的 `3b79af2` 标签源码：`src/domain/functionComparisonDimensions.ts`、`website/evidence/dx-ball-sound-pan.md`。
