# 对自己程序的动态分析：调试器、跟踪，以及证实静态论断

> 一次命中断点，只是对某一次构建的某一次运行作出的观察。本课构建一套测试台（harness），把这些观察转化为针对静态课程所提论断的三种判定：已确认、已推翻、未观察到。

**Type:** Build
**Languages:** Python, C, arm64 assembly, lldb command language
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~130 minutes

## 学习目标

- 解释 Linux（`ptrace`）和 macOS（Mach 任务端口与异常端口）上的调试器需要操作系统提供什么，以及为什么沙盒中的 shell 无法启动被调试进程。
- 按函数名和按清单地址设置软件断点，设置一个硬件观察点，并说明处理器对两者施加的限制。
- 用 Python 以批处理模式驱动 lldb，配合停止钩子在每次停顿时记录寄存器、内存和栈帧变量，并把会话记录解析回一条时间线。
- 用这条时间线检验第 32、33、34、41 课的静态论断：参数寄存器、结构体字段偏移、条件分支走了哪条边，以及发生了哪些调用。
- 对 `-O2` 构建运行同一份计划，读懂那些能解析却从未命中的断点，并说明这能证明什么、不能证明什么。
- 保存并加载核心文件，并指出栈式核心文件遗漏了哪些内容。
- 对照 gdb 与 lldb 的命令集，并把 `strace`、`ltrace`、`bpftrace`、`dtrace` 和 `rr` 放到正确的操作系统上，注明版本与许可证。
- 陈述让 lldb 能调试你自己构建的二进制文件的 macOS 代码签名事实，以及它拒绝调试的两种条件。

## 问题

第 41 课从 DWARF 中读出 `struct account`，并说 `balance` 位于偏移 4。第 33 课构建了 `classify` 的控制流图，并说当余额为负时，第一个分支走向 `return -1` 块。第 32 课说前两个参数分别通过 `x0` 和 `x1` 传入。这些句子全都从字节推导而来，没有一句经过实际观察。

接着，一个 Agent 读取同一个程序，报告「`main` 调用了 `bump`、`classify` 和 `sum_balances`」。符号表列出了这三个函数，所以这句话看起来稳妥。你用 `-O2` 构建程序，在每个函数上设置断点并运行。进程打印出四行输出后退出。每个断点都报告 `hit count = 0`。

Agent 对源码的判断没错，错的是它被问及的那次构建。编译器内联了全部三个调用，并把 `sum_balances(rows, 3)` 折叠为常量 315。要弄清这台机器上的这个二进制文件究竟如何，唯一的方法就是在调试器下运行并观察。

本课构建一套测试台来做这件事。它编译 `account.c`，读取反汇编清单，规划一次 lldb 会话，运行会话，并针对停顿显示的内容给每条静态论断评分。评分有三种结果。`confirmed` 表示某次停顿表明该论断在这次运行中为真。`refuted` 表示某次停顿表明它为假。`not-observed` 表示没有任何停顿能检验它，记录中会写明原因。随后，第 35 课把每个判定保存为一条 `observed` 记录，权威级别为 `controlled-replay`。

```figure
re-42-run-timeline
```

## 调试器是三项系统服务加一张表

调试器对自己并不拥有其代码的进程做四件事：让进程停下，读写它的寄存器，读写它的内存，并在进程再次停下时得到通知。内核把每一项都作为服务提供，而不同操作系统的服务各不相同。

在 Linux 上，这项服务是 `ptrace(2)`。手册页说它「提供了一种手段，使一个进程（跟踪器）可以观察并控制另一个进程的执行」。`PTRACE_TRACEME` 让子进程声明其父进程将跟踪它；`PTRACE_ATTACH` 接管一个无关进程；`PTRACE_PEEKDATA` 读取「被跟踪进程内存中地址 addr 处的一个字」；`PTRACE_GETREGS` 读取寄存器；`PTRACE_SINGLESTEP` 执行一条指令。Yama 模块限制 attach：`/proc/sys/kernel/yama/ptrace_scope` 默认值 1 表示「目标进程必须是调用者的后代」，2 表示只有管理员可以 attach，3 表示任何人都不能。

在 macOS 上，`ptrace` 是更小的接口。Apple 的手册列出 `PT_TRACE_ME`、`PT_ATTACHEXC`、`PT_STEP`、`PT_CONTINUE` 和 `PT_KILL`，并说明使用 `PT_ATTACHEXC` 时「来自子进程的信号会作为 Mach 异常发送给父进程」。内存和寄存器完全不经过 `ptrace`。lldb 的 `debugserver` 调用 `task_for_pid()` 获取进程的 Mach 任务端口，再调用 `task_set_exception_ports()`，让断点以异常消息的形式到达。系统守护进程 `taskgated` 决定谁可以调用 `task_for_pid`。

这种设计的代价，可以从本课保留的一份会话记录中看出来。该会话的 shell 运行在 Seatbelt 沙盒中。同一份计划在沙盒外产生十一次停顿，在沙盒内则产生：

```text
(lldb) run
error: process exited with status -1 (attach failed ((os/kern) invalid address))
```

进程被创建了，但调试器无法接管它。这个错误关乎调试器的权限，与程序本身无关。测试台把它记录为启动失败：零次停顿，没有任何判定。

这张表就是断点表。调试器记住它在哪个地址替换了哪个指令字节，这样在你继续执行时就能把原始字节放回去，并按位置统计命中次数。`breakpoint list` 打印这张表，测试台在每次运行结束时读取它。

## 断点：替换一个字节或一个字

软件断点是一条会让处理器陷入的指令，被写到你想停下的那条指令的前几个字节上。在 x86-64 上它是 `INT3`，操作码 `CC`，长一个字节，Intel 手册将其描述为「用于调用调试异常处理程序」。在 arm64 上它是 `BRK #imm16`，一条四字节指令，其异常「使用 EC 值 0x3C 把异常记录在 ESR_ELx 中」。Arm 手册补充说，首选返回地址就是 `BRK` 指令本身，这就是为什么停顿报告的是被替换指令的地址，而不是其后一条指令的地址。

lldb 可以按函数名、按文件和行号、或按地址设置断点。按函数名设置时会跳过函数序言：

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O0`bump + 12 at account.c:13:19, address = 0x00000001000004bc
```

`bump + 12` 是第四条指令。它前面的三条是 `sub sp, sp, #0x10`、`str x0, [sp, #0x8]` 和 `str w1, [sp, #0x4]`，这些栈帧建立指令被 DWARF 行表归到函数的起始行。停顿时，`x0` 和 `x1` 仍保存着参数，因为序言只是复制了它们。

按地址设置时，lldb 需要知道该地址属于哪个模块。在进程存在之前裸写 `breakpoint set -a 0x100000504`，得到的是一个背后没有段的加载地址；在第一次测试中，它在整个运行期间都保持 `unresolved`，`hit count = 0`。测试台使用的形式会指定模块：

```text
(lldb) breakpoint set -a 0x100000504 -s account-O0
Breakpoint 4: address = 0x0000000100000504
```

有了 `-s`，这个数字就是 `account-O0` 内部的文件地址，lldb 会加上加载偏移，运行结束后该位置显示为 `resolved, hit count = 2`。第 33 课的清单给出的是文件地址，所以正是这种形式把静态清单与动态停顿连接起来。

### 手工解码分支

`classify` 在 `-O0` 下的第一个条件分支，是 `0x1000004f0` 处的字 `36f800a8`，`llvm-objdump` 把它打印为 `tbz w8, #0x1f, 0x100000504`。按 Arm A64 的 `TBZ` 编码手工解码它。

1. 第 31 位是 `b5`，即位号的高位。它是 0。
2. 第 30 到 25 位是 `011011`，即测试并分支指令组。第 24 位是 0，选择 `TBZ` 而非 `TBNZ`。
3. 第 23 到 19 位是 `b40`，即位号的低五位：`11111` = 31。结合 `b5` = 0，被测试的位是 31，即 32 位值的符号位。
4. 第 18 到 5 位是 `imm14`：`00000000000101` = 5。分支偏移为 `imm14 * 4` = 20 = `0x14`，所以目标地址是 `0x1000004f0 + 0x14 = 0x100000504`。
5. 第 4 到 0 位是 `Rt`：`01000` = 8，所以寄存器是 `w8`。

`w8` 由前一条指令 `ldr w8, [x8, #0x4]` 加载，也就是偏移 4 处的 `a->balance`。`TBZ` 在测试位为零时跳转。符号位清零时分支跳转，即 `balance >= 0`。源码写的是 `if (a->balance < 0) return -1;`，所以编译器反转了条件：跳转边跳过 `return -1` 块，而 `0x1000004f4` 处的顺序执行则进入该块。因此测试台携带的静态论断是一条带谓词的边论断：`a->balance >= 0` 时跳转，否则顺序执行。第二个分支的提升方式相同。`subs w8, w8, #0xc8` 计算 `balance - 200`，`b.le 0x100000524` 在该值至多为零时跳转，即 `balance <= 200`。

## 观察点是硬件资源，而且数量有限

观察点在某个地址的内存被访问时让进程停下。它不能是替换后的指令，因为任何位置上的任何指令都可能发起这次访问。因此处理器带有一小组地址比较器。在 x86-64 上它们是调试寄存器：`DR0` 到 `DR3` 保存「最多 4 个断点的地址」，`DR7` 启用每一个，并设置其条件（执行、写入或读写）和长度（1、2、4 或 8 字节）。在 arm64 上，成对出现的是 `DBGWVR<n>_EL1`，它「保存用于观察点匹配的数据虚拟地址」，以及用于控制的 `DBGWCR<n>_EL1`；Arm 手册说「一个实现可以包含 2 到 64 个观察点」。

lldb 会报告它得到的数量：

```text
(lldb) watchpoint set variable a->balance
Watchpoint created: Watchpoint 1: addr = 0x16fdfe00c size = 4 state = enabled type = m
(lldb) watchpoint list
Number of supported hardware watchpoints: 4
```

这台机器上是四个。被观察的地址 `0x16fdfe00c` 是 `x0 + 4`：`bump` 停顿处的结构体指针加上 DWARF 给出的 `balance` 偏移。`size = 4` 对应 `int`。`type = m` 表示修改（modify）。槽位用尽后，lldb 会拒绝下一个观察点；由于未对齐或长度不是 2 的幂的区域可能占用两个槽位，你能设置的观察点数量可能少于四个。

数据观察点在访问之后报告。Intel 手册说「写内存的指令会先覆盖原始数据，然后才产生数据断点引发的调试异常」，lldb 把旧值保存在自己的缓存里，以便同时打印两者：

```text
Watchpoint 1 hit:
old value: 250
new value: 255
* thread #1, queue = 'com.apple.main-thread', stop reason = watchpoint 1
    frame #0: 0x00000001000004d0 account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:14:12
```

停顿位于 `0x1000004d0`，即 `0x1000004cc` 处 `str w8, [x9, #0x4]` 之后的指令。解析输出之前，有一个怪癖值得了解。刚执行 `watchpoint set` 后，lldb 会打印 `Watchpoint 1 hit:`，旧值为空，`new value: 250`。那是初始读取，测试台会忽略没有旧值的命中。

## 单步执行、停止钩子与批处理模式

lldb 的命令语法是 `<noun> <verb> [-options] [arguments]`。单步动词有 `thread step-in`（`s`）、`thread step-over`（`n`）、`thread step-inst`（`si`）、`thread step-inst-over`（`ni`）和 `thread step-out`（`finish`）。`thread until 12` 运行到某一行。`thread return <expr>` 带值提前返回。测试台不做单步。它在计划的地址上停下，让停止钩子去读取：

```text
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
```

`-o` 可以重复，命令在每次停顿时按顺序执行。`register read` 读取两个参数寄存器和程序计数器。`memory read -s4 -fx -c24 $x0` 从 `x0` 中的地址打印 24 个四字节字（十六进制），即 96 字节，三行 `struct account`。`frame variable -P 1` 打印参数和局部变量，并把指针展开一层，因此 `a` 会显示它的五个字段。

批处理模式是 `lldb -b`。手册说 `--batch` 执行来自 `-s`、`-S`、`-o` 和 `-O` 的命令然后退出，并且如果某条命令「因信号或崩溃」而停止，「调试器会返回交互提示符」。测试台把计划写入文件并用 `-s` 传入。有一项设置很重要：`settings set interpreter.stop-command-source-on-error false`。没有它，进程退出后的第一个 `continue` 就会结束整个文件，`breakpoint list` 永远不会执行。有了它，lldb 会打印 `error: Process must be launched.` 然后继续。

输出形状的两个事实决定了解析器如何工作。第一，钩子输出在钩子运行时打印，而带 `frame #0` 行的 `Process N stopped` 块在其后打印。因此解析器把寄存器、内存和变量行收集到一个待定块中，并把该块附加到它看到的下一条 `frame #0` 行。如果某个块的程序计数器位于 `dyld` 的 `_dyld_start` 中，那它就是程序运行前的入口停顿，不会有任何 `frame #0` 认领它。第二，带 `--auto-continue` 的断点根本不报告停顿，停止钩子也不会为它运行。本测试台的第一个版本用了 auto-continue，结果什么也没记录。计划改用普通断点，并配固定数量的 `continue` 命令。

## 计划来自反汇编清单

`Plan.commands()` 根据它将要运行的二进制的 `llvm-objdump -d` 文本构建命令文件。对于 `account-O0`：

```text
breakpoint set -n bump
breakpoint set -n classify
breakpoint set -n sum_balances
breakpoint set -a 0x100000504 -s account-O0
breakpoint set -a 0x1000004f4 -s account-O0
breakpoint set -a 0x100000524 -s account-O0
breakpoint set -a 0x100000514 -s account-O0
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
run
watchpoint set variable a->balance
watchpoint list
continue
...
breakpoint list
watchpoint list -v
```

三个按名断点是函数入口。四个地址断点是 `Listing.conditional_branches()` 在 `classify` 中找到的每个条件分支的目标与顺序执行位置：`0x1000004f0` 处的 `tbz` 和 `0x100000510` 处的 `b.le`。观察点在第一次停顿时设置，计划预计那是在 `bump`。随后是十六条 `continue` 命令，多于任何运行所需；退出后的剩余命令会无害地失败。这个预算是运行次数的上限，不是关于程序的知识。

论断由 `claims_for()` 从相同的输入构建：

| Id | 类型 | 论断 | 来源 |
|---|---|---|---|
| `arg-registers` | argument-register | 每次入口时前两个参数通过 `x0` 和 `x1` 传入 | 第 32 课与 AAPCS64 |
| `field-offsets` | field-offset | `struct account` 为 32 字节，`kind` 在 0，`balance` 在 4，`code` 在 8，`rate` 在 16，`label` 在 24 | 第 41 课与 DWARF 转储 |
| `branch-1` | branch-taken | `0x1000004f0` 处的 `tbz` 在 `a->balance >= 0` 时跳到 `0x100000504`，否则到 `0x1000004f4` | 第 33 课的边与第 34 课的提升 |
| `branch-2` | branch-taken | `0x100000510` 处的 `b.le` 在 `a->balance <= 200` 时跳到 `0x100000524`，否则到 `0x100000514` | 第 33 课的边与第 34 课的提升 |
| `calls` | call-edge | `main` 调用 `bump` 一次、`classify` 三次、`sum_balances` 一次 | 源码文本与 `nm` |
| `store-once` | store | `bump` 写入 `a->balance` 一次，新值是旧值加 `by` | 第 34 课的伪代码 |
| `stride` | struct-stride | `list[i].balance` 位于 `list + 32*i + 4` | 第 41 课与 psABI 聚合规则 |
| `output` | behaviour | 程序打印 `class 0 1 -1` 和 `sum 315` | 源码算术 |

字段偏移来自 `struct_layout()`，它从 dSYM 转储中读取 `DW_TAG_structure_type`（`DW_AT_byte_size (0x20)`）以及五个 `DW_TAG_member` 条目及其 `DW_AT_data_member_location`。计划从不硬写数字 4，而是读取它。

## -O0 下的运行：十一次停顿

第一次停顿是 `bump` 断点。钩子在停顿块之前打印了这些：

```text
      x0 = 0x000000016fdfe008
      x1 = 0x0000000000000005
      pc = 0x00000001000004bc  account-O0`bump + 12 at account.c:13:19
0x16fdfe008: 0x00000073 0x000000fa 0x00000009 0x00000000
0x16fdfe018: 0x00000000 0x3ff40000 0x00000719 0x00000001
0x16fdfe028: 0x0000006c 0xffffffd8 0x00000003 0x00000000
0x16fdfe038: 0x00000000 0x40000000 0x00000721 0x00000001
(account *) a = 0x000000016fdfe008 {
  kind = 's'
  balance = 250
  code = 9
  rate = 1.25
  label = 0x0000000100000719 "savings"
}
(int) by = 5
```

对照 DWARF 偏移读第一行的八个字。偏移 0 处的字 0 是 `0x73`，即 `'s'`，`kind` 字节，后面跟三个填充字节。偏移 4 处的字 1 是 `0xfa` = 250，即 `balance`。偏移 8 处的字 2 是 `0x9`，即 `code`，一个 `short` 位于低半部分，上方有两个填充字节。偏移 12 处的字 3 是 8 字节对齐 `double` 之前的填充。偏移 16 处的字 4 和 5 是 `0x3ff4000000000000`，即 1.25 的 IEEE-754 编码。偏移 24 处的字 6 和 7 是 `0x100000719`，即指向 `__TEXT` 的 `label` 指针。接下来的八个字以 `0x6c` = `'l'` 和 `0xffffffd8` = -40 开头：第三行 `loan`，位于 32 字节之后。每个数字都与 `frame variable` 一致，因此 `_check_layout()` 在这次停顿时把 `field-offsets` 记录为 `confirmed`，并在 `x0` 持有结构体指针的全部十一次停顿中重复此操作。

`x0` 等于 `a`，`x1` 等于 `by`，因此 `arg-registers` 在这次入口得到确认。测试台只比较声明过的参数。在 `classify` 的停顿时，`x1` 仍读出 `0x5`，这是 `bump` 的遗留值；而 `classify` 只有一个参数，所以不会拿它作比较。没有参数使用的寄存器只是噪声，第 32 课的论断也没有对它作出任何陈述。

第二次停顿是观察点，旧值 250，新值 255，`by` = 5。整个运行只命中一次，因此 `store-once` 得到确认。接着是 `classify` 的三次激活，每次都有入口停顿以及一到两次边停顿：

| 停顿 | 地址 | 停止原因 | 检验内容 |
|---|---|---|---|
| 2 | `0x1000004e8` | breakpoint 2.1 | 入口，`a->balance` = 100，`x0` = `a` |
| 3 | `0x100000504` | breakpoint 4.1 | `tbz` 跳转，100 >= 0，`branch-1` 确认 |
| 4 | `0x100000524` | breakpoint 6.1 | `b.le` 跳转，100 <= 200，`branch-2` 确认 |
| 5 | `0x1000004e8` | breakpoint 2.1 | 入口，`a->balance` = 255 |
| 6 | `0x100000504` | breakpoint 4.1 | `tbz` 跳转，255 >= 0 |
| 7 | `0x100000514` | breakpoint 7.1 | `b.le` 顺序执行，255 > 200 |
| 8 | `0x1000004e8` | breakpoint 2.1 | 入口，`a->balance` = -40 |
| 9 | `0x1000004f4` | breakpoint 5.1 | `tbz` 顺序执行，-40 < 0 |
| 10 | `0x100000544` | breakpoint 3.1 | `sum_balances` 入口，`x1` = 3 |

`_check_branches()` 把停顿按激活分组，在入口读取余额，并对每个边停顿计算提升后的谓词预测的后继。五次边停顿，五次一致：`branch-1` 确认三次，`branch-2` 确认两次。停顿 5 显示的余额是 255，正是观察点看到被写入的值。这次运行针对一个程序，各项证据彼此连贯。

第 10 次停顿时，钩子从 `list` 读取了 24 个字。`_check_stride()` 取字 1、字 9 和字 17，对应偏移 4、36 和 68：100、255 和 -40。它把每个值与 `classify` 在同一地址看到的余额相比较，求和得到 315，再与程序打印的 `sum 315` 行比较。然后进程以状态 0 退出，`breakpoint list` 显示按名断点的命中次数为 1、3、1，四条边断点为 2、1、1、1。全部八条论断最终都是 `confirmed`，背后有 25 条记录，其中 23 条与某次停顿绑定，2 条与整个运行绑定。

## -O2 下的运行：零次停顿

同一份源码在 `-O2` 下是另一个程序。`main` 的 `llvm-objdump -d` 有四条 `bl` 指令，全部指向 `printf`。代替那些调用的是 `mov w8, #-0x1`、`mov w8, #0x1`、通过 `xzr` 写入的零，以及 `mov w8, #0x13b`，即 315。编译器在编译期计算了每个结果。`nm` 仍列出 `_bump`、`_classify` 和 `_sum_balances`，因为它们是外部函数，行外副本必须存在，供其他目标文件调用。dSYM 为全部四个函数包含 `DW_TAG_subprogram` 条目，而 `DW_TAG_inlined_subroutine` 条目为零，所以连调试信息也没有记录这次内联。

这个构建的计划有三个按名断点，没有边断点。`-O2` 下的 `classify` 是六条指令：`ldr`、`cmp`、`cset`、`cmp`、`csinv`、`ret`，其中没有条件分支。`-O0` 清单中的边论断在这个二进制里没有地址，测试台不会把它们带过来。关于某条边的论断，属于其清单中存在该边的那个构建。

运行：

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O2`bump at account.c:13:16, address = 0x0000000100000460
(lldb) run
sizeof 32 align 8
offsets 0 4 8 16 24
class 0 1 -1
sum 315
Process 48444 exited with status = 0 (0x00000000)
(lldb) watchpoint set variable a->balance
error: Command requires a process which is currently stopped.
(lldb) breakpoint list
1: name = 'bump', locations = 1, resolved = 1, hit count = 0
```

每个断点都解析成功。地址存在，符号存在，lldb 在那里写入了 `BRK`。命中次数为零，因为没有任何指令取指到达该地址。判定如下：

| 论断 | 判定 | 记录 |
|---|---|---|
| `calls` | refuted | 完整运行并以退出状态 0 结束后，入口停顿 bump 0、classify 0、sum_balances 0 |
| `output` | confirmed | 四行输出与退出状态 0 |
| `arg-registers` | not-observed | 没有停顿检验它，命中次数全为 0 |
| `field-offsets` | not-observed | 没有停顿检验它 |
| `store-once` | not-observed | `bump` 中没有停顿，所以观察点从未设置 |
| `stride` | not-observed | `sum_balances` 中没有停顿 |

对这次推翻的含义要说得精确。原论断是 `main` 在这个程序中调用了这三个函数。对这个构建和这次运行而言，它没有调用；测试台检查了运行已经完成，因为退出前命中次数为零只意味着运行尚未到达该调用。论断并没有说这些函数是死代码，运行也没有这样说：它们被导出，其他目标文件仍可能调用。另外四条论断既未确认也未推翻。在这个构建中，结构体的 `balance` 仍在偏移 4，`-O2` 的 DWARF 也这样说，但这次运行中没有任何停顿读取内存来展示它。`not-observed` 保留这一区别，第 35 课也会让这个未知保持开放，直到某次运行真正观察到它。

## 核心转储

核心文件是写入磁盘的进程内存和寄存器，这样即使程序已经消失，调试器也能读取一个已停止的程序。Apple 的 `core(5)` 说「少数导致进程异常终止的信号也会导致进程核心内状态的记录被写入磁盘」，写入 `/cores/core.pid`，「前提是被终止的进程对该目录有写权限，且目录存在」。在这台机器上，`sysctl` 报告 `kern.coredump: 1` 和 `kern.corefile: /cores/core.%P`，`/cores` 存在，权限模式为 `drwxrwxr-t root admin`，而 shell 的 `ulimit -c` 为 0。在你用 `ulimit -c unlimited` 提高限制（或用 `launchctl limit core unlimited` 在系统范围提高）之前，不会写入内核核心文件。

lldb 可以在没有崩溃的情况下按需写入核心文件。在 `bump` 断点处：

```text
(lldb) process save-core --style stack account-O0-stack.core
Saving 49152 bytes of data for memory region at 0x16fdf4000

Modified-memory or stack-memory only corefile created.  This corefile may 
not show library/framework/app binaries on a different system, or when 
those binaries have been updated/modified. Copies are not included
in this corefile.  Use --style full to include all process memory.
```

文件大小为 53,248 字节。加载它需要把二进制文件放在旁边：

```text
(lldb) target create "./account-O0" --core "account-O0-stack.core"
(lldb) bt
* thread #1, stop reason = ESR_EC_BRK_AARCH64 (fault address: 0x0)
  * frame #0: 0x00000001000004bc account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:13:19
    frame #1: 0x0000000100000640 account-O0`main at account.c:40:5
    frame #2: 0x000000019a949d54 dyld`start + 7184
(lldb) frame variable *a
(account) *a = (kind = 's', balance = 250, code = 9, rate = 1.25, label = "")
(lldb) frame variable a->label
(const char *) a->label = 0x0000000100000719 ""
```

回溯和整数字段完好，因为它们位于核心文件所保存的栈上。`label` 打印为空字符串，但指针完好。指针 `0x100000719` 是对的；它指向的字节是可执行文件 `__TEXT` 段中的字面量 `"savings"`，而栈式核心文件不包含该段。指针是观察结果，字符串不可用，lldb 也不会凭空造出它。`--style full` 包含全部进程内存；在这台研究机器上，这个 34 KB 程序的完整核心文件达到 2.1 GB，因为进程映射了共享缓存。停止原因显示 `ESR_EC_BRK_AARCH64`：核心文件记录了断点引发的 `BRK` 异常。

在 Linux 上，内核按 `/proc/sys/kernel/core_pattern` 写入核心文件，该模板中 `%p` 表示 PID，`%e` 表示命令名，`%s` 表示信号，`%t` 表示时间；开头的 `|` 会把转储通过管道传给一个程序。systemd 发行版用这个管道交给 `systemd-coredump`，`coredumpctl list`、`info`、`dump` 和 `debug` 可以取回它保存的内容。`gdb prog core` 和 `lldb -c core prog` 都能打开结果。

## gdb 与 lldb 对照

两个调试器做同样的工作，用词不同。测试台中的 `COMMAND_MAP` 有 35 行，由 `python3 code/main.py commands` 打印。本课用到的部分：

| 任务 | gdb | lldb | lldb 别名 |
|---|---|---|---|
| 在函数处设断点 | `break bump` | `breakpoint set -n bump` | `b bump` |
| 在某个模块内的地址处设断点 | `break *0x1000004f0` | `breakpoint set -a 0x1000004f0 -s account-O0` | |
| 在每次停顿时运行命令 | `define hook-stop` ... `end` | `target stop-hook add -o '...'` | |
| 列出断点及命中次数 | `info breakpoints` | `breakpoint list` | `br l` |
| 观察变量的写入 | `watch a->balance` | `watchpoint set variable a->balance` | `wa s v a->balance` |
| 列出观察点 | `info watchpoints` | `watchpoint list -v` | `wa l -v` |
| 读取一个寄存器 | `p $x0` | `register read x0` | |
| 按字读取内存 | `x/8xw 0x16fdfe008` | `memory read -s4 -fx -c8 0x16fdfe008` | `x -s4 -fx -c8 0x16fdfe008` |
| 显示参数 | `info args` | `frame variable --no-locals` | |
| 单步执行一条指令 | `stepi` | `thread step-inst` | `si` |
| 运行到函数返回 | `finish` | `thread step-out` | `finish` |
| 写核心文件 | `gcore core` | `process save-core --style stack core` | |
| 打开核心文件 | `gdb account-O0 core` | `lldb -c core account-O0` | |

lldb 的 `p` 是 `dwim-print` 的别名，它通过 `frame variable` 路径打印普通变量，对其他内容回退到 `expression`。`v` 是 `frame variable`。在测试台中优先使用 `frame variable`：它读取 DWARF 位置，从不在被调试进程中运行代码。

版本与许可证信息截至 2026-10-14。lldb 沿用 LLVM 版本号，当前 llvm.org 发布版是 2026-10-06 的 LLVM 23.1.3，采用 Apache License 2.0（含 LLVM 例外）。Homebrew 将其作为独立 formula `lldb` 提供。这台机器上的 Command Line Tools 附带 `lldb-1703.0.236.21`，这是 Apple 自己的编号，内嵌 Python 3.9.6，且 Apple 的 lldb 已签名，可直接调试。GDB 18.1 于 2026-09-25 发布，采用 GPL-3.0-or-later。其发行说明列出了 Windows、`.gdb_index`、AArch64 `FPMR` 和 Python `gdb.Corefile` 的改动，没有 Darwin 条目。截至 18.1，GDB 对 Apple silicon 进程没有原生支持：2025 年 10 月发布了一组 15 个补丁的「Initial AArch64-Darwin support」系列，2026 年 2 月一位维护者说明了通过版权转让和先合入 BFD 补丁来落地的路线。Homebrew 的 `gdb` formula 有 arm64 bottle，并以伪造的 `--target=x86_64-apple-darwin20` 构建，添加这些 bottle 的 pull request 说「无法通过 GDB 调试 aarch64-darwin 二进制文件」。macOS 上任何非 Apple 调试器还需要自签名证书和 `com.apple.security.cs.debugger` entitlement，否则 `task_for_pid` 会失败并报「(os/kern) failure (0x5)」。gdb 在本课程中的位置是 Linux，Debian stable 为 arm64 提供 gdb 16.3，`rr` 也生活在那里。

## 跟踪：程序向内核请求了什么

调试器展示程序在停顿时的状态。跟踪器（tracer）展示程序对系统的影响，而不会长时间停下它。工具按操作系统区分。

| 工具 | 操作系统 | 版本 | 许可证 | 记录内容 |
|---|---|---|---|---|
| `strace` | Linux | 7.2，2026-08-18 | LGPL-2.1-or-later | 带参数和返回值的系统调用 |
| `ltrace` | Linux | 最后的 tarball 为 0.7.3，Debian 中为 git 快照 | GPL-2.0 | 动态库调用 |
| `bpftrace` | Linux | v0.27.0，2026-09-10 | Apache-2.0 | 通过 eBPF 的内核与用户探针 |
| `dtrace` 与 `dtruss` | macOS | macOS 26.2 中的 `Sun D 1.19` | CDDL，来自 Sun 的 OpenSolaris | D 探针；`dtruss` 打印系统调用 |

`dtruss` 在其手册中说「由于使用 DTrace，只有具备 root 权限的用户才能运行此命令」。Apple 的 `dtrace(1)` 补充说「在默认的系统完整性保护（SIP）设置下，D 程序不允许访问内核地址值或内核内存内容」，并且 SIP 下不允许破坏性操作。本课没有在这台机器上运行 `sudo dtruss ./account-O0`，因为测试台不以 root 运行，课程将这一点记录为未验证。跟踪回答的是与测试台不同的问题：「这次运行是否为每一行 `printf` 发起了 `write`」是跟踪问题，「`bump` 处 `x1` 是否为 5」是调试器问题。两者都只对单次运行有效。

## 记录与回放

`rr` 记录一次运行，然后在 gdb 下确定性地回放，并支持反向执行：在你关心的值上设置观察点，`reverse-cont` 就会反向运行到写入处。该项目这样描述自己：「你记录一次失败，然后就可以确定性地调试这份记录，想调试多少次都行」。当前版本是 2025-02-13 的 5.9.0，采用 MIT 许可证，并附带一个 BSD-2-Clause 条款块。它仅支持 Linux。它需要硬件性能计数器来统计分支，因此可在 Intel Nehalem 及之后、某些 AMD Zen 核心，以及 Linux 下的「某些 AArch64 微架构（如 ARM Neoverse N1 或 Apple Silicon M 系列）」上运行。在虚拟机中，计数器必须被虚拟化，项目 wiki 说「Mac 上的其他虚拟化方案都不行」，所以这台机器上的 Docker 和 Colima 都跑不了它。Pernosco 是基于 rr 记录构建的商业服务。

gdb 有自己的软件记录器。`record full` 记录「进程执行的日志」，默认限制为 200,000 条指令；`record btrace` 使用 Intel 分支跟踪，「不记录数据」。两者在 lldb 中都不可用。在这台 macOS 机器上，测试台保存的会话记录就是记录：同一份计划作用于同一个二进制文件，产生相同顺序的相同停顿，fixture 测试固定了这个顺序。

## 一次运行能证明什么

一次运行只能证明一个构建上的一个输入。C 标准把未定义行为定义为「在使用不可移植或错误的程序构造，或错误数据时，本文档不施加任何要求的行为」，因此一个让有符号 `int` 溢出的程序，在 `-O0` 和 `-O2` 下合法地打印不同结果。第 36 课用 `percent_change` 溢出单元遇到过这一点。UBSan 把这个问题变成一次停顿：`-fsanitize=signed-integer-overflow -fsanitize-trap=undefined` 让溢出「执行一条陷阱指令」，在 lldb 中表现为一次带原因的停顿。

优化后的代码还会移动你想观察的内容。GDB 手册说「优化器重新排列了你的代码；调试器展示的是实际存在的内容」，并且被优化得不存在的变量是「GDB 永远看不到」的变量。GDB 把这种值打印为 `<optimized out>`，其源码将它与 `<unavailable>` 区分开：optimized out「在程序中实际不存在」，unavailable「在程序中存在（在机器层面），但 GDB 无法获取它」。当程序计数器位于变量 DWARF 位置列表的每个范围之外时，lldb 打印 `<variable not available>`。对存放在寄存器中的局部变量设置观察点永远不会触发，因为寄存器不是内存。上面的 `-O2` 运行展示了极端情况：整个函数被折叠掉，而一个从未设置的观察点保持沉默，什么也证明不了。

因此测试台写三种记录，绝不写第四种。检验了论断并一致的停顿是 `confirmed`。检验了但结果不一致的停顿是 `refuted`。没有停顿能检验的论断是 `not-observed`，并附上原因，断点命中次数是原因的一部分。第 35 课的账本接受前两种为 `observed` 记录，权威级别 `controlled-replay`，并把第三种保留为开放未知。

## 在 macOS 上只调试自己构建的程序

在 Apple silicon 上，链接器会替你签名。Apple 的 `ld(1)` 对 `-adhoc_codesign` 说：「Apple Silicon 二进制的默认行为是进行 ad-hoc 代码签名。」对 fixture 运行 `codesign -dvvv account-O0` 打印：

```text
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=387 flags=0x20002(adhoc,linker-signed) hashes=9+0 location=embedded
Signature=adhoc
TeamIdentifier=not set
```

`0x20002` 是 `CS_ADHOC`（`0x2`）加上 `CS_LINKER_SIGNED`（`0x20000`），在 Apple 的代码签名头文件中意为「由链接器自动签名」。`codesign -d --entitlements -` 在可执行文件名之后不打印任何内容：该二进制没有 entitlements。lldb 启动它，命中了每个断点，设置了硬件观察点，并写出了核心文件；当时 `DevToolsSecurity -status` 报告「Developer mode is currently disabled」，系统完整性保护处于启用状态。

有两个 entitlement 控制谁可以调试什么，它们位于相反的两侧。`com.apple.security.cs.debugger` 用于调试器：「一个布尔值，表示该应用是否为调试器，是否可以附加到其他进程或获取任务端口」。Apple 的 `debugserver` 携带它的私有形式，对 Command Line Tools 副本运行 `codesign -d --entitlements -` 会列出 `com.apple.private.cs.debugger`、`com.apple.private.set-exception-port`、`com.apple.private.thread-set-state` 和 `com.apple.private.logging.diagnostic`。`com.apple.security.get-task-allow` 用于目标进程。Xcode 把它注入调试构建，在导出时移除，公证会拒绝仍然带有它的二进制文件。Apple 的工程师陈述了拒绝规则：当目标启用了强化运行时（hardened runtime）且没有 `get-task-allow` 时，任务端口会被拒绝。普通的 `clang` 构建没有强化运行时，两个条件都不触发。如果你曾用 `codesign --options runtime` 给 fixture 签名，请在调试前用 `codesign -s - -f --entitlements get-task-allow.plist` 重新签名。

本节以另外三个事实收尾。第一，在一个登录会话中首次可能看到对话框「Developer Tools Access needs to take control of another process for debugging to continue」；`sudo DevToolsSecurity -enable` 可以为 `admin` 或 `_developer` 组成员免除额外密码。第二，调试自己的代码不需要禁用 SIP。Apple 说「你不需要禁用 SIP 就能从 Xcode 运行和调试应用」，唯一涉及调试的 SIP 组件由 `csrutil enable --without debug` 放宽，针对的是他人的进程。第三，本会话中的 Seatbelt 沙盒完全阻止 `task_for_pid`，fixture `sandbox-O0.lldb.txt` 展示了结果。无法启动的运行是一个环境事实，测试台也将其作为环境事实报告。

本课的一切都适用于你自己编译的程序。同样的命令只有在进程所有者主动允许时才能附加到其他进程，本课不教授如何改变这一点。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
```

输出是一份基于已保存会话记录的 JSON 报告。对每个构建，它列出断点及其解析状态和命中次数、硬件观察点数量、每次停顿及其寄存器和参数、退出状态、程序输出、lldb 打印的错误，以及各条论断及其判定和背后的证据记录。它还包含核心文件往返、Python API 结果、沙盒失败，以及这台机器上缺少哪些外部工具。

预期形状：

```json
{
  "program": "account.c",
  "variants": {
    "O0": {
      "binary": "account-O0",
      "breakpoints": {"1": {"spec": "name=bump", "address": "0x1000004bc", "resolved": true, "hits": 1}, "...": "..."},
      "hardwareWatchpoints": 4,
      "watchpoints": {"1": {"address": "0x16fdfe00c", "size": 4, "type": "m", "hits": [[250, 255]], "hitCount": 1}},
      "stops": [{"index": 0, "pc": "0x1000004bc", "function": "bump", "kind": "entry", "registers": {"x0": "0x16fdfe008", "x1": "0x5"}, "...": "..."}],
      "exitStatus": 0,
      "claims": [{"id": "arg-registers", "verdict": "confirmed", "confirmed": 5, "refuted": 0, "not-observed": 0, "...": "..."}],
      "evidence": [{"claim": "arg-registers", "verdict": "confirmed", "stop": 0, "observation": "bump: x0 = 0x16fdfe008 and a = 0x000000016fdfe008, x1 = 0x5 and by = 5"}]
    },
    "O2": {"stops": [], "claims": [{"id": "calls", "verdict": "refuted", "...": "..."}], "...": "..."}
  },
  "core": {"savedAt": {"function": "bump", "pc": "0x1000004bc"}, "labelString": "", "...": "..."},
  "types": {"python": "3.9.6", "size": 32, "...": "..."},
  "sandbox": {"stops": 0, "errors": ["process exited with status -1 (attach failed ((os/kern) invalid address))"]},
  "absentTools": ["bpftrace", "gdb", "llvm-objdump", "ltrace", "rr", "strace"]
}
```

按以下顺序阅读 `code/main.py`：

1. `Listing` 把 `llvm-objdump -d` 文本解析为函数和指令，`Function.conditional_branches()` 找出分支及其目标和顺序执行位置。
2. `struct_layout()` 从 `llvm-dwarfdump` 文本中读取结构体和成员条目。
3. `lift_condition()` 把一个条件分支转换为谓词，即第 34 课针对 `tbz`、`cbz` 以及比较并分支形式的步骤。
4. `Plan.commands()` 根据清单写出 lldb 命令文件，`Plan.lldb_command()` 是批处理调用。
5. `parse_transcript()` 按顺序遍历输出，构建带有寄存器、内存、变量和观察点值的 `Stop` 记录，以及断点表和退出状态。
6. `claims_for()` 陈述八条论断，`evaluate()` 用 `_check_*` 函数逐次停顿评分。
7. `rollup()` 把记录汇总为每条论断一个判定，`refuted` 优先于 `confirmed`。
8. `capture()` 和 `capture_extras()` 生成了 fixture；当 `clang` 和 `lldb` 可以运行时，`live()` 在全新构建上重复整个循环。

另外还有四条命令。`python3 code/main.py plan O0` 打印某个构建的命令文件。`python3 code/main.py commands` 打印 gdb 与 lldb 对照表。`python3 code/main.py tools` 报告 `PATH` 上有哪些外部工具。`python3 code/main.py live O0` 把 `account.c` 编译到临时目录，根据新的清单制定计划，运行 lldb，并针对那次运行给论断评分。在沙盒中，它会报告 lldb 无法启动进程，附上错误，并以状态 1 退出。

## 使用它

按以下顺序把测试台用于你自己的程序。

1. 用 `-g` 和你将要发布的优化级别构建它。把二进制、其 dSYM 或 DWARF，以及 `llvm-objdump -d` 文本放在一起。
2. 在运行之前写下论断。每条论断都要写明函数、寄存器或地址或偏移、期望值或边，以及它来自哪一课或哪份文档。看到停顿之后才写的论断只是在描述停顿，什么也检验不了。
3. 从清单生成计划并阅读它。计划中的每个地址都应能追溯到清单中的一行。如果某个断点找不到对应行，就删掉它。
4. 以批处理模式运行一次，并保存会话记录，头部带上命令行和调试器版本。会话记录就是证据。判定由它推导而来，任何人都能重新推导。
5. 评分。先读每条 `not-observed` 记录及其原因，再读确认项。原因会告诉你哪些论断需要另一个输入、另一个构建或另一个断点。
6. 用同样的论断运行优化构建。预期看到编译内联之处的断点已解析但零命中，并让这些论断保持开放。
7. 把已确认和已推翻的记录作为 `observed`、权威级别 `controlled-replay` 送入第 35 课的账本，每次停顿一条记录。把未观察到的保留为未知，并附上断点命中次数。

## 交互实验

### 实验 A：手工读取一次停顿

在 `code` 目录中打开 Python：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program/code
python3 -q
```

运行：

```python
from main import *
t = load_transcript("session-O0.lldb.txt")
s = t.stops[0]
hex(s.pc), s.function, s.reason, s.registers
[hex(w) for w in s.memory[:8]]
s.struct_pointer().fields
```

在执行第二行代码之前，写下你认为哪个字保存 250，哪两个字保存 1.25。然后运行 `struct.unpack("<d", struct.pack("<II", s.memory[4], s.memory[5]))` 并比较。接着查看 `t.stops[1].watch_hit` 和 `t.stops[1].pc`，在 `account-O0.objdump.txt` 中找到那个地址：它是存储指令之后的那条指令。

### 实验 B：分支及其后继

```python
plan = make_plan("O0")
plan.branches()
classify = plan.listing.function("classify")
[lift_condition(classify, b, "a->balance")[0] for b in plan.branches()]
[(hex(s.pc), s.struct_pointer().field_int("balance")) for s in t.stops_in("classify")]
```

在查看证据之前，把每次边停顿与其入口停顿配对，判断谓词预测的是哪个后继。然后运行 `[e.to_dict() for e in evaluate(plan, t, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim.startswith("branch")]` 并比较。

### 实验 C：从未停下的运行

```python
o2 = load_transcript("session-O2.lldb.txt")
len(o2.stops), o2.exit_status, o2.hits()
o2.errors[:3]
make_plan("O2").branches()
rollup(evaluate(make_plan("O2"), o2, struct_layout(open("fixtures/account-O2.dwarfdump.txt").read(), "account")))
```

用你自己的话解释每条 `not-observed`，然后打开 `account-O2.objdump.txt`，找出 `main` 中替代调用的四个常量。再用 `count_tags()` 统计 `account-O2.dwarfdump.txt` 中的 `DW_TAG_inlined_subroutine`。

### 实验 D：破坏证据

```python
text = open("fixtures/session-O0.lldb.txt").read().replace("x1 = 0x0000000000000005", "x1 = 0x0000000000000006", 1)
broken = parse_transcript(text)
[e.to_dict() for e in evaluate(plan, broken, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim == "arg-registers"][:2]
```

一个被修改的寄存器会让第一条 `arg-registers` 记录变成 `refuted`，汇总结果也随之改变，而另外四条记录仍为 `confirmed`。然后删除观察点停顿前的 `old value: 250` 行，看 `store-once` 降为 `not-observed`。一个无法被弄失败的评分器，就不是在评分。

如果你的 shell 中 lldb 能启动进程，最后运行 `python3 main.py live O0` 和 `python3 main.py live O2`。栈地址会与 fixture 不同，判定不应不同。

## 实践练习

给测试台增加一条关于 `sum_balances` 中循环的论断。

要求：

- 用 `conditional_branches()` 在 `-O0` 清单中找到 `sum_balances` 的循环回边，并在循环头部添加一个断点。
- 增加一条论断：对于 `count` = 3，头部执行 `count + 1` 次，也就是第 33 课的回边加上退出测试。
- 扩展停止钩子或计划，使钩子在每次头部停顿时读取循环计数器的栈槽，并编写 `_check_loop()` 记录计数器值的序列。
- 在 lldb 能启动的机器上用 `python3 main.py capture O0` 重新采集会话记录，并重新运行测试套件。在提交的命令文件与新计划一致之前，计划测试会失败，这正是该测试的意义。
- 增加一个测试，修改会话记录中的一个计数器值，并断言该论断变为 `refuted`。

成功意味着报告按顺序显示头部停顿及其计数器值，并且 `-O2` 运行让新论断保持 `not-observed`，原因是循环被向量化或折叠掉了。

## 课程产出物

本课包含 `outputs/skill-dynamic-analysis-of-your-own-program.md`。用它把你构建的程序、其清单和一组静态论断转化为一份保存的调试器会话记录，以及每条论断一个判定，并给出产生每个判定的停顿，以及每条仍为 not-observed 的论断的原因。它适用于 Claude、Cursor、Codex、OpenClaw、Hermes 或任何读取 SKILL.md 的 Agent。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

测试套件无需 lldb 即可运行。它读取一次采集到 `code/fixtures/` 中的会话记录。验证应证明：

- 清单解析器找到四个函数及其范围，以及 `-O0` 下 `classify` 的两个条件分支及其目标和顺序执行位置
- `tbz w8, #0x1f` 的提升结果是 `a->balance >= 0`，`subs` 加 `b.le` 的提升结果是 `a->balance <= 200`
- `-O0` 下的 `main` 调用 `bump`、三次 `classify` 和 `sum_balances`，而 `-O2` 下的 `main` 只调用 `printf`
- x86-64 的 `j<cc>` 助记符是条件跳转，`jmp` 和 `callq` 不是
- DWARF 转储给出一个 32 字节结构体，成员位于 0、4、8、16 和 24
- `-O2` 的 DWARF 有四个子程序和零个内联子程序条目，`nm` 仍列出三个函数
- 计划与两个构建提交的命令文件一致，会话记录头部记录运行它的 lldb 命令
- `-O0` 会话记录在预期的函数中有十一次停顿，退出状态 0，并包含四行输出
- 钩子输出附加到其后的停顿，`x1` = 5，内存从 `x0` 开始，并包含标签字符串
- 断点表带有解析状态和命中次数 1、3、1、2、1、1、1
- 观察点在 `x0 + 4` 处是 4 字节的修改类型，命中一次，旧值 250、新值 255，并报告四个硬件槽位
- `-O2` 会话记录有零次停顿、三个已解析但零命中的断点、没有观察点，且每个未使用的 `continue` 产生一个错误
- 沙盒中的启动在任何停顿之前失败，并给出 attach 错误
- 核心文件往返保留回溯和 `balance` = 250，并丢失 `label` 背后的字符串
- Python API 在没有进程的情况下读取五个字段、大小 32 和函数范围
- 手写的 x86-64 会话记录能用 `rdi`、`rsi` 和 `rip` 解析
- 在 `-O0` 下每条论断都得到确认，记录数分别为 5、11、3、2、1、1、1、1
- 分支证据给出值、条件和边
- 字段偏移证据通过 DWARF 布局解码这些字
- 存储与步长证据带有旧值、新值和三个行的余额
- `-O2` 运行推翻调用边，并让其他论断保持 not-observed
- 被修改的寄存器值会被推翻，被删除的观察点命中会变成 not-observed，且推翻在汇总中胜出
- 命令映射的每一行都有 gdb 和 lldb 形式，硬件表列出四个 x86 调试寄存器
- 演示列出缺失的工具而不是失败，live 模式按名称报告缺失的工具
- 图形数据来自两个会话，站点模块嵌入相同的数据

## 生产环境故障模式

| 故障 | 弱报告会怎么说 | 测试台必须证明什么 |
|---|---|---|
| 已解析但零命中的断点 | 「函数运行了」或「函数是死代码」 | 一次带退出状态的完整运行、命中次数，并只对调用边给出 `refuted` 判定 |
| 在错误的停顿处读取寄存器 | 「`classify` 内部 `x1` = 5」 | 只在入口停顿时比较声明过的参数 |
| 不按偏移读取结构体 | 一列十六进制字 | 每个字都映射到 `DW_AT_data_member_location`，并与 `frame variable` 比较 |
| 根据助记符给分支判定 | 「`tbz` 表示负数时跳转」 | 从编码提升出的谓词，以及后继处的停顿 |
| 对寄存器中的局部变量设观察点 | 「没有发生写入」 | 观察点设在某个地址上且确实命中，否则记录为 `not-observed` |
| auto-continue 断点 | 一份没有停顿的干净记录 | 普通断点、停止钩子和 continue 预算 |
| 不带模块的地址断点 | `unresolved, hit count = 0` | 用 `-s MODULE`，让清单地址解析为加载地址 |
| 第一条命令出错就结束批处理 | 末尾没有 `breakpoint list` | 设置 `interpreter.stop-command-source-on-error false` |
| 把栈式核心文件当作完整内存读取 | 把 `label = ""` 报告为程序的值 | 核心文件样式、指针及其指向的段 |
| 沙盒中的 shell | 「lldb 坏了」 | 把 `attach failed ((os/kern) invalid address)` 记录为环境事实 |
| 没有 `get-task-allow` 的强化运行时目标 | 「macOS 禁止调试」 | Apple 规则中的两个条件，以及对自己二进制重新签名 |
| 评分后丢弃会话记录 | 没有来源的判定 | 会话记录文件、其命令行和调试器版本 |

## 操作规则

从清单规划停顿，以批处理模式运行一次，保留会话记录，并在每个能检验它的停顿处给每条静态论断评分。只根据停顿写 confirmed 或 refuted；写 not-observed 时附上命中次数和原因；绝不要让一个已解析的断点代替一次运行。只调试你自己构建的程序。

## 延伸阅读

- [LLDB Tutorial: command grammar, breakpoints, watchpoints, stop hooks](https://lldb.llvm.org/use/tutorial.html)
- [lldb(1) manual page: --batch, --one-line, --source, --core](https://lldb.llvm.org/man/lldb.html)
- [GDB to LLDB Command Map](https://lldb.llvm.org/use/map.html)
- [LLDB Python Reference: SBTarget, SBFrame, SBValue, and the embedded interpreter](https://lldb.llvm.org/use/python-reference.html)
- [LLDB Caveats: the Python build lldb links against](https://lldb.llvm.org/resources/caveats.html)
- [LLVM 23.1.3 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.3)
- [GDB 18.1 release announcement, 2026-09-25](https://sourceware.org/pipermail/gdb-announce/2026/000150.html)
- [GDB manual: Process Record and Replay](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Process-Record-and-Replay.html)
- [GDB manual: Debugging Optimized Code](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Optimized-Code.html)
- [GDB wiki: Permissions on Darwin, code signing a debugger](https://sourceware.org/gdb/wiki/PermissionsDarwin)
- [gdb-patches: Initial AArch64-Darwin support series, October 2025](https://sourceware.org/pipermail/gdb-patches/2025-October/222196.html)
- [Homebrew gdb formula pull request adding arm64 bottles](https://github.com/Homebrew/homebrew-core/pull/209753)
- [ptrace(2), Linux manual page](https://man7.org/linux/man-pages/man2/ptrace.2.html)
- [lldb debugserver MachTask.mm: task_for_pid and task_set_exception_ports](https://github.com/llvm/llvm-project/blob/main/lldb/tools/debugserver/source/MacOSX/MachTask.mm)
- [Intel SDM Volume 2: INT n, INTO, INT3, INT1](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Intel SDM Volume 3: debug registers DR0 to DR7 (Debug, Branch Profile, TSC, and Resource Monitoring Features)](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm A64 ISA: BRK, Breakpoint instruction](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/BRK--Breakpoint-instruction-)
- [Arm A64 ISA: TBZ, Test bit and branch if zero](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/TBZ--Test-bit-and-branch-if-zero-)
- [Arm Architecture Reference Manual, D2.9 Watchpoint exceptions](https://developer.arm.com/documentation/ddi0487/latest/)
- [Apple ld(1): -adhoc_codesign](https://keith.github.io/xcode-man-pages/ld.1.html)
- [Apple ld64 LinkEdit.hpp: CS_LINKER_SIGNED](https://github.com/apple-oss-distributions/ld64/blob/59a99ab60399c5e6c49e6945a9e1049c42b71135/src/ld/LinkEdit.hpp)
- [Apple: Debugging Tool Entitlement, com.apple.security.cs.debugger](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.debugger)
- [Apple: Hardened Runtime](https://developer.apple.com/documentation/security/hardened-runtime)
- [Apple: Resolving common notarization issues, the get-task-allow entitlement](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [Apple: Disabling and Enabling System Integrity Protection](https://developer.apple.com/documentation/security/disabling-and-enabling-system-integrity-protection)
- [Apple Developer Forums: debugserver may not get the taskport of a hardened process](https://developer.apple.com/forums/thread/127544)
- [Apple DevToolsSecurity(8)](https://keith.github.io/xcode-man-pages/DevToolsSecurity.8.html)
- [Apple core(5)](https://keith.github.io/xcode-man-pages/core.5.html)
- [Apple Technical Note TN2124: Mac OS X Debugging Magic, core dumps](https://developer.apple.com/library/archive/technotes/tn2124/_index.html)
- [lldb `process save-core` help text](https://github.com/llvm/llvm-project/pull/143126)
- [core(5), Linux manual page](https://man7.org/linux/man-pages/man5/core.5.html)
- [coredumpctl(1)](https://man7.org/linux/man-pages/man1/coredumpctl.1.html)
- [Apple dtrace(1)](https://keith.github.io/xcode-man-pages/dtrace.1.html)
- [Apple dtruss(1m)](https://keith.github.io/xcode-man-pages/dtruss.1m.html)
- [strace releases](https://github.com/strace/strace/releases)
- [ltrace](https://ltrace.org/)
- [bpftrace CHANGELOG](https://github.com/bpftrace/bpftrace/blob/master/CHANGELOG.md)
- [rr project](https://rr-project.org/)
- [rr 5.9.0 release](https://github.com/rr-debugger/rr/releases/tag/5.9.0)
- [rr wiki: Building And Installing, hardware and VM requirements](https://github.com/rr-debugger/rr/wiki/Building-And-Installing)
- [Clang UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ISO/IEC 9899:2024 working draft N3220, clause 3.5 undefined behavior](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3220.pdf)
- [DWARF Debugging Information Format Version 5](https://dwarfstd.org/doc/DWARF5.pdf)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64)](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI)
