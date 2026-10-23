# 逆向工程全景：历史、工具家族与反编译器的构建方式

> 五个工具读取同一个目标文件，给出四个地址、三种基本块计数，以及对同一个 switch 的两种答案；本课构建一个对照框架，说明哪个数字对应哪种情况。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~150 minutes

## 学习目标

- 在 1973 到 2026 年的编年时间线上定位该领域的主要工具，并说出每个工具所属的家族。
- 说出反编译器的八个阶段，以及 Ghidra、Binary Ninja、Hex-Rays、radare2、rizin 和 angr 对每个阶段使用的术语。
- 把一条指令提升为 ESIL、RzIL、P-code 和 VEX，并解释其中哪些立即写标志位，哪些只保存一个 thunk。
- 把五个工具的地址归约到同一个 `.text` 偏移，并在一个函数上证明这一归约。
- 构建一个对照框架，读取捕获的工具输出，报告各工具在指令起始点、基本块、边和函数起始点上的一致情况。
- 根据捕获的证据，指出每个工具在何处凭空发明代码、漏掉代码，或在间接跳转处停下。

## 问题

你让一个智能体总结三个逆向工程工具对 `ledger-x86_64-O1.o` 中 `grade` 的说法。智能体运行每个工具，读取三份清单，然后报告：函数在一个工具中位于 `0x08000050`，在另一个工具中位于 `0x400010`，在第三个工具中位于 `0x100010`。它得出结论说，这些工具对函数位置的意见不一致。

这些工具其实是一致的。该目标文件是可重定位的，没有加载地址，所以每个工具各自选了一个。减去每个工具给 `.text` 的地址，所有答案都是 `0x10`。智能体比较的是原始地址，因为没有任何东西告诉它这个地址在每个工具里意味着什么。

后来，同一个智能体读取 `-O0` 构建中的 `grade`，从四个工具分别得到 3、8、8、9 个基本块。四个数字都是真实的。一个工具在 `jmp rax` 处停下。两个工具通过重定位读出了跳转表。一个工具为了放置标签，在跳转处把基本块拆开了。没有背后原因的基本块计数，只是一个没有论点的数字。

本课就是你在信任这些数字之前需要的地图。第 32 到 36 课分别构建了一个解码器、一个提升器、一份证据记录和一次重建。本轨道的后半部分会把这些成果与人们实际使用的工具放在一起。要做到这一点，你需要该领域的历史、工具所属的家族、每个反编译器都要经过的流水线，以及一个能在你自己编译的目标文件上比较各工具说法的对照框架。

```figure
re-37-field-map
```

## 五十三年，以日期为刻度

这个领域比它的大多数工具都要古老。下面每个日期在「延伸阅读」中都有原始出处。

### 1973 到 1991：先是调试器，然后是反汇编器

最早的反编译学位论文出自 1973 年。斯坦福的 Hollander 和普渡的 Housel 撰写了它们，今天两者都因被引用而为人所知。那个时期的实用工具是调试器。Maranzano 和 Bourne 于 1977 年在贝尔实验室编写了 `adb` 教程，该调试器随 1979 年的 UNIX V7 一同发布。Tim Paterson 的 `DEBUG` 带一个 8086 反汇编器，于 1981 年 8 月随 PC-DOS 1.00 发布。Richard Stallman 的 GDB 于 1986 年进入发行版。SoftICE 于 1987 年在 DOS 下提供了内核调试器。

批处理反汇编器随后出现。V Communications 于 1988 年推出的 Sourcer 对 DOS 程序执行多轮扫描，并写出带注释的清单。GNU binutils 始于 1988 年的无编号快照，`objdump` 带有 1990 年的版权。Ilfak Guilfanov 于 1991 年 1 月写下 IDA 的第一行代码，并于 1991 年 5 月 20 日以共享软件形式发布 IDA 0.1。

### 1991 到 2009：交互式反汇编器与最早的反编译器

Cristina Cifuentes 于 1994 年 7 月在 QUT 完成「Reverse Compilation Techniques」。她的反编译器 `dcc` 能把 i80286 DOS 程序转换为 C，她与 Gough 在 1995 年的论文确立了此后每个反编译器都保留的各个阶段。REC 于 1997 年跟进，Boomerang 于 2002 年出现，后者是第一个开源反编译器，采用 BSD 许可证。

IDA 于 1996 年在 DataRescue 成为商业产品，1999 年获得插件 SDK，随后转到 2005 年成立的 Hex-Rays 旗下。Hex-Rays Decompiler 1.0 于 2007 年 9 月 17 日发布。同年，Mike Van Emmerik 提交了「Static Single Assignment for Decompilation」，这篇论文把 SSA（静态单赋值）置于反编译器数据流的中心。Halvar Flake 的「Structural Comparison of Executable Objects」（2004）开创了差分家族，BinDiff 于 2004 年 8 月开始销售。

radare 始于 2006 年 2 月，最初是一个支持 64 位偏移的取证十六进制编辑器。radare2 是它在 LGPL-3.0 下的模块化重写，于 2009 年 3 月 7 日首次发布。

### 2009 到 2019：框架与中间表示

LLDB 于 2010 年 6 月 8 日作为 LLVM 子项目公布。BAP（Binary Analysis Platform）及其 BIL 中间语言在 CAV 2011 上亮相。Google 于 2011 年 3 月 1 日收购了 BinDiff 的开发者 zynamics。Hopper 于 2011 年底前后上架 Mac App Store。Matt Godbolt 于 2012 年启动 Compiler Explorer，让 C++ 程序员能看到编译器生成的汇编。

angr 于 2013 年 8 月写下第一行代码。Phoenix 论文（Schwartz、Lee、Woo、Brumley，USENIX Security 2013）展示了保持语义的结构化方法，在 8,676 个函数中只留下 40 条 `goto` 语句。DREAM（Yakdan 等，NDSS 2015）用与模式无关的结构化方法去除了这些 `goto` 语句。Vector 35 于 2015 年 1 月 14 日成立，Binary Ninja 于 2016 年 7 月 23 日开始销售，带有 Low Level IL 和 Python API。angr 于 2015 年公开发布，并在 2016 年 8 月的 DARPA Cyber Grand Challenge 中获得第三名。BinDiff 于 2016 年 3 月改为免费。Cutter 1.0 于 2017 年 12 月为 radare2 提供了 Qt 图形界面。2018 年，Guilfanov 在 Recon 上发表了「Decompiler internals: microcode」，这是对 Hex-Rays IR 的首次公开描述。

### 2019 到 2026：开源反编译器与研究转向

NSA 于 2019 年 3 月 5 日在 RSA 大会上以 Apache-2.0 发布 Ghidra 9.0，一个月后在 GitHub 上公开完整源码。angr 8.19.2.4 于同年加入了初步的反编译器。Binary Ninja 2.0 于 2020 年 5 月 20 日发布其 High Level IL 反编译器。Rizin 于 2020 年 12 月 8 日宣布成为 radare2 的分支，Cutter 团队随后迁移过去。IDA Freeware 7.6 于 2021 年 5 月加入了云端 x64 反编译器。

BinDiff 8 于 2023 年 9 月 25 日以 Apache-2.0 开源。SAILR（Basque 等，USENIX Security 2024）成为 angr 的默认结构化器，并主张只应删除虚假的 `goto` 语句，因为 Linux 6.1 的源码中有 3,754 条。Binary Ninja 4.0 于 2024 年 2 月加入 Free 版本，IDA 9.0 加入了用于无头运行的 `idalib`。学习型反编译器的论文从 SLaDe（2023）经 LLM4Decompile（2024）到 DecompileBench（2025），再到 2026 年的两项研究：一项关于智能体辅助的可读性，另一项关于「重编译更多、保留更少」的反编译器。

本课运行的版本是 2026 年 10 月 14 日当时的最新版本：Ghidra 12.1.4（2026 年 9 月 21 日）、radare2 6.2.4（2026 年 10 月 5 日）、rizin 0.9.1（2026 年 6 月 29 日）和 angr 10.0.3（2026 年 10 月 10 日）。Binary Ninja 6.0「Krypton」（2026 年 9 月 3 日）和 IDA 9.4（2026 年 7 月 13 日）是当前的闭源工具。本课没有运行它们。

## 工具家族

工具按产出物划分家族。这些家族在同一个程序中会重叠，而重叠之处正是大多数混淆的起点。

### 反汇编器

线性扫描（linear sweep）从代码段的第一个字节解码到最后一个字节。GNU `objdump -d` 和 `llvm-objdump -d` 就是这样工作的，两者都不构建控制流图。GNU `objdump` 把符号当作指令边界，并在符号之间扫描。binutils 2.47（2026 年 7 月 26 日）采用 GPL-3.0-or-later。LLVM 23.1.3 采用 Apache-2.0 及 LLVM 例外条款。Apple 的 Command Line Tools 以「Apple LLVM version 17.0.0」提供 `llvm-objdump`，本课捕获的就是这一份。

交互式反汇编器从已知入口执行递归下降，为函数起始点加入启发式规则，并维护一个数据库。IDA Pro（处理器模块、自动分析队列、FLIRT 签名）、Ghidra（`Disassembler` 加 Auto Analysis 分析器）、Binary Ninja（先递归下降，再对未知区域做线性扫描）、Hopper、radare2 与 rizin（`aa`、`aaa`、`af`，图形界面为 Cutter 和 iaito）都属于这个家族。Andriesse 等（USENIX Security 2016）在编译器生成的二进制上测量了其中九个。指令准确率接近 100%，而真正困难的基本任务是函数起始点，假阴性率达到 20% 或更高。

### 反编译器

商业反编译器：Hex-Rays Decompiler（从 x86 到 RISC-V 的十二个目标家族）、Binary Ninja 的 HLIL 及其 Pseudo C 渲染，以及 Hopper 的伪代码视图。开源反编译器：Ghidra 反编译器，一个 Apache-2.0 下的 C++ 引擎，r2ghidra 和 rz-ghidra 将其作为插件复用，pypcode 将其作为提升器复用；angr 反编译器基于 AIL，在 BSD-2-Clause 下使用 SAILR 结构化；JavaScript 实现的 r2dec 和 jsdec；以及作为历史的 Boomerang 和 `dcc`。radare2 还内置了 `pdc`，即「以类 C 语法输出伪反汇编」。rizin 没有内置反编译器。`pdc` 命令在 rizin 0.9.1 中并不存在，`pdg` 需要 rz-ghidra 插件。

### 调试器

GDB（1986）及其 GEF 和 pwndbg 前端、LLDB（2010）、带 Time Travel Debugging 的 WinDbg（2017）、OllyDbg（2014 年停止开发）、x64dbg（GPL-3.0，2014）、带 Python 连接器的 Ghidra Debugger、Binary Ninja 调试器，以及 radare2 和 rizin 的内置调试器。第 42 课会在你自己的构建上使用其中两个。

### 二进制分析框架

angr（VEX IR、符号执行、`CFGFast` 与 `CFGEmulated`、反编译器）、BAP（BIL，OCaml）、Dyninst、Jakstab 和 ByteWeight 是 Andriesse 点名的框架。Ghidra 可通过 PyGhidra 作为库使用，Binary Ninja 通过 `import binaryninja`，IDA 通过 `idalib`。

### 格式与引擎库

LIEF 1.0.0（2026 年 7 月 12 日，Apache-2.0）可解析 ELF、PE、Mach-O、COFF 以及 Android 格式。CLE（「CLE Loads Everything」）是 angr 的加载器。Capstone 5.0.9（2026 年 5 月 28 日，BSD-3-Clause）负责解码；Keystone 0.9.2（GPL-2.0）负责汇编；Unicorn 2.1.4（GPL-2.0）负责模拟执行。radare2 6.2.4 捆绑 Capstone 5，rizin 0.9.1 内置 Capstone、Zydis 和 tree-sitter。

### 差分工具与编译器侧查看器

BinDiff 8（Apache-2.0）及其 BinExport 12 插件，支持 IDA Pro、Binary Ninja 和 Ghidra；Diaphora 3.4.3（2026 年 10 月 11 日，AGPL-3.0）作为 IDA 插件；以及 `radiff2` 和 `rz-diff`。Compiler Explorer（BSD-2-Clause）展示编译器为一段源码生成的汇编，这是本列表中其他一切的正向方向。第 44 课会使用差分家族。

### 版本、许可证与本课实际运行的工具

| 工具 | 2026-10-14 读取的版本 | 许可证 | 无头入口 | 本课是否运行 |
|---|---|---|---|---|
| Ghidra | 12.1.4 | Apache-2.0 | `analyzeHeadless`、`pyghidra.start()` | 是，四个目标文件全部运行 |
| radare2 | 6.2.4 | LGPL-3.0 | `r2 -qc '<cmds>' file`、`rabin2` | 是 |
| rizin | 0.9.1 | LGPL-3.0 | `rizin -qc '<cmds>' file`、`rz-bin` | 是 |
| angr | 10.0.3 | BSD-2-Clause | Python 中的 `angr.Project` | 是 |
| llvm-objdump | Apple LLVM 17.0.0 | Apache-2.0 及 LLVM 例外条款 | `llvm-objdump -d -r` | 是 |
| Binary Ninja | 6.0「Krypton」 | 商业；非商业用途有 Free 版本 | `binaryninja.load()`，需 Commercial、Ultimate 或无头许可证 | 否，仅依据文档 |
| IDA Pro | 9.4 | 商业订阅；非商业 x86 与 x86-64 可用 IDA Free | `idat`、`ida -A -S`、`idalib` | 否，仅依据文档 |
| Hopper | 6.6.0 | 商业，永久许可 | 无文档记录 | 否，仅依据文档 |
| Cutter | 2.5.0 | GPL-3.0 | rizin 之上的图形界面 | 否 |

Binary Ninja 的 Free 版本只能反编译 x86、x86-64、ARMv7 和 AArch64，且没有 API。IDA Free 没有 IDAPython、没有 SDK、也没有 `idalib`，其反编译器在云端运行。本课中关于这两个工具的每一条陈述都来自它们的文档和发布说明，图里也将它们标记为未运行。

## 反编译器流水线，逐阶段拆解

Cifuentes 在 1994 年命名了这些阶段：语法分析、语义分析、中间代码生成、控制流图生成、数据流分析、控制流分析和代码生成。她将它们划分为与机器相关的前端、与机器和语言无关的中端，以及与语言相关的后端。此后的每个反编译器都保留这种形态，只是重命名了各部分。下表给出本课使用的八个阶段，以及每个工具对这些阶段的称呼。图展示了同一张表，工具列可以切换选择。

| 阶段 | Ghidra | Binary Ninja | Hex-Rays / IDA | radare2 | rizin | angr |
|---|---|---|---|---|---|---|
| 加载 | `ElfLoader`、`MachoLoader`、`PeLoader`；`.ldefs` 选择语言 | `BinaryView`、`BinaryViewType` | 加载器模块（`LDSC`） | RBin（`rabin2`） | RzBin（`rz-bin`） | CLE（`cle.Loader`、`main_object`） |
| 反汇编 | `Disassembler`；SLEIGH `.slaspec` 编译为 `.sla`；分析器「Disassemble Entry Points」「Function Start Search」 | 架构插件；Recursive Descent，然后是 linear sweep 阶段 | 处理器模块（`ana.cpp`、`emu.cpp`、`out.cpp`）；自动分析队列；FLIRT | RArch 插件到 `RAnalOp`；`aa`、`aaa`、`af` | RzAnalysis 插件；`aa`、`aaa`、`af` | Capstone 基本块；`CFGFast` 函数起始点搜索 |
| CFG 恢复 | 「Generate Basic Blocks and the CFG」；`BlockGraph`；「Decompiler Switch Analysis」 | `Function`、`BasicBlock`；「Control Flow Graph (CFG) Analysis」阶段 | `mbl_graph_t`，在 `MMAT_LOCOPT` 就绪 | `afb`、`agf`、`aab` | `afb`、`agf`、`aab`（Nucleus） | `CFGFast`、`CFGEmulated`；`kb.functions`；IRSB |
| 提升 | 「Generate Raw P-code」；Varnode 位于 `ram`、`register`、`const`、`unique` | Lifted IL、LLIL、Mapped MLIL、MLIL | 微码（`minsn_t`、`mop_t`）；`MMAT_GENERATED` | ESIL（`asm.esil`、`ae`） | RzIL（`aoi`、`aez`） | VEX（`pyvex.lift`、IRSB），然后是 AIL |
| SSA | `Heritage`、`ActionHeritage`；`MULTIEQUAL`、`INDIRECT` | LLIL SSA、MLIL SSA、HLIL SSA；`HLIL_VAR_PHI` | 没有具名 SSA；在 `MMAT_GLBOPT1` 到 `3` 中做 use-def 数据流 | 无；ESIL 是为模拟执行而构建的 | 无；RzIL 是为模拟执行而构建的 | `_stage_transform_to_ssa_level0` 与 `level1`；`SeqNodeDephication` |
| 数据流与类型 | `ActionDeadCode`、`ActionInferTypes`、`Rule` 池；原型来自 `.cspec` | MLIL 常量传播、变量与类型；`analyzeReturns` | 「Transform microcode」；`MMAT_CALLS`、`MMAT_GLBOPT`、`MMAT_LVARS`；TIL | `afv`、`afc`、`aaft`；ESIL 模拟执行 | `afv`、`afc`、`aft` | `CallingConventionAnalysis`、`VariableRecoveryFast`、`Typehoon` |
| 结构化 | `CollapseStructure`；`BlockIf`、`BlockWhileDo`、`BlockSwitch`、`BlockGoto` | `HLIL_IF`、`HLIL_WHILE`、`HLIL_SWITCH`；AST 位于 `root` | 「Generate ctree」；结构分析；goto 区域 | `pdc`；`pdd`（r2dec）；`pdg`（r2ghidra） | `pdg`（rz-ghidra）；jsdec | `RegionIdentifier`；`RecursiveStructurer`，配合 `SAILRStructurer` |
| 输出 | 「Emit Final C Tokens」；`PrintC`、`ActionNameVars`、`ActionSetCasts` | HLIL 渲染；Pseudo C、Pseudo Rust、Pseudo Python | 「Beautify ctree; Print ctree」；`cfunc_t` | `pdc`、`pdd`、`pdg` 的文本 | `pdg` 文本 | `CStructuredCodeGenerator`；`codegen.text` |

### 加载

加载器解析容器、映射段并读取符号和重定位。在可重定位目标文件中，还没有任何东西拥有地址，因此加载器会选定一个。radare2 和 rizin 把文件映射到 `0x08000000`，于是 `.text` 位于 `0x08000040`，因为该段从文件偏移 `0x40` 开始。CLE 把 `.text` 放在 `0x400000`。Ghidra 的 `ElfLoader` 把它放在 `0x100000`。`llvm-objdump` 只打印段偏移，完全不给基址。

各加载器处理重定位的方式也不同。Ghidra 在导入时处理重定位，CLE 会应用它们。radare2 打开目标文件时会打印提示：重定位未应用，设置 `bin.relocs.apply=true` 可以改变这一点。加载器如何处理重定位，决定了后续每个阶段能看到什么；你会在下面的 `-O0` 跳转表中遇到这一影响。

### 反汇编

第 33 课构建了这两种方法。表中除 `llvm-objdump` 外的每个工具都先执行递归下降，再加入函数起始点搜索。Binary Ninja 的文档按顺序列出了这些阶段：Recursive Descent、Call Target Analysis、作为「未知区域的线性反汇编」的 CFG Analysis，以及 Tail Call Analysis。Ghidra 按优先级运行分析器，其中「Disassemble Entry Points」和「Function Start Search」较早执行。IDA 维护自动分析队列，每个队列有优先级，并为库代码保留 FLIRT 签名。angr 的 `CFGFast` 记录了四种函数起始点来源：符号、序言扫描、调用目标，以及对剩余字节的完整扫描。最后一种来源正是 angr 在填充区凭空发明函数的地方。

### CFG 恢复

每个工具都像第 33 课那样切分基本块，但有一处差异值得你测量。angr 在 `call` 处结束基本块，因为它的 IRSB 是单入口、多出口的基本块，而调用是一个出口。Ghidra、radare2 和 rizin 会穿过调用继续。对于 `-O0` 下的 `grade_label`（它调用 `grade`），angr 报告 2 个基本块，另外三个工具报告 1 个。指令是相同的。

### 提升

Binary Ninja 把提升定义为「从较低层语言翻译到被认为更高层的中间语言」。这些 IR 的用途各不相同。P-code 和 VEX 是为分析与反编译而构建的。ESIL 和 RzIL 是为模拟执行而构建的，两者都没有 SSA 阶段。Hex-Rays 微码最初是带 72 个操作码的类 RISC 代码，并会经历从 `MMAT_GENERATED` 到 `MMAT_LVARS` 的成熟度级别。Binary Ninja 先提升到 Lifted IL，然后在 LLIL 中把标志位折叠进条件，而 MLIL 会「把寄存器和内存访问翻译成变量」。下一节会把其中四种 IR 放在同一条指令上并排比较。

### SSA

Van Emmerik 的论文列出了 SSA 为反编译器带来的好处：表达式传播、在不对 ABI 做假设的情况下识别参数与返回值、保持性分析、把类型分析作为稀疏数据流问题，以及间接跳转与调用的分析。Ghidra 把它的 SSA 构造称为 `Heritage`，并加入两个只存在于图中的操作：用于 phi 的 `MULTIEQUAL` 和用于副作用的 `INDIRECT`。Binary Ninja 在每个 IL 层级都暴露 SSA 形式。angr 在 `Clinic` 内部有两个 SSA 阶段。Hex-Rays 没有命名 SSA 形式，而是在全局优化过程中做 use-def 数据流分析。ESIL 和 RzIL 则没有。

### 数据流与类型

Ghidra 的主简化循环依次运行「Eliminate Dead Code」「Propagate Local Types」和「Perform Term Rewriting」，直到不再发生变化；重写由一组 `Rule` 对象完成。Hex-Rays 变换微码，在 `MMAT_CALLS` 处检测调用参数，并在 `MMAT_LVARS` 处分配局部变量。angr 运行 `CallingConventionAnalysis`、`VariableRecoveryFast` 和 `Typehoon`，后者是 Noonan、Loginov 和 Cok 多态类型推断的实现。第 41 课讲的就是这个阶段。

### 结构化

Ghidra 的 `CollapseStructure` 搜索与某种结构匹配的子图，把它们折叠成一个节点；当它卡住时，会删除一条边并将其标记为非结构化。Hex-Rays 把 CFG 归约为 `if`、`while`、`do` 和 `switch`，其余部分留作 goto 区域。angr 识别单入口、单出口区域，默认用 `SAILRStructurer` 做结构化，`PhoenixStructurer` 和 `DreamStructurer` 作为备选。第 34 课用手写方式实现了一个结构化器。

### 输出

Ghidra 在 `ActionNameVars` 和 `ActionSetCasts` 之后，通过 `PrintC` 和 Oppen 美化打印器输出 C token。Hex-Rays 美化并打印 ctree。Binary Ninja 把 HLIL 渲染为 Pseudo C、Pseudo Rust 或 Pseudo Python。angr 返回 `codegen.text`。radare2 打印 `pdc`、`pdd` 或 `pdg`。

## 一条指令，四种 IR

取 `48 83 ec 10`，即 `subq $0x10, %rsp`。用第 33 课的规则在纸上解码它。

1. `48` 是 REX 前缀，`W = 1`。操作数是 64 位。
2. `83` 是带 8 位立即数的第一组算术操作码。ModRM 的 reg 字段选择具体操作。
3. `ec` 是 ModRM `11 101 100`。`mod = 11` 表示寄存器形式。`reg = 101` 即 `/5`，对应 `sub`。`r/m = 100` 是 `%rsp`。
4. `10` 是立即数 16。

这四个字节被交给四个提升器。fixture 中保存了确切的命令和完整文本。

radare2 6.2.4 给出的 ESIL 是一个以逗号分隔的字符串，共 30 个 token：

```text
16,0x8000000000000000,-,!,16,rsp,-=,63,$o,^,of,:=,63,$s,sf,:=,$z,zf,:=,$p,pf,:=,64,$b,cf,:=,4,$b,af,:=
```

它是一个栈机。`16,rsp,-=` 原地相减。随后每个标志位都用内部运算符写入：`$o` 溢出、`$s` 符号、`$z` 零、`$p` 奇偶、`$b` 借位。

rizin 0.9.1 给出的 RzIL 是一个带 14 个 `set` 效应的 S 表达式：

```text
(seq (set op1 (var rsp)) (set op2 (bv 64 0x10)) (set sub (- (var op1) (var op2))) (set rsp (var sub))
     (set _result (var sub)) (set pf (! (lsb (let _val (cast 8 false (var _result)) ...))))
     (set zf (is_zero (var _result))) (set sf (msb (var _result))) ... (set cf (|| ...)) (set of (|| ...)) (set af (|| ...)))
```

每个标志位都是位向量上的布尔表达式。仅奇偶标志就是一个四层 `let`。

通过 pypcode 4.0.1、来自 SLEIGH 规范 `x86:LE:64:default` 的 P-code 在 `IMARK` 之后是九个操作：

```text
CF = RSP < 0x10
OF = sborrow(RSP, 0x10)
RSP = RSP - 0x10
SF = RSP s< 0x0
ZF = RSP == 0x0
unique[58300:8] = RSP & 0xff
unique[58400:1] = popcount(unique[58300:8])
unique[58500:1] = unique[58400:1] & 0x1
PF = unique[58500:1] == 0x0
```

寄存器传输，临时变量位于 `unique` 空间。标志位在指令处立即计算，`CF` 和 `OF` 在减法覆写 `RSP` 之前算出。

pyvex 10.0.3 给出的 VEX 在 `IMark` 之后是七条语句：

```text
t2 = GET:I64(offset=48)
t0 = Sub64(t2,0x0000000000000010)
PUT(offset=144) = 0x0000000000000008
PUT(offset=152) = t2
PUT(offset=160) = 0x0000000000000010
PUT(offset=168) = 0x0000000000000000
PUT(offset=48) = t0
```

没有出现任何标志位。VEX 用寄存器在客户机状态（guest state）中的偏移来命名寄存器，偏移 48 是 `rsp`。偏移 144 到 168 分别是 `cc_op`、`cc_dep1`、`cc_dep2` 和 `cc_ndep`。提升器存入操作码 8（即 64 位减法）和两个操作数。后续读取标志位的分支会从这个 thunk 计算它。这就是为什么在框架报告中 `pcode.writesFlags` 为 true、`vex.computesFlags` 为 false，也是 VEX 成为四者中最短的原因。

同样四个提升器处理 `48 63 c7`，即 `movslq %edi, %rax`，`-O1` 下 `grade` 的第一条指令，得到 5 个 ESIL token（`32,edi,~,rax,=`）、1 个 RzIL `set`、1 个 P-code 操作（`RAX = sext(EDI)`）和 4 条 VEX 语句。没有提升器写入标志位，因为该指令没有标志位。Ghidra 通过无头脚本捕获的自身清单把同一操作打印为 `(register, 0x0, 8) INT_SEXT (register, 0x38, 4)`：寄存器偏移 0 是 `RAX`，偏移 `0x38` 是 `RDI`，大小分别是 8 和 4 字节。

`83 f9 04`，即 `cmpl $0x4, %ecx`，来自 `grade` 中的边界检查，得到 30 个 ESIL token、13 个 RzIL `set` 效应、10 个 P-code 操作，以及 7 条带 `cc_op` 为 7（32 位减法）的 VEX 语句。没有任何东西写入 `%ecx`。你会在 VEX 中遇到的两种减法宽度是：32 位用 7，64 位用 8，你构建的翻译器会携带这张表。

## 同一个函数的四种地址基址

框架在比较任何东西之前，必须先就函数的位置与自身达成一致。先为 `ledger-x86_64-O1.o` 中的 `grade` 手动做一次算术。

| 工具 | `.text` 的地址 | `grade` 的地址 | 偏移 |
|---|---|---|---|
| `llvm-objdump` | `0x0` | `0x10` | `0x10 - 0x0 = 0x10` |
| radare2、rizin | `0x08000040` | `0x08000050` | `0x08000050 - 0x08000040 = 0x10` |
| angr | `0x400000` | `0x400010` | `0x400010 - 0x400000 = 0x10` |
| Ghidra | `0x100000` | `0x100010` | `0x100010 - 0x100000 = 0x10` |

符号表显示 `grade` 位于 `.text + 0x10`，大小为 46。一旦去掉基址，所有工具都一致。框架从每份捕获中读取基址：radare2 和 rizin 用 `iSj`，angr 捕获中的 `.text` 段条目，Ghidra 捕获中的 `.text` 内存块，线性扫描则为零。测试 `test_every_tool_places_grade_at_offset_0x10_from_its_own_base` 固定了这一事实。

## 工具在哪里一致，在哪里分歧

`code/main.py` 中的框架读取捕获、归约每个地址并进行比较。下面就是它报告的数字，测试会逐一断言。

### 一致的情形：-O1 下的 ledger

四个函数，线性扫描共 56 条指令，符号内 54 条，填充区 2 条。每个工具在每个函数中报告的指令起始点都相同：9、14、13、18。四个图构建器报告 3、3、5、4 个基本块和 2、3、5、4 条边，而且 radare2、rizin、angr 和 Ghidra 的边集合两两完全相同。`total` 的五个基本块和五条边正是第 33 课手绘的结果。

唯一的差异在填充区。函数之间的 3 个字节（`grade` 之后的 `66 90`，`total` 之后的 `90`）被线性扫描解码，因为它没有理由停下；也被 angr 解码，它会创建没有符号的 `sub_40003e` 和 `sub_40005f`。radare2、rizin 和 Ghidra 都不认领它们。

### 向量化的情形：-O2 下的 ledger

`total` 在四个图构建器的每一个中都是 41 条指令、10 个基本块、13 条边。第 33 课中的 SSE2 循环和标量剩余循环在各处都以相同方式切分。angr 又会在填充区增加两个函数。

### 跳转表：-O0 下的 grade

这里工具出现了分歧，而分歧本身就是本课的内容。

| 工具 | 基本块 | 边 | 捕获记录了什么 |
|---|---|---|---|
| radare2 | 3 | 2 | `0x62` 处的基本块没有后继；`0x76` 到 `0x9a` 的字节从未到达；缺少 8 个指令起始点 |
| rizin | 8 | 11 | `0x62` 处的基本块带有一个 `switch_op`，包含 5 个 case 和 4 个不同目标 |
| angr | 8 | 7 | case 基本块存在，而 `0x62` 处的转移指向一个无法解析的目标节点 |
| Ghidra | 9 | 12 | `0x74` 处的 `switchD` 标签拆分了基本块；四条 `COMPUTED_JUMP` 流向 `0x76`、`0x7f`、`0x88`、`0x91` |

四者到达时都解码了同样的 33 条指令。radare2 在 `jmp rax` 处停下，留下 36 个字节无人认领，框架把该范围标记为漏掉的代码。rizin 和 Ghidra 找到了四个不同目标，正是第 33 课从 `.rela.rodata` 的五个重定位中读出的。angr 列出了 `grade` 内的 case 基本块，却仍然没有从跳转基本块指向它们的边；它的捕获记录 `0x62` 处的转移通向一个无法解析的目标节点，其反编译器输出了一个计算得到的 `goto`。对于同样的控制流，Ghidra 比 rizin 多一个基本块、多一条边，因为它的 switch 分析在 `jmp rax` 上放置了标签，而该基本块模型会在标签处开始一个新基本块。

第 33 课的图有 8 个基本块和 11 条边。rizin 与之完全吻合。要点在于看到：9 与 11、7 与 2 并不是对「找到了多少代码」的四种测量。它们是四个答案，回答的是工具跟随一条间接跳转走了多远，以及它选择在哪里切分。

### 代码中的数据：sweep-trap

`lookup` 是 24 字节，`0x10` 处有 4 个表字节 `31 c0 90 48`。线性扫描在该范围内报告 8 个指令起始点。框架列出它与多数派的差异：在 `0x10`、`0x12`、`0x13` 处多出起始点，在 `0x14` 处漏掉一个起始点。那就是第 33 课的 `addq $0x1, %rax` 提前了一个字节。radare2、rizin 和 Ghidra 报告 6 个起始点、2 个基本块、1 条边，并把 `0x10` 到 `0x14` 留作无人认领。angr 为 `lookup` 报告同样的 2 个基本块和 1 条边，然后又在表字节上凭空发明了 `sub_400010`：一个结束于 `0x14` 的 4 字节基本块中的 2 条指令，而 `lookup` 的第二个基本块正是从那里开始，因此 `0x13` 处的 REX 字节不属于任何指令。框架对该目标文件的最后一条发现是：没有任何图构建器解码 `0x10` 到 `0x14`，所以这些字节位于函数内部，是数据。

### 各工具凭空发明与漏掉的代码

跨越四个目标文件，框架指出 angr 凭空发明了 9 个没有符号的函数，每一个都位于填充区或陷阱表处。它指出 radare2 漏掉的一个范围，即 `-O0` 下 `grade` 的 36 个 case 字节。它指出 9 个线性扫描会解码、而没有任何图构建器当作代码的范围：八个填充间隙加上陷阱表。它没有在任何工具中指出漏掉的函数。

## 三个反编译器看 grade

捕获的工具中有三个会为 `-O1` 下的 `grade` 输出类 C 文本。源码是 `switch (score / 10)`，包含五个 case，编译后变成一次边界检查和一张返回值表。

Ghidra 12.1.4：

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

angr 10.0.3：

```c
extern unsigned int g_4000b4[4];

unsigned int grade(int a0)
{
    unsigned int v1;  // ecx
    unsigned int v2;  // eax

    v1 = (unsigned int)((long long)(a0 * 1717986919) >> 0x22) + (unsigned int)(a0 * 1717986919 >> 63) - 6;
    v2 = 0;
    if (v1 <= 4)
        v2 = g_4000b4[v1];
    return v2;
}
```

radare2 6.2.4，`pdc`：

```c
void sym.grade (int64_t arg1) {
    rax = edi
    rax = rax * 0x66666667
    rcx = rax
    rcx >>= 0x22
    rax >>>= 0x3f
    ecx += eax
    ecx += 0xfffffffa
    eax = 0
    v = ecx - 4
    if (((unsigned) v) <= 0) {
        eax = ecx
        rcx = rip
        eax = dword [rcx + rax*4]
    }
    return
}
```

把三者对照流水线来看。Ghidra 的项重写把乘以 `0x66666667` 和右移 34 位折叠回 `param_1 / 10`，这正是第 34 课的惯用法。angr 提升了同样的字节，却把算术保留为 `a0 * 1717986919 >> 0x22`。radare2 的 `pdc` 是 ESIL 模拟执行加伪语法：只有寄存器名，没有变量，没有类型，边界检查在无符号减法之后被打印成 `<= 0`。Ghidra 通过 `DAT_001000b4` 为表加载赋予类型；angr 声明了 `g_4000b4[4]`，只有四个元素，而源码中有五个。Ghidra 和 angr 都没有恢复出 case 的枚举，因为在 `-O1` 下字节中已经不存在 switch。在存在跳转表的 `-O0` 下，Ghidra 打印出 `switch(param_1 / 10)`，case 为 6 到 10 并带一个 default，而 angr 打印的是计算得到的 `goto`。

rizin 没有自己的反编译器，捕获记录也表明那里不存在 `pdc`。Binary Ninja 的 HLIL 和 Hex-Rays 的 ctree 是本课无法捕获的两种输出。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
```

输出是一份 JSON 报告。开头是捕获时记录的工具版本，以及你机器上存在的工具——它们可能全部缺失。然后对四个目标文件分别给出符号范围、填充间隙、每个工具使用的基址、每个工具找到或凭空发明的函数起始点，以及每个函数的指令起始点一致情况、基本块计数、边计数和每个工具缺少的边。结尾是每个工具的字节核算、工具自身的注释、一组用平实句子写成的发现，以及 IR 部分：翻译器对三条指令的输出，与在捕获的提升结果上测得的计数并列。

预期结构：

```json
{
  "captureTools": {"llvm-objdump": "Apple LLVM version 17.0.0", "radare2": "radare2 6.2.4 ...", "rizin": "rizin 0.9.1 ...", "angr": "10.0.3", "ghidra": "12.1.4"},
  "localTools": {"radare2": "absent", "...": "..."},
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "bases": {"llvm-objdump": "0x0", "radare2": "0x8000040", "angr": "0x400000", "ghidra": "0x100000", "...": "..."},
      "functions": {"grade": {"blocks": {"radare2": 3, "rizin": 8, "angr": 8, "ghidra": 9}, "edges": {"radare2": 2, "rizin": 11, "angr": 7, "ghidra": 12}, "...": "..."}},
      "findings": ["radare2 never reaches bytes 0x76-0x9a inside grade that the other graph builders decode", "..."]
    }
  },
  "ir": {"translator": [{"text": "subq $0x10, %rsp", "flags": {"pcodeLike": "eager", "vexLike": "thunk"}, "...": "..."}], "observed": {"...": "..."}}
}
```

按以下顺序阅读 `code/main.py`：

1. `read_text_symbols()` 读取 `.text` 的段头和 `STT_FUNC` 符号。符号表是框架唯一的事实基准，`TextSection.gaps()` 从中推导出填充区。
2. `parse_objdump()`、`parse_r2_family()`、`parse_angr()` 和 `parse_ghidra()` 各自把一份捕获转换为一个 `ToolView`：一个基址、以「偏移到长度」表示的指令，以及每个函数一个 `FunctionView`，包含基本块、边和任何反编译文本。每个解析器只懂一个工具的词汇，别的什么都不懂。
3. `majority()` 和 `ranges()` 是每次比较都会用到的两个辅助函数。
4. `function_report()` 先比较指令起始点与图构建器的多数派，再比较基本块计数，然后比较边集合，并列出每个工具缺少的东西。
5. `byte_report()` 把每个字节核算为符号内代码、填充、无人认领、凭空发明或漏掉，并指出符号内没有任何图构建器解码的字节。
6. `findings()` 把它们转换成句子，`compare()` 为每个目标文件组装一份报告。
7. `decode_small()`、`to_pcode_like()` 和 `to_vex_like()` 是翻译器：三种寄存器形式编码，一种立即风格，一种 thunk 风格。
8. `observed_ir()` 测量捕获的提升结果，`figure_data()` 组装时间线、阶段表以及图所展示的证据。

还有三个命令。`python3 code/main.py compare FIXTURE.o` 打印一个目标文件的报告。`python3 code/main.py ir 4883ec10` 打印翻译器对一条指令的两种形式。`python3 code/main.py figure-data` 打印嵌入图中的数据。

捕获由 `code/fixtures/capture_tools.py` 一次性生成，它会把每条命令和版本记录在 `captures.json` 中。框架和测试从不启动任何工具。如果你有这些工具，可以运行捕获脚本刷新文件，并阅读 fixture 的 README 了解所需环境。

## 使用它

按以下顺序在新目标文件上使用该框架。

1. 自己构建目标文件，并保留编译器命令。在打开任何工具之前先读它的符号表，这样你就能知道有多少个函数、填充区在哪里。
2. 用无头模式把每个工具各捕获一次。把原始输出和确切命令与目标文件放在一起。无法重新阅读的分歧不算发现。
3. 检查基址。选一个函数，确认减去各工具的 `.text` 地址后，每个工具都把它放在同一偏移。如果有一个不是，就停下来，弄清那个工具加载了什么。
4. 先比较指令起始点。如果图构建器在这里就有分歧，那么后面的比较是关于解码的，而不是关于分析的。
5. 比较基本块和边，并在评判一个计数之前先读每个工具的注释。`switch_op`、`COMPUTED_JUMP` 或无法解析的目标节点解释了大多数差异。
6. 核算字节。工具认领的填充区说明启发式规则在起作用。工具留下的符号内字节说明遍历停住了。没有任何图构建器认领的符号内字节是数据。
7. 写下你没有运行哪些工具。来自文档的描述有用，但与捕获是不同种类的证据。

## 交互实验

### 实验 A：一个函数，四种基址

从 `code` 目录打开 Python：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape/code
python3 -q
```

运行：

```python
from main import *
views = load_views("ledger-x86_64-O1.o")
{tool: hex(view.base) for tool, view in views.items()}
{tool: hex(view.functions["grade"].start + view.base) for tool, view in views.items()}
```

在运行第二行之前，先把四个原始地址写下来，并与本课的表核对。然后把 `grade` 换成 `grade_label`，手动计算它在 Ghidra 中的位置。

### 实验 B：从四个角度看跳转表

```python
report = compare("ledger-x86_64-O0.o")
report["functions"]["grade"]["blocks"]
report["functions"]["grade"]["edgesMissingPerTool"]["radare2"]
report["notes"]["rizin"]
report["notes"]["ghidra"]
report["bytes"]["radare2"]["misses"]
```

数一数 radare2 缺少的边，并把每一条与第 33 课 `-O0` 清单中的一行对应起来。然后阅读 Ghidra 的注释，并在 Ghidra 捕获中找到 `switchD` 标签。

### 实验 C：立即标志位对比 thunk

```python
translate("48 83 ec 10")["pcodeLike"]
translate("48 83 ec 10")["vexLike"]
translate("83 f9 04")["vexLike"]
{row["name"]: (row["pcode"]["ops"], row["vex"]["statements"]) for row in observed_ir()["instructions"]}
```

在两个输出中找到 `cc_op` 常量，并解释为什么一个是 8、另一个是 7。然后尝试 `translate("48 01 c8")`，读取 `TranslateError`。

### 实验 D：五个工具下的陷阱

```python
report = compare("sweep-trap.o")
report["functions"]["lookup"]["instructionStarts"]
report["bytes"]["dataInsideSymbols"]
report["functionStarts"]["angr"]
```

在纸上按线性扫描的方式解码字节 `31 c0 90 48 83 c0 01`，再按下降在 `jmp` 之后的方式解码一次。把你得到的两份清单与框架报告的多余和缺失起始点对应起来。

## 实践练习

增加第五份捕获并扩展框架。

要求：

- 选择一个本课没有捕获、且你可以在自己机器上运行的工具：GNU `objdump`、带 Capstone 的 LIEF、手动导出的 Binary Ninja Free 会话，或者使用不同分析器集合的 Ghidra。在全部四个目标文件上捕获它的输出，并把确切命令和版本写入 `captures.json`。
- 编写一个解析器，把捕获转换为 `ToolView`，包含工具的基址、指令、函数、工具具备的基本块，以及工具具备的边。
- 把该工具加入 `TOOLS`；如果它会构建图，再加入 `DESCENT_TOOLS`。运行演示，阅读多数派集合发生了什么变化。
- 编写一个测试，断言该工具在 `ledger-x86_64-O1.o` 上的基本块计数；再写一个测试，断言它如何处理陷阱表。
- 为课程写一段文字，用该工具自己的说法，说明它如何称呼加载器、反汇编器和 CFG。

成功的标准是：`python3 code/main.py` 在 `captureTools` 中报告新工具，`ledger-x86_64-O1.o` 中的每个函数都保持 `allDescentToolsAgree` 为 true，并且 `sweep-trap.o` 的发现说明了新工具对 `0x10` 到 `0x14` 字节的处理结果。

## 课程产物

本课包含 `outputs/skill-the-reverse-engineering-landscape.md`。用它在你构建的一个目标文件上运行多个工具，把它们的地址归约到同一个基址，并逐函数报告它们在指令起始点、基本块和边上的哪些地方一致，以及每个工具凭空发明或漏掉了哪些字节。它需要带命令和版本记录的捕获、从文件中读取的符号表，以及一份说明哪些工具仅根据文档描述的声明。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- ELF 读取器在 `ledger-x86_64-O1.o` 中找到四个函数符号，大小分别为 16、46、31、63，以及两个填充间隙
- 四个目标文件的线性扫描捕获分别包含 86、56、84、11 条指令
- radare2 和 rizin 在 `ledger-x86_64-O1.o` 上报告相同的指令、基本块和边，且只有 radare2 有 `pdc` 文本
- angr 把 `sub_40003e` 和 `sub_40005f` 标记为没有符号的函数
- Ghidra 对 `total` 的基本块就是第 33 课的五个基本块，且其调用流不是图的边
- 去掉基址后，每个工具都把 `grade` 放在偏移 `0x10`
- 对 `ledger-x86_64-O1.o` 中每个函数的每个指令起始点，所有工具都一致
- 四个图构建器在该文件上报告 3、3、5、4 个基本块和 2、3、5、4 条边，且边集合完全相同
- `-O2` 下的 `total` 在每个图构建器中都有 10 个基本块和 13 条边
- `-O0` 下的 `grade` 在 radare2、rizin、angr 和 Ghidra 中分别有 3、8、8、9 个基本块和 2、11、7、12 条边，原因见各工具的注释
- `-O0` 下的 `grade_label` 除 angr 外到处都是一个基本块；angr 在调用处结束基本块
- angr 在每个填充间隙发明一个函数，并在陷阱表上再加一个；其他工具既不发明也不漏掉函数
- 线性扫描在陷阱处多出起始点 `0x10`、`0x12`、`0x13`，并漏掉 `0x14`，且没有图构建器解码 `0x10` 到 `0x14`
- 发现中指出了 radare2 漏掉的范围，以及 angr 发明的每个函数
- 清单为全部 21 个捕获文件记录了版本和命令
- 三个工具反编译了 `grade`，且只有 Ghidra 折叠了除法惯用法
- 翻译器在一种风格中立即写入 CF、OF、SF、ZF、PF，在另一种风格中写入 `cc_op` thunk；对 `cmp` 不写寄存器，对 `movslq` 不写标志位
- 翻译器拒绝内存形式、其他操作码、多余的 REX 位和错误长度
- 翻译器的 `cc_op` 数字 8 和 7 与捕获的 VEX 中相同
- 捕获的提升结果对三条指令分别给出 9、1、10 个 P-code 操作和 7、4、7 条 VEX 语句
- 时间线按顺序从 1973 年到 2026 年；流水线有八个阶段和六个工具，每个单元格都有术语；两个闭源工具标记为未运行
- 交互式图嵌入与框架产出的相同数据

## 生产环境失败模式

| 失败模式 | 薄弱报告会怎么说 | 框架必须证明什么 |
|---|---|---|
| 来自不同工具的原始地址 | 「各工具对 grade 的位置意见不一」 | 去掉各自基址后是同一个偏移，并在一个函数上验证 |
| 把基本块计数当作找到多少代码的度量 | 「Ghidra 找到的代码最多」 | 指令起始点相同，多出的基本块由标签解释 |
| 把单个工具当作权威 | 「objdump 表明这些表字节是指令」 | 线性扫描就是线性扫描；三个图构建器把这些字节留作无人认领 |
| 把填充区函数算作发现 | 「angr 多找到了七个函数」 | 每一个都位于填充间隙或数据上，且都没有符号 |
| 把停住的遍历报告为死代码 | 「四个 case 基本块不可达」 | 36 个字节被一个工具漏掉、被三个工具到达 |
| 不看注释就比较边计数 | 「rizin 和 Ghidra 对 grade 意见不一」 | 同样是四个计算目标，只是多了一个在标签处的拆分 |
| 把反编译器的算术当作程序本身 | 「该函数乘以 1717986919」 | 那是第 34 课的惯用法，一个工具折叠了它，另一个保留了它 |
| 把仅描述过的闭源工具说成运行过 | 「Binary Ninja 报告五个基本块」 | 图和表将其标记为仅有文档 |
| 没有命令的捕获 | 「我们运行了 radare2」 | 每个文件都有版本、命令和日期的清单 |
| 把没有标志位的 IR 读成有损提升 | 「VEX 丢掉了标志位」 | `cc_op`、`cc_dep1`、`cc_dep2` 中的 thunk，在分支处求值 |

## 操作规则

每个工具都用其命令和版本捕获一次，把每个地址归约到同一个基址，并且先比较指令起始点、再比较基本块、最后比较边。把每个字节核算为已认领、填充、数据或漏掉。在评判一个计数之前先读工具自己的注释，并说明哪些工具你只是读过资料。

## 延伸阅读

- [Cifuentes, "Reverse Compilation Techniques", PhD thesis, QUT (1994)](https://eprints.qut.edu.au/36820/)
- [Cifuentes and Gough, "Decompilation of Binary Programs", Software: Practice and Experience (1995)](https://doi.org/10.1002/spe.4380250706)
- [Van Emmerik, "Static Single Assignment for Decompilation", PhD thesis, University of Queensland (2007)](https://espace.library.uq.edu.au/view/UQ:158682)
- [Schwartz, Lee, Woo, Brumley, "Native x86 Decompilation Using Semantics-Preserving Structural Analysis and Iterative Control-Flow Structuring" (USENIX Security 2013)](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Yakdan et al., "No More Gotos" (NDSS 2015)](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Basque et al., "Ahoy SAILR! There is No Need to DREAM of C" (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/basque)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004)](https://dl.gi.de/items/7f573931-cb51-49ef-a12c-0452a9fb0ea4)
- [Maranzano and Bourne, "A Tutorial Introduction to ADB", Bell Labs (1977)](http://squoze.net/UNIX/v7/files/doc/18_adb.pdf)
- [Stallman, KTH lecture on GDB entering distribution (30 October 1986)](https://www.gnu.org/philosophy/stallman-kth.html)
- [Hex-Rays, "IDA: celebrating 30 years of binary analysis innovation"](https://hex-rays.com/blog/ida-celebrating-30-years-of-binary-analysis-innovation)
- [Guilfanov, "Decompiler internals: microcode", Recon Brussels 2018 (slides)](https://www.recon.cx/2018/brussels/resources/slides/RECON-BRX-2018-Decompiler-internals-microcode.pdf)
- [Hex-Rays, microcode view and maturity levels](https://docs.hex-rays.com/core/decompiler/how-tos/microcode-view)
- [Hex-Rays, IDA 9.4 release notes](https://docs.hex-rays.com/release-notes/9_4)
- [Hex-Rays, IDA Free](https://hex-rays.com/ida-free)
- [Ghidra releases (12.1.4)](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Ghidra P-code reference](https://github.com/NationalSecurityAgency/ghidra/blob/master/GhidraDocs/languages/html/pcoderef.html)
- [Ghidra decompiler source, main work flow and Action classes](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/Decompiler/src/decompile/cpp)
- [Ghidra analyzeHeadless README](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/RuntimeScripts/support/analyzeHeadlessREADME.md)
- [Binary Ninja, BNIL overview](https://docs.binary.ninja/dev/bnil-overview.html)
- [Binary Ninja, "Architecture Agnostic Function Detection in Binaries" (2017)](https://binary.ninja/2017/11/06/architecture-agnostic-function-detection-in-binaries.html)
- [Binary Ninja, batch processing and headless licences](https://docs.binary.ninja/dev/batch.html)
- [Binary Ninja 6.0 "Krypton" release post](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [Binary Ninja Free](https://binary.ninja/free/)
- [radare2 book, ESIL](https://book.rada.re/emulation/esil.html)
- [radare2 book, decompilers](https://book.rada.re/arch/decompile.html)
- [radare2 releases (6.2.4)](https://github.com/radareorg/radare2/releases)
- [rizin, RzIL design notes](https://github.com/rizinorg/rizin/blob/dev/doc/rzil.md)
- [rizin releases (0.9.1)](https://github.com/rizinorg/rizin/releases)
- [Announcing Rizin (8 December 2020)](https://rizin.re/posts/announcing-rizin/)
- [Cutter releases (2.5.0)](https://github.com/rizinorg/cutter/releases)
- [angr documentation, CFG recovery](https://docs.angr.io/en/latest/analyses/cfg.html)
- [angr documentation, the decompiler and its analysis passes](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [angr documentation, intermediate representation (VEX)](https://docs.angr.io/en/latest/advanced-topics/ir.html)
- [angr on PyPI (10.0.3)](https://pypi.org/project/angr/)
- [pypcode on PyPI](https://pypi.org/project/pypcode/)
- [llvm-objdump command guide](https://llvm.org/docs/CommandGuide/llvm-objdump.html)
- [GNU objdump manual](https://sourceware.org/binutils/docs/binutils/objdump.html)
- [LIEF documentation](https://lief.re/doc/stable/index.html)
- [Capstone releases](https://github.com/capstone-engine/capstone/releases)
- [BinDiff releases (v8, open source)](https://github.com/google/bindiff/releases/tag/v8)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [Hopper Disassembler downloads](https://www.hopperapp.com/download.html)
- [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)
- [LLDB announced as an LLVM subproject (2010)](https://blog.llvm.org/2010/06/new-lldb-debugger.html)
- [Boomerang decompiler](https://boomerang.sourceforge.net/)
- [Tan et al., "LLM4Decompile" (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.203/)
- [Gao et al., "DecompileBench" (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, "When LLM Decompilers Recompile More and Preserve Less" (2026)](https://arxiv.org/abs/2609.05370)
