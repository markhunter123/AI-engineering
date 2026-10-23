# 二进制差异比较与符号执行：匹配函数并求解输入

> 两个函数之间的匹配，是由某一种签名以某个质量等级给出的断言；一个解出的输入，则是某条路径条件的一个模型。本课把这两者都构建出来，并对照 radiff2、Ghidra、angr 和 z3 进行校验。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), 34 (lifting to pseudocode), 36 (rebuild and prove equal)
**Time:** ~150 minutes

## 学习目标

- 从一个目标文件和它的反汇编列表构造五种函数签名：精确字节、屏蔽重定位字段后的字节、助记符序列、流图形状和符号名。
- 运行一个 BinDiff 风格的分阶段贪心匹配器：只有当某个签名在两侧都唯一时才接受匹配，并记录每个匹配背后的阶段和质量等级。
- 复现这样一种情形：两个字节完全相同的函数在加载镜像上无法通过字节关联器，屏蔽重定位字段后则通过；同时给出解释那四个变化字节的重定位算术。
- 用 Levenshtein 距离复现 radiff2 的相似度比值，并解释为什么其中一个比值只有在比较使用 radare2 实际到达的 82 个字节时才出现。
- 在一个小型 IR 上枚举函数的路径，为每条路径携带一个条件，并用有界搜索为每个叶子求解输入。
- 在一个带符号边界的循环上度量路径爆炸，并将路径数量、分类和模型与 angr 10.0.3 和 z3 5.1.0 对比。

## 问题

你在 `-O2` 下重新构建了 `ledger.c`，然后问一个智能体：与 `-O1` 构建相比有没有变化。智能体跑了一个差异工具，报告 `grade` 被修改了，因为它在这两个已加载程序中的字节不同。接着它花掉大量预算去解释一个并不存在的改动。`grade` 的 46 个字节在两个目标文件里完全相同。差异工具比较的是重定位之后的镜像，而在 `-O2` 程序中 `.rodata` 的位置比原来远了 0x70 个字节，于是 `grade` 里的一个位移量增大了 0x70。

同一天下午，你问「哪个 score 会让 `grade` 返回 4」。智能体读了伪代码，回答「90」。这只是一个答案。该函数对 90 到 109 的每一个 score 都返回 4，而问题没有给出边界，于是智能体给出的答案只是样本，你需要的却是一个集合。

这两个失败源自同一个缺口。一个匹配和一个解出的输入，背后都各自有一套方法支撑的断言。本课用标准库 Python 构建这两套方法，并在旁边运行生产工具。工具与代码一致的地方，你就有了两种推导。不一致的地方，你就有了一个地址、一个字节或一个边界可以去查。

```figure
re-44-matches-and-paths
```

## 函数签名是一条带有质量等级的断言

Flake 在 2004 年把二进制差异比较定义为：在两个相似可执行文件的函数集合之间构造同构。Dullien 和 Rolles 在 2005 年将其扩展到基本块和指令。BinDiff 的概念页描述了这项工作中产生的策略。为两侧的每个函数计算同一种属性。若某个值在每一侧都恰好出现一次，它就是一个匹配。若某个值出现多次，则它是歧义的，需要在这个等价类内部用下一种属性继续下钻。没有对应值的函数保持未匹配。每一轮之后，新匹配的调用者和被调用者构成小集合，下一种属性在这些小集合里很可能唯一，于是用下一种属性重复该过程。

每种匹配算法都带有质量等级。该页面把原始函数字节的哈希匹配和名称哈希匹配评为很好（very good），把流图 MD 索引和素数签名匹配（每个助记符对应一个小素数，对整个函数取乘积）评为好（good），把字符串引用评为中等（medium），把循环次数或地址序列评为差（poor）。一个匹配的置信度，是产生它的那些算法质量的平均值，并经过 sigmoid 加权。函数相似度大约由 25% 的匹配流图边、15% 的匹配基本块、10% 的匹配指令、以及 50% 的流图 MD 索引差异构成，再乘以置信度。

本课保留其中五种属性，以及那条真正重要的规则：两侧都唯一才匹配，否则不匹配。它舍弃了 MD 索引，因为概念页只给出名字而未给出公式，改用普通的度数序列代替。质量标签保留下来，因为一个匹配的好坏，只取决于产生它的最弱事实。

## 从一个函数记录得到五种签名

差异工具对每个目标文件读取两样东西。ELF 提供字节、符号大小和 `.rela.*` 记录。`llvm-objdump -d` 列表提供指令边界和助记符，而第 33 课已经证明，一个解码器在这些目标的每条指令上都与那份列表一致。`function_record()` 检查列表中每条指令的字节是否等于目标文件在该地址处的字节，不等就抛出异常。列表一旦过期，就说明测试夹具（fixture）已经变了。

以 `ledger-x86_64-O1.o` 中的 `grade` 为例。符号给出偏移 0x10、大小 46，因此记录覆盖 0x10 到 0x3e，并排除用于对齐下一个函数的两字节 `nop`。

精确字节签名就是这 46 个字节的 SHA-256。屏蔽字节签名则先把每个重定位字段清零。这里只有一个字段：位于偏移 0x36 的 `R_X86_64_PC32 .rodata+0x10` 记录，宽四个字节，即 0x33 处 `leaq (%rip), %rcx` 的位移量。在 RELA 目标文件中，这四个字节本来就已为零，所以在本课的所有目标文件中，两种签名对每个函数都一致。屏蔽机制是为下一节准备的。

助记符签名对 14 个助记符的序列做哈希：

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

流图形状签名来自第 33 课的基本块和带类型的边，并限制为入口可达的块。`-O1` 下的 `grade` 有三个块和三条边。块 0x10 以 `ja 0x3d` 结束，带一条跳转边和一条直落（fallthrough）边；块 0x31 直落到 0x3d；0x3d 返回。度数序列按排序顺序列出每个块的 (in, out)，所以签名写作 `3b 3e [0/2,1/1,2/0]`。对于 `-O0` 下的 `grade`，记录通过五条 `.rela.rodata` 条目解析间接跳转，方式与第 33 课完全相同，得到八个块、十一条边，其中四条是 `table` 边。

名称签名就是符号名。符号存在时它是最强的事实；构建被剥离符号时，它什么也不是。

还有一个字段会随记录保留，供后面的补丁案例使用。当一个函数包含指向 `.rodata` 的 `lea` 重定位、一次边界检查和一次带索引的加载时，`referenced_table()` 会像第 36 课那样读取这张表：`ja` 之前的 `cmpl $0x4` 给出五个条目，加载指令 `movl (%rcx,%rax,4), %eax` 给出每条目四个字节。对于 `-O1` 下的 `grade`，`.rodata+0x14` 处的表读作 `1, 2, 3, 4, 4`。

## 贪心匹配器

`match_functions()` 按顺序运行各个阶段。对每个阶段，它把两侧仍未匹配的函数按该阶段的签名分组，并接受那些在每一侧都恰好只有一个函数的取值。匹配记录保存阶段、该阶段的质量等级、其他所有同样一致的签名，以及一个相似度比值。

```python
for stage in order:
    for signature, candidates in left_groups.items():
        partners = right_groups.get(signature, [])
        if len(candidates) != 1 or len(partners) != 1:
            continue
        a, b = candidates[0], partners[0]
        agreeing = [s.id for s in STAGES if a.signatures()[s.id] == b.signatures()[s.id]]
        matches.append(Match(a, b, stage, agreeing, radare2_ratio(a.raw, b.raw), data_equal))
```

本课的顺序先运行四个结构阶段，最后运行名称阶段，这样结构阶段先接受检验，再由名称在事后为其评级。对于 `-O1` 对 `-O2`：

| 函数 | 阶段 | 一致的签名 | 相似度 |
|---|---|---|---|
| `percent_change` | 精确字节 | 全部五种 | 1.000 |
| `grade` | 精确字节 | 全部五种 | 1.000 |
| `total` | 名称 | 仅名称 | 0.171 |
| `grade_label` | 精确字节 | 全部五种 | 1.000 |

`total` 从 31 字节、13 条指令增长到 140 字节、41 条指令，因为 clang 用 SSE2 向量化了循环，并保留了一个标量剩余循环。它的形状从五个块、五条边变成十个块、十三条边。结构上没有任何东西保留下来，只有符号把两者联系起来。

对于 `-O0` 对 `-O1`，同样的顺序会产生一个错误的匹配：

| -O0 下的函数 | 匹配到 -O1 的函数 | 阶段 | 名称是否一致 |
|---|---|---|---|
| `percent_change` | `grade_label` | 流图形状 | 否 |
| `grade` | `grade` | 名称 | 是 |
| `total` | `total` | 名称 | 是 |
| `grade_label` | 未匹配 | | |

`-O0` 下的 `percent_change` 是一个四块菱形：入口根据 `before == 0` 分支，有两条臂，以及一个返回的汇合点。它的度数序列是 `0/2, 1/1, 1/1, 2/0`。`-O1` 下的 `grade_label` 内联了 `grade`，在边界检查周围有同样的菱形。该形状在两侧都唯一，规则于是接受了它。中等的质量标签就是警告，而随后运行的名称阶段否定了这个匹配。这个错误匹配还占用了右侧的 `grade_label`，导致真正的 `percent_change` 与 `grade_label` 配对始终未能匹配。

存在真实符号名时，BinDiff 的顺序把名称匹配作为第二个阶段，代码中的 `BINDIFF_ORDER` 也是如此。按这个顺序，四个 `-O0` 函数都通过名称匹配到各自的 `-O1` 版本，这正是下文 Ghidra 报告的结果。本课在演示中保留两种顺序，让这个失败可见。四个节点的流图形状，是每一个在返回前汇合的 `if` 与 `else` 的形状。

## 相同字节，重定位后的操作数

Ghidra 在导入目标文件时会应用重定位。因此它的字节关联器比较的是加载后的镜像，第 36 课正是出于这个原因屏蔽了重定位字段。`load_image()` 复现了导入过程。它按头部顺序遍历已分配的节，将每个节按其 `sh_addralign` 对齐，并从 0x100000 开始布局。然后它应用每一条记录：对 `R_X86_64_PC32` 和 `R_X86_64_PLT32`，把 `S + A - P` 写成有符号 32 位值；对 `R_X86_64_64`，把 `S + A` 写成 64 位值。

该布局在两个程序中每个节的起始地址上都与 Ghidra 12.1.4 一致，测试套件会拿捕获的版本跟踪输出核对这一点：

| 节 | `-O1` 程序 | `-O2` 程序 |
|---|---|---|
| `.text` | 0x100000，159 字节 | 0x100000，271 字节 |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

`-O2` 下的 `.text` 长了 112 个字节，即 `total` 的增长量，因此 `.rodata` 的起始位置远了 0x70 个字节。`grade` 在两个程序中都位于 0x100010。它在偏移 0x33 处的 `leaq`，位移字段在 0x36，指令结束于 0x3a。重定位加数 0x10 加上通常的 4，把表定位到 `.rodata+0x14`。

在 `-O1` 镜像中：`0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`。在 `-O2` 镜像中：`0x100110 + 0x14 - 0x10003a = 0xea = 234`。字段字节在一个镜像中是 `7a 00 00 00`，在另一个中是 `ea 00 00 00`。46 个字节中有一个不同，Levenshtein 比值为 0.978，精确字节阶段失败。屏蔽字节阶段把该字段清零后通过。`grade_label` 和它的表一起移动了相同的 0x70，所以它的位移量保持相等，在两种视图下都能通过精确字节匹配。

Ghidra 报告了同样的划分。在 `-O1` 对 `-O2` 上，Exact Function Bytes Match 找到了 `percent_change` 和 `grade_label`，Exact Function Instructions Match 找到这两者再加上 `grade`，Exact Function Mnemonics Match 找到同样的三个。指令关联器会屏蔽操作数，这就覆盖了位移量。本课的镜像视图给出的精确字节匹配，恰好是 Ghidra 字节关联器找到的那两个函数；给出的屏蔽字节一致，恰好对应它的指令关联器找到的那三个函数。测试套件对两个等式都做了断言。

rea 6.5.0 的 `compare_functions` 对 provider 文本做行级差异比较。在汇编维度上，它附加了「relocation normalization is unavailable」这一限制。本课的加载器和屏蔽机制提供了这种规范化。

## radiff2 度量的是什么

radare2 6.2.4 自带 `radiff2`。使用 `-AC` 时，它分析两个文件并为每个函数打印一行：左侧大小和地址、MATCH 或 UNMATCH、一个比值，以及右侧大小和地址。对我们的目标文件捕获到的行如下：

| 配对 | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` 对 `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` 对 `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` 对 `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

radare2 源码中的比值是 `1 - distance / max(len_a, len_b)`，其中距离是两个函数指纹之间的 Levenshtein 距离，指纹则是把函数的基本块按地址顺序拼接而成。匹配阈值是 0.7。`code/main.py` 中的 `radare2_ratio()` 在原始函数字节上实现了同样的公式，并把 0.130435、0.106061、0.269841 和 0.171429 复现到小数点后六位。

它无法复现 `-O0` 下 `grade` 的 0.426829。按该函数的 118 个字节计算，比值是 0.296610。radare2 在大小列里打印的是 `118`。它的指纹更短。在没有应用重定位的情况下，它无法解析跳转表，因此分析只到达入口块、以 `jmpq *%rax` 结束的块、默认分支和返回：一共 82 个字节，正是第 33 课纯递归下降所认定的那 82 个字节。这 82 个字节与 `-O1` 下的 46 个字节之间的 Levenshtein 距离是 47，`1 - 47/82 = 0.426829`。加上 `-e bin.relocs.apply=true` 后，radare2 找到全部 118 个字节并打印 0.288136，这等于本课在重定位镜像上得到的比值：在那里，四个位移字节与目标文件中的零字段不同。两个工具只有在比较同样的字节时才会一致，而大小列并不会说明到底哪些字节参与了比较。

`-O` 把指纹限制为操作码字节，并把这些操作数清零。`-O0` 对 `-O1` 的比值升到 0.239130、0.512195、0.303030 和 0.523810，`-O1` 对 `-O2` 的 `total` 升到 0.192857。整文件距离也存在：`-s`（Myers）报告两个目标文件之间的相似度 0.956、距离 162，`-ss`（Levenshtein）报告 0.927 和 137。这些数字包含头部、符号表和重定位节，对函数本身什么也没说明。

## Ghidra 版本跟踪关联器

Ghidra 的版本跟踪（Version Tracking）工具用关联器（correlator）把源程序与目标程序匹配起来。函数关联器有 Exact Function Bytes Match、Exact Function Instructions Match（屏蔽操作数）、Exact Function Mnemonics Match 和 Duplicate Function Instructions Match（1:N 和 N:M）。数据关联器有 Exact、Duplicate 和 Similar Data Match。符号关联器按精确名称、重复名称或相似度匹配名称。引用关联器利用已接受的匹配找到更多匹配。函数必须在目标程序中已定义，函数最小长度默认为 10 字节。Auto Version Tracking 依次运行 Exact Symbol Name、Exact Data、Exact Function Bytes、Exact Function Instructions、Exact Function Mnemonics、Duplicate Function Instructions，然后是一个引用关联器，并为已接受的匹配创建隐含匹配。11.2 版本加入了 BSim 关联器。

本课以无头（headless）方式捕获了这些关联器的结果。`capture_ghidra_vt.sh` 把三个目标文件导入一个项目，并运行 `ListVersionTrackingMatches.java`。这是一个简短脚本，继承 `AbstractGhidraVersionTrackingScript`，按路径打开源程序和目标程序，按 Auto Version Tracking 的顺序运行六种非引用关联器，并为每个块、函数、关联器和匹配打印一行。本节中的每个数字都来自那两个文本文件。

| 关联器 | `-O0` 对 `-O1` | `-O1` 对 `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 个函数 | 4 个函数，5 个数据项（`labels.rel`、`.L.str.1` 到 `.L.str.4`） |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2（`percent_change`、`grade_label`） |
| Exact Function Instructions Match | 0 | 3（增加 `grade`） |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

在 `-O0` 与 `-O1` 之间，每个函数都发生了变化，所以只有名称匹配上了，这正是本课匹配器在 BinDiff 顺序下给出的结果。每个匹配都打印出相似度 1.000 和置信度 10.000。帮助页面把精确关联器描述为相似度 1.0、置信度 1.0，因此打印出的置信度用的是哪种量纲，是一个本课记录但未解决的开放问题。

Tools 菜单下的 Program Differences 是另一回事。它比较两个具有相同地址空间的程序在字节、标签、代码单元、引用、注释以及相同地址上的函数方面的差异。它是一种按地址对齐的差异比较，凡位置移动过的对象都不会匹配。

## 依据文档介绍的 Diaphora 与 Binary Ninja

Diaphora 是一个 IDA Pro 插件，2015 年首次发布，截至 2026-08-18 版本为 3.4.2。自 2.0 起采用 AGPL-3.0，且需要 IDAPython，所以它能在 IDA Home 或 IDA Pro 中运行，不能在 IDA Free 中运行。它把每个数据库导出为 SQLite，并把启发式规则作为 SQL 查询来执行。每条启发式规则都有一个名称、一个类别（Best、Partial 或 Unreliable）、一种比值类型，以及可选标志，例如 SAME_CPU 或 SLOW。当前源码中的名称读起来就像本课这些阶段，只是数量更多：Bytes hash、Same address and mnemonics、Same cleaned assembly、Mnemonics small-primes-product、Same nodes, edges, loops and strongly connected components、Topological sort hash，以及需要 Hex-Rays 反编译器的伪代码模糊哈希。它的发布说明展示了调优历史：慢速启发式只在函数数量低于约 2,000 时运行，「Same graph」被移到了不可靠类别。那次移动正是本课的四块菱形在大规模下的样子。

Binary Ninja 6.0 于 2026-09-03 发布，加入了带 provider 和 resolver 的 Binary Similarity 框架。它随附 BinDiff 作为结构 provider，以及用于精确函数匹配、可容忍可重定位操作数的 WARP；其指南说，当函数除可重定位指令之外都匹配时用 WARP，当函数在重新编译后仍保持结构相关时用 BinDiff。该功能仅在 Ultimate 版本中提供。这里的 Binary Ninja 没有可供包管理器安装的途径，也未运行，所以本段内容仅来自其文档，别无其他依据。

BinDiff 本身自 2023-09-25 起以 Apache-2.0 开源，最新版本是 v8。它需要一个反汇编器和 BinExport 插件，后者支持 IDA Pro 9.4 或更高版本、Binary Ninja 和 Ghidra。学习者免费可走的路径是 Ghidra + BinExport + BinDiff 命令行。本课没有运行它，实践练习会要求你来做。

## 相同代码，不同常量

第 36 课保留了一个 `grade` 的候选重建版本，它的表最后一个条目是 3，而原始版本是 4。`grade_case10_wrong.o` 与 `-O1` 下的 `grade` 拥有相同的 46 个代码字节。匹配器按精确字节把它们配成一对，所有签名都一致。引用表检查通过一个重定位读到 `1, 2, 3, 4, 4`，通过另一个重定位读到 `1, 2, 3, 4, 3`，并报告 `dataEqual: false`。另外三个 `-O1` 函数在左侧未匹配，因为候选目标文件只包含一个函数。

这就是补丁分析的形态。一个止步于字节的函数差异工具会说没有任何变化。Ghidra 的 Exact Data Match 和 BinDiff 的字符串引用匹配正是出于同一原因而存在。本课的版本很小：对每一对匹配，比较两个函数的重定位分别指向的字节，条目数量来自边界检查，条目宽度来自加载指令。

## 符号执行：路径、条件、求解器

King 在 1976 年描述了符号执行。输入变成符号。每条语句更新一个符号存储。每个分支为路径添加一个条件，求解器判定该路径是否可行，并产生一个沿其执行的具体输入。带分支的函数成为一棵路径树，叶子就是一个附带了条件的返回。

Baldoni、Coppa、D'Elia、Demetrescu 和 Finocchi 在 2018 年综述了这一领域。他们的第 5 节指出，路径爆炸是核心代价：执行器可能在每个分支处分叉，因此状态数量可能随分支数量指数增长，而边界依赖符号输入的循环会产生无界的路径集合。他们列出的缓解手段包括：通过在每个分支处求解来剪除不可实现的路径、为函数和循环提供摘要（summary）、状态合并，以及欠近似（under-approximation）。Concolic 执行同时以具体和符号两种方式运行程序，记录具体运行沿途的条件，取反其中一个，再向求解器请求下一个输入。DART（2005）、CUTE（2005）和 SAGE（2008）是文中点名的系统。

KLEE（Cadar、Dunbar、Engler，OSDI 2008）是一个构建在 LLVM bitcode 之上的符号虚拟机，因此它的输入是 `clang -emit-llvm` 的输出，而不是二进制文件。它自带 POSIX 环境模型和一个重放库。在 89 个 GNU coreutils 程序上，它达到 84.5% 的行覆盖率，而开发者测试为 67.7%，并发现了 56 个严重 bug。KLEE 3.2 于 2025-12-23 在 NCSA 许可下发布，推荐使用 LLVM 16。

angr 作用于二进制文件。`angr.Project` 持有一个 CLE 加载器（映射目标文件并应用重定位）、一个提升器（把基本块转换为 VEX IR，即 `block.vex`，另一个可选实现是 pypcode），以及一个状态工厂。`SimState` 保存寄存器、内存和一个求解器。`SimulationManager` 推进名为 `active`、`deadended`、`found`、`avoided` 和 `errored` 的状态集合（stash），`explore(find=..., avoid=...)` 则封装了它。每一步中，各引擎按顺序运行：failure、syscall、hook、用于具体数据的 unicorn，以及作为兜底的 VEX。claripy 是 Z3 之上的求解器前端；在 angr 10.0 中，它是一个随包内置的 Rust 实现，通过 `from angr import claripy` 导入。angr 10.0.3 于 2026-10-10 上传，采用 BSD-2-Clause 许可，需要 Python 3.12 或更高版本。Z3 5.1.0 于 2026-08-16 在 MIT 许可下发布，是 angr、Triton 以及 KLEE 可选后端背后的求解器。

## 基于第 34 课 IR 的微型执行器

`code/main.py` 中的执行器使用一个小型 IR，语句形态取自第 34 课：`Assign`、`Branch`、`Jump`、`Return`，再加上第 34 课实践练习要求过的 `Switch` 终结符。表达式包括 `Const`、`Var`、带位宽的 `Bin`、`Cast`、从具名具体表读取的 `Load`，以及 `Not`。四个测试夹具函数都在第 34 课化简后所达到的层面上手工提升，此时 `0x66666667` 乘法已变成 `score / 10`，每个 IR 块都列出它来源的指令地址。`check_ir_against_listing()` 断言这些地址恰好覆盖该函数的列表一次。

在该 IR 中，`-O0` 下的 `grade` 如下：

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

switch 的目标来自 `jump_table_targets()`，与第 33 课是同样的重定位算术，因此这段 IR 读出的就是夹具内容。

`execute()` 维护一个路径状态的工作列表。一个状态包含当前块、块轨迹、条件列表，以及一个把每个变量映射为输入上表达式环境。`Assign` 把环境代入源表达式并存储结果。`Branch` 把环境代入条件，然后向求解器询问两次：一次加上该条件，一次加上其否定。求解器无法满足的一侧被丢弃，这就是综述所说的及早剪枝。`Switch` 按目标对表条目分组，对每个不同目标用条目等式的析取分叉一次，angr 处理符号跳转目标时也是如此。`Return` 把环境代入返回值并闭合叶子。轨迹超过步数上限时，该路径被标记为截断并停止。

手工跟踪通向 `return 4` 的路径。入口赋值 `local_10 = score / 10 - 6`。在这条路径上，分支条件 `local_10 >u 4` 为假，因此条件列表保存 `!(score / 10 - 6 >u 4)`。switch 的条目 3 和 4 都指向 0x76，因此到 0x76 的分叉加入 `score / 10 - 6 == 3 || score / 10 - 6 == 4`。块 0x76 赋值 `local_4 = 4`，跳到 0xa1，返回时替换为 `4`。叶子就是该条件与该值组成的一对。

求解器是 `BoundedSearch`。它为每个输入持有一个有限论域，枚举各论域的笛卡尔积，并按 C 语义求值每个约束：32 位回绕、截断除法、有符号与无符号比较，以及算术移位或逻辑移位。除零或表索引落在表外，会使该赋值无法满足约束。当 `score` 在 -200 到 1200 之间时，`grade` 的五条路径共耗费 44,460 次求值，用时几毫秒。为每个叶子求解输入集合，得到：

| 叶子 | 返回值 | 条件 | 边界内的 score |
|---|---|---|---|
| 0x76 | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 到 109，20 个输入 |
| 0x7f | 3 | `!(q >u 4) && q == 2` | 80 到 89，10 个输入 |
| 0x88 | 2 | `!(q >u 4) && q == 1` | 70 到 79，10 个输入 |
| 0x91 | 1 | `!(q >u 4) && q == 0` | 60 到 69，10 个输入 |
| 0x9a | 0 | `q >u 4` | 1,351 个输入，边界内的其余部分 |

`q` 表示 `score / 10 - 6`。`-O1` 下的 `grade` 得到相同的分类：其 IR 只有两条路径，查表路径返回 `lookup_14[q]`，默认路径返回 0；因为不存在可达的 case 块，问题就变成对返回值的约束。表 `lookup_14` 从目标文件的 `.rodata` 偏移 0x14 处读取。

边界就是这种方法的极限。有界搜索在其论域内完备，在论域外则什么也不说。在 -200 到 1200 之内，60 到 109 的 score 没有哪个返回 0，110 及以上的 score 也没有哪个返回非零值。z3 对全部 2^32 个 score 证明了这两点，下一节将展示两种方法在何处给出不同的见证（witness）。

## 实测路径爆炸

`-O1` 下的 `total` 就是第 33 课的循环：对 `count` 的一次测试，一个把 `values[rdx]` 累加并递增 `rdx` 的循环体，以及一个在 `rdx != rcx` 时跳回循环体的 `jne`。在 IR 中，`rcx` 是 `zext64(count)`，`rdx` 从 0 开始。每次经过回边时，都在 `rdx != count` 处分叉，而 `rdx` 是常量，因此 `count = c` 的路径执行循环体 `c` 次后退出。当 `count` 限定在 0 到 2 时，执行器找到 3 条路径。0 到 4 时找到 5 条，0 到 8 时找到 9 条。每条路径返回其 count 的三角和：(0, 0)、(1, 1)、(2, 3)，一直到 (8, 36)。angr 报告 3、5 和 9 个 deadended 状态，并在全部三个优化级别下给出同样的九组 count 与和，包括 `-O2`：它的向量化循环要求 count 至少为 8，因此在这个范围内从不改变路径数量。

没有边界时，执行器永远不会结束，因为每个迭代次数都对应多一条可行路径。本课给每条路径设了 24 个块的上限；在 `count` 最大到 64 时运行，返回 22 条路径，截断 2 条，并且从未启动 65 条路径中其余的 41 条。报告给出这三个数字。求解成本也在增长：边界为 2 时 33 次求值，为 4 时 115 次，为 8 时 531 次，带上限的那次运行为 6,900 次。

## 用 angr 和 z3 回答同样的问题

`capture_angr.py` 在三个目标文件上运行了 angr 10.0.3。CLE 用 ELF 后端把每个可重定位目标文件加载到 0x400000，对 `-O0` 应用了 13 处重定位、对其余文件应用了 8 处，并解析了四个符号。对于带一个符号 64 位参数的 `grade`，`run()` 在 `-O0` 下以 5 个 deadended 状态结束，在 `-O1` 和 `-O2` 下各以 2 个状态结束，耗时分别为 0.475、0.408 和 0.318 秒。给每个 deadended 状态加上 `eax == v`，并向求解器询问有符号最小值与最大值：在每个优化级别下，`v` 从 1 到 4 分别得到 60 到 69、70 到 79、80 到 89、90 到 109，`v` 为 0 时得到 -2147483648 到 2147483647。两个交叉检查的结果都是 unsat。

到达指定块的形式在 `-O0` 的 `grade` 上使用 CFGFast，列出第 33 课找到的八个块，然后用 `explore(find=0x400076)` 定位存储 4 的那个块。它在 0.47 秒内找到一个状态，另有三条仍处于活动状态；参数的最小值为 90、最大值为 109，样本为 91、93、94、97 和 101。

对于 `percent_change`，得到两个 deadended 状态，约束分别为 `before[31:0] != 0` 和 `before[31:0] == 0`。向非零状态询问 `eax == 50`，`-O0` 下得到 `before = 21055744, after = 332315210`，`-O1` 下得到 `40566804, 834296877`，`-O2` 下得到 `-24484867, 993934694`。在 -16 到 16 范围内的有界搜索给出 `before = -10, after = -15`。就第 34 课的列表向 z3 询问同样的问题，它回答 `before = -25092599, after = 305712160`。测试套件用 IR 求值器验证全部五对，每一对都得到 50。一个模型只证明某个输入存在，它并不是那个输入；三个正确的工具给出了三个不同的输入。

`capture_z3.py` 用 32 位位向量对 `-O1` 下 `grade` 的列表建模，包含乘法、两次移位和查表，并再提出四个问题。「列表与源码 `switch` 是否存在某个不同的 32 位 score」：0.068 秒内得到 unsat，也就是说编译后的函数在全部 2^32 个输入上与源码等价。「列表与第 36 课的阈值重建是否不同」：在 score 706346536 处为 sat，此时 `score / 10` 为 70634653，switch 落入默认分支，而阈值版本返回 4。同一差异限定在 0 到 109：unsat，这就是第 36 课的网格没能发现它的原因。对于 `percent_change`，`before == 3` 且结果为 -133 是可满足的，`after = 1073741823`，也就是第 36 课的溢出单元；而有界搜索在其论域内找到了第二个见证 `after = -1`，此时 `(-1 - 3) * 100 / 3` 截断为 -133，完全无溢出。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

输出是一份 JSON 报告。`diffs` 保存 `-O0` 对 `-O1` 和 `-O1` 对 `-O2` 的目标文件匹配与加载镜像匹配，外加按 BinDiff 顺序得到的 `-O0` 配对。`toolComparison` 为每个函数把本课的阶段与 radiff2 的判定和比值、以及 Ghidra 的关联器结果放在一起。`patch` 是 `grade_case10_wrong.o` 案例。`radare2Ratios` 复现 radiff2 的数字，`gradeRelocationCase` 展示布局和两个位移量，`symbolic` 保存路径、分类和爆炸计数。

按以下顺序阅读 `code/main.py`：

1. `ElfObject` 读取节、符号和 `.rela.*` 记录。`parse_listing()` 读取 `llvm-objdump` 文本。
2. `jump_table_targets()` 和 `build_cfg()` 给出可达块和带类型的边。`FunctionRecord.signatures()` 构建五种签名。
3. `load_image()` 布局已分配的节，并应用 `S + A - P` 和 `S + A`。
4. `match_functions()` 是分阶段匹配器。`levenshtein()` 和 `radare2_ratio()` 给出相似度。
5. `parse_radiff2()` 和 `parse_ghidra_vt()` 读取捕获的工具输出用于比较。
6. `evaluate()`、`substitute()` 和 `BoundedSearch` 是表达式语义和求解器。
7. `execute()` 枚举路径。`solve_returns()` 为每个叶子求解输入集合。
8. `grade_ir_o0()`、`grade_ir_o1()`、`percent_change_ir_o1()` 和 `total_ir_o1()` 是手工提升，附有各自的指令地址。

还有三个命令。`python3 code/main.py diff A.o B.o` 比较两个目标文件，它们的列表文件就放在旁边；`--image` 比较它们的加载镜像。`python3 code/main.py paths grade O0` 打印一个函数的路径和分类，`paths total 8` 打印带边界的循环。`python3 code/main.py figure-data` 打印交互图内嵌的数据。

## 使用它

对你自己程序的两个构建，按以下顺序使用差异工具和执行器。

1. 为每个目标文件捕获一份列表，并让 `function_record()` 对照字节校验它。在做其他任何事情之前先修好列表。
2. 先比较目标文件。阅读每个匹配的阶段，在第二种签名也一致之前，把流图形状匹配当作建议。
3. 比较加载后的镜像。一个函数从精确字节降级到屏蔽字节，说明它相对其数据发生了移动，而位移算术会指出是哪个节移动了。
4. 对每个匹配，比较被引用的表。代码相同而常量不同，就是函数被修改了。
5. 运行名称阶段并对比。当构建带有符号时，按 BinDiff 顺序运行，并且只把结构结果用于给各阶段评级。
6. 当某个工具报告比值时，先弄清哪些字节参与了计算，再与自己的结果比较。
7. 对你想要输入的函数，编写每条语句带一个地址的 IR，对每个符号循环计数器设定边界后运行 `execute()`，并求解每个叶子。报告结果时一并报告边界。
8. 在可用时通过 angr 或 z3 运行同样的查询，并比较分类边界。一个样本只是一个模型。

## 交互实验

### 实验 A：手工构造五种签名

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

运行：

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

在打印 `grade.raw` 之前，先写下该函数偏移 0x26 处的四个字节。然后对 `ledger-x86_64-O0.o` 中的 `grade` 运行同样的语句，并数出 `grade.edges` 中有几条 `table` 边。

### 实验 B：观察重定位如何破坏字节哈希

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

在读 `displacements` 之前在纸上算出 `0x100110 + 0x14 - 0x10003a`。然后把 `load_image()` 中的基址改为 0x400000，确认位移量不变，因为两个操作数一起移动。

### 实验 C：错误匹配以及避免它的顺序

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

打印 `-O0` 下 `records(...)["percent_change"].cfg_shape()` 与 `-O1` 下 `grade_label` 的结果，阅读这两个度数序列。然后写第六个阶段，对排序后的助记符做哈希，使顺序不再重要，看看它能否把它们区分开。

### 实验 D：路径与边界

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

把 `SCORE_DOMAIN` 改为 `range(0, 110)` 并重新运行 `grade_paths("O0")`。0 对应的分类缩小到 60 个输入，两个交叉检查仍然成立。然后把论域设为 `range(90, 100)`，看看哪些叶子完全没有模型。

## 实践练习

在测试夹具上运行 BinDiff，并与你已有的三个结果进行对比。

要求：

- 在 Ghidra 12.1.4 中安装 BinExport 扩展，导出三个 ledger 程序。记录 BinExport 的发布标签和 Ghidra 版本。
- 安装 BinDiff 8 命令行，使用仅相似度（similarity-only）选项生成 `-O0` 对 `-O1` 和 `-O1` 对 `-O2` 的结果。
- 用 `build_fixtures.sh` 中的确切命令把逐函数相似度和置信度捕获到 `code/fixtures/`。
- 给 `tool_comparison()` 增加一个 `bindiff` 列，并添加一个测试，把 BinDiff 在 `-O1` 对 `-O2` 上的匹配集合与本课的屏蔽字节集合进行比较。
- 写一段文字，指出 BinDiff 为 `grade` 使用的算法，以及它是否比较了加载后的镜像。

成功的标准是：演示为每个函数打印四列工具结果，且新测试陈述出一个等式或一个得到解释的差异。

## 课程产物

本课包含 `outputs/skill-binary-diffing-and-symbolic-execution.md`。用它把两个构建和一个函数变成一份分阶段差异报告：每个匹配都注明质量等级，每个未匹配或发生变化的函数都有解释，并附一份路径报告，给出每个叶子解出的输入集合以及产生该集合的边界。

## 验证它

运行演示和测试套件：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应当证明：

- 符号表给出四个函数，大小在 `-O0` 下为 46、118、66、39，在 `-O1` 下为 16、46、31、63，在 `-O2` 下为 16、46、140、63
- `-O1` 目标文件的 `.rela.text` 记录与 `llvm-objdump -r` 打印的描述一致
- 列表中的每条指令都位于其符号范围内，字节与目标文件一致，而改动任一字节都会抛出异常
- `-O1` 下的 `total` 有五个块、五条边和一个自环，其形状字符串是确定性的
- `-O0` 下的 `grade` 从 `.rela.rodata` 恢复出五个跳转表目标，并有八个块和十一条边
- 没有跳转表时，`-O0` 下的 `grade` 只到达四个块共 82 个字节
- `-O0` 下的 `percent_change` 与 `-O1` 下的 `grade_label` 具有相同的四块形状
- 在可重定位目标文件的每个函数上，精确哈希与屏蔽哈希一致
- `-O1` 与 `-O2` 有三个函数字节相同，`-O0` 与 `-O1` 没有任何函数字节相同
- `grade` 引用的表读作 `1, 2, 3, 4, 4`，候选版本的表读作 `1, 2, 3, 4, 3`
- 加载器在两个程序中把 `.text`、`.rodata` 和 `.rodata.str1.1` 放在 Ghidra 放置它们的位置
- `R_X86_64_PC32` 按 `S + A - P` 写入并给出位移 122，`.data` 中的 `R_X86_64_64` 条目获得字符串地址
- `grade` 的字节在目标文件中一致，在镜像中因 0x70 的偏移而不同，屏蔽后匹配恢复
- `-O1` 对 `-O2` 按精确字节匹配三个函数，`total` 按名称匹配；镜像视图按屏蔽字节匹配 `grade`，相似度为 1 - 1/46
- 本课顺序在 `-O0` 对 `-O1` 上产生一个错误的流图形状匹配，而 BinDiff 顺序按名称匹配全部四个
- 补丁候选在字节上匹配，在被引用数据上不匹配
- Levenshtein 比值复现 0.130435、0.106061、0.269841、0.171429，以及 82 个字节上的 0.426829，还有 radiff2 在应用重定位后打印的重定位字节比值
- radiff2 的 MATCH 集合等于精确字节阶段，Ghidra 的字节关联器和指令关联器结果等于镜像视图的精确字节集合与屏蔽字节集合
- Ghidra 在 `-O0` 对 `-O1` 上只找到名称匹配，每个匹配的相似度为 1.0、置信度为 10.0
- 手工提升的 IR 恰好覆盖每条列表指令一次，且求值器遵循 C 语义
- `grade` 在 `-O0` 下有五条带 switch 条件的路径，在 `-O1` 下有两条路径，且分类相同
- 有界分类 60 到 69、70 到 79、80 到 89、90 到 109 与 angr 在三个级别上的结果以及 z3 的结果相同，deadended 状态分别为 5 个和 2 个，到达指定块的取值范围为 90 到 109
- 两个交叉检查在边界内、在 angr 中、在 z3 中都没有模型，且 z3 证明列表与源码 switch 等价
- 来自有界搜索、angr 和 z3 的每个 `percent_change` 模型求值结果都是所请求的值
- `total` 给出 3、5、9 条带三角和的路径，与 angr 相同，步数上限返回 22 条并截断 2 条
- 图数据包含两种视图、重定位数字，以及五个叶子及其输入集合，站点模块内嵌同样的数据

## 生产环境中的失效模式

| 失效情形 | 弱工具报告什么 | 分析必须证明什么 |
|---|---|---|
| 对加载镜像做字节哈希 | 代码完全相同的两个构建之间「grade 被修改」 | 字段偏移、两个位移量以及节偏移；屏蔽哈希一致 |
| 对可重定位目标文件做屏蔽哈希 | 「屏蔽没有区别」 | 字段本来就已经是零；重定位记录和被引用常量承载了变化 |
| 仅靠流图形状 | 一对自信但互不相关的函数 | 第二种签名一致，或者名称相矛盾从而丢弃该匹配 |
| 在剥离符号的构建上先名称后结构 | 什么都匹配不上 | 带质量等级的结构阶段结果，以及未匹配列表 |
| 跨工具比较比值 | 「radiff2 说 0.43，本课说 0.30」 | 每一侧究竟对哪些字节做了指纹；不应用重定位是 82 字节，应用后是 118 字节 |
| 代码相同、常量不同 | 「精确匹配」 | 通过重定位和边界检查读取的表 |
| 把一个模型当作输入报告 | 「score 是 90」 | 90 到 109 的分类及其数量，以及找到它的边界 |
| 把有界搜索报告为 unsat | 「没有输入能得到 -133」 | 只在边界内 unsat；z3 在边界外找到 1073741823 |
| 无边界的符号循环 | 超时 | 边界或上限、返回的路径数、截断的路径数，以及从未启动的路径数 |
| 把闭源工具说成已运行 | 没有捕获记录的数字 | 附有命令和版本的捕获文件，或者一句说明仅依据文档的话 |

## 操作规则

一次只用一种签名进行匹配，只接受在两侧都唯一的取值，并让匹配携带阶段和质量等级。对目标文件和加载镜像都做差异比较，并解释每个在两种视图之间变了阶段的函数。在每个分支处用求解器枚举路径，为每个符号循环设定边界，并报告输入集合及其边界。没有边界的单个模型只是一个样本。

## 延伸阅读

- [BinDiff 概念：匹配策略、算法与质量评级](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [BinDiff 仓库与 v8 开源版本](https://github.com/google/bindiff/releases/tag/v8)
- [BinExport README：IDA Pro、Binary Ninja 和 Ghidra 导出器](https://github.com/google/binexport)
- [Flake《Structural Comparison of Executable Objects》（DIMVA 2004）与 Dullien、Rolles《Graph-based comparison of Executable Objects》（SSTIC 2005），收录于 BinDiff 仓库](https://github.com/google/bindiff/tree/main/docs/papers)
- [Diaphora README 与启发式源码](https://github.com/joxeankoret/diaphora)
- [Ghidra 版本跟踪关联器帮助页](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Ghidra Auto Version Tracking 帮助页](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Ghidra 12.1.4 发布](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Binary Ninja 6.0「Krypton」发布说明](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html) 与 [Binary Similarity 用户指南](https://docs.binary.ninja/guide/similarity.html)
- [radare2 手册：radiff2](https://book.rada.re/tools/radiff2/intro.html) 与 [radare2 6.2.4 源码](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King《Symbolic execution and program testing》（CACM 1976）](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni 等《A Survey of Symbolic Execution Techniques》（ACM Computing Surveys 2018）](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar、Sen《Symbolic execution for software testing: three decades later》（CACM 2013）](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar、Dunbar、Engler《KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs》（OSDI 2008）](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf) 与 [KLEE 3.2 发布](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili 等《SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis》（IEEE S&P 2016）](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [angr 文档：加载、模拟管理器与模拟流水线](https://docs.angr.io/en/latest/core-concepts/toplevel.html) 与 [PyPI 上的 angr 10.0.3](https://pypi.org/project/angr/)
- [Z3 5.1.0 发布](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Triton 动态二进制分析库](https://github.com/JonathanSalwan/Triton)
- [System V 应用二进制接口，AMD64 架构处理器补充，表 4.9：重定位类型](https://gitlab.com/x86-psABIs/x86-64-ABI)
