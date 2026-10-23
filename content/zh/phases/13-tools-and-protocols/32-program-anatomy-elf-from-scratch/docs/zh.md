# 程序解剖：从零编写 ELF 读取器

> 编译后的程序是一组相互指向的表。你亲手写的读取器，其输出可以逐字节核对。

**Type:** Build
**Languages:** Python (stdlib), C (fixture only)
**Prerequisites:** Phase 0 · 01 (Dev Environment) with clang installed, Python 3.10 or later, hexadecimal arithmetic
**Time:** ~120 minutes

## 学习目标

- 说明本路线中每个目标的授权规则，以及 17 U.S.C. 1201(f) 的四项限制。
- 用 `struct.unpack` 读取 ELF 头、节头表以及两张字符串表。
- 依据 gABI 与 x86-64 psABI 中的名称，解码 `.symtab` 条目和 `.rela` 记录。
- 返回指定函数的精确字节，并把每个文件偏移映射到拥有它的结构。
- 读取程序头，并通过 `PT_LOAD` 把虚拟地址映射到文件偏移。
- 读取 Mach-O 目标文件的加载命令，并指出每一条对应的 ELF 概念。

## 开始之前：授权与法律

本课程第 32 至 36 课的每一个目标，都是你自己编写并编译的程序。源码是 `code/fixtures/ledger.c`，包含四个简短的 C 函数。你拥有这份源码、编译命令以及它生成的目标文件。没有需要阅读的许可证，没有需要绕过的保护，也没有任何第三方代码被你打开。本路线从不涉及软件保护、许可证校验，或你不拥有的程序。

这条规则之所以存在，是因为法律将反汇编视为复制。在 *Sega Enterprises Ltd. v. Accolade, Inc.*，977 F.2d 1510（第九巡回法院，1992）一案中，法院认定反汇编过程中产生的中间副本属于《版权法》意义上的副本。随后法院认定：「当反汇编是获取受版权保护的计算机程序中所体现的思想与功能要素的唯一途径，并且存在寻求这种访问的正当理由时，反汇编在法律上属于对受版权保护作品的合理使用。」

第二个案例是 *Sony Computer Entertainment, Inc. v. Connectix Corp.*，203 F.3d 596（第九巡回法院，2000）。Connectix 在构建模拟器时复制了 PlayStation BIOS，法院认定这些副本属于合理使用。两个案例保护的都是研究行为。二者都没有保护最终产品中复制的代码。

两部成文法划定了边界。17 U.S.C. 1201(f) 允许「已合法获得使用某计算机程序副本权利」的人规避访问控制，「其唯一目的是识别和分析程序中为实现独立创作的计算机程序与其他程序的互操作性所必需的要素」。这里叠加了四项限制：合法副本、互操作性目的、该要素此前并非「易于获得」，以及分析本身不构成侵权。第 1201(f) 条只是反规避规则的例外。复制问题仍由合理使用原则管辖。

在欧盟，指令 2009/24/EC 第 5(3) 条涵盖研究行为。拥有副本使用权的人可以「观察、研究或测试程序的运行，以确定构成程序任何要素之基础的思想和原理」。该权利在加载、显示、运行、传输或存储程序时均适用。

第 6 条允许在「为获得独立创作的计算机程序之互操作性所必需的信息不可或缺」时，未经权利人授权进行反编译。其条件与 1201(f) 一致。行为由被许可人实施，信息此前并非易于获得，且行为不超出所需的部分。第 6(2) 条禁止将结果用于任何其他目的，或用于在表达上实质相似的程序。第 8 条规定，与第 6 条或第 5(3) 条相抵触的合同条款无效。

| 来源 | 允许的行为 | 条件 |
|---|---|---|
| 17 U.S.C. 1201(f) | 规避访问控制以分析程序 | 合法副本、互操作性目的、信息并非易于获得、不构成侵权 |
| 指令 2009/24/EC 第 5(3) 条 | 观察、研究和测试运行中的程序 | 拥有副本使用权，行为在正常加载和运行范围内 |
| 指令 2009/24/EC 第 6 条 | 反编译 | 被许可人、互操作性目的、信息并非易于获得、限于所需部分 |
| *Sega v. Accolade* | 反汇编属于合理使用 | 是接触思想和功能要素的唯一途径，且有正当理由 |
| *Sony v. Connectix* | 逆向工程中的中间副本 | 最终产品不构成侵权 |

这些说明来自成文法与判决书，不构成法律意见。当未来的目标不属于你时，在打开文件之前先取得书面授权，并把授权与证据一并保存。《Reverse Engineering 101》手册第 2.1 节给出了完整引证和 CJEU 案例。

## 问题

你编译了 `ledger.c`，并对目标文件运行了 `llvm-objdump -t`。输出中 `percent_change` 的值为 `0000000000000000`，大小为 `0000000000000010`。在这个 1,768 字节的文件里，哪 16 个字节属于该函数？清单还显示，偏移 `0x36` 处有 `R_X86_64_PC32 .rodata+0x10`，而文件中该位置的四字节是零。它们会变成什么，为什么当表从 `0x14` 开始时加数是 `0x10`？

rea 用 `inspect_binary_layout` 回答布局问题，而该工具只存在于 Linux x64、pwntools 4.15.0，位于 `REA_PWNTOOLS_PYTHON` 之后。在这台 Mac 上它并不存在。`address_to_file_offset` 需要一个 Ghidra 会话，以及一份 SHA-256 匹配的原始文件字节完整副本。`inspect_artifact` 按路径和内容哈希列出容器中有什么。它从不打开目标文件内部的表。于是你用 `struct.unpack`、零依赖，在大约 950 行 Python 中自己写这个读取器。测试将每一张表与一次性捕获进 fixture 的 `llvm-objdump` 文本进行比对。

```figure
elf-byte-map
```

## 文件是指向表的表

System V gABI 第 4 章图 4-1 画出了同一个文件的两种视图。链接视图包含 ELF 头、可选的程序头表、节，以及必需的节头表。执行视图包含 ELF 头、必需的程序头表、段，以及可选的节头表。gABI 还加了一句话，统摄整个读取器：「只有 ELF 头在文件中有固定位置」。其他每一个结构都要通过存放在另一个结构中的偏移来找到。

gABI 还固定了标量类型，并指出「ELF 不使用位域」。`Elf64_Addr` 与 `Elf64_Off` 是 8 字节，`Elf64_Half` 是 2 字节，`Elf64_Word` 是 4 字节，`Elf64_Xword` 是 8 字节。x86-64 psABI 第 4.1.1 节要求 `ELFCLASS64`、`ELFDATA2LSB`、`e_machine` 为 `EM_X86_64`（62），且 `e_flags` 为 0。小端加固定宽度，意味着带 `<` 前缀的 `struct.unpack` 一次调用就能读取每条记录。

`ledger-x86_64-O1.o` 的前 64 字节：

```text
00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............
00000010: 0100 3e00 0100 0000 0000 0000 0000 0000  ..>.............
00000020: 0000 0000 0000 0000 2804 0000 0000 0000  ........(.......
00000030: 0000 0000 4000 0000 0000 4000 0b00 0100  ....@.....@.....
```

对照 gABI 的「ELF Header」页面来读。`e_ident` 是 16 个字节：`7f 45 4c 46`（魔数）、class 2（64 位）、data 1（小端）、version 1、OS ABI 0（System V），以及填充。

接着 `e_type` `0x0001` 是 `ET_REL`，即可重定位目标文件。`e_machine` `0x003e` 是 62，即 `EM_X86_64`。`e_entry` 为零，因为目标文件没有入口点。`e_phoff` 为零，因为目标文件没有程序头。`e_shoff` 是 `0x428`，即节头表的文件偏移。最后一行包含 `e_ehsize` `0x40`、`e_phentsize` 0、`e_phnum` 0、`e_shentsize` `0x40`、`e_shnum` `0x0b`（11 个节头），以及 `e_shstrndx` `0x0001`。

读取器把这一行变成一次调用：

```python
fields = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
```

格式字符串就是从左到右读 gABI 图：两个半字、一个字、三个 8 字节值、一个字、六个半字。读取器会拒绝魔数、class 或字节序不符的文件，也会拒绝 `e_shentsize` 或 `e_phentsize` 不是 64 和 56 的文件。宽度一旦错误，后续所有偏移都会错，所以这项检查放在最前面。

## 节头与两张字符串表

每个节头是一个 64 字节的 `Elf64_Shdr`（gABI 图 4-8）：`sh_name`、`sh_type`、`sh_flags`、`sh_addr`、`sh_offset`、`sh_size`、`sh_link`、`sh_info`、`sh_addralign`、`sh_entsize`。格式字符串是 `<IIQQQQIIQQ`。`sh_name` 是索引为 `e_shstrndx` 的节内的偏移，名称在第一个零字节处结束。在这个目标文件中 `e_shstrndx` 是 1，clang 把该节命名为 `.strtab`，所以一张表同时服务节名与符号名。

读取器打印 -O1 目标文件的 11 个节头：

```text
Idx Name             Type         Flags   Offset     Size       Addr Link Info Align
  0                  NULL                    0x0      0x0        0x0    0    0     0
  1 .strtab          STRTAB                0x368     0xbe        0x0    0    0     1
  2 .text            PROGBITS     AX        0x40     0x9f        0x0    0    0    16
  3 .rela.text       RELA         I        0x2a8     0x48        0x0   10    2     8
  4 .rodata          PROGBITS     A         0xe0     0x50        0x0    0    0     8
  5 .rela.rodata     RELA         I        0x2f0     0x78        0x0   10    4     8
  6 .rodata.str1.1   PROGBITS     AMS      0x130      0xa        0x0    0    0     1
  7 .comment         PROGBITS     MS       0x13a     0x2f        0x0    0    0     1
  8 .note.GNU-stack  PROGBITS              0x169      0x0        0x0    0    0     1
  9 .llvm_addrsig    LLVM_ADDRSIG E        0x368      0x0        0x0   10    0     1
 10 .symtab          SYMTAB                0x170    0x138        0x0    1    9     8
```

索引 0 是保留的空节头。`.text` 在文件偏移 `0x40` 处存放 159 字节代码，紧跟在头之后，标志为 `A`（在内存中分配）和 `X`（可执行）。`.rodata` 存放 80 字节只读数据。`.rodata.str1.1` 存放五个标签字符串 `F\0D\0C\0B\0A\0`，共 10 字节，标志为 `M`（可合并）和 `S`（字符串）。

`.comment` 存放 47 字节的编译器标记 `\0Apple clang version 17.0.0 (clang-1700.6.3.2)\0`。`.llvm_addrsig` 的类型是 `0x6fff4c03`，落在 gABI 保留的 OS 专用范围 `SHT_LOOS` 到 `SHT_HIOS` 内，LLVM 的 `ELF.h` 为它命名。它的标志 `E` 是 `SHF_EXCLUDE`，所以链接器会将其丢弃。

`sh_link` 和 `sh_info` 两列是表与表之间的指针。`.rela.text` 的 `sh_info` 为 2、`sh_link` 为 10：它修补第 2 个节，并从第 10 个节取符号。`.symtab` 的 `sh_link` 为 1，即符号名所用的字符串表；`sh_info` 为 9，即第一个非局部符号的索引。每一行的 `sh_addr` 都是零，因为目标文件尚无地址。地址由链接器分配。

gABI 的两条规则让读取器能画出完整的文件图。节「连续且不重叠」，文件「可以包含非活动空间」。读取器按偏移对所有结构排序，并填补空洞：

```text
     0x0 +64    header   ELF header
    0x40 +159   section  .text
    0xdf +1     gap      inactive space
    0xe0 +80    section  .rodata
   0x130 +10    section  .rodata.str1.1
   0x13a +47    section  .comment
   0x169 +7     gap      inactive space
   0x170 +312   section  .symtab
   0x2a8 +72    section  .rela.text
   0x2f0 +120   section  .rela.rodata
   0x368 +190   section  .strtab
   0x426 +2     gap      inactive space
   0x428 +704   table    section header table
```

三个空洞合计 10 字节，每个都是对齐造成的。`.rodata` 的 `sh_addralign` 为 8，所以从 `0xe0` 而不是 `0xdf` 开始。`.symtab` 的对齐为 8，所以 `.comment` 之后有七个字节未被使用。节头表本身按 8 对齐。

类型为 `SHT_NOBITS` 的节不占任何文件字节，所以映射图会跳过它们。这个目标文件里没有这种类型的节。链接后的程序中的 `.bss` 属于此类，若读取器在那里相信 `sh_size`，就会读到文件末尾之外。

## 符号：绑定、类型、所属节、值、大小

一个 `Elf64_Sym` 是 24 字节（gABI 图 4-18）：`st_name`、`st_info`、`st_other`、`st_shndx`、`st_value`、`st_size`，格式为 `<IBBHQQ`。`st_info` 打包了两个字段：高四位是绑定（binding），低四位是类型。gABI 将绑定命名为 `STB_LOCAL` 0、`STB_GLOBAL` 1、`STB_WEAK` 2，将类型命名为 `STT_NOTYPE` 0、`STT_OBJECT` 1、`STT_FUNC` 2、`STT_SECTION` 3、`STT_FILE` 4。`st_other` 的低两位保存可见性，这里自始至终都是 `STV_DEFAULT` 0。

```text
Idx            Value   Size Bind   Type    Section          Name
  0              0x0      0 LOCAL  NOTYPE  *UND*
  1              0x0      0 LOCAL  FILE    *ABS*            ledger.c
  2              0x0     20 LOCAL  OBJECT  .rodata          labels.rel
  3              0x0      0 LOCAL  SECTION .rodata          .rodata
  4              0x2      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.1
  5              0x4      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.2
  6              0x6      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.3
  7              0x8      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.4
  8              0x0      0 LOCAL  SECTION .rodata.str1.1   .rodata.str1.1
  9              0x0     16 GLOBAL FUNC    .text            percent_change
 10             0x10     46 GLOBAL FUNC    .text            grade
 11             0x40     31 GLOBAL FUNC    .text            total
 12             0x60     63 GLOBAL FUNC    .text            grade_label
```

索引 0 是保留的未定义符号。`st_shndx` 指明定义该符号的节，有三个保留值：`SHN_UNDEF` 0、用于文件名的 `SHN_ABS` `0xfff1`，以及 `SHN_COMMON` `0xfff2`。节符号的名字为空，所以 `llvm-objdump` 为它们打印节自身的名称，读取器也这样做。gABI 规则「所有 STB_LOCAL 绑定的符号都排在弱符号和全局符号之前」体现在前九行中，`.symtab` 的 `sh_info` 记录为 9 正是这个原因。

在可重定位目标文件中，`st_value` 是符号所在节内的偏移，因为 `sh_addr` 为零。因此函数的文件偏移是 `sh_offset + st_value - sh_addr`，长度是 `st_size`：

```python
section = elf.sections[symbol.st_shndx]
file_offset = section.sh_offset + (symbol.st_value - section.sh_addr)
code = elf.data[file_offset : file_offset + symbol.st_size]
```

对 `percent_change` 来说就是 `0x40 + 0`，16 个字节为 `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`。大小与起点同样重要。`grade` 结束于 `.text` 偏移 `0x3e`。紧随其后的两字节 `66 90` 是编译器添加的 `nop`，好让 `total` 从 `0x40` 这个 16 字节边界开始。这两个字节不属于任何符号。第 33 课把 `st_size` 作为反汇编器的停止条件，第 36 课在比较重新构建的结果之前会把它们屏蔽掉。

## 重定位：那些零字节意味着什么

一个 `Elf64_Rela` 是 24 字节（gABI 图 4-23）：`r_offset`、`r_info` 和有符号的 `r_addend`，格式为 `<QQq`。gABI 用两个宏拆分 `r_info`：`ELF64_R_SYM(i)` 是 `i >> 32`，`ELF64_R_TYPE(i)` 是 `i & 0xffffffff`。在可重定位文件中，`r_offset` 是表的 `sh_info` 所指节内的偏移。类型名称与公式来自 x86-64 psABI 表 4.9。其中 `S` 是符号值，`A` 是加数，`P` 是被修补的位置，`L` 是 PLT 条目。`R_X86_64_64` 为 `S + A`，`R_X86_64_PC32` 为 `S + A - P`，`R_X86_64_PLT32` 为 `L + A - P`。

-O1 目标文件有八条记录：

```text
  .rela.text       0x36 R_X86_64_PC32          .rodata+0x10
  .rela.text       0x84 R_X86_64_PC32          .rodata+0x24
  .rela.text       0x93 R_X86_64_PC32          .rodata-0x4
  .rela.rodata      0x0 R_X86_64_PC32          .rodata.str1.1
  .rela.rodata      0x4 R_X86_64_PC32          .L.str.1+0x4
  .rela.rodata      0x8 R_X86_64_PC32          .L.str.2+0x8
  .rela.rodata      0xc R_X86_64_PC32          .L.str.3+0xc
  .rela.rodata     0x10 R_X86_64_PC32          .L.str.4+0x10
```

拆解第一条。偏移 `0x36` 位于 `grade` 内，该函数跨 `0x10` 到 `0x3e`。`0x33` 处的指令是 `48 8d 0d 00 00 00 00`，一条 RIP 相对寻址的 `lea`，其四字节位移位于 `0x36`。处理器把位移加到下一条指令的地址 `0x3a` 上。

编译器希望结果是 `.rodata + 0x14`，那里开始一张含五个 32 位值 `1, 2, 3, 4, 4` 的表。因此存储值必须是 `(.rodata + 0x14) - 0x3a`。公式 `S + A - P` 在 `P = 0x36` 时给出 `.rodata + A - 0x36`。令两者相等得到 `A = 0x14 - 4 = 0x10`。加数携带的是目标偏移减去位移与指令结尾之间的四个字节。

这张表就是 clang 在 -O1 下编译 `grade` 中 `switch` 的方式。代码用乘以 `0x66666667` 加两次移位计算 `score / 10`，减去 6，与 4 比较，然后索引该表。-O1 的 ELF 目标文件没有为该表保留符号。引用由节符号 `.rodata` 加一个加数构成，只有同一份源码的 arm64 构建把它命名为 `l_switch.table.grade`。第二条记录位于 `grade_label` 的 `0x84`，指向 `.rodata + 0x28` 处另一张 8 字节值表。第三条位于 `0x93`，指向 `.rodata + 0x0`，即 `labels.rel` 的起点。

五条 `.rela.rodata` 记录填充 `labels.rel`。每个条目位于偏移 `4i`，加数为 `4i`，所以 `S + A - P` 化简为字符串地址减去表的基址。`grade_label` 用 `movslq` 读取条目，再加回表基址，就得到了指针。第一条记录以加数 0 命名节符号 `.rodata.str1.1`，因为字符串 `F` 是该节的第一个字节。其余记录命名 `.L.str.1` 到 `.L.str.4`，即编译器为 `D`、`C`、`B`、`A` 生成的局部标签。

同一份源码在三个优化级别下编译成三张不同的表。读取器无需反汇编器就能显示差异：

| 构建 | `labels` 所在位置 | `grade` 如何选择 | `grade_label` 如何到达 `grade` |
|---|---|---|---|
| -O0 | `.data`，五个 8 字节指针，五条 `R_X86_64_64` 记录 | `.rodata` 中的跳转表：五条指向 `.text` 的 `R_X86_64_PC32` 记录 | 带 `R_X86_64_PLT32 grade-0x4` 的调用 |
| -O1 | `.rodata` 中的 `labels.rel`，五个 32 位相对条目 | `.rodata+0x14` 处的值表，一条 `R_X86_64_PC32` | 内联，无调用 |
| -O2 | 与 -O1 相同 | 与 -O1 相同 | 内联，`total` 增长到 140 字节 |

在 -O0 下，`switch` 是真正的跳转表，五个条目指回 `.text` 的 `+0x91`、`+0x8c`、`+0x87`、`+0x82` 和 `+0x86`。对 `grade` 的调用带有同样的 `-0x4` 加数，原因与 `lea` 相同。在 -O1 下，优化器看出每个分支都返回常量，于是用一张结果表替换了跳转。第 33 课会解码这两种形态。

## 程序头：加载器的视角

可重定位目标文件没有程序头，所以 `e_phnum` 为 0。可执行文件或共享目标文件则有，而且加载器不读其他任何内容。一个 `Elf64_Phdr` 是 56 字节（gABI 图 5-1）：`p_type`、`p_flags`、`p_offset`、`p_vaddr`、`p_paddr`、`p_filesz`、`p_memsz`、`p_align`，格式为 `<IIQQQQQQ`。gABI 为 `PT_LOAD` 条目固定了三条规则。它们按 `p_vaddr` 升序出现；`p_memsz` 大于 `p_filesz` 意味着零填充。`p_vaddr` 与 `p_offset` 在模 `p_align` 下同余。

本路线的每个目标都是你自己构建的程序，而这台 Mac 上的链接器无法从 `ledger.o` 生成 Linux 可执行文件。因此 fixture `tiny-x86_64` 来自 `code/fixtures/build_tiny_elf.py`，它用 `struct.pack` 写出 736 字节。文件包含一个 ELF 头、两个程序头、一个 36 字节的 `_start`、3 字节的 `.rodata`、一张符号表，以及六个节头。`_start` 向文件描述符 1 写入 `ok\n`，并以状态 42 退出。在 Linux x86-64 上它可以运行。在这里，它是程序头路径的 fixture。

```text
Program headers:
  PT_LOAD          r-x offset=0x0 vaddr=0x400000 filesz=0x2e0 memsz=0x2e0 align=0x1000
  PT_GNU_STACK     rw- offset=0x0 vaddr=0x0 filesz=0x0 memsz=0x0 align=0x10

entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

唯一的 `PT_LOAD` 把整个 736 字节文件映射到 `0x400000`。读取器把地址映射到文件偏移的方法是：找到包含该地址的 `PT_LOAD`，计算 `p_offset + (address - p_vaddr)`。对入口点来说就是 `0 + (0x4000b0 - 0x400000) = 0xb0`，即 `.text` 的起点。

这就是 rea 的 `address_to_file_offset` 背后的计算。对于 Ghidra 目标，该工具返回 `{address, file_offset}`。Hopper provider 还会返回 `provider_file_offset` 和 `image_base_file_offset`。该工具需要一份完整的原始文件副本，且副本的 SHA-256 必须与目标一致。rea 的 `inspect_binary_layout` 文档中有一句值得记住的话：「链接地址不是运行时地址」。位置无关可执行文件加载到内核选择的基址上，文件偏移是唯一稳定的坐标。

程序头也解释了为什么剥离过的可执行文件仍能运行。链接器可以完全移除节头表，`e_shoff` 随之变为零。遍历节的工具于是什么也看不到。加载器从未看过它们。

## Mach-O 中的相同概念

Apple 没有为 Mach-O 发布文字规范。XNU 源码中的头文件就是规范：`EXTERNAL_HEADERS/mach-o/loader.h`、`nlist.h` 和 `reloc.h`。fixture `ledger-arm64-O1.o` 是同一份 `ledger.c` 用 `clang -target arm64-apple-macos -O1 -fno-asynchronous-unwind-tables -c` 构建的结果。

一个 `mach_header_64` 是 32 字节：`magic`、`cputype`、`cpusubtype`、`filetype`、`ncmds`、`sizeofcmds`、`flags`、`reserved`。魔数是 `0xfeedfacf`，即 `MH_MAGIC_64`。`cputype` 是 `0x0100000c`，即 `CPU_TYPE_ARM64`。`filetype` 是 1，即 `MH_OBJECT`。`ncmds` 为 5，`sizeofcmds` 为 616：加载命令占据头之后的 616 字节，每条都以 `{cmd, cmdsize}` 开头。标志 `0x2000` 是 `MH_SUBSECTIONS_VIA_SYMBOLS`。

```text
Load commands:
  0 LC_SEGMENT_64                cmdsize=472 at 0x20
  1 LC_BUILD_VERSION             cmdsize=24 at 0x1f8
  2 LC_LINKER_OPTIMIZATION_HINT  cmdsize=16 at 0x210
  3 LC_SYMTAB                    cmdsize=24 at 0x220
  4 LC_DYSYMTAB                  cmdsize=80 at 0x238
```

`LC_SEGMENT_64`（0x19）是 Mach-O 对两张 ELF 表的回答。它的 `segment_command_64` 包含 `segname`、`vmaddr`、`vmsize`、`fileoff`、`filesize`、`maxprot`、`initprot`、`nsects` 和 `flags`。随后在同一命令中跟着 `nsects` 条类型为 `section_64` 的记录，每条 80 字节。`loader.h` 中关于 `MH_OBJECT` 的注释说，节位于「一个没有段填充的未命名段中」。fixture 正是如此：一个名字为空的段，`fileoff` 为 648，`filesize` 为 456，以及五个节。

```text
  Idx Section                    Addr   Size Offset Reloff Nreloc Type
    1 __TEXT,__text               0x0   0xd0    648   1104      6 S_REGULAR
    2 __DATA,__const             0xd0   0x28    856   1152      5 S_REGULAR
    3 __TEXT,__cstring           0xf8    0xa    896      0      0 S_CSTRING_LITERALS
    4 __TEXT,__const            0x108   0x40    912      0      0 S_REGULAR
    5 __LD,__compact_unwind     0x148   0x80    976   1192      4 S_REGULAR
```

每个节在 `segname` 中指明其父段，所以两个都叫 `__const` 的节是不同的，因为一个是 `__DATA,__const`，另一个是 `__TEXT,__const`。重定位按节通过 `reloff` 和 `nreloc` 组织，而 ELF 中每个目标节都有单独的 `.rela.X` 节。

每条 `relocation_info` 是 8 字节。先是有符号的 `r_address`。随后是一个 32 位字段，打包为 `r_symbolnum`（24 位）、`r_pcrel`（1 位）、`r_length`（2 位）、`r_extern`（1 位）和 `r_type`（4 位）。六条 `__text` 记录是三对 `ARM64_RELOC_PAGE21` 与 `ARM64_RELOC_PAGEOFF12`，每个表引用一对。arm64 的 `adrp` 加 `add` 完成一条 x86-64 `lea` 的工作，所以每个引用需要两条记录。五条 `__DATA,__const` 记录是 `ARM64_RELOC_UNSIGNED`，`r_length` 为 3，每条修补一个 8 字节绝对指针。

在 -O1 下，arm64 构建把 `labels` 保留为普通指针，而 x86-64 构建则做成了相对表。

`LC_SYMTAB`（0x2）包含 `symoff` 1240、`nsyms` 17、`stroff` 1512、`strsize` 176，而 1512 加 176 等于 1688，即文件大小。每条 `nlist_64` 是 16 字节：`n_strx`、`n_type`、`n_sect`、`n_desc`、`n_value`。`n_type` 是位字段。`N_TYPE`（掩码 `0x0e`）等于 `N_SECT`（`0xe`）表示定义在序号（从 1 起）为 `n_sect` 的节中。`N_EXT`（`0x01`）标记外部符号。

`nlist_64` 没有大小字段。要得到 `_percent_change` 的 24 字节，读取器取同一节中的下一个符号地址，即 `0x18` 处的 `_grade`。这里的 C 符号带有前导 `_`，所以函数名是 `_percent_change`。

`LC_DYSYMTAB`（0xb）记录了与 ELF `sh_info` 相同的局部符号在前规则：`ilocalsym` 0、`nlocalsym` 13、`iextdefsym` 13、`nextdefsym` 4。`LC_BUILD_VERSION`（0x32）以打包的 `XXXX.YY.ZZ` 整数记录 `platform` 1（macOS）、`minos` 26.0 和 `sdk` 26.2。这条命令正是 arm64 fixture 以字节形式提交的原因。在 SDK 不同的 Mac 上构建会产生不同的文件。

读取器的 `compare_formats` 把两个文件并排显示：

| 概念 | ELF，`ledger-x86_64-O1.o` | Mach-O，`ledger-arm64-O1.o` |
|---|---|---|
| 文件标识 | `e_ident` `7f 45 4c 46`，class 64，data LSB | `magic` `0xfeedfacf` |
| 机器 | `e_machine` 62，`EM_X86_64` | `cputype` `0x100000c`，`CPU_TYPE_ARM64` |
| 文件种类 | `e_type` 1，`ET_REL` | `filetype` 1，`MH_OBJECT` |
| 表的位置 | `e_shoff` `0x428`，11 个节头 | 5 条加载命令，头之后 616 字节 |
| 代码 | `0x40` 处的 `.text`，大小 `0x9f`，标志 `AX` | 648 处的 `__TEXT,__text`，大小 `0xd0`，位于未命名段中 |
| 符号名 | `.symtab`（第 10 节），名称经 `sh_link` 1 指向 `.strtab` | `LC_SYMTAB` `symoff` 1240，`nsyms` 17，名称位于 `stroff` 1512 |
| 符号大小 | `st_size`：`percent_change` 为 16 字节 | 无此字段：下一个符号减去当前符号，24 字节 |
| 重定位 | `.rela.text`，3 条记录，`R_X86_64_PC32` | `__text` `reloff` 1104，6 条记录，`PAGE21` 与 `PAGEOFF12` |
| 函数字节 | `percent_change`：`0x40` 处的 16 字节 | `_percent_change`：`0x288` 处的 24 字节 |

rea 通过 `inspect_macho` 读取 Mach-O，它经由 `xcrun` 运行 `otool -h -l`、`nm -gjU`、`dyld_info -imports -exports` 和 `dwarfdump --uuid`。你的读取器不需要其中任何一个，它读取的 `symoff` 和 `stroff` 与 `otool -l` 打印的相同。rea 的 `open_binary` 工具从魔数字节识别 ELF、MZ、PE 以及 thin 或 universal Mach-O。它自己也读取偏移 18 处的 `e_machine`。那正是你在这里构建内容的前 20 字节。对于没有头的文件，你要提供 `format: raw-binary`、架构和入口地址。Ghidra provider 会拒绝通用二进制并要求瘦身切片。你的读取器在看到 `0xcafebabe` 时也这样做。

## 构建它

运行标准库读取器：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
```

示例读取 `ledger-x86_64-O1.o`，打印其头、节、符号、重定位和文件偏移图。然后打印四个函数的字节串，读取 `ledger-arm64-O1.o`，并打印对照表。最后以 `tiny-x86_64` 的程序头结束。预期输出行：

```text
percent_change: 16 bytes at file offset 0x40: 85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3
grade: 46 bytes at file offset 0x50: 48 63 c7 48 69 c0 67 66 66 66 ...
entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

传入路径即可读取单个文件：`python3 code/main.py code/fixtures/ledger-x86_64-O0.o`。读取器根据前四个字节选择 ELF 或 Mach-O。

按以下顺序阅读 `code/main.py`：

1. `ElfFile._read_header()` 检查魔数、class、字节序和条目大小，然后解包头。
2. `ElfFile._read_sections()` 和 `read_cstring()` 读取节头表，并通过 `e_shstrndx` 解析名称。
3. `ElfFile._read_symbols()` 拆分 `st_info`，通过 `sh_link` 解析名称，并让保留的节索引保持可读。
4. `ElfFile._read_relocations()` 遍历每个 `SHT_RELA` 节，拆分 `r_info`，并按表 4.9 命名类型。
5. `ElfFile.function_span()` 和 `function_bytes()` 把符号转换为文件范围和其中的字节。
6. `ElfFile.file_offset_map()` 按偏移排序所有结构，并填补非活动空间。
7. `ElfFile._read_program_headers()` 和 `address_to_file_offset()` 处理可执行文件。
8. `MachOFile` 读取头、加载命令、带节的 `LC_SEGMENT_64`、`LC_SYMTAB`、`LC_BUILD_VERSION` 以及按节的重定位。
9. `compare_formats()` 用从两个文件中读到的值，为同一个概念命名。

## 使用它

在把本课的 `code` 目录加入 `sys.path` 之后，后续课程可以用 `from main import ElfFile` 导入该读取器。第 33 课向它索取 `function_bytes` 和 `relocations_for(".text")`。第 35 课把它的输出记录为观测证据。第 36 课向它索取在字节比较前需要屏蔽的重定位偏移。

用同样的 `clang -target x86_64-unknown-linux-gnu -c` 命令把你自己的 C 编译成目标文件，并在其上运行读取器。只有当 x86-64 psABI 表 4.9 列出某个重定位名称时才添加它。只有当 gABI 或产出它的工具链有文档时，才添加某个节类型。当读取器遇到没有名字的值时，它打印十六进制数字，而不是猜测。

## 交互实验

### 实验 A：手工读文件头

在 `code` 目录中打开 Python：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch/code
python3 -q
```

运行：

```python
import struct
from main import ElfFile, FIXTURES
data = (FIXTURES / "ledger-x86_64-O1.o").read_bytes()
data[:4], data[4], data[5]
struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
elf = ElfFile(data)
elf.header.e_shoff, elf.header.e_shnum, elf.sections[elf.header.e_shstrndx].name
```

元组显示 `e_type` 1、`e_machine` 62，以及 `e_shoff` 1064，即 `0x428`。最后一行表明第 1 个节 `.strtab` 保存节名。

### 实验 B：switch 背后的值表

```python
rodata = elf.section_bytes(elf.section(".rodata"))
[int.from_bytes(rodata[i : i + 4], "little") for i in range(0x14, 0x28, 4)]
[int.from_bytes(rodata[i : i + 8], "little") for i in range(0x28, 0x50, 8)]
[r for r in elf.relocations_for(".text") if r.symbol == ".rodata"]
```

两张表读出来都是 `1, 2, 3, 4, 4`。对照 `ledger.c` 验证：`grade(65)` 计算 `65 / 10 - 6 = 0`，读取条目 0，即 1，也就是 `case 6` 的值。`grade(100)` 读取条目 4，即 4，是 `case 10` 与 `case 9` 共用的值。

### 实验 C：同一份源码的三种构建

```python
for level in ("O0", "O1", "O2"):
    obj = ElfFile.from_path(FIXTURES / f"ledger-x86_64-{level}.o")
    print(level, [s.name for s in obj.sections if s.name in (".data", ".rodata")])
    print(level, sorted({r.type_name for r in obj.relocations}))
    print(level, {s.name: s.st_size for s in obj.symbols if s.type == "FUNC"})
```

-O0 目标文件有一个 `.data` 节和三种重定位类型。-O1 和 -O2 目标文件没有 `.data`，只有一种类型。`total` 在 -O0 下是 66 字节，-O1 下 31 字节，-O2 下 140 字节。

### 实验 D：Mach-O 中的同一个函数

```python
from main import MachOFile
macho = MachOFile.from_path(FIXTURES / "ledger-arm64-O1.o")
[(c.name, c.cmdsize) for c in macho.load_commands]
macho.function_span("_percent_change")
macho.function_bytes("_percent_change")[:4].hex()
```

范围是文件偏移 648 处的 24 字节。前四个字节是 `a0000034`，即指令 `0x340000a0` 的小端形式，`llvm-objdump -d` 将其打印为 `cbz w0, 0x14`。这就是 `before == 0` 的守卫，与 x86-64 构建编码为 `85 ff 74 09` 的是同一个判断。

## 实践练习

把读取器扩展到链接后的程序。在 Linux x86-64 机器上，构建 `clang -O1 -shared -fPIC code/fixtures/ledger.c -o libledger.so` 和 `clang -O1 code/fixtures/ledger.c -o ledger_main`（后者需自行添加 `main`），然后在两者上运行读取器。

要求：

- 用同样的 `Elf64_Sym` 布局读取 `.dynsym`，因为 `SHT_DYNSYM` 是 11，记录格式不变。
- 把 `.dynamic` 读取为 `d_tag` 和 `d_val` 的 `Elf64_Dyn` 记录，并按 gABI「Dynamic Section」页面命名 `DT_NEEDED`、`DT_STRTAB`、`DT_SYMTAB`、`DT_RELA` 和 `DT_JMPREL`。
- 把 `st_shndx` 为 `SHN_UNDEF` 的未定义符号列为导入，把 `.rela.plt` 中的 `R_X86_64_JUMP_SLOT` 记录列为到达它们的调用点。
- 把每个 `PT_LOAD` 段映射到它包含的节，使用 gABI 规则「一个目标文件段包含一个或多个节」。
- 用 `address_to_file_offset` 证明共享对象中每个已定义函数的字节，就是 `function_bytes` 返回的字节。
- 把两个文件的 `llvm-objdump -h -t -r -p` 与 `llvm-objdump -T` 文本捕获到 `code/fixtures/`，并在测试中与读取器比对。

成功意味着导入列表、段映射和每个函数范围都与捕获的工具输出一致，且测试期间不运行任何工具。

## 评分实验

关于本课 fixture 的五个问题。用你构建的代码找出每个答案，然后写入 `code/lab_answers.json`。数字写成数字或 `"0x40"` 这样的十六进制字符串，列表写成 JSON 列表。

1. `t1`：ledger-x86_64-O1.o 的 e_shoff，以数字表示。
2. `t2`：ledger-x86_64-O1.o 中 percent_change 的 st_size。
3. `t3`：ledger-x86_64-O1.o 中 .text 的文件偏移，用十六进制。
4. `t4`：ledger-x86_64-O0.o 中的重定位类型名称，排序后以列表表示。
5. `t5`：ledger-x86_64-O1.o 中 .rodata+0x14 处的五个 int 值，以列表表示。

检查你的答案：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/lab.py
```

检查器根据已提交的 fixture，用 `code/main.py` 计算每个期望值。它为每个任务打印 `PASS`、`FAIL` 或 `TODO` 以及分数，只有五项全部通过时才以 0 退出。`python3 code/lab.py my-answers.json` 检查另一个文件。`code/tests/test_lab.py` 证明检查器接受正确答案并拒绝错误答案。

### 看它运行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) 这台机器把本课作为它的前两个站点运行。转盘读取包含 `percent_change` 的目标文件的头，机柜打开 `.text` 并读取符号。关闭 Symbols 开关，看看符号表携带了哪些事实。

## 课程产物

本课产出 `outputs/skill-program-anatomy-elf-from-scratch.md`。可与 Claude、Cursor、Codex、OpenClaw、Hermes 或任何读取 SKILL.md 的智能体配合使用。该技能把用户自行构建的编译目标文件或可执行文件转换成布局报告。报告给出每个结构、符号和重定位的文件范围，并与捕获的工具输出核对。

## 验证它

运行示例与确定性测试套件：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

测试套件证明：

- 三个 x86-64 目标文件和 tiny 可执行文件的每个节索引、名称、大小、地址和类型，都与捕获的 `llvm-objdump -h` 文本一致
- 每个符号的值、绑定、类型、所属节、大小和名称，都与捕获的 `llvm-objdump -t` 文本一致
- 每条重定位的偏移、类型名称、符号和加数，都与捕获的 `llvm-objdump -r` 文本一致，包括 -O0 下的 `R_X86_64_PLT32` 和 `R_X86_64_64`
- `tiny-x86_64` 的两个程序头与捕获的 `llvm-objdump -p` 文本一致，而目标文件没有程序头
- `percent_change` 是文件偏移 `0x40` 处的 16 字节，字节符合预期，且每个函数都位于 `.text` 内
- 文件偏移图恰好覆盖每个 fixture 的每个字节一次，-O1 对象的三个对齐空洞各就各位
- `tiny-x86_64` 的入口点经 `PT_LOAD` 映射到文件偏移 `0xb0`，而可重定位目标文件拒绝地址映射
- Mach-O 头、每条加载命令、段、五个节、`LC_SYMTAB` 和 `LC_BUILD_VERSION` 都与捕获的 `--private-headers` 文本一致
- Mach-O 符号和 15 条重定位与捕获的 `llvm-objdump` 文本一致，`_percent_change` 是偏移 648 处的 24 字节
- 错误的魔数、32 位 class、大端文件、截断的表、通用二进制和 C 源文件都会抛出 `FormatError`
- `build_tiny_elf.py` 逐字节重现已提交的 `tiny-x86_64`
- 图示模块 `site/figures-re-32.js` 携带 `ledger-x86_64-O1.o` 的精确字节

## 生产环境失败模式

| 失败 | 弱读取报告的内容 | 读取器必须证明的内容 |
|---|---|---|
| 把 `st_value` 当作文件偏移 | 「`percent_change` 在 0」 | 偏移为 `sh_offset + st_value - sh_addr`，这里是 `0x40` |
| 把填充当作代码 | 「`grade` 是 48 字节」 | `st_size` 是 46，其后的 `66 90` 不属于任何符号 |
| 从错误的表读取节名 | 乱码名称或异常 | `e_shstrndx` 指定表，这里是第 1 节 |
| 对 `SHT_NOBITS` 相信 `sh_size` | 读取越过文件末尾 | `.bss` 没有文件字节，映射图跳过它 |
| 把零字节读成操作数 | 「`lea 0(%rip)`」 | `0x36` 处的 `.rela.text` 记录指名 `.rodata+0x10` |
| 把加数读成目标偏移 | 「表从 `0x10` 开始」 | 结合指令结尾的 `S + A - P` 得出 `0x14` |
| 在已剥离的可执行文件上遍历节 | 「文件没有代码」 | `PT_LOAD` 映射加载器使用的每个地址 |
| 把链接地址当作运行时地址 | 随机基址下的错误偏移 | 来自 `PT_LOAD` 的文件偏移是稳定坐标 |
| 仅用 `sectname` 查找 Mach-O 节 | 两个 `__const` 节合二为一 | 名称是 `segname,sectname` |
| 期望 `nlist_64` 有大小字段 | 异常或零字节 | 同一节中的下一个符号结束该函数 |
| 把通用二进制读成单个 Mach-O | 「魔数 `0xcafebabe` 未知」 | 拒绝它并要求瘦身切片 |
| 在另一台 Mac 上重建 arm64 目标文件 | 「fixture 变了」 | `LC_BUILD_VERSION` 记录 SDK，所以提交的字节就是 fixture |

## 操作准则

读取你信得过的头，跟随其中保存的偏移，并用定义每个字段的规范为其命名。把比对用的工具输出保留为已提交的 fixture，这样比较无需工具即可运行。当某个值在规范中没有名称时，就打印数字。你能逐字节核对的读取器，才是后续每节课赖以构建的读取器。

## 延伸阅读

- [System V gABI 第 4 章：Object Files](https://www.sco.com/developers/gabi/latest/ch4.intro.html) 以及 [ELF Header](https://www.sco.com/developers/gabi/latest/ch4.eheader.html)、[Sections](https://www.sco.com/developers/gabi/latest/ch4.sheader.html)、[Symbol Table](https://www.sco.com/developers/gabi/latest/ch4.symtab.html)、[Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html) 页面
- [System V gABI 第 5 章：Program Header](https://www.sco.com/developers/gabi/latest/ch5.pheader.html) 与 [Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V ABI，AMD64 架构处理器补充](https://gitlab.com/x86-psABIs/x86-64-ABI)，表 4.9 重定位类型
- [Apple XNU `loader.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)、[`nlist.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h) 与 [`reloc.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/reloc.h)
- [17 U.S.C. 1201](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section1201&num=0&edition=prelim) 与 [指令 2009/24/EC](https://eur-lex.europa.eu/eli/dir/2009/24/oj/eng)
- [Sega v. Accolade, 977 F.2d 1510](https://law.resource.org/pub/us/case/reporter/F2/977/977.F2d.1510.92-15655.html) 与 [Sony v. Connectix, 203 F.3d 596](https://law.resource.org/pub/us/case/reporter/F3/203/203.F3d.596.99-15852.html)
- [rea 6.5.0 `docs/binary-diagnostics.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/binary-diagnostics.md) 与 [`docs/ghidra-provider.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
