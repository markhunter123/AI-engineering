# 链接、加载与剥离符号的二进制文件：PLT、GOT、导入、修正与函数识别

> 一个已链接的程序会调用它自身并不包含的函数。本课用三种格式阅读链接编辑器为这些调用写下的表，跟踪一次调用到达加载器填充的槽位，并在一份构建产物失去名字之后找回它的函数。

**Type:** Build
**Languages:** Python, C, x86-64 and arm64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow), 38 (calling conventions)
**Time:** ~160 minutes

## 学习目标

- 根据一条重定位记录和符号表，判断链接编辑器会把它变成什么：一个固定距离、一个带 GOT 槽的 PLT 条目，还是一条动态重定位。
- 阅读位置无关 ELF 可执行文件的动态段、PLT 和 GOT，并用 psABI 公式跟踪一次调用穿过它们。
- 阅读 COFF 目标文件（20 字节文件头、40 字节节条目、18 字节符号、10 字节重定位），以及 PE32+ 映像的导入、导出和基址重定位目录。
- 逐字解码 Mach-O 的链式修正链，并把每个字与 `dyld_info` 对照。
- 说明 `strip` 会移除什么、每种格式中又有什么存活下来，并运行剥离后的程序证明它仍然可用。
- 用 `LC_FUNCTION_STARTS`、序言扫描和调用目标恢复剥离后的 arm64 构建的函数起点，对照未剥离孪生文件的符号表给每种方法打分，并用 FLIRT 形态的签名给函数命名。

## 问题

你把 `main.c` 链接到 `libledger.dylib`，然后问一个智能体：`main` 在 `0x1000005a0` 调用了哪个函数。智能体反汇编了这条指令，读到 `bl 0x100000694`。那个地址上是十二个字节：`adrp x16`、`ldr x16, [x16, #0x38]`、`br x16`。智能体顺着这次加载读到 `0x100004038`，读出八个字节 `00 00 00 00 00 00 10 80`，然后报告说这是一次经由损坏指针跳入内核空间的调用。

接着你剥离了同一程序的静态构建，问它有多少个函数。智能体运行 `nm`，只看到一个名字 `__mh_execute_header`，于是回答：一个。

两个答案都把文件当成了正在运行的进程来读。`0x100004038` 处的那个字是一个链式修正。第 63 位表示 bind，十二位指出下一个修正的位置，低位说明要绑定哪一个导入。在 `main` 运行之前，dyld 会用 `_puts` 的地址替换这个字。剥离后的文件失去了符号表，却保留了 `LC_FUNCTION_STARTS`，后者用九个字节列出了全部五个函数起点。

本课用标准库 Python 构建五个读取器。一个用于我们自己目标文件的 COFF 读取器，一个用于导入、导出和基址重定位目录的 PE32+ 读取器。一个用于加载命令、存根、间接符号表、链式修正链和导出 trie 的 Mach-O 读取器，并逐字对照 `dyld_info` 校验。一个用于动态段、PLT 和 GOT 的 ELF 读取器，外加一个规划器，说明对我们这些目标文件做 PIE 链接会创建什么。一个用于剥离构建的函数查找器，带四种方法和各自的评分。一个 FLIRT 形态的签名构建器，带通配的重定位字节和尾部的 CRC16。

这台机器上只有 Mach-O 能链接。arm64 可执行文件和那个 dylib 是链接、剥离并运行过的。ELF 和 COFF 目标文件在这里用 `clang -target` 编译，但没有安装 ELF 或 PE 链接编辑器，所以那两种格式的两个已链接映像，是由 fixtures 文件夹里的两个 Python 脚本按照规范逐字节写出的。`llvm-objdump` 能读它们，库探针也能读。它们没有被运行，正文在相关之处会说明这一点。

```figure
re-40-call-through-three-formats
```

## 链接编辑器如何处理一条重定位

第 32 课读过重定位记录，第 33 课把 `S + A - P` 应用到一个 `lea` 上。一条记录指名一个位置 `P`、一个类型、一个符号 `S` 和一个加数 `A`。x86-64 psABI 表 4.9 为每种类型给出一个公式，其中 `L` 表示符号 PLT 条目的位置，`G` 表示它的 GOT 偏移，`GOT` 表示 GOT 地址，`B` 表示映像的加载基址。三种结果覆盖了我们的目标文件。

链接编辑器把值写入字段，记录随即消失。对输入中已定义的符号使用 `R_X86_64_PC32` 就属于这种情况。同一映像中两个字节之间的距离永远不会变，无论加载地址是什么。

链接编辑器创建一个 PLT 条目和一个 GOT 槽。对一个未定义函数使用 `R_X86_64_PLT32` 属于这种情况。调用得到 `L + A - P`，即到该条目的距离。条目再通过槽跳转。`.rela.plt` 中的一条 `R_X86_64_JUMP_SLOT` 记录告诉动态链接器去填充这个槽。

链接编辑器为加载器保留一条重定位。`R_X86_64_64` 是八字节绝对地址，而位置无关可执行文件在加载之前并不知道自己的地址。这条记录在 `.rela.dyn` 中变成公式为 `B + A` 的 `R_X86_64_RELATIVE`。

`main-x86_64-O1.o` 有 18 条记录：14 条在 `.text`，2 条在 `.data.rel.ro`，2 条在 `.rodata`。`code/main.py` 中的 `plan_dynamic_link()` 把这些规则与 `ledger-x86_64-O1.o` 一起应用：

| 记录 | 数量 | 决策 | 原因 |
|---|---|---|---|
| 指向 `.rodata`、`.L.str`、`fns`、`.rodata` 中表的 `R_X86_64_PC32` | 16 | 原地 | 符号在输入中已定义 |
| 指向 `percent_change`、`grade`、`total`、`grade_label` 的 `R_X86_64_PLT32` | 4 | 直接，无 PLT | 在 `ledger-x86_64-O1.o` 中定义，因此链接编辑器把 PLT32 降为 PC32 |
| 两次指向 `puts`、两次指向 `printf` 的 `R_X86_64_PLT32` | 4 | PLT 条目与 GOT 槽 | 未定义，由共享库提供 |
| 指向 `fns[]` 中 `grade` 的 `R_X86_64_64` | 2 | `R_X86_64_RELATIVE` | PIE 中的绝对字段 |

两个导入产生 48 字节的 `.plt`（PLT0 加两个 16 字节条目）、40 字节的 `.got.plt`（三个保留槽加两个），以及 48 字节的 `.rela.plt`。两条相对记录产生 48 字节的 `.rela.dyn`。这些数字来自把规范应用到我们的记录上。手工构建的 `tiny-dyn-x86_64` 为同样两个导入实现了这个计划的 PLT 部分，其尺寸与计划一致。

自 binutils 2.31（H.J. Lu，2018-02-13）和 LLVM 7.0.0（D43383，2018-02-20）起，编译器对每次调用都写 `PLT32`，无论是否定义。由链接编辑器决定。因此目标文件中的 `PLT32` 只表示「一次调用」，对是否会有 PLT 条目只字不提。其他构建中还会出现另外两种类型。`R_X86_64_GOTPCRELX` 和 `R_X86_64_REX_GOTPCRELX`（`G + GOT + A - P`）要求一个不带 PLT 的 GOT 槽，`-fno-plt` 会产生它们；当符号是局部符号时，链接编辑器可以把它们放松为直接的 `lea` 或 `call`（psABI 附录 B）。`R_X86_64_32` 和 `R_X86_64_32S` 在 PIE 中无法容纳地址，链接会失败。这两种都不出现在我们的两个目标文件中。

## 动态段、PLT 与 GOT

gABI 第 5 章说，参与动态链接的可执行文件有一个 `PT_INTERP` 程序头，它必须排在所有可加载段之前；还有一个 `PT_DYNAMIC`，覆盖 `.dynamic` 节。该节是一个 `Elf64_Dyn` 对数组，每对一个有符号 tag 和一个值，以 `DT_NULL` 结束。

`tiny-dyn-x86_64` 有 5,184 字节，由 `build_tiny_dyn_elf.py` 依据 gABI 和 psABI 写出。它的动态段在 `0x1000` 处有十二对：

| Tag | 编号 | 值 | 含义 |
|---|---|---|---|
| `DT_NEEDED` | 1 | `libc.so.6` | 一个字符串表偏移 `0xb`，要加载的库 |
| `DT_HASH` | 4 | `0x1a0` | SysV 哈希表 |
| `DT_STRTAB` | 5 | `0x188` | `.dynstr` |
| `DT_SYMTAB` | 6 | `0x140` | `.dynsym`，每项 24 字节 |
| `DT_STRSZ` | 10 | `0x15` | 21 字节字符串 |
| `DT_SYMENT` | 11 | `0x18` | 24 |
| `DT_PLTGOT` | 3 | `0x10c0` | `.got.plt` |
| `DT_PLTRELSZ` | 2 | `0x30` | 两条 24 字节记录 |
| `DT_PLTREL` | 20 | 7 | 记录是 `DT_RELA` |
| `DT_JMPREL` | 23 | `0x1b8` | `.rela.plt` |
| `DT_FLAGS_1` | `0x6ffffffb` | `0x08000000` | `DF_1_PIE`，一个 GNU tag |
| `DT_NULL` | 0 | 0 | 结束 |

gABI 解释了 `DT_JMPREL` 为什么单独存在：它「持有仅与过程链接表关联的重定位条目的地址。把这些重定位条目分开，可以让动态链接器在进程初始化期间忽略它们，前提是启用了惰性绑定。」`DT_FLAGS` 中的 `DF_BIND_NOW`、`DT_FLAGS_1` 中的 `DF_1_NOW`，或环境中的 `LD_BIND_NOW`，都会关闭惰性绑定，动态链接器会在启动时处理它们。

### PLT0 与 puts 的条目

psABI 第 5.2 节在图 5.2 中给出两种形态。以下是我们文件的字节：

```text
     1f0: ff 35 d2 0e 00 00    pushq  0xed2(%rip)      # 0x10c8 = GOT+8
     1f6: ff 25 d4 0e 00 00    jmpq   *0xed4(%rip)     # 0x10d0 = GOT+16
     1fc: 0f 1f 40 00          nopl   (%rax)

     200: ff 25 d2 0e 00 00    jmpq   *0xed2(%rip)     # 0x10d8 = GOT+24, the slot for puts
     206: 68 00 00 00 00       pushq  $0x0             # index 0 into DT_JMPREL
     20b: e9 e0 ff ff ff       jmp    0x1f0            # PLT0
```

算术只做一次。RIP 相对位移相对于下一条指令。`0x1f6 + 0xed2 = 0x10c8`，即 `.got.plt + 8`。`0x1fc + 0xed4 = 0x10d0`，即 `.got.plt + 16`。`0x206 + 0xed2 = 0x10d8`，第四个槽。`0x210 - 0x20 = 0x1f0`，即 PLT0。每个条目 16 字节，所以 `exit` 的条目从 `0x210` 开始，压入 `$0x1`，并读取 `0x10e0`。

GOT 保留三个槽。psABI 5.2：第零个条目保存 `_DYNAMIC` 的地址，并且「全局偏移表中的第一和第二个条目也是保留的」。我们的 `.got.plt` 读出来是 `0x1000, 0, 0, 0x206, 0x216`。glibc 在加载文件时，把槽一填成它的 `struct link_map`，把槽二填成 `_dl_runtime_resolve`。槽三在文件里是 `0x206`，也就是 `puts@plt` 里的那条 `pushq`。psABI 说：该槽「初始……持有紧随其后的 pushq 指令的地址」。

于是第一次调用执行 `call 0x200`，`jmpq *0x10d8` 落到 `pushq $0`，`jmp 0x1f0` 压入 GOT+8 并通过 GOT+16 跳进解析器；解析器读取 `DT_JMPREL` 的记录 0，即 `R_X86_64_JUMP_SLOT puts`，在 `libc.so.6` 中查找 `puts`，把它的地址写入 `0x10d8`，然后跳过去。第二次调用执行 `call`、`jmpq *0x10d8`，直接到达。Drepper 写道：「因此除第一次之外，所有调用的开销『只有』一次间接跳转」。psABI 关于 `%r11` 的脚注解释了第 38 课中的一个寄存器选择：它是暂存寄存器，所以「PLT 中的代码无需溢出任何寄存器」。

`.text` 中的调用点在 `0x227` 处读作 `e8 d4 ff ff ff`，而 `0x22c - 0x2c = 0x200`。`llvm-objdump` 为它打印 `callq 0x200 <puts@plt>`，并从 `.rela.plt` 记录中取得条目名字，`plt_entries()` 也以同样方式命名。

### 文件中没有的东西

文件中没有 `PT_GNU_RELRO`。Jakub Jelinek 于 2004-01-06 在 binutils 邮件列表上提出 RELRO：把仅在重定位期间写入的节移到可写段开头，这样动态链接器之后就可以对它们执行只读的 `mprotect`。GNU ld 的默认脚本把 `.init_array`、`.fini_array`、`.data.rel.ro`、`.dynamic` 和 `.got` 放进去，只把 `.got.plt` 的三个保留槽（24 字节）留在里面，因为其他槽是惰性写入的。这就是部分 RELRO。`-z now` 设置 `DF_BIND_NOW` 并把整个 `.got.plt` 移入，即完全 RELRO，因为启动之后不再写入任何东西。glibc 的 `_dl_protect_relro` 对按页对齐的范围调用 `mprotect(start, end - start, PROT_READ)`。`-fno-plt` 在编译器一侧更进一步：调用通过 `jmpq *name@GOTPCREL(%rip)` 进行，惰性绑定不可能发生，GCC 手册说「所有外部符号都在加载时解析」。

文件同样没有版本节。针对 `libc.so.6` 的真实链接会加入 `.gnu.version`（每个 `.dynsym` 条目一个 `Elfxx_Half`）和 `.gnu.version_r`（每个库一个 `Verneed`，并为每个版本名带 `Vernaux` 记录）。`printf@GLIBC_2.2.5` 的意思是：`.gnu.version` 中 `printf` 的条目持有一个索引，其 `vna_other` 等于该索引的 `Vernaux` 有 `vna_name`「GLIBC_2.2.5」，它位于 `vn_file` 为「libc.so.6」的 `Verneed` 之下。索引 0 是局部，1 是全局，第 15 位（`VERSYM_HIDDEN`、`0x8000`）对链接编辑器隐藏某个版本。

哈希表是 SysV `DT_HASH`，一个桶加一条三项的链，共 24 字节。动态链接器需要 `DT_HASH` 或 `DT_GNU_HASH` 才能在文件中查找符号。这个文件不导出任何东西，而且这里没有运行它，所以 glibc 是否接受单桶表尚未验证。

### .dynsym、.symtab 与 strip

gABI 把 `.dynsym` 和 `.dynstr` 设为可分配节，把 `.symtab` 和 `.strtab` 设为不可分配节。动态链接器只读前一对，从不读第二对。Drepper：「普通符号表在运行时并不使用，因此没有必要从二进制中剥离它。它对性能没有影响。」

GNU `strip --strip-all` 移除 `.symtab`、`.strtab`、`.debug_*` 和 `.comment`。所有可分配节都保留：`.interp`、`.dynsym`、`.dynstr`、`.gnu.hash`、`.gnu.version*`、`.rela.dyn`、`.rela.plt`、`.plt`、`.text`、`.rodata`、`.eh_frame`、`.init_array`、`.dynamic`、`.got`、`.got.plt`、`.data`、`.bss` 以及各 note。节头会保留，除非你加上 `--strip-section-headers`（binutils 2.41）或运行 `llvm-strip --strip-sections`。`--strip-debug` 保留符号。`--only-keep-debug` 写出调试文件，`--add-gnu-debuglink` 会在剥离后的文件中指向它，并在 `.gnu_debuglink` 中写入 CRC32；`ld --build-id` 写出 `.note.gnu.build-id`，便于调试器匹配两者。

## COFF 目标文件

`clang -target x86_64-pc-windows-msvc -O1 -fno-asynchronous-unwind-tables -c` 生成了 `main-x86_64.obj` 和 `ledger-x86_64.obj`。PE Format 页面定义了这个目标文件：20 字节文件头、无可选头、由 40 字节条目组成的节表、由 18 字节记录组成的符号表，以及每节 10 字节的重定位记录。

`code/main.py` 中的 `CoffObject` 读取这四者。文件头写明 machine 为 `0x8664`、10 个节、37 条符号记录、`SizeOfOptionalHeader` 为 0。节表列出 `.text`、`.data`、`.bss`、六个名为 `.rdata` 的节，以及 `/16`。以 `/` 开头的名字是字符串表中的十进制偏移，字符串表以它自己的四字节长度开头；`/16` 解析为 `.llvm_addrsig`，也就是 `llvm-objdump -h` 打印的名字。

37 条记录中包含辅助记录。节符号（存储类 3）带一个辅助记录，其中保存节长度、重定位数量、一个校验和以及一个 COMDAT 选择。`.rdata` 中有五个节的选择为 2（「any」），每个节在 MSVC 修饰名之下保存一个字符串字面量：`??_C@_04JDDJEAHN@even?$AA@` 就是字符串「even」。链接编辑器为每个保留一份副本。最后一条记录是 `.file`，其辅助记录为「main.c」。`llvm-objdump -t` 把它们打印为 `[ 0]`、`AUX scnlen ...`、`[ 2]` 等，并跳过辅助索引，测试把我们的索引与名字对与该文本比较。

重定位方面：`.text` 中有 14 条 `IMAGE_REL_AMD64_REL32`，`.rdata` 中有 4 条 `IMAGE_REL_AMD64_ADDR64`。`REL32` 是「从重定位后字节开始的 32 位相对地址」。它就是内置了 `-4` 的 `R_X86_64_PC32`：ELF 记录带有加数 `-4`，因为记录有加数字段，而 COFF 记录没有。`ledger-x86_64.obj` 在 `0x36`、`0x84` 和 `0x93` 处有三条 `REL32`，第 32 课的 `ledger-x86_64-O1.o` 在同样三个偏移处有三条 `R_X86_64_PC32`。代码相同，两种格式只在那个 4 写在哪里上有差别。`ADDR64` 是八字节绝对地址，即字符串指针的 `labels` 表，它在映像中变成一条基址重定位，正如 `R_X86_64_64` 变成 `R_X86_64_RELATIVE`。

| | ELF `Elf64_Rela` | COFF 重定位 | Mach-O `relocation_info` |
|---|---|---|---|
| 记录大小 | 24 字节 | 10 字节 | 8 字节 |
| 字段 | offset、`r_info` 中的类型与符号、addend | offset、符号索引、类型 | 地址，然后一个 32 位字：symbolnum:24、pcrel:1、length:2、extern:1、type:4 |
| 加数 | 显式 | 无，存在于字段中 | 无，或真实记录之前的一条 `ARM64_RELOC_ADDEND` 记录 |
| 对 `puts` 的调用 | `R_X86_64_PLT32 puts-0x4` | `IMAGE_REL_AMD64_REL32 puts` | `ARM64_RELOC_BRANCH26 _puts` |

## PE 映像：导入、导出与基址重定位

`tiny-x86_64.exe` 有 2,560 字节，由 `build_tiny_pe.py` 写出。它与那个微型 ELF 任务相同：从 `msvcrt.dll` 调用 `puts` 和 `exit`。它还导出一个名字并携带一条基址重定位，所以本课需要的三个目录都在。`pefile` 2024.8.26 和 `llvm-objdump -p` 都能读它，两份捕获都是 fixture。

走一遍头部。MS-DOS 存根以 `MZ` 开头，在偏移 `0x3c` 处保存 `e_lfanew`，值为 `0x40`。在 `0x40` 处是签名 `PE\0\0`。然后是 COFF 文件头：machine `0x8664`、4 个节、`SizeOfOptionalHeader` 240、特征 `0x22`（可执行映像、大地址感知）。可选头以 `Magic` 字段开头，值为 `0x20b`，表示 PE32+。`AddressOfEntryPoint` 是 `0x1000`，`ImageBase` 是 `0x140000000`，`SectionAlignment` `0x1000`，`FileAlignment` `0x200`，`SizeOfImage` `0x5000`，`Subsystem` 3（控制台）。`DllCharacteristics` `0x8160` 解码为 `HIGH_ENTROPY_VA 0x0020`、`DYNAMIC_BASE 0x0040`、`NX_COMPAT 0x0100` 和 `TERMINAL_SERVER_AWARE 0x8000`。接着是 16 个数据目录，每个是 RVA 加大小。其中四个被设置：Export 在 `0x2080`，大小 `0x4d`；Import 在 `0x2000`，大小 `0x28`；Base Relocation 在 `0x4000`，大小 `0xc`；IAT 在 `0x2040`，大小 `0x18`。

映像中的地址是 RVA，即相对加载基址的偏移。节表把它们映射到文件偏移：`.text` RVA `0x1000` 对应文件 `0x200`，`.rdata` RVA `0x2000` 对应 `0x400`，`.data` RVA `0x3000` 对应 `0x600`，`.reloc` RVA `0x4000` 对应 `0x800`。`PeImage.rva_to_offset()` 就是这张表。

### 导入目录

RVA `0x2000` 处是一个 20 字节导入描述符数组，以全零描述符结束。我们的只有一个：Import Lookup Table `0x2028`、时间戳 0、转发链 0、名字 `0x2068`、Import Address Table `0x2040`。名字 RVA 指向「msvcrt.dll」。查找表保存以零结束的 64 位条目。第 63 位置位表示按序号导入，序号在 0 到 15 位。第 63 位清零表示 30 到 0 位是 hint/name 条目的 RVA：一个 16 位 hint，「导出名字指针表中的一个索引」，然后是 ASCII 名字。我们的是 `0x2058`（hint 0，「puts」）和 `0x2060`（hint 0，「exit」）。

规范说 IAT「在映像被绑定之前与导入查找表内容相同」。我们在 `0x2040` 的 IAT 保存同样两个 RVA，测试检查 `0x2028` 处的 24 字节等于 `0x2040` 处的 24 字节。「在绑定期间，导入地址表中的条目会被覆盖为正在导入符号的 32 位（PE32）或 64 位（PE32+）地址。」Windows 加载器在入口点运行之前完成这件事。没有惰性绑定，也没有存根。代码直接访问槽：

```text
    100b: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1011 + 0x102f = 0x2040, the IAT slot of puts
    1013: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1019 + 0x102f = 0x2048, the IAT slot of exit
```

`FF /2` 是间接近调用。两条指令字节相同，因为每条都位于另一条之后八个字节，并读取另一条之后八个字节处的槽。`iat_calls()` 扫描代码节中的 `ff 15`，加上 6 和位移，并从导入目录给槽命名。

### 导出目录

RVA `0x2080` 处是 40 字节的导出目录表：flags、时间戳、版本、指向「tiny-x86_64.exe」的名字 RVA、序号基 1、1 个地址表条目、1 个名字指针。随后是三张表。`0x20a8` 处的 Export Address Table 保存一个 RVA，`0x1000`。`0x20ac` 处的 Export Name Pointer Table 保存一个指向字符串「tiny_entry」的 RVA。`0x20b0` 处的 Export Ordinal Table 保存一个 16 位无偏索引 0。名字 `k` 映射到地址条目 `ordinals[k]`，导出的序号是 `ordinals[k] + ordinal base`。规范还补充了一条我们没用到的规则：指向导出节内部的地址条目是转发器字符串，如 `MYDLL.expfunc`；名字指针表「按字典序排列以支持二分查找」。

### 基址重定位

`.reloc` 保存块。每个块以页 RVA 和块大小开头，然后是 16 位条目，高四位是类型，低十二位是页内偏移。我们的一个块：页 `0x3000`、大小 12、条目 `0xa000`、类型 10（`IMAGE_REL_BASED_DIR64`）、偏移 0，以及一个类型 0（`IMAGE_REL_BASED_ABSOLUTE`）的填充条目。RVA `0x3000` 处的八个字节保存 `0x1400020d0`，即映像基址加上字符串「linked by hand」的 RVA。如果加载器把映像放在别处，它会把差值加到该字段上。规范：「映像文件不包含 COFF 重定位……映像以 .reloc 节中基址重定位的形式包含重定位信息（除非映像带有 IMAGE_FILE_RELOCS_STRIPPED 属性）。」当前 link.exe 默认开启 `/DYNAMICBASE`，所以映像带有 `.reloc` 和 `DYNAMIC_BASE`。

PE 映像没有可剥离的符号表。文件头的 `PointerToSymbolTable`「对于映像应为零，因为 COFF 调试信息已弃用」，名字保存在单独的 PDB 中。每个映像中存活下来的是导入目录、导出目录，以及对 64 位代码而言的 `.pdata`，其 `RUNTIME_FUNCTION` 条目指明每个分配栈或调用其他函数的起止位置。叶子函数不在其中。

## Mach-O 映像：加载命令、存根与间接符号表

`ledger_main` 是链接后的 arm64 可执行文件。`otool -h` 打印 19 条加载命令和标志 `0x00200085`。`MH_NOUNDEFS 0x1`、`MH_DYLDLINK 0x4`、`MH_TWOLEVEL 0x80` 和 `MH_PIE 0x200000` 之和正是这个值。这 19 条命令是四条 `LC_SEGMENT_64`（`__PAGEZERO`、`__TEXT`、`__DATA_CONST`、`__LINKEDIT`）、两条 `LC_LOAD_DYLIB`（先 `@rpath/libledger.dylib`，后 `/usr/lib/libSystem.B.dylib`），以及各一条 `LC_DYLD_CHAINED_FIXUPS`、`LC_DYLD_EXPORTS_TRIE`、`LC_SYMTAB`、`LC_DYSYMTAB`、`LC_LOAD_DYLINKER`、`LC_UUID`、`LC_BUILD_VERSION`、`LC_SOURCE_VERSION`、`LC_MAIN`、`LC_RPATH`、`LC_FUNCTION_STARTS`、`LC_DATA_IN_CODE` 和 `LC_CODE_SIGNATURE`。`MachoFile.command_names()` 返回与 `otool -l` 相同顺序的相同列表。

`__TEXT` 从 `0x100000000` 开始。`__text` 是 `0x100000558` 处的 256 字节，只有 `main`，因为四个 ledger 函数位于 dylib 中。`__stubs` 是 `0x100000658` 处的 84 字节，七个存根，每个十二字节。`__DATA_CONST` 从 `0x100004000` 开始，含 64 字节的 `__got`（八个指针槽，节类型 `S_NON_LAZY_SYMBOL_POINTERS`）和 32 字节的 `__const`（`main.c` 的 `fns[]` 和 `names[]` 表）。

### 手工拆解一个存根

`_puts` 的存根在 `0x100000694`：

```text
    30 00 00 90    adrp x16, 0x100004000
    10 1e 40 f9    ldr  x16, [x16, #0x38]
    00 02 1f d6    br   x16
```

`0x90000030` 是 `adrp`。寄存器是低五位，16。`immlo` 是 30 到 29 位，0。`immhi` 是 23 到 5 位，1。值为 `immhi << 2 | immlo = 4`，以 4,096 字节为一页，所以目标页是指令所在页 `0x100000000` 加 `0x4000`。`0xf9401e10` 是带无符号 12 位偏移（按 8 缩放）的 `ldr`：21 到 10 位是 7，乘 8 得 `0x38`，所以加载读取 `0x100004038`。`0xd61f0200` 是 `br x16`。ld64 源码中的 arm64 存根正是这个三指令 `StubAtom`。

哪个名字属于哪个槽来自间接符号表。`LC_DYSYMTAB` 指向 15 个符号表索引。存根节在 `reserved1` 中说明其索引串的起点，在 `reserved2` 中说明每个存根的大小（12）。指针节使用 `reserved1` 和每槽八字节。`__stubs` 从索引 0 开始，`__got` 从索引 7 开始，所以存根 5 映射到符号表条目 10，即 `_puts`，而 `__got` 的槽 7 映射到同一条目。`otool -Iv` 打印这两段，`indirect_table()` 复现每一行。

### 两级命名空间

`nm -m` 打印 `(undefined) external _puts (from libSystem)` 和 `(undefined) external _grade (from libledger)`。库来自符号记录本身。`nlist.h` 把 `GET_LIBRARY_ORDINAL(n_desc)` 定义为 `n_desc` 的 15 到 8 位：1 表示第一条 `LC_LOAD_DYLIB`，2 表示第二条，`0xfe` 表示动态查找，`0xff` 表示可执行文件。`ld(1)`：「使用 -two_levelnamespace（默认），链接器只在命令行上的 dylib 中搜索符号，并记录在哪个 dylib 中找到它们。」`nm_lines()` 从 `n_type`、`n_sect`、`n_desc` 和 `n_value` 渲染这十二条记录，测试把输出逐行与捕获的 `nm -m` 比较。用 `-Wl,-flat_namespace` 重新链接，再读 `otool -h`，就能看到 `MH_TWOLEVEL` 位消失。

## 链式修正，逐字节

`LC_DYLD_CHAINED_FIXUPS` 是一个 `linkedit_data_command`：`dataoff 0x8000`、`datasize 208`。这 208 字节遵循 `fixup-chains.h`。

头是七个 32 位字段：`fixups_version 0`、`starts_offset 32`、`imports_offset 80`、`symbols_offset 112`、`imports_count 8`、`imports_format 1`（`DYLD_CHAINED_IMPORT`）、`symbols_format 0`（未压缩）。`llvm-objdump --macho --chained-fixups` 打印这七个值，测试会对它们进行比较。

在偏移 32 处是 `dyld_chained_starts_in_image`：`seg_count 4` 和四个偏移，每个段一个，没有修正的段为零。只有 `__DATA_CONST` 有。它的 `dyld_chained_starts_in_segment` 读出 `size 24`、`page_size 0x4000`、`pointer_format 6`（`DYLD_CHAINED_PTR_64_OFFSET`，「target 是 vm 偏移」）、`segment_offset 0x4000`、`max_valid_pointer 0`、`page_count 1`，以及一个为 0 的 `page_start`。`DYLD_CHAINED_PTR_START_NONE`（`0xffff`）表示该页没有修正。

链从文件偏移 `0x4000 + 0` 开始。每个 64 位字携带两件事：往那里写什么，以及下一个字在哪里。

```text
  dyld_chained_ptr_64_bind    ordinal:24  addend:8  reserved:19  next:12  bind:1 = 1
  dyld_chained_ptr_64_rebase  target:36   high8:8   reserved:7   next:12  bind:1 = 0
```

读第一个字 `0x8010000000000000`。第 63 位置位，是一次绑定。`next` 是 62 到 51 位，`0x002`，单位是 4 字节，所以下一个修正往后 8 字节。序号是 0。读 `0x4058` 处的第十二个字 `0x00000000000006db`。第 63 位清零，是一次重定基（rebase）。`next` 为 0，链结束。target 是 `0x6db`，一个 vm 偏移，映像基址 `0x100000000` 使它成为 `0x1000006db`，即 `__cstring` 中的字符串「odd」。它前面的字 `0x00100000000006d6` 把 `names[0]` 重定基为「even」。

这次遍历访问十二个字：`__got` 中八个序号 0 到 7 的绑定，`__const` 中两个序号 0 的绑定（`fns[]` 的两个条目都指向 `_grade`），以及 `names[]` 的两个重定基。`chained_fixups()` 返回它们及其文件偏移和原始字，测试把这十二个 `(offset, raw)` 对与 `dyld_info -fixup_chain_details` 比较，后者为第一个打印 `raw: 0x8010000000000000 bind: (next: 002, bindOrdinal: 0x000000)`。

偏移 80 处的导入表有八个 32 位条目：`lib_ordinal:8`、`weak_import:1`、`name_offset:23`，名字在偏移 112。序号 0 到 3 是来自库 1 的 `_grade`、`_grade_label`、`_percent_change`、`_total`，4 到 7 是来自库 2 的 `___stack_chk_fail`、`___stack_chk_guard`、`_printf`、`_puts`。`fixup_rows()` 把两张表连接成 `dyld_info -fixups` 打印的行，例如 `__DATA_CONST __got 0x100004038 bind libSystem/_puts`，测试比较全部十二行。

所以 `0x100004038` 处的「损坏指针」其实是导入 7。dyld 在加载时遍历一次这条链，把 `_puts` 的地址写入槽中。Apple 的 WWDC22 演讲这样表述：「在 DATA 的 64 位指针位置中，一些位包含下一个修正位置的偏移。那里还打包了一个位，说明该修正是绑定还是重定基。」Xcode 13 让这种格式成为部署目标 macOS 12 或 iOS 15 的默认值，Xcode 13.3 把它扩展到 macOS 11 和 iOS 13.4。这种格式没有惰性绑定。每个槽都在 `main` 运行之前绑定。

### 它取代的布局

`ledger_main_nochain` 是用 `-Wl,-no_fixup_chains` 进行的同一链接。它改用 `LC_DYLD_INFO_ONLY`，带四张字节编码表：rebase 8 字节、bind 72、lazy bind 120、export 64。它还有链式文件没有的三个节：`__TEXT,__stub_helper`（108 字节）、仅 16 字节的 `__got`（保存 `___stack_chk_guard` 和 `dyld_stub_binder`）以及 `__DATA,__la_symbol_ptr`（56 字节，七个槽）。`dyld_info -fixups` 列出七行 `lazy-bind` 和七条重定基，每条把对应的惰性槽指向 `__stub_helper` 中它自己的跳板。跳板压入一个索引并跳到 `dyld_stub_binder`，后者解析名字并写入槽。这就是上一节 PLT 和 GOT 的 Mach-O 形态：`__la_symbol_ptr` 是 `.got.plt`，`__stub_helper` 是 PLT0 和 `pushq $index`，`dyld_stub_binder` 是 `_dl_runtime_resolve`。`code/main.py` 报告这个文件的偏移和节，但不解码 bind 操作码；实践练习要求你完成这一点。

### 导出 trie

`LC_DYLD_EXPORTS_TRIE` 是一棵前缀树。每个节点是一个 uleb128 终端大小，若非零，则还有 uleb128 标志和一个 uleb128 偏移，然后一个字节的子节点计数，以及对每个子节点的一条字符串边和一个 uleb128 节点偏移。`exports_trie()` 遍历它。在 `libledger.dylib` 上它返回 `_percent_change 0x410`、`_grade 0x428`、`_total 0x464`、`_grade_label 0x498`，即 `dyld_info -exports` 打印的那四个偏移。在可执行文件上它返回 `__mh_execute_header`、`_fns` 和 `_main`。

## strip 留下了什么

Apple 的 `strip(1)` 说，对于动态链接的可执行文件，不带选项时它行为等同于 `-u -r`：保留未定义符号和动态引用的符号。`-x` 移除局部符号。`-N` 移除整个符号表和字符串表。「对于动态共享库，最大程度的剥离通常是 -x」，因为「动态链接过程只使用全局符号」。

在我们的 fixture 上观察到的结果。`strip ledger_main` 把 `LC_SYMTAB` 从 12 条记录减到 10 条，移除了 `_main`、`_fns` 和 `_names`。八个未定义导入保留下来，因为 dyld 需要它们的名字来绑定；`__mh_execute_header` 保留，因为它是 `REFERENCED_DYNAMICALLY`。第十条记录是类型为 `N_OPT`、名为 `radr://5614542` 的 stab，Apple 的工具会写它，以免表为空；`nm` 不打印它。剥离后的程序打印与原始程序相同的三行并退出 0。`LC_FUNCTION_STARTS` 仍有一个起点，`main`。

不带选项的 `strip libledger.dylib` 比 man 页通常说的 `-x` 走得更远：`LC_SYMTAB` 有 0 条记录，`nm` 打印「no symbols」。导出 trie 仍以相同偏移列出四个函数，因为 dyld 通过 trie 绑定，从不通过 `LC_SYMTAB`。`LC_FUNCTION_STARTS` 仍列出四个起点。实验 C 把新的 `main.c` 链接到剥离后的 dylib，以表明链接编辑器同样由 trie 服务。

| 项目 | `--strip-all` 后的 ELF | PE 映像 | `strip` 后的 Mach-O |
|---|---|---|---|
| 静态符号表 | 消失 | 映像中从不存在 | 局部和已定义名字消失，未定义导入保留 |
| 加载器绑定的名字 | `.dynsym` 和 `.dynstr` 保留 | 导入目录保留 | 导入保留在 `LC_SYMTAB` 和链式导入表中 |
| 导出名字 | `.dynsym` 保留 | 导出目录保留 | 导出 trie 保留 |
| 函数起点 | `.eh_frame` 的 FDE，除非以无 unwind 表方式构建 | 非叶子函数的 `.pdata` | `LC_FUNCTION_STARTS` 保留 |
| 调试信息 | `.debug_*` 消失 | 在 PDB 中，从不在映像中 | 在 dSYM 中，从不在映像中 |
| 节头或加载命令头 | 保留，除非 `--strip-section-headers` | 保留 | 保留 |

## 在没有符号的情况下寻找函数

`static-arm64-O2` 把 `main-arm64-O2.o` 和 `ledger-arm64-O2.o` 链接成一个可执行文件，`__text` 中有五个函数。未剥离文件的 `nm -n` 是真值：`_main` 在 `0x100000500`（244 字节）、`_percent_change` 在 `0x1000005f4`（24）、`_grade` 在 `0x10000060c`（60）、`_total` 在 `0x100000648`（248）、`_grade_label` 在 `0x100000740`（72）。`static-arm64-O2.stripped` 在 `__text` 内没有 `N_SECT` 符号。`FunctionFinder` 对它运行四种方法，`score()` 把每个列表与真值比较。

### LC_FUNCTION_STARTS

该命令指向 `__LINKEDIT` 中的九个字节：`80 0a f4 01 18 3c f8 01 00`。格式是 uleb128 增量，第一个相对于 `__TEXT` 地址。`80 0a` 是 `0x500`，所以第一个起点是 `0x100000500`。`f4 01` 是 `0xf4`，所以是 `0x1000005f4`。`18` 给出 `0x10000060c`，`3c` 给出 `0x100000648`，`f8 01` 给出 `0x100000740`，`00` 结束列表。五个全中，没有误报，在 `-O0` 和 `-O2` 都是如此。链接编辑器为展开器和崩溃报告器写这张表，`strip` 不会动它。Ghidra 在它的「Mach-O Function Starts」分析器中读取它，`dyld_info -function_starts` 打印它，LIEF 把它暴露为 `function_starts`。

### 序言扫描

Ghidra 的 AARCH64 起点模式以 `fd 7b bf a9` 开头，即 `stp x29, x30, [sp, #-0x10]!`。在我们的文件上，它在两个优化级别都找到 0/5。clang 先发出 `sub sp, sp, #N`，然后发出带正偏移的 `stp x29, x30, [sp, #M]`，所以预索引形式从未出现。一个针对 `sub sp, sp, #imm` 的掩码 `(word & 0xff0003ff) == 0xd10003ff`，在 `-O0`（每个函数都有栈帧）找到 5/5，在 `-O2` 只找到 1/5，因为此时 `percent_change`、`grade`、`total` 和 `grade_label` 都是没有栈帧的叶子函数。`percent_change` 以 `cbz w0, ...` 开头。序言扫描报告的是编译器选择发出的东西，而第 38 课已经展示过，那常常是什么都没有。

这里两种扫描都没有产生误报，因为程序很小。像 x86-64 的 `push rbp` 的 `55` 这样的单字节模式会命中其他指令内部，所以 Ghidra 的 x86-64 模式文件把 `55 48 89 e5` 这样的后置模式与 `ret` 或 `nop` 这样的前置模式配对，后者必须结束前一个函数。

### 调用目标

`__text` 中的每个 `bl` 都指向一个函数起点，因为编译器不会为 C 发出跳进函数中间的 `bl`。编码是 `0x94000000` 加上有符号 26 位字偏移。`0x1000005a0` 处的 `bl` 读作 `3d 00 00 94`：`0x3d` 是 61，乘 4 得 `0xf4`，所以目标是 `0x100000694`。在 `ledger_main` 中，那是 `_puts` 的存根。在静态构建中，`call_targets()` 收集 `__text` 内的目标，丢弃落在 `__stubs` 中的，再加上 `LC_MAIN` 入口，因为没有东西调用 `main`。两个优化级别都是 5/5。Nucleus（Andriesse、Slowinska、Bos，2017）把函数检测建立在这个思路上，面向编译器生成的代码；Andriesse 等人在 2016 年测出，函数边界是反汇编中困难的部分，比指令边界更难。

## FLIRT 形态的签名

Hex-Rays 在其文档中描述 FLIRT：一个模式是「函数的前 32 字节，其中所有可变字节都被标记」，可变字节来自修正和外部引用，模式存在于一棵树中，模式之外「我们计算从位置 33 开始直到第一个可变字节的 CRC16」，并存储该长度。模式与 CRC 相同的两个函数，先靠一个它们不同的位置上的尾部字节区分，再靠它们引用的名字区分。无法区分的碰撞「由自然智能解决」。短于四字节且无外部引用的函数被排除，「签名文件不包含原始库中的任何字节，除了函数名」。

`signatures_from_object()` 从 `ledger-arm64-O2.o` 和 `main-arm64-O2.o` 构建这种签名，方式与 FLAIR 从库的目标文件构建相同。每条 `relocation_info` 记录覆盖一个四字节字，这些字节成为可变字节。`.pat` 行的布局记录在 rizin 的解析器中，依次是模式、CRC 长度、CRC16、函数长度、`:0000` 和名字，以及每个被引用名字的 `^offset name`：

```text
A00000342800004B890C8052087D091B000DC01AC0035FD6 00 0000 0018 :0000 _percent_change
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 003C :0000 _grade ^0000 l_switch.table.grade ^0001 l_switch.table.grade
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 0048 :0000 _grade_label ^0000 l_switch.table.grade_label ^0001 l_switch.table.grade_label ^0002 _labels ^0003 _labels
```

`_percent_change` 是 24 字节，所以它的模式就是整个函数，CRC 长度为 0。`_grade` 是 60 字节，第一个重定位在第 36 字节，所以 CRC 覆盖四个字节。CRC16 是公开 `.pat` 解析器使用的那一个：多项式 `0x8408`、初值 `0xffff`、取补，然后交换两个字节。它对 ASCII 字符串「123456789」的值是 `0x6e90`，测试固定了这一点。

`match_signatures()` 在剥离构建的每个函数起点尝试每个签名，先比模式，再比 CRC。在 `-O0`，五个函数都得到名字，没有碰撞。在 `-O2`，上面两行发生碰撞。`grade_label` 内联了 `grade`，所以两个函数共享前 36 字节和四个 CRC 字节。两个签名在 `0x10000060c` 和 `0x100000740` 都匹配。尾部把它们分开：在第 44 字节，`_grade` 是 `20`，`_grade_label` 是 `28`，即 `ldr w0, [x9, w8, uxtw #2]` 对 `ldr x8, [x9, w8, uxtw #3]`，长度是 60 对 72。报告记录两次碰撞以及决定它们的字节。两个优化级别都命名了 5/5。

测试 `test_object_bytes_equal_linked_bytes_outside_relocated_words` 检查这个前提：每个目标函数中每个非可变字节，都等于已链接可执行文件同一偏移处的字节，因此链接编辑器只改动了被重定位的字。`_total` 完全没有重定位，也没有可变字节。

### 工具如何应用同一思路

IDA Pro 9.4 为每种处理器附带 FLIRT 签名以及构建它们的 `sigmake` 工具；IDA Free 只覆盖 x86 和 x86-64，且仅限非商业使用。Lumina 是一个服务器，用函数的 md5 作为函数元数据的键，跳过短于 32 字节的函数。Ghidra 12.1.4（Apache-2.0）有 Function ID：一个基于助记符和寻址模式的完整哈希，「对链接导致的变化具有鲁棒性」；还有一个特定哈希，加入常量操作数，按匹配指令打分，并用父函数和子函数作为平局裁决；它自带的数据库只覆盖 Visual Studio 库，你需要从自己的二进制填充自己的库。Binary Ninja 6.0 Krypton（商业，有免费版）使用 WARP（Apache-2.0）：函数 GUID 是对基本块 GUID 的 UUIDv5，每个基本块 GUID 又是对块字节的 UUIDv5，字节在「所有可重定位指令」清零并丢弃 NOP 之后计算，并把被调用者、调用者和相邻函数作为约束。BinDiff 8（Apache-2.0）以固定的步骤顺序在两个二进制之间匹配函数，从哈希匹配经流图 MD 索引到调用序列匹配，用匹配的边、块、指令和流图 MD 索引加权相似度。Diaphora 3.4.3（AGPL-3.0，仅 IDA）加入小素数乘积：每个助记符映射到一个素数，一个函数就是乘积，形成与顺序无关的多重集指纹。radare2 6.2.4（LGPL-3.0）有 zignatures（`zg`、`z/`、`zb`），并用 `zfs` 读取 FLIRT `.sig` 文件；rizin 0.9.1（LGPL-3.0）用 `Fc` 从打开的二进制创建 `.pat` 或 `.sig`，用 `Fs` 应用。

IDA 和 Binary Ninja 是闭源的，这里依据它们的文档描述。radare2 和 rizin 没有安装在这台机器上，其命令引自它们的书籍和源码。在这些 fixture 上运行过的工具是 Command Line Tools 中的 `clang`、`ld`、`llvm-objdump`、`otool`、`nm`、`strip` 和 `dyld_info`，以及通过探针使用的 LIEF 1.0.0、pyelftools 0.33 和 pefile 2024.8.26。LIEF 读取了这里的每种格式，包括它自 0.17.0 起支持的 COFF 目标文件，并在 `ledger_main` 上报告十个绑定和两个重定基，即同样的十二个修正。pyelftools 只读 ELF 文件。pefile 拒绝 `main-x86_64.obj`，报 `PEFormatError: 'DOS Header magic not found.'`，但读取了微型映像。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
```

输出是一份包含五部分的 JSON 报告。`elf` 保存两个目标文件的重定位计数、链接计划，以及 `tiny-dyn-x86_64` 的动态段、PLT 和 GOT。`coff` 保存两个目标文件的节、重定位、函数和未定义名字。`pe` 保存 `tiny-x86_64.exe` 的头部、目录、导入、导出、基址重定位和 IAT 调用。`macho` 保存 `ledger_main` 及其存根、链式修正、修正行和导出，`ledger_main_nochain` 的旧布局，那个 dylib，以及两份 strip 报告。`identification` 保存在 `-O0` 和 `-O2` 下的真值、带评分的四种方法，以及带碰撞的签名。

预期形态：

```json
{
  "elf": {"linkPlan": {"pltSymbols": ["puts", "printf"], "sizes": {".plt": 48, ".got.plt": 40, ".rela.plt": 48, ".rela.dyn": 48}, "...": "..."}, "tiny-dyn-x86_64": {"...": "..."}},
  "coff": {"main-x86_64.obj": {"...": "..."}},
  "pe": {"imports": [{"dll": "msvcrt.dll", "iat": "0x2040", "...": "..."}], "...": "..."},
  "macho": {"ledger_main": {"flags": "0x200085", "fixupRows": ["__DATA_CONST __got 0x100004000 bind libledger/_grade", "..."], "...": "..."}, "strip": {"...": "..."}},
  "identification": {"O2": {"methods": {"functionStarts": {"recall": "5/5"}, "prologueSub": {"recall": "1/5"}, "...": "..."}, "correctNames": 5}}
}
```

按以下顺序阅读 `code/main.py`：

1. `ElfFile` 读取头部、节、两张符号表和重定位，`dynamic_entries()`、`jump_slots()`、`plt_entries()` 和 `plt0()` 读取已链接映像。
2. `plan_dynamic_link()` 把表 4.9 和第 5.2 节应用到可重定位目标文件的重定位记录上。
3. `CoffObject` 读取文件头、节表、带辅助记录的 18 字节符号以及 10 字节重定位。
4. `PeImage` 读取 MS-DOS 存根、PE 签名、可选头、数据目录、节，以及导入、导出和基址重定位目录，`iat_calls()` 解码 `ff 15`。
5. `MachoFile` 读取加载命令、段、`nlist` 记录、间接符号表、存根、链式修正、导出 trie 和目标文件重定位。
6. `FunctionFinder` 运行四个起点查找器，`score()` 把每个与真值比较。
7. `signatures_from_object()` 和 `match_signatures()` 构建并应用 FLIRT 形态的签名。
8. `parse_*` 函数读取测试要对照的捕获工具文本。

更多命令：`python3 code/main.py elf FILE`、`coff FILE`、`pe FILE` 和 `macho FILE` 打印一份报告。`python3 code/main.py plan OBJECT...` 为任意一组 ELF 目标文件打印链接计划。`python3 code/main.py functions STRIPPED UNSTRIPPED OBJECT...` 在你自己的另一对构建上给查找器和签名打分。

## 使用它

在你构建的程序上按这个顺序使用这些读取器。

1. 读取头部，说出格式和种类：可重定位目标文件、可执行文件还是共享库。读取器拒绝它不认识的东西，这种拒绝本身就是信息。
2. 列出导入及其提供者：`DT_NEEDED` 和 `.dynsym`、导入目录、或链式导入表与 `LC_LOAD_DYLIB`。对这份列表上任何名字的每次调用都会离开映像。
3. 手工跟踪一次调用到它的槽。写下调用字节、存根或 PLT 字节、槽地址、文件中的字，以及加载器往那里写什么。如果文件中的字看起来像垃圾，先把它当作修正来解码，再称它为垃圾。
4. 把符号表与导出和函数起点对比。加载器需要的名字在 `strip` 后存活。只有调试器需要的名字不会。说明缺失的名字属于哪一组。
5. 在剥离构建中，运行你能运行的查找器，并在你还有未剥离孪生文件时给它们打分。优先使用链接编辑器写的表——`LC_FUNCTION_STARTS`、`.pdata` 或 `.eh_frame`——而不是字节模式，并用调用目标填补空缺。
6. 从你自己的目标文件构建签名并应用它们。记录每一次碰撞以及分开它的字节。两个签名都声称拥有的名字，在尾部或被引用名字做出决定之前，都是未知。

把捕获的工具文本与文件放在一起。工具链变化时重新捕获，因为变化的清单就是变化的 fixture。

## 交互实验

### 实验 A：手工解析 PLT

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code
python3 -q
```

运行：

```python
from main import *
elf = load_elf("tiny-dyn-x86_64")
[(e.tag_name, e.text) for e in elf.dynamic_entries()]
elf.plt0()
[(e.symbol, hex(e.address), hex(e.slot), hex(e.slot_initial)) for e in elf.plt_entries()]
[hex(w) for w in elf.got_plt_words()]
```

在每次调用之前，用上文中的字节算出答案。然后把 `build_tiny_dyn_elf.py` 中的 `IMPORTS` 改成三个名字，重新构建，在读取之前预测 `.plt`、`.got.plt` 和 `.rela.plt` 的新尺寸。

### 实验 B：一个链式修正

```python
exe = load_macho("ledger_main")
stub = next(s for s in exe.stubs() if s.symbol == "_puts")
stub.lines()
raw = int.from_bytes(exe.read(stub.slot, 8), "little")
hex(raw), raw >> 63, (raw >> 51) & 0xfff, raw & 0xffffff
[(i.index, i.name, i.library) for i in exe.chained_fixups()["imports"]]
```

先在纸上解码这个绑定字：bind 位、以四字节为单位的 `next` 字段、序号。然后读取往后八个字节的字，确认它是下一个修正。然后读取 `0x100004058` 处链的最后一个字，并解释为什么它的 `next` 是零。

### 实验 C：strip 与重新链接

在 fixtures 目录中，把新程序链接到完全剥离的 dylib：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code/fixtures
clang -target arm64-apple-macos -O1 main.c -L. -lledger.stripped -Wl,-rpath,@executable_path -o /tmp/relinked
nm /tmp/relinked | grep ' U '
dyld_info -exports libledger.stripped.dylib
```

链接成功，而那个库的 `LC_SYMTAB` 是空的，因为链接编辑器读取导出 trie。然后运行 `python3 ../main.py macho libledger.stripped.dylib`，比较 `symtab`、`exports` 和 `functionStarts`。然后改用 `-N` 剥离静态构建，看看 `dyld_info -function_starts` 还会打印什么。

### 实验 D：碰撞

```python
from main import *
stripped = load_macho("static-arm64-O2.stripped")
sigs = signatures_from_object(load_macho("ledger-arm64-O2.o"))
[s.pat_line() for s in sigs]
starts = FunctionFinder(stripped).function_starts()
match_signatures(sigs, stripped, starts)
```

读两条碰撞记录。然后从 `ledger-arm64-O0.o` 构建签名，把它们与 `static-arm64-O0.stripped` 匹配，并从 `.pat` 行解释为什么 `-O0` 没有碰撞。然后把 `PATTERN_LENGTH` 缩短到 16，数一数有多少函数碰撞。

## 实践练习

解码旧式绑定表，并加入 PE 查找器。

要求：

- 把 `ledger_main_nochain` 的 `LC_DYLD_INFO_ONLY` rebase、bind 和 lazy bind 操作码流解码成带段、节、地址、类型和目标的行，并让它们等于捕获的 `dyld_info -fixups` 文本中的 21 行。
- 给 `build_tiny_dyn_elf.py` 加入一个 `PT_GNU_RELRO` 程序头和一个带 `DF_BIND_NOW` 的 `DT_FLAGS`，把 `.got.plt` 移入覆盖范围，并让 `llvm-objdump -p` 显示新头。在 fixtures README 中说明这复现了完全 RELRO 的哪一部分，哪一部分只有 glibc 能完成。
- 为 PE32+ 映像写一个 `.pdata` 读取器，以及一个使用 `RUNTIME_FUNCTION` 起始地址的函数查找器。用不带 `-fno-asynchronous-unwind-tables` 的 `clang -target x86_64-pc-windows-msvc -c` 构建一个目标文件，读取它的 `.pdata` 重定位，并说明它的哪些函数是表省略的叶子函数。
- 为每个读取器写测试，与捕获的工具文本比较。

成功的标志是：`python3 code/main.py macho ledger_main_nochain` 打印与 `dyld_info` 相同的 21 行修正，微型 ELF 带有 `llvm-objdump` 能列出的 `PT_GNU_RELRO`，`.pdata` 查找器能给它能命名的函数命名，并列出它无法命名的叶子函数。

## 课程产物

本课包含 `outputs/skill-linking-loading-and-stripped-binaries.md`。用它把你构建的程序转成一份清单：导入及其提供者；一次跟踪到加载器填充槽位的调用；`strip` 移除了什么、什么存活下来的说明；以及对剥离构建中函数起点和名字的带评分恢复。它要求以未剥离孪生文件为真值，为每个断言提供捕获的工具清单，并记录每一次碰撞。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
python3 code/tests/test_main.py
```

验证应证明：

- `tiny-dyn-x86_64` 的动态段逐 tag 与 `llvm-objdump -p` 相符，且 `DT_FLAGS_1` 携带 `DF_1_PIE`
- 两条 `R_X86_64_JUMP_SLOT` 记录与 `llvm-objdump -R` 相符，并命名 `puts` 和 `exit`
- PLT0 压入 GOT+8 并通过 GOT+16 跳转，每个条目读取自己的槽并压入自己的索引，每个槽持有它自己 `pushq` 的地址
- 微型 ELF 的节和程序头与 `llvm-objdump -h -p` 相符
- 两个 ELF 目标文件的重定位记录与 `llvm-objdump -r` 相符
- 链接计划找到两个 PLT 符号、四次直接调用、两条相对重定位，以及微型映像的尺寸
- COFF 节表、37 条带辅助记录的符号记录和重定位与 `llvm-objdump -h -t -r` 相符，且 `/16` 解析为 `.llvm_addrsig`
- `ledger` 的三条 COFF `REL32` 记录位于三个 ELF `PC32` 偏移处
- PE 头、`DllCharacteristics` 和数据目录与 `llvm-objdump -p` 相符
- 导入目录、hint/name 条目和 IAT 与 `llvm-objdump` 和 `pefile` 相符，且绑定前 IAT 等于查找表
- 导出目录和 `DIR64` 基址重定位解码正确，被重定位的指针命名字符串「linked by hand」
- RVA `0x100b` 处的 `ff 15` 解析为 `puts` 的 IAT 槽
- Mach-O 标志、19 条加载命令名和两个库与 `otool` 相符
- 七个存根解码到八个 `__got` 槽，间接符号表与 `otool -Iv` 相符
- 链式修正头与 `llvm-objdump --macho --chained-fixups` 相符
- 链遍历复现 `dyld_info -fixup_chain_details` 的全部十二个原始字
- 修正行和导入表与 `dyld_info -fixups` 和 `-imports` 相符
- 导出 trie 对 dylib 及其剥离副本与 `dyld_info -exports` 相符
- `nm_lines()` 对可执行文件及其剥离副本等于 `nm -m`
- `strip` 移除了 `_main`、`_fns` 和 `_names`，保留了九个名字和一个 `radr://5614542` stab，剥离后的程序打印相同输出
- 旧布局有 `LC_DYLD_INFO_ONLY`、七个惰性指针，以及指向 `__stub_helper` 的重定基
- arm64 目标文件重定位与 `llvm-objdump -r` 相符
- `LC_FUNCTION_STARTS` 在两个优化级别都恢复 5/5，且等于 `nm` 和 `dyld_info -function_starts`
- `sub sp` 扫描在 `-O0` 找到 5/5，在 `-O2` 找到 1/5，`stp` 栈帧扫描找不到
- `bl` 目标加上 `LC_MAIN` 入口恢复 5/5
- 签名在两个级别命名全部五个函数，并记录由第 44 字节分开的 `grade` 碰撞
- 「123456789」的 CRC16 是 `0x6e90`，且 `main` 的 GOT 加载对之下的八个字节是可变字节
- 每个非可变目标字节都等于链接后的字节
- 演示与图形数据报告相同地址，站点模块以 ES5 内嵌相同数据

## 生产环境失败模式

| 故障 | 弱工具报告什么 | 分析必须证明什么 |
|---|---|---|
| 一条 `bl` 或 `call` 落入存根 | 「调用了 0x694 处未命名函数」 | 存根字节、它加载的槽，以及该槽绑定的导入 |
| 从文件中读取一个 GOT 字 | 「损坏指针 0x8010000000000007」 | 第 63 位是 bind，`next` 是 2，序号 7 是来自库 2 的 `_puts` |
| 目标文件中的 `PLT32` | 「每次调用都经过 PLT」 | 已定义符号被降为直接距离，只有未定义符号获得条目 |
| 把部分 RELRO 读成完全 RELRO | 「GOT 是只读的」 | 没有 `-z now` 时，三个保留槽之后的 `.got.plt` 槽位于 `PT_GNU_RELRO` 之外 |
| 把 COFF `REL32` 与 ELF `PC32` 相比 | 「Windows 构建有不同重定位」 | 偏移相同，`-4` 存在于 ELF 加数和 COFF 字段中 |
| 加载前读取 IAT | 「导入指向字符串」 | 加载器绑定之前 IAT 等于查找表，那些字符串是 hint/name 条目 |
| 一个剥离的 dylib | 「没有符号，没有导出」 | 导出 trie 和 `LC_FUNCTION_STARTS` 列出每个函数 |
| 对剥离可执行文件运行 `nm` | 「一个函数」 | `LC_FUNCTION_STARTS` 列出五个起点，共九个字节 |
| 在 `-O2` 做序言扫描 | 「一个函数」 | 四个叶子函数没有栈帧，调用目标和起点表能找到它们 |
| 两个函数共享 36 字节 | 「grade_label 就是 grade」 | 第 44 字节的尾字节和 60 与 72 的长度分开这个碰撞 |
| 在另一台机器的构建上运行工具 | 本文中的数字 | 从你自己的 fixture 重新捕获每一份清单 |

## 操作准则

跟踪每次外部调用，从调用点到加载器填充的槽，并在称槽中的字为地址之前，按格式定义解码它。先记录加载器仍然需要什么，再进行 strip。先使用链接编辑器写的表恢复函数起点，再扫描字节；用带符号的孪生文件给每个查找器打分，并记录每一次签名碰撞及分开它的字节。

## 延伸阅读

- [System V gABI，第 5 章：程序加载与动态链接](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V 应用程序二进制接口，AMD64 架构处理器补充：第 5.2 节与表 4.9](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Drepper，《如何编写共享库》，版本 4.1.2](https://akkadia.org/drepper/dsohowto.pdf)
- [Drepper，《ELF 符号版本控制》](https://akkadia.org/drepper/symbol-versioning)
- [Jelinek，《针对漏洞加固 DSO/可执行文件的小改动》，binutils 邮件列表，2004-01-06](https://www.sourceware.org/ml/binutils/2004-01/msg00070.html)
- [Taylor，《Linker relro》，2008](https://www.airs.com/blog/archives/189)
- [binutils 提交 bd7ab16b：用于调用的 PLT32，2018-02-13](https://gnu.googlesource.com/binutils-gdb/+/bd7ab16b4537788ad53521c45469a1bdae84ad4a) 和 [LLVM D43383](https://reviews.llvm.org/D43383)
- [GNU Binutils：strip](https://sourceware.org/binutils/docs/binutils/strip.html) 和 [llvm-objcopy](https://llvm.org/docs/CommandGuide/llvm-objcopy.html)
- [Microsoft，PE Format](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [CALL 指令参考，FF /2](https://www.felixcloutier.com/x86/call)
- [Apple，mach-o/loader.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h) 和 [mach-o/nlist.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h)
- [Apple dyld，mach-o/fixup-chains.h](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/fixup-chains.h)
- [Apple ld64，stub_arm64.hpp](https://github.com/apple-oss-distributions/ld64/blob/main/src/ld/passes/stubs/stub_arm64.hpp)
- [Xcode 13 发行说明：链式修正](https://developer.apple.com/documentation/xcode-release-notes/xcode-13-release-notes) 和 [WWDC22 第 110362 场，《Link fast: Improve build and launch times》](https://developer.apple.com/videos/play/wwdc2022/110362/)
- [strip(1)](https://keith.github.io/xcode-man-pages/strip.1.html)、[ld(1)](https://keith.github.io/xcode-man-pages/ld.1.html) 和 [dyld(1)](https://keith.github.io/xcode-man-pages/dyld.1.html)
- [Hex-Rays，IDA F.L.I.R.T. Technology: In-Depth](https://docs.hex-rays.com/core/flirt/concepts/ida-f.l.i.r.t.-technology-in-depth) 和 [Lumina 概览](https://docs.hex-rays.com/add-ons/lumina/overview)
- [Ghidra Function ID 帮助](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/FunctionID/src/main/help/help/topics/FunctionID/FunctionID.html) 和 [AARCH64 函数起始模式](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/patterns/AARCH64_LE_patterns.xml)
- [Vector 35，WARP](https://github.com/Vector35/warp) 和 [Binary Ninja WARP 指南](https://docs.binary.ninja/guide/warp.html)
- [BinDiff 手册](https://www.zynamics.com/bindiff/manual/) 和 [BinDiff 源码](https://github.com/google/bindiff)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [rizin pat.c，.pat 解析器](https://github.com/rizinorg/rizin/blob/dev/librz/sign/pat.c)、[radare2 flirt.c](https://github.com/radareorg/radare2/blob/master/libr/anal/flirt.c) 和 [radare2 书：Zignatures](https://book.rada.re/signatures/zignatures.html)
- [Andriesse 等，《An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries》（USENIX Security 2016）](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse) 和 [Andriesse、Slowinska、Bos，《Compiler-Agnostic Function Detection in Binaries》（EuroS&P 2017）](https://danielandriesse.com/papers/eurosp-2017.pdf)
- [Bao 等，ByteWeight（USENIX Security 2014）](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/bao) 和 [Shin、Song、Moazzezi，《Recognizing Functions in Binaries with Neural Networks》（USENIX Security 2015）](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/shin)
- [LIEF 变更日志](https://lief.re/doc/latest/changelog.html)、[pyelftools](https://github.com/eliben/pyelftools) 和 [pefile](https://github.com/erocarrera/pefile)
