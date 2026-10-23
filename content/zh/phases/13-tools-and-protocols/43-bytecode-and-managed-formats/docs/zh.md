# 超越原生代码：Python、JVM、.NET 与 WebAssembly 字节码

> 字节码文件是程序本身，外加编译器为它写下的注解；本课要构建的读取器，就是用来看清每种容器到底保留了哪些注解。

**Type:** Build
**Languages:** Python, Java, C#, WebAssembly text
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow)
**Time:** ~150 minutes

## 学习目标

- 按 PEP 552 读取 16 字节的 `.pyc` 文件头，并在不借助 `marshal` 模块的情况下反序列化代码对象，覆盖四种解释器版本写出的文件。
- 按 JVMS 第 4 章解析 JVM class 文件：常量池、字段、方法、`Code` 属性，以及 `javac -g` 追加的那些表。
- 反汇编 `Ledger.class` 中的 JVM 操作码，并用 `javap -c` 在四份同源构建上验证清单。
- 逐节读取 WebAssembly 模块，解码每个函数体，并用 `wasm-objdump -d` 验证结果。
- 从 PE 数据目录一路走到 `MethodDef` 行和每个方法的 CIL，遍历 .NET 程序集的 ECMA-335 元数据，并与 ILSpy 所使用的读取库对账。
- 针对同一个函数在五种容器中的形态，测量每种格式保留与丢失了哪些名称、类型和控制结构。

## 问题

你构建了 `ledger.py`，并在 `__pycache__` 里发现了 `ledger.cpython-314.pyc`。你问智能体这个文件里有什么。智能体跑了一个反编译器。工具打印出 `Unsupported opcode`，还有一行 `# WARNING: Decompyle incomplete`。智能体由此断定：文件被混淆了。

文件并没有被混淆。它的前两个字节是 `2b 0e`，小端字为 3627。这是 CPython 3.14 写下的数字，而该反编译器的操作码表只到 3.13。是工具过时了。智能体没有读文件头，自然分辨不出这两种情况。

第 33 课展示了原生代码会丢失什么：参数名、类型，以及被编译成跳转表的 `switch`。字节码容器丢得少一些，而且每种容器丢的东西各不相同。pyc 保留所有变量名，但不保留任何类型。class 文件保留类型，只有你明确提出时才保留局部变量名。Wasm 模块保留控制结构，除非自定义节携带名称，否则名称全部丢弃。.NET 程序集把参数名存进一张表，局部变量名则留给另一个文件。在你弄清某种容器原本记了哪些注解之前，无法评判任何反编译器在它上面的输出。

本课用 Python 标准库构建四个读取器。每个读取器都从第一个字节开始读容器，列出一个函数，并在一批你自己构建的样例文件上与格式自带的工具对照。最后一节把 `ledger` 里的 `grade` 放进五种容器里并排比较。

```figure
re-43-five-containers
```

## pyc 文件头由四个字组成

PEP 552 把文件头定义为四个 32 位字。第一个是魔法数（magic number）。第二个是位字段。置位 bit 0 表示文件基于哈希。置位 bit 1 表示 `check_source`：解释器在导入时对源文件做哈希并比对。bit 0 为零时，第三、四个字是源文件的 mtime 和源文件大小；bit 0 为一时，它们保存源文件的 64 位 SipHash。

魔法数定义在 `Include/internal/pycore_magic_number.h` 中。自 3.11 起，规则是 Python 3.n 从 2900 + 50n 起步，发布周期内每次字节码变更就加一。写入文件的 token 是 16 位数后面跟 `\r\n`，所以用文本模式打开的 pyc 会读错魔法数。最终取值为：3.11 是 3495，3.12 是 3531，3.13 是 3571，3.14 是 3627，3.15 是 3666。Python 3.15.0 于 2026-10-09 发布，但构建这些样例的机器上没有安装，所以它的数字来自头文件，而不是本地的某个文件。

每种解释器对应两个 pyc 样例。`compile_pyc.py` 用 `py_compile.compile(..., dfile="ledger.py")` 写出它们，使存储的文件名是 `ledger.py`，而不是构建机器的路径。下面是 3.14 两个文件的前 16 个字节：

```text
checked-hash   2b 0e 0d 0a  03 00 00 00  c7 16 c7 fc c5 1d 81 01
timestamp      2b 0e 0d 0a  00 00 00 00  8d 60 cf 6a  f1 01 00 00
```

逐字读这份时间戳文件头。`03 00 00 00` 或 `00 00 00 00` 是标志字。`8d 60 cf 6a` 是 0x6acf608d = 1791975565，即构建时 `ledger.py` 的 mtime。`f1 01 00 00` 是 497，即源文件大小。带校验哈希的文件用的是标志 3，外加八个哈希字节。四个解释器在文件头上除了魔法字以外完全一致：

| 解释器 | 魔法数字节 | 字 | 文件字节数 | `grade.co_code` 字节数 | `grade` 指令数 |
|---|---|---|---|---|---|
| 3.11.14 | `a7 0d 0d 0a` | 3495 | 1054 | 80 | 31 |
| 3.12.12 | `cb 0d 0d 0a` | 3531 | 937 | 62 | 26 |
| 3.13.12 | `f3 0d 0d 0a` | 3571 | 943 | 70 | 26 |
| 3.14.6 | `2b 0e 0d 0a` | 3627 | 965 | 96 | 35 |

`code/main.py` 中的 `read_pyc_header()` 返回这些字段，并根据字值给出解释器版本名。`importlib.util.MAGIC_NUMBER` 对当前解释器给出同样的四个字节。`python3 code/main.py compare` 会用 `PATH` 上找到的每个 `python3.11` 到 `python3.15` 编译源文件。对每个解释器，它把我们的读取器看到的文件头，与该解释器自报的事实并排打印。

## marshal 逐字段写入代码对象

每个样例的第 16 个字节都是 `e3`。它是 `0x80 | ord('c')`：带 `FLAG_REF` 的代码对象的 marshal 类型码，告诉读取器记住这个对象，供之后的 `r` 反向引用使用。marshal 的文档明说该格式有意不公开文档，且代码对象在不同版本之间不兼容。这两句话都属实，而这个格式仍小到可以手读。`Python/marshal.c` 列出了全部类型码：`N` 表示 None，`i` 表示 32 位 int，`s` 表示 bytes，`z` 和 `Z` 表示短 ASCII 字符串，`(` 和 `)` 表示元组，`c` 表示代码对象，`r` 表示反向引用。容器会先为自身预留引用槽，再写子对象，读取器也按同样顺序处理。

`w_complex_object()` 按固定顺序写入代码对象。对 3.11 到 3.15，顺序是：以 32 位整数写 `argcount`、`posonlyargcount`、`kwonlyargcount`、`stacksize`、`flags`，再以对象形式写 `code`、`consts`、`names`、`localsplusnames`、`localspluskinds`、`filename`、`name`、`qualname`，然后写 `firstlineno`，最后写 `linetable` 和 `exceptiontable`。下面从第 17 个字节起，手工解码 3.14 文件的模块代码对象：

```text
00 00 00 00   argcount 0
00 00 00 00   posonlyargcount 0
00 00 00 00   kwonlyargcount 0
01 00 00 00   stacksize 1
00 00 00 00   flags 0
f3 22 00 00 00   type 's' with FLAG_REF, 34 bytes of co_code follow
80 00  52 06  74 00  52 01  17 00  74 01 ...
```

代码字节每条指令两个字节：`80 00` 是 `RESUME 0`，`52 06` 是 `LOAD_CONST 6`，`74 00` 是 `STORE_NAME 0`，`17 00` 是 `MAKE_FUNCTION`。样例中 3.14 的清单打印出同样的五条模块级语句。`code` 之后是常量，其中装着四个嵌套的代码对象；再之后是 `names`：`('LABELS', 'percent_change', 'grade', 'total', 'grade_label')`。

`code/main.py` 里的 `Unmarshaller` 读取这一布局，也读取 3.9 和 3.10 使用的旧布局——那时 `nlocals`、`varnames`、`freevars`、`cellvars` 还是各自独立的字段。它构建的 `CodeObject` 使用与真实对象相同的属性名。`varnames` 按 `localspluskinds` 中 `CO_FAST_LOCAL` 位过滤 `localsplusnames`。对四个文件里的每一个 `grade`，它都返回 `('score', 'bucket')`、`firstlineno` 为 10，以及非空的 `linetable`。测试 `test_unmarshaller_agrees_with_the_running_interpreter` 会把每个字段与 `marshal.loads()` 读取当前解释器所写文件的结果做比对。另外三个文件我们的代码能读，而当前解释器下的 `marshal` 读不了。

## 四个解释器，grade 的四种字节码

`dis` 文档在每个版本上都带着同一条警告：字节码是 CPython 的实现细节，不保证保持不变。样例展示了这对十一行 Python 代码意味着什么。每个解释器都捕获了 `python3.1N -m dis ledger.py` 的输出。

在 3.11 中，`grade` 开头是这样的：

```text
 11           2 LOAD_FAST                0 (score)
              4 LOAD_CONST               1 (10)
              6 BINARY_OP                2 (//)
             10 STORE_FAST               1 (bucket)
 12          12 LOAD_FAST                1 (bucket)
             14 LOAD_CONST               2 (9)
             16 COMPARE_OP               5 (>=)
             22 POP_JUMP_FORWARD_IF_FALSE     2 (to 28)
```

`BINARY_OP` 位于偏移 6，下一条指令在 10。中间的两个字节是 PEP 659——特化自适应解释器——写入的内联缓存条目。偏移 16 的 `COMPARE_OP` 后面接着 22，说明它占两个缓存条目。除非传入 `show_caches=True`，这些缓存不会显示。3.12 用 `RETURN_CONST` 取代 `LOAD_CONST` 加 `RETURN_VALUE`，把跳转指令改名为 `POP_JUMP_IF_FALSE`，并把 `grade` 从 80 字节压到 62 字节。3.13 打印 `L1` 到 `L4` 标签而不再打印偏移，并给 CLI 增加了 `-C`（显示缓存）和 `-O`（显示偏移）选项。3.14 又移除了 `RETURN_CONST`，加入 `LOAD_SMALL_INT` 和 `LOAD_FAST_BORROW`，并在每个条件跳转之后插入 `NOT_TAKEN`：

```text
 12         20       LOAD_FAST_BORROW         1 (bucket)
            22       LOAD_SMALL_INT           9
            24       COMPARE_OP             188 (bool(>=))
            26       CACHE                    0 (counter: 0)
            28       POP_JUMP_IF_FALSE        3 (to L1)
            30       CACHE                    0 (counter: 0)
            32       NOT_TAKEN
```

`-C -O` 清单显示 `grade` 内部有 13 个 `CACHE` 条目：`BINARY_OP` 之后五个，每个 `COMPARE_OP` 之后一个，每个 `POP_JUMP_IF_FALSE` 之后一个。自适应样例展示了缓存的用途。调用 200 次之后，`dis.dis(grade, adaptive=True)` 在普通清单打印 `RESUME` 和 `COMPARE_OP` 的位置打印 `RESUME_CHECK` 和 `COMPARE_OP_INT`，`total` 则显示 `FOR_ITER_LIST`、`BINARY_OP_ADD_INT` 和 `JUMP_BACKWARD_NO_JIT`。特化形式就住在同一段 `co_code` 字节里。pyc 从不包含它们，因为特化发生在内存中。

这就是反编译器存在版本上限的原因。uncompyle6 3.9.3 声明支持 Python 1.0 到 3.8 的字节码。decompyle3 3.9.3 支持 3.7 和 3.8。二者背后的跨版本反汇编器 xdis 6.3.0 支持到 3.13。pycdc 的构建文件里操作码表最高到 `python_3_13.cpp`，遇到未知操作码就打印 `Unsupported opcode` 并把输出标记为不完整。PyLingual 声称支持 3.6 以来所有已发布版本，并用 3.8 到 3.15 的解释器重新编译、校验自己的输出。面对 3.14 文件，前三个工具要么拒绝要么失败，只有后两个用得上。先读魔法字，再选工具。

## class 文件保存着它自己的字典

JVMS Java SE 25 第 4.1 节把布局定义为一个结构：`magic`、`minor_version`、`major_version`、`constant_pool_count`、常量池、`access_flags`、`this_class`、`super_class`、接口、字段、方法、属性。魔法数是 `0xCAFEBABE`。Java SE 25 虚拟机接受 45 到 69 的 major 版本，minor 为 65535 表示预览特性文件。`Ledger-g.class` 的前十个字节是 `ca fe ba be 00 00 00 41 00 5e`：major 65 表示 Java 21，`0x5e` = 94 是常量池计数。用 javac 25.0.2 从同一份源文件构建的 `Ledger-jdk25.class` 开头是 `ca fe ba be 00 00 00 45`：major 69。

常量池就是 class 文件存放自身字典的地方。`-g` 构建里的第 `#13` 项是一个 `Methodref`，指向类 `#8` 和名称与类型项 `#14`。顺着索引走，得到 `Ledger.grade:(I)I`，这正是 `javap -v` 在 `//` 之后打印的内容。描述符 `(I)I` 表示一个 `int` 参数、一个 `int` 返回值。`total` 上的 `([I)I` 表示参数是 `int` 数组。代码触及的每个方法、字段和类都有这样一个条目；第 4.10 节的校验器会拒绝指令与这些条目不一致的文件。`code/main.py` 中的 `ClassFile` 类读取表 4.4 的全部 17 种常量标签，并把 `Long` 或 `Double` 当作两个槽位。遇到未知标签，或最后一个属性之后还有多余字节，它都会抛错。

属性可以挂在类、每个字段、每个方法以及 `Code` 属性自身之上。JVMS 规定虚拟机必须忽略不认识的属性，因此调试表可以增删而不改变方法体。javac 手册说明：`-g` 生成包括局部变量在内的全部调试信息；默认只生成行号和源文件名；`-g:none` 什么都不生成。三份构建正好量化了这一点：

| 构建 | 字节数 | 常量池计数 | `LineNumberTable` | `LocalVariableTable` | `SourceFile` | `grade` 代码字节数 |
|---|---|---|---|---|---|---|
| `javac -g` | 1832 | 94 | 7 个方法 | 6 个方法 | 有 | 50 |
| `javac` | 1581 | 82 | 7 个方法 | 无 | 有 | 50 |
| `javac -g:none` | 1404 | 79 | 无 | 无 | 无 | 50 |

三份文件中 `grade` 的 50 个代码字节完全相同，测试会检查这一点。`-g` 为 `grade` 增加的是一行 `LocalVariableTable`：start 0、length 50、slot 0、name `score`、descriptor `I`。没有它，反编译器只看到 0 号槽位是 `int` 类型，不得不自行编一个名字。`StackMapTable` 在每份构建中都保留，因为校验器需要它；`BootstrapMethods` 属性也保留，因为 `main` 通过 `invokedynamic` 做字符串拼接。

## 手工解码 tableswitch

JVMS 第 6 章把每条指令定义为一个操作码字节加上该操作码指定的操作数。分支偏移是相对分支操作码地址的 16 位有符号值，这一点与 x86-64 不同——后者的位移相对下一条指令。`tableswitch` 和 `lookupswitch` 开头有 0 到 3 个填充字节，使其 32 位操作数起始于 4 的倍数地址。

下面是 `Ledger-g.class` 中 `grade` 从 pc 4 开始的字节：

```text
aa                 tableswitch at pc 4
00 00 00           padding to pc 8
00 00 00 2c        default: 4 + 44 = 48
00 00 00 06        low 6
00 00 00 0a        high 10
00 00 00 2a        case 6:  4 + 42 = 46
00 00 00 28        case 7:  4 + 40 = 44
00 00 00 26        case 8:  4 + 38 = 42
00 00 00 24        case 9:  4 + 36 = 40
00 00 00 24        case 10: 4 + 36 = 40
```

三十六个字节，一条指令，就是整个 `switch (score / 10)`，9 和 10 共用同一个分支。第 33 课里同一个 `switch` 的原生构建需要边界检查、一个 `lea`、一个 `movslq`、一个 `add`、一次间接跳转，以及另一个节里的五处重定位。而在这里，跳转目标就在指令内部，它们属于一体这个事实也在指令内部。

`disassemble_jvm()` 覆盖全部 202 个已定义操作码及其操作数宽度，遵循填充规则，并按 `javap` 的风格渲染操作数。`compare_with_javap()` 解析捕获的 `javap -c -p` 文本——包括多行的 `tableswitch` 块——再逐条核对偏移、助记符和操作数。四份构建在 108 条指令上全部一致。测试 `test_branch_offsets_are_relative_to_the_branch_instruction` 检查 pc 1 的 `ifne` 落在 6、13 处的 `if_icmpge` 落在 33、30 处的 `goto` 落回 10。

radare2 6.2.4 是为这些课程从源码构建的，它也能读同一个文件。`rabin2 -I` 报告 `arch java`、`lang java 21`、`linenum true`、`lsyms true`，这是它对该版本和两张调试表的解读。`pd 14 @ sym.Ledger.grade` 打印 `tableswitch` 及其指向五个目标的箭头。它还会在这个文件上打印 `Unable to parse code attribute: 2`——这是工具自己承认的局限，也是把 `javap` 留作参照工具（oracle）的理由。与 Python 相比，这一格式的反编译器已经相当成熟。CFR 0.152 是 MIT 许可。Procyon 0.6.0 和 Vineflower 1.12.0 是 Apache-2.0，Vineflower 是 Fernflower 的分支，支持 Java 21。Fernflower 本身位于 IntelliJ Community 仓库中。JADX 1.5.6 面向 Android DEX 格式，JD-GUI 最后一次发布是在 2019 年。Harrand、Soto-Valero、Monperrus 和 Baudry 在 2020 年测量了其中八个工具：最好的在 84% 的类上产出语法正确的输出，在 78% 的类上产出语义等价的输出。他们的元反编译器恢复了 37.6% 的任何单一工具都处理不了的类。类型、名称和经过校验的栈使这一格式可反编译。但它们并不能让任何单一反编译器做到完整。

## Wasm 模块是一列节

核心规范把模块定义为前导码 `00 61 73 6d 01 00 00 00`（字符串 `\0asm` 加版本 1），其后跟随若干节。每一节是一个 id 字节、一个 LEB128 编码的 `u32` 长度，以及节内容。id 1 到 13 是已知节，每节至多出现一次并按顺序排列。id 0 是自定义节，带名称，可出现在任意位置。所有整数都是 LEB128，所以 `e4 00` 是 100，`80 08` 是 1024。对 `ledger.wasm` 运行 `wasm-objdump -h` 列出七个节，`WasmModule` 找到的也是同样七个节、同样的载荷偏移：

| 节 | 载荷 | 数量 | 对 ledger 而言保存的内容 |
|---|---|---|---|
| Type | 10 到 26 | 3 | `(i32) -> nil`、`(i32, i32) -> i32`、`(i32) -> i32` |
| Import | 28 到 39 | 1 | 类型为 0 的 `env.log` |
| Function | 41 到 46 | 4 | 每个已定义函数的类型索引 |
| Memory | 48 到 51 | 1 | 一页，无最大值 |
| Export | 53 到 110 | 5 | `memory` 和四个函数名 |
| Code | 113 到 271 | 4 | 大小分别为 23、63、44、23 字节的函数体 |
| Data | 273 到 290 | 1 | 偏移 1024 处的 `F\0D\0C\0B\0A\0` |

Function 节和 Code 节之所以分开，是为了让浏览器能在整个文件到达之前就开始编译。每个代码条目由长度、局部变量声明向量和一个表达式组成。对 `grade` 而言，局部变量向量是 `01 7f`：一组、一个 `i32`。那就是 `bucket`，有类型，没有名字。

控制流在构造上就是结构化的。规范规定分支只能指向外层的 `block`、`loop` 或 `if` 结构。标签按嵌套深度索引。`br_table` 通过一个操作数做间接跳转，该操作数索引一张标签表。解码 `grade` 在文件偏移 `0xa3` 处的 switch：

```text
0e                  br_table
0b                  11 labels follow
04 04 04 04 04 04   buckets 0 to 5: label 4, the outermost block, returns 0
03 02 01 00         buckets 6, 7, 8, 9: labels 3, 2, 1, 0
00                  bucket 10: label 0, shared with 9
04                  default label 4
```

每个标签是一个深度。标签 0 是最内层的 `block`，其 `end` 之后的代码是 `i32.const 4`，随后 `br 4` 跳出结果块。表之前的六条 `block` 指令、十一个标签，以及表之后的五条 `br` 指令，合起来就是整个 `switch`；任何离开结构的跳转都会被校验器拒绝。`disassemble_wasm()` 跟踪深度，使清单像 `wasm-objdump` 那样缩进；它覆盖从 `unreachable` 到 `i64.extend32_s` 的所有单字节操作码，并在两份构建上与捕获的 `wasm-objdump -d` 在 77 条指令上全部一致。

名称是普通模块唯一丢弃的东西。`wasm-objdump` 仍打印 `<grade>`，因为 Export 节把这个字符串映射到函数索引 2，读取器也照做。在 290 字节的文件里，局部变量名根本不存在。`wat2wasm --debug-names` 会添加一个名为 `name` 的自定义节，其子节包含模块名、函数名和局部变量名。文件增长到 425 字节。读取器为函数 2 找到 `{0: 'score', 1: 'bucket'}`，为函数 3 找到 `{0: 'values', 1: 'count', 2: 'sum', 3: 'i'}`。`wasm-strip` 会再次移除该节。tool-conventions 仓库还描述了另外两种载体：DWARF 可以作为 `.debug_*` 自定义节嵌入；`sourceMappingURL` 自定义节可以指向 source map，其列号就是二进制中的字节偏移。

工具链在 2026 年发生了变化。2026-09-15 发布的 wabt 1.0.42 移除了 `wasm-decompile`，本机 Homebrew 构建恰好只剩其余十一个工具。wasm-tools 1.261.0 提供 `print`、`objdump` 和 `demangle`。Binaryen version_133 提供 `wasm-dis`。radare2 6.2.4 也能读这个模块，它的 `rabin2 -I` 报告 `arch wasm`。它给 `grade` 的符号起始于 `0x8e`，即局部变量向量开始处，而 `wasm-objdump` 把函数标在 `0x8b`，即长度字段开始处。它的清单使用较旧的助记符 `get_local`、`set_local` 和 `brtable 11 4 4 ...`，并打印标签数量。三个工具看到的字节相同，拼写不同，所以做比对时必须说明清单是用哪个工具捕获的。为编写本课所做的调研在 Ghidra 的 processor 目录里没有找到 WebAssembly 处理器模块，且该目录列表并未完整阅读。第三方插件 ghidra-wasm-plugin v2.4.0 面向 Ghidra 12.0。规范本身是一份仍在演进的文档。W3C 的 `wasm-core-2` 页面是 2026 年 10 月 13 日的候选推荐草案，摘要描述的是 3.0 版；编辑草案日期相同。

## ECMA-335 元数据是程序自带的数据库

.NET 程序集是一个 PE 文件。`LedgerNet.dll` 以 `MZ` 开头，有三个节：`.text`、`.rsrc` 和 `.reloc`；它声明的 machine 为 `0x14c`，因为编译器生成的是平台中立代码。可选头的数据目录 14，也就是 CLR Runtime Header，指向 RVA `0x2008`。ECMA-335 II.25.3.3 定义了那里的 72 个字节：

```text
48 00 00 00   Cb 72
02 00 05 00   MajorRuntimeVersion 2, MinorRuntimeVersion 5
78 21 00 00   MetaData RVA 0x2178
84 07 00 00   MetaData size 1924
01 00 00 00   Flags: COMIMAGE_FLAGS_ILONLY
05 00 00 06   EntryPointToken 0x06000005, MethodDef row 5, which is Main
```

位于 RVA `0x2178` 的元数据根以签名 `42 53 4a 42` 开头，读作 `BSJB`；接着是版本 1.1 和字符串 `v4.0.30319`。其后是五个流头：偏移 `0x6c` 处的 `#~`，然后是 `#Strings`、`#US`、`#GUID` 和 `#Blob`。II.24.2 规定每种流至多出现一次。`#~` 流以一个 `HeapSizes` 字节开头（此处为零，所以每个字符串、GUID 和 blob 索引都是两字节宽），随后是一个 64 位 `Valid` 掩码，标明存在哪些表。共有十四个位被置位：

| 表 | Id | 行数 | 行大小 | 行内容 |
|---|---|---|---|---|
| Module | 0x00 | 1 | 10 | 模块名及其 GUID |
| TypeRef | 0x01 | 21 | 6 | `System.Object`、`System.Console`，另有 19 个 |
| TypeDef | 0x02 | 4 | 14 | `<Module>`、`Ledger`，以及两个编译器辅助类型 |
| Field | 0x04 | 2 | 6 | `Labels` 和一个静态数组初始化器 |
| MethodDef | 0x06 | 6 | 14 | 五个方法和 `.cctor` |
| Param | 0x08 | 5 | 6 | `before`、`after`、`score`、`values`、`score` |
| MemberRef | 0x0a | 16 | 6 | 12 个属性构造器和 `Main` 调用的 4 个方法 |
| CustomAttribute | 0x0c | 12 | 6 | 编译器和 SDK 添加的属性 |
| StandAloneSig | 0x11 | 3 | 2 | 三个局部变量签名 |
| Assembly, AssemblyRef, ClassLayout, FieldRva, NestedClass | | 1, 2, 1, 1, 1 | | 标识、引用、布局 |

行大小不存储在任何地方。II.22 规定每一列的类型，II.24.2.6 给出行大小规则。堆索引按 `HeapSizes` 取 2 或 4 字节。简单表索引在该表行数少于 65536 时取 2 字节。一个编码索引（coded index）若能指向 n 张表、带 b 个标签位，则当所有这些表的行数都少于 2^(16-b) 时取 2 字节。`CliAssembly` 根据这些规则和行数算出 14 个行大小。样例中 System.Reflection.Metadata 的转储报告了相同的行大小和表偏移，测试正是用它验证这套算术。

III.1.9 定义了 token：高字节是表，低三字节是从 1 开始的行号。`0x06000002` 是 `MethodDef` 第 2 行。该行的 RVA 为 `0x2060`，名称为 `Grade`，签名 blob 为 `00 01 08 08`：默认调用约定、一个参数、返回 `int32`、参数 `int32`。它的 `ParamList` 指向 `Param` 表第 3 行，名字是 `score`。正因为有 Param 表，ILSpy 才能从剥离的 release 构建里打印出 `Grade(int score)`。在 RVA `0x2060` 处，方法体以一个 fat 头开始：

```text
13 30         flags 0x13: fat, init locals; size 3 dwords
02 00         MaxStack 2
2d 00 00 00   CodeSize 45
01 00 00 11   LocalVarSigTok 0x11000001
```

`StandAloneSig` 第 1 行保存 `07 01 08`：`LOCAL_SIG`、一个局部变量、`int32`。这个局部变量有类型、没有名字。ECMA-335 把局部变量名放在 Partition V 的调试交换格式里，.NET SDK 会把它们写入 PDB。本样例用 `DebugType` none 构建，所以没有任何表、也没有任何文件携带 `bucket` 这个名字。`PercentChange` 反而使用 tiny 头：它的第一个字节 `3a` 是 `0x0e << 2 | 2`，表示 14 字节的方法体、无局部变量、默认栈深 8。

`Grade` 的 45 个代码字节按 Partition III 的操作码表解码。`switch` 的操作码是 `0x45`，后跟一个 `u32` 计数和相应数量的 `int32` 目标；目标相对表后那条指令计算：

```text
IL_0000: 02           ldarg.0
IL_0001: 1f 0a        ldc.i4.s 10
IL_0003: 5b           div
IL_0004: 0a           stloc.0
IL_0005: 06           ldloc.0
IL_0006: 1c           ldc.i4.6
IL_0007: 59           sub
IL_0008: 45 05 00 00 00 08 00 00 00 06 00 00 00 04 00 00 00 02 00 00 00 02 00 00 00
                      switch (IL_0029, IL_0027, IL_0025, IL_0023, IL_0023)
IL_0021: 2b 08        br.s IL_002b
IL_0023: 1a 2a        ldc.i4.4, ret
```

编译器在 `switch` 之前减去了 6，所以 case 6 对应第一个目标。`disassemble_cil()` 通过表来渲染 token：`ldsfld Labels`、`call Grade`、`call System.Console::WriteLine`，以及从 `#US` 堆取出的 `ldstr ' '`。这一切的参照工具是 `MetaDump`——样例中的一个短 C# 程序，基于 `System.Reflection.Metadata`（ILSpy 内部使用的库）构建。它打印每张表的行数、大小和偏移，每个方法的 RVA、签名、参数名和 IL 字节，以及 `System.Reflection.Emit.OpCodes` 的全部 226 个字段。`compare_with_metadump()` 在全部 246 项检查上一致。运行时表有 218 个操作码，外加 8 个 `prefix` 占位符。本课的表有 219 个条目，因为它包含 III.2.2 的 `no.`，而 Emit 表从未有过这一项。

这一格式的工具：ILSpy 11.0（MIT，2026-08-12 发布）通过其 Avalonia 桌面构建在 macOS 上运行。`ilspycmd` 11.1.0.9782 用 `-il` 打印 IL，用 `-m 0x06000002` 按 token 选择方法。dnSpyEx 6.6.0 只在 Windows 上延续 dnSpy。IL 反汇编器 `ildasm` 以 NuGet 包形式发布，而不在 SDK 里。radare2 6.2.4 也能读这些元数据：`rabin2 -I` 报告 `arch cil` 和 `class PE32`，`is` 列出 `Ledger.Grade(int:score):int`，参数名和两个类型都取自 Param 和签名表。这个构建里没有 CIL 反汇编器。本样例用 .NET SDK 8.0.205 构建，因为构建机器上安装的是这个 SDK。.NET 10 自 2025-11-11 起是当前版本，它写出的 ECMA-335 结构相同。

## 每种容器保留了什么

把五种容器里的 `grade` 放进同一张表。每一行都来自 `python3 code/main.py keeps`，该命令在样例上运行这些读取器：

| 容器 | 函数体字节数 | 函数名 | 参数名 | 参数与返回类型 | 局部变量名 | 行号 | 控制流 |
|---|---|---|---|---|---|---|---|
| ELF 目标文件，clang -O0 | 118 | `.symtab` 中的 `grade` | 无 | 无 | 无 | 无，没有 `.debug_*` 节 | 5 次跳转，以及 1 次经重定位表的间接跳转 |
| CPython 3.14 pyc | 96 | `co_qualname` | `score` | 无 | `bucket` | `co_linetable`，首行 10 | 4 组比较与 `POP_JUMP_IF_FALSE` |
| JVM class，`javac -g` | 50 | 常量池 | `LocalVariableTable` 中的 `score` | `(I)I` | 除参数外无 | `LineNumberTable`，6 行 | 1 个 `tableswitch` |
| WebAssembly | 63 | Export 节 | 仅在有 name 节时 | `(i32) -> i32` | 仅在有 name 节时 | 无 | 6 个嵌套 block、1 个 `br_table`、4 个 `br` |
| .NET 程序集 | 45 | `MethodDef` | `Param` 中的 `score` | `static int32(int32)` | 无，类型仅在 `StandAloneSig` | 无 PDB 时没有 | 1 个 `switch`、1 个 `br.s` |

这张表解释了你会看到的反编译结果。第 34 课把 `int total(const int *, int)` 与 Ghidra 从原生代码推断出的 `int total(long param_1, uint param_2)` 做了对比，因为原生代码一列都不保留。pyc 保留所有名称、不保留类型，其控制流是每次发布都会变形的跳转，所以 Python 反编译器按版本失效，却能免费恢复名称。class 文件保留类型和经过校验的栈，开启 `-g` 时保留名称，并把 `switch` 表达成一条指令。Wasm 模块在构造上保留结构，所以 `wasm2wat` 是无损往返，唯一丢失的是名称。.NET 程序集保留最完整的描述，因为元数据表是规范性的，但它仍然需要 PDB 才能得到一个局部变量的名字。

这张表告诉你反编译器手里有什么可用，以及它的哪些输出是照抄存储的事实、哪些是推断。ILSpy 输出里的参数名是照抄；同一份输出里的局部变量名则是推断，除非有 PDB。第 35 课把这一差别记为每条结论的证据等级。

## 构建它

运行演示：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
```

输出是一份 JSON 报告。对每个 pyc，它给出两个文件头、文件大小和 `grade` 大小、来自捕获清单的指令数与操作码名称，以及当前解释器能否读取。对每份 class 构建，它给出 major 版本、大小、调试表计数和与 `javap` 的一致情况。对每份 Wasm 构建，它给出节、类型、函数、name 节内容，以及与 `wasm-objdump` 的一致情况。对程序集，它给出 CLI 头、流、表行数、方法，以及与 System.Reflection.Metadata 的一致情况。最后以 keeps 表和 `PATH` 上找到的解释器列表收尾。

按以下顺序阅读 `code/main.py`：

1. `Reader` 持有游标和整数宽度。`uleb()` 和 `sleb()` 是 Wasm 与 .NET blob 堆所需的 LEB128 读取器。
2. `read_pyc_header()`、`Unmarshaller` 和 `PycFile` 读取 pyc。`parse_dis_text()` 读取捕获的 `dis` 清单，使跨版本比较在测试时无需四个解释器。
3. `ClassFile` 读取 class 文件，`disassemble_jvm()` 解码其代码。`parse_javap_code()` 和 `compare_with_javap()` 是与参照工具的比对。
4. `WasmModule` 读取各节，`disassemble_wasm()` 带着深度解码表达式。`compare_with_wasm_objdump()` 是与参照工具的比对。
5. `CliAssembly` 走遍 PE、CLI 头、元数据根、流和表。`TABLE_SCHEMAS` 和 `CODED_INDEXES` 就是 II.22 和 II.24.2.6 的数据化形式。`disassemble_cil()` 解码方法体，`compare_with_metadump()` 负责核对。
6. `NativeObject` 读取足够多的 ELF 以列出函数符号，并把第 33 课的清单当作文本。`what_each_keeps()` 和 `figure_data()` 构建对比。

另有五个命令。`python3 code/main.py pyc FILE`、`class FILE`、`wasm FILE`、`dll FILE` 会打印结构并列出你给定文件中的每个函数。`python3 code/main.py compare` 用 `PATH` 上的每个解释器编译 `ledger.py`，打印各自写出的内容。

## 使用它

对一个你自己构建的容器，按以下顺序使用这些读取器。

1. 读前几个字节，判定格式。两个字节后跟 `0d 0a` 是 pyc；`ca fe ba be` 是 class 文件；`00 61 73 6d` 是模块；`4d 5a` 且数据目录 14 已设置则是程序集。
2. 读文件头，写下它指明的生产者版本。对 pyc 是魔法字，对 class 文件是 major 版本，对程序集是运行时和元数据版本字符串。
3. 在运行任何反编译器之前，先核对该版本是否在它的支持范围内。记录上限，以及工具在何处记录了它。
4. 在同一文件上运行你的读取器和格式自带的工具，逐行比对。不一致要么是读取器的 bug，要么是工具变了；在清单成为证据之前，必须先把这两种情况弄清楚。
5. 为你关心的函数填出 keeps 表。把每个事实标记为保留或丢失，并写明它来自哪张表、哪个属性或哪个节。
6. 只有到这时才读反编译器的输出。把它里面的每个名称和类型标记为对已保留事实的照抄，或推断。

把参照工具的文本文件与容器一起保留。当解释器、JDK、SDK 或 wabt 版本变化时，用 `build_fixtures.sh` 重建并重新捕获，因为清单变了就意味着样例变了。

## 交互实验

### Lab A：先读文件头，再读代码对象

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats/code
python3 -q
```

运行：

```python
from main import *
pyc = load_pyc("ledger.cpython-311.pyc")
pyc.header.to_dict()
pyc.module.names
pyc.functions()["grade"].varnames
len(pyc.functions()["grade"].code)
```

在第一次调用之前写下魔法字。然后加载 `ledger.cpython-314.pyc`，比较两个 `code` 长度。再在你的解释器下对 3.11 文件尝试 `import marshal; marshal.loads(pyc.data[16:])`，读一读报错信息。

### Lab B：常量池链

```python
cls = load_class("Ledger-g.class")
cls.constant(13)
cls.constant(14)
cls.describe_constant(13)
[row["text"] for row in jvm_listing(cls, "grade")]
```

在你调用 `describe_constant` 之前，先手工沿 `#13` 追到 `#8` 和 `#14`，再追到 `Utf8` 条目。然后加载 `Ledger-gnone.class`，对两者都调用 `cls.method("grade").attribute_names()`。`Code` 数组相同，属性不同。

### Lab C：深度与标签

```python
module = load_wasm("ledger.wasm")
[(row["depth"], row["text"]) for row in wasm_listing(module, "grade")]
module.function_name(2)
load_wasm("ledger-names.wasm").names["locals"][2]
```

找到 `br_table` 那一行，读出它的深度。数一数它后面的 `end` 行，把每一个与表中的标签号对应起来。然后打开 `ledger.wat`，找到标签 3 所指的 `block $six`。

### Lab D：从 token 到字节

```python
asm = load_assembly("LedgerNet.dll")
asm.method("Grade")
body = asm.method_body(asm.method("Grade")["RVA"])
body.fat, body.max_stack, body.code_size, hex(body.local_signature_token)
asm.token_text(0x11000001)
[row["text"] for row in cil_listing(asm, "GradeLabel")]
```

手工从 `.text` 节头算出 RVA `0x2060` 的文件偏移，并与 `asm.rva_to_offset(0x2060)` 对比。然后查看 `asm.row(0x08, 3)`，找出 `score` 这个名字存放在哪里。

## 实践练习

给比较再加一种源语言，或给表再加一个事实。

要求：

- 用一种你有编译器的语言写一个带循环和局部变量的函数，例如用 Kotlin 编译成 class 文件，或用 Rust 编译成 Wasm 模块。把产物和参照工具清单连同确切命令一起提交到 `build_fixtures.sh`。
- 让对应的读取器在每条指令上都与参照工具一致。只在样例需要时添加操作码。
- 扩展 `what_each_keeps()`，加入新容器，并为每个容器再加一个事实：函数体是否携带异常表或 `try` 区域。
- 把新容器加入 `figure_data()`，并重新生成 `site/figures-re-43.js`。
- 写一个测试：当读取器把某个事实报告为保留、却没有指明承载它的表、属性或节时，测试失败。

成功标准：`python3 code/main.py keeps` 显示新行，且每个单元格都有证据；所有参照比对仍然一致。

## 课程产物

本课包含 `outputs/skill-bytecode-and-managed-formats.md`。用它把字节码容器变成一份文件头报告、一份经过核对的单函数清单，以及一张把每个名称、类型和结构标记为保留或丢失的事实表。它要求以格式自带的工具作为参照，在任何反编译之前先做版本检查，并说明每个丢失的事实原本应存放在哪里。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应当证明：

- 四个 pyc 文件头携带魔法字 3495、3531、3571、3627，标志 3，以及 8 字节哈希；时间戳文件携带标志 0、mtime 1791975565、大小 497
- 3.11 及以后每个版本的魔法字，都落在 `2900 + 50n` 规则为其版本给出的区间内
- 反序列化器返回的名称、常量、代码字节、变量名和行表，与 `marshal.loads()` 读取当前解释器所写文件的结果一致
- 四个文件都保留模块名、`grade` 的 `('score', 'bucket')`、首行 10，以及文件名 `ledger.py`
- `grade.co_code` 在四个版本中分别 80、62、70、96 字节，捕获清单分别有 31、26、26、35 条指令
- 3.11 清单有 `POP_JUMP_FORWARD_IF_FALSE`，3.12 和 3.13 有 `RETURN_CONST`，3.14 有 `LOAD_SMALL_INT`、`LOAD_FAST_BORROW` 和 `NOT_TAKEN`
- 3.14 的 `-C -O` 清单在 `grade` 中显示 13 个缓存条目，其中 5 个在 `BINARY_OP` 之后；自适应清单显示 `COMPARE_OP_INT`
- 缺少 `0d 0a` 的文件头、被截断的文件头，以及不是代码对象的函数体，都会抛出 `FormatError`
- 指令清单只对当前解释器写出的 pyc 存在，且与捕获的 `dis` 输出一致
- class 文件的 major 为 65 或 69，大小分别为 1581、1832、1404、1581，常量池计数分别为 82、94、79、82
- 常量 `#13` 解析为 `Method Ledger.grade:(I)I`，`#47` 解析为 `String F`
- JVM 反汇编器在四份构建的全部 108 条指令上都与 `javap -c -p` 一致
- pc 4 处的 `tableswitch` 占 36 字节，low 6、high 10、default 48，目标为 46、44、42、40、40
- 三个调试级别改变属性计数，而 `grade` 的 50 个代码字节保持不变
- `ifne`、`if_icmpge` 和 `goto` 的目标都相对分支操作码计算
- 错误的魔法数、尾随字节和未定义操作码都会抛出 `FormatError`
- 七个 Wasm 节的载荷偏移与 `wasm-objdump -h` 打印的一致；带名称的构建在 293 到 425 之间新增一个 `name` 自定义节
- 类型、`env.log` 导入、五个导出和偏移 1024 的数据段，都按 `ledger.wat` 中所写解码
- Wasm 反汇编器在两份构建的全部 77 条指令上都与 `wasm-objdump -d` 一致
- `grade` 有 6 个嵌套 block、1 个标签为 `4 4 4 4 4 4 3 2 1 0 0` 且 default 为 4、位于深度 6 的 `br_table`，以及 1 个 `i32` 局部变量
- 没有 name 节时，函数名来自导出；局部变量名只有带 name 节时才存在
- LEB128 的 `e4 00` 是 100，`7f` 是 -1，`80 08` 是 1024；块类型的渲染与 `wasm-objdump` 一致
- 错误的前导码、版本 2 和未知操作码都会抛出 `FormatError`
- 程序集是 PE32，machine 为 `0x14c`，CLI 头在 RVA `0x2008`，运行时 2.5，`ILONLY`，入口点 `0x06000005`
- 元数据版本是 `v4.0.30319`，存在五个流，14 张表的行数和行大小与参照工具报告一致
- .NET 读取器在全部 246 项检查上都与 System.Reflection.Metadata 一致
- `Grade` 的参数为 `score`，签名为 `static int32(int32)`，fat 头中 `MaxStack` 为 2、代码 45 字节，并有 1 个没有名字的 `int32` 局部变量
- CIL 的 `switch` 目标、`br.s`、`ldsfld`、`call` 和 `ldstr` 操作数都能通过表和堆解析
- CIL 操作码表覆盖运行时全部操作码，且只额外包含 `no.`
- keeps 表在四种容器中报告参数名，原生代码一栏为无；在三种容器中报告类型
- 演示中的每一行参照比对都读到 n 对 n
- 交互图嵌入的数据与 `figure_data()` 产生的相同

## 生产环境故障模式

| 故障 | 弱工具会报告什么 | 分析必须证明什么 |
|---|---|---|
| 反编译器比解释器旧 | `Unsupported opcode`、输出被标记为不完整，或声称文件被混淆 | 魔法字、它指明的版本，以及工具记录的支持上限 |
| 用错误的解释器读 pyc | `marshal.loads` 抛错或返回垃圾数据 | 我们的反序列化器能从任何 3.9 到 3.15 的文件读取名称、常量和大小，指令则来自匹配的 `dis` |
| 把特化操作码当成文件内容 | 清单里出现 `COMPARE_OP_INT` | pyc 保存的是未特化形式，特化在预热之后发生于内存 |
| 默认 javac 构建却报告局部变量名 | 把 `var1`、`var2` 当作源码中的名字 | `LocalVariableTable` 只在 `-g` 时存在，两种情况下的 50 个代码字节完全相同 |
| 分支目标按指令长度算错 | JVM 清单的跳转落在指令中间 | 偏移相对分支操作码计算，并与 `javap` 核对 |
| 跳过 switch 填充 | `tableswitch` 的 default 错误、case 整体偏移 | 0 到 3 个填充字节使 pc 对齐到 4，五个 case 共 36 字节 |
| 以为 Wasm 名称已被剥离 | 「没有符号」 | 导出节为每个导出函数命名，局部变量需要 `name` 节，`wasm-strip` 只移除该节 |
| 三个工具三种拼写 | `get_local` 对 `local.get`，函数在 `0x8b` 还是 `0x8e` | 字节相同，比对时必须说明清单捕获自哪个工具 |
| 猜测表的行大小 | 第一张表之后的每一行都读错 | 行大小由 `HeapSizes`、行数和编码索引宽度决定，且与参照工具一致 |
| 在 release 程序集里期待局部变量名 | 把 `V_0` 当作源码中的名字 | 局部变量在 `StandAloneSig` 中只有类型，名字只在 PDB 里；`DebugType` none 没有写出 PDB |
| 读取器猜测未知操作码或类型码 | 给出看似合理的助记符 | 抛出带偏移的 `FormatError`，参照比对就此停止 |
| 参照清单捕获自另一份构建 | 测试在过时文本上通过 | `build_fixtures.sh` 记录每个工具版本，并写出 `versions.txt` 和 `sha256.txt` |

## 操作规则

先读文件头，说出它指向的生产者。用一个对未知内容一律拒绝的读取器读容器，并在同一文件上把它的清单与格式自带的工具核对。然后，对反编译器输出里的每个名称和类型，说明它是容器存储的，还是工具推断的。

## 延伸阅读

- [PEP 552: Deterministic pycs](https://peps.python.org/pep-0552/)
- [CPython `Include/internal/pycore_magic_number.h`](https://github.com/python/cpython/blob/main/Include/internal/pycore_magic_number.h)
- [Python `marshal` module documentation](https://docs.python.org/3/library/marshal.html)
- [CPython `Python/marshal.c`](https://github.com/python/cpython/blob/main/Python/marshal.c)
- [Python 3.14 `dis` module documentation](https://docs.python.org/3.14/library/dis.html)
- [PEP 659: Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/)
- [PEP 626: Precise line numbers for debugging and other tools](https://peps.python.org/pep-0626/)
- [PEP 790: Python 3.15 Release Schedule](https://peps.python.org/pep-0790/)
- [uncompyle6 on PyPI](https://pypi.org/project/uncompyle6/), [decompyle3 on PyPI](https://pypi.org/project/decompyle3/), [xdis on PyPI](https://pypi.org/project/xdis/), [pycdc on GitHub](https://github.com/zrax/pycdc), [PyLingual on GitHub](https://github.com/syssec-utd/pylingual)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 4: The class File Format](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 6: The Java Virtual Machine Instruction Set](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-6.html)
- [javac manual, Java SE 25](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javac.html) and [javap manual](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javap.html)
- [CFR](https://www.benf.org/other/cfr/), [Procyon](https://github.com/mstrobel/procyon), [Vineflower](https://github.com/Vineflower/vineflower), [Fernflower in intellij-community](https://github.com/JetBrains/intellij-community/tree/master/plugins/java-decompiler/engine), [JADX](https://github.com/skylot/jadx)
- [Harrand, Soto-Valero, Monperrus, Baudry, "Java Decompiler Diversity and its Application to Meta-decompilation" (JSS 2020)](https://arxiv.org/abs/2005.11315)
- [ECMA-335, Common Language Infrastructure, 6th edition (June 2012)](https://ecma-international.org/publications-and-standards/standards/ecma-335/)
- [PE Format, Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [System.Reflection.Metadata.Ecma335.TableIndex](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.metadata.ecma335.tableindex)
- [ILSpy on GitHub](https://github.com/icsharpcode/ILSpy), [ilspycmd on NuGet](https://www.nuget.org/packages/ilspycmd), [dnSpyEx on GitHub](https://github.com/dnSpyEx/dnSpy)
- [WebAssembly Core Specification: Binary Format, Modules](https://webassembly.github.io/spec/core/binary/modules.html) and [Values](https://webassembly.github.io/spec/core/binary/values.html)
- [WebAssembly Core Specification: Custom Sections (name section)](https://webassembly.github.io/spec/core/appendix/custom.html)
- [WebAssembly tool-conventions: Debugging](https://github.com/WebAssembly/tool-conventions/blob/main/Debugging.md)
- [wabt 1.0.42 release notes](https://github.com/WebAssembly/wabt/releases/tag/1.0.42), [wasm-tools](https://github.com/bytecodealliance/wasm-tools), [Binaryen](https://github.com/WebAssembly/binaryen), [ghidra-wasm-plugin](https://github.com/nneonneo/ghidra-wasm-plugin)
- [radare2 on GitHub](https://github.com/radareorg/radare2), version 6.2.4 built from source for this lesson
