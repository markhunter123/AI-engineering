/** zh translations — figures-re-32 … figures-re-37 (reverse-engineering lessons, 6 figures) */
module.exports = {
  'elf-byte-map': '条带按比例画出整个目标文件——左端 offset 0x0，右端最后一个字节。蓝色是选中的结构或函数，浅蓝是容纳选中函数的 section，灰框是其他结构，空白是闲置空间。条带下方的行是选中内容的真实字节——每行 16 个，左侧是文件 offset。',
  're-33-sweep-vs-descent': '蓝格是被方法解码成指令的字节。琥珀格是最后一个 retq 之后被当成代码解码的 padding 字节。灰格是无人认领的字节。实线框是方法找到的 basic block，虚线框是它还没找到的 block。箭头是方法已知的边，虚线箭头来自 .rodata 里的跳转表。',
  'lifting-stages-percent-change': '蓝色标记来自 ledger.c 第 3 行（那个除法）的字节、指令、IR 语句、SSA 值和 C 行。黑色标记第 2 行的零守卫。灰色标记 block 标签、花括号和签名——它们没有源行。面板下方的计数由屏幕上的行算出。十六个字节里有九个变成一行 C。',
  're-evidence-ledger': '蓝色实线：通过或一条观测到的记录。蓝色虚线：推导得出。琥珀虚线：未知——只有一部分运行。琥珀实线：一条悬空的链接。灰色：无支撑或未决。verdict 数的是它上方的那些行。',
  're-36-prove-equal': '蓝格和蓝字节与原件一致。琥珀标记不一致的格或字节。虚线列是 relocation 字段——比较前两侧都清零。constants 行放的是 relocation 指向的五个表项。verdict 数格数、未掩码字节和表项。',
  're-37-field-map': '蓝色标记和蓝列属于选中的工具。虚线按钮是本课没有运行的工具。evidence 行报告的是 code/main.py 里的 harness 对选中工具在课例 fixture 上测到的东西：ledger 在 -O1 下的 block 数、grade 在 -O0 下的跳转表、sweep-trap 里的数据字节，以及凭空发明、没有符号的函数。',
};
