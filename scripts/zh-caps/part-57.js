/** zh translations — figures-python (1 figure) */
module.exports = {
  's0-list-vs-generator': '两个循环读同样的八行。list 版本在第一个训练步开始前就把所有行读进内存。generator 版本只握着一个两行的 batch，batch 备好第一步就开跑。换成真实数据集：list 需要装下整个文件的内存，generator 只需要装下一个 batch。',
};
