/** zh translations — figures-nlp5 (8 figures) */
module.exports = {
  'n5-subword-merge': 'Byte-pair encoding 从字符出发，反复合并最高频的相邻对：先 (l,o) 和 (e,r)，再 (lo,w)，然后 (low,er)。高频词最终成为单个 token，生僻词拆成熟悉的碎片——任何输入都不会 out of vocabulary。',
  'n5-crosslingual-bridge': '跨一百种语言预训练的单个 encoder，把同义的句子放进同一个共享空间里彼此相邻。在英文标签上微调一个任务头，再跨桥带过去：零乌尔都语标签也能给乌尔都语文本打分，几百条目标语言样本关掉剩下的大部分差距。',
  'n5-chunk-cuts': '每一行的文档都相同，不同的只是下刀的位置。固定窗口便宜但会把句子拦腰斩断（标记的刀口）；recursive splitting 从段落边界退到句子再退到空格；semantic splitting 在相邻句相似度骤降处下刀。recursive 512-token 分块仍是基准默认。',
  'n5-judge-gauge': 'judge 模型读 query、检索到的 context 和答案，套一条 rubric，返回 0 到 1 的分数。RAGAS、DeepEval、G-Eval 全建立在这个循环上；要做的信任工作是冻结 judge 版本、并审计它偏袒长答案的偏差。',
  'n5-slot-tracker': '三轮对话、对同一张 slot-value 词典的三次修改：先填 price 和 area，然后一次更正把 cheap 改成 moderate，接着 cuisine 到场。joint goal accuracy 只有每个 slot 全对才给这轮记分——对话中途的更正因此是追踪最难的部分。',
  'n5-patch-stream': 'vision transformer 从不卷积：图像切成固定 patch，每个 patch 摊平、线性投影，排成一列 token 排在学出来的 [CLS] token 后面。position embedding 是对原始网格仅存的记忆；再往后就是标准的 transformer encoder。',
  'n5-mel-decode': 'Whisper 用眼睛听声音：波形变成 log-mel spectrogram，encoder 像 ViT 读 patch 那样读谱帧，decoder 一次一个 token 敲出转写，由选择语言和任务的特殊 token 引导。',
  'n5-block-stack': '结业项目把整个阶段叠进一个 decoder-only 模型：底部是 embedding，L 个重复 block 各含 causal attention 和一个 SwiGLU feed-forward，最后是 norm 加 LM head，逐字符采样莎士比亚。小到笔记本跑得动，正确到能继续放大。',
};
