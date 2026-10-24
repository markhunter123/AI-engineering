/** zh translations — figures-multimodal2 (8 figures) */
module.exports = {
  'mm-patch-n-pack': '每张图像保留原生宽高比，各自贡献自己的 patch 数：宽的图表、高的收据、小的图标各不相同。Patch-n-Pack 把它们全部接进同一条 transformer 序列，用块对角 mask 保证一张图只注意自己的 patch。不用缩放成正方形，也不浪费 padding token。',
  'mm-llava-projector': 'LLaVA 用最简单的桥取代 Q-Former 瓶颈：一个两层 MLP，把每个冻结 ViT 的 patch embedding 从视觉维度映射到语言模型的 embedding 维度。每个 patch 变成 LLM 输入序列里的一个 token，直接在语言模型损失上训练。简单者胜。',
  'mm-mrope-axes': 'Qwen2-VL 给每个 token 一个三分量位置，三个分量各绕自己的轴旋转：时间、高度、宽度。时间轴转得慢，一小时后的帧也落在不同的相位上；空间轴转得快，用来编码 patch 网格。一套不依赖绝对位置表的方案，同时覆盖单图、多图 batch 和长视频。',
  'mm-video-token-budget': '视觉 token 数随采样帧数增长，帧数随 FPS 乘时长增长。采样率翻倍，token 数翻倍，一小时的片子会冲垮任何固定的上下文窗口。三条出路：硬扛的百万 token 上下文、跨设备的 ring attention，以及激进的 pooling 或 Agent 式检索——绝不一次性加载整段视频。',
  'mm-action-tokens': 'vision-language-action 模型要发出电机指令，但 transformer 只会说 token。RT-2 把每个归一化的关节目标离散成 256 个 bin 之一，再把 bin 映射成词表 id，于是 10 自由度的动作变成十个普通 token。同一个能给图像写说明的 decoder，现在也能写出控制轨迹——这就是 web 级知识能迁移到机器人身上的原因。',
  'mm-doc-layout': '文档不是照片。标题、正文、图表、表格的含义都来自它们在页面上的位置。Donut、Nougat 这类 OCR-free 模型直接读页面图像、输出结构化标记；layout-aware encoder 则同时融合三路输入：文本内容、bounding-box 布局、图像 patch。「Total: $1,245」摆在哪儿，本身就是答案的一部分。',
  'mm-maxsim': 'Text-RAG 把整页压成一个向量，图表和表格就此丢失。ColPali 给每个图像 patch 保留一个向量、每个查询词保留一个向量，再用 MaxSim 打分：每个查询词在所有页面 patch 上取最大相似度，页面得分是这些最大值之和。关于图表的查询能直接咬住承载它的那个 patch，中间不需要 OCR。',
  'mm-agent-loop': 'computer-use Agent 循环运转：感知屏幕、围绕目标推理、发出结构化动作（点击坐标或键入字符串），然后观察输出的截图，再来一轮。每一轮就是一次多模态模型调用，难的是把动作 grounding 到正确的像素。错误会污染下一次观察，所以恢复能力与最初的计划同样重要。',
};
