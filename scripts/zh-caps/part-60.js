/** zh translations — figures-systems3 (8 figures) */
module.exports = {
  'masked-diffusion-unmask': '文本 token 靠 next-token prediction 从左到右一次生成一个。图像 token 起始全被 mask、并行恢复：每一步同时预测所有被 mask 的格子，留下最有把握的、其余重新 mask。几波之后整幅图填满——步数远少于自回归的图像解码。',
  'any-to-any-stream': '文本、图像、语音、音乐各自 token 化进同一个共享词表，再交织成一条序列、由一个 causal transformer 消费。因为每种模态都只是 token，模型可以输出任何模态——解码流交替 token 类型，流出速度快到能对话。',
  'video-diffusion-denoise': '3D VAE 把片段压成时空 patch，diffusion transformer 对它们去噪。灰色斑点是模型一步步除掉的噪声；蓝色形状跨帧浮现并移动——那是网络必须建模的时间连贯性：同一个物体、光照和运动贯穿整条带。',
  'inpaint-mask-reinject': 'inpainting 只对被 mask 的区域去噪，同时在每一步重新注入已知像素，边界因此保持一致。行进的虚线是 mask 边缘；内侧噪声解析成新内容，周围的上下文被按死、逐像素不变。',
  'agentic-rag-loop': '基础 RAG 检索一次就作答。Agentic RAG 循环：检索候选、推理它们是否真的回答了 query、再行动——改写 query、重排、或者再检索。环绕的 token 是一个 query 在循环里打转，直到 context 好到可以作答——多跳问题要的正是这个。',
  'mcp-nxm-collapse': 'MCP 之前，每个 host 和每个 server 讲一套私有协议，是一张 N×M 集成矩阵。MCP 是中间的一份 JSON-RPC spec：写一个 server，任何合规 host 都能发现并调用它的工具、资源和 prompt。请求流入、结果流回，走同一种线格式。',
  'a2a-task-lifecycle': '一个 Agent 用 SendMessage 发消息，远端 Agent 开一个 Task。调用方只盯它的 status.state：TASK_STATE_SUBMITTED、TASK_STATE_WORKING、有时 TASK_STATE_INPUT_REQUIRED，然后 TASK_STATE_COMPLETED。远端 Agent 的内部推理保持不透明——调用方看到的是状态变化，以及最后作为输出返回的 artifact。',
  'rvq-codec-cascade': '神经音频 codec 不用一个巨型 codebook，而是级联多个小 codebook：第一个量化 encoder 输出，后面每个量化剩下的残差。递减的长条就是残差沿栈变小。强令 codebook 0 承载语言内容（semantic）、其余承载声学细节——token 化语音模型由此成立。',
};
