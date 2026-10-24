/** zh translations — figures-llmstack5 (11 figures) */
module.exports = {
  'l5-data-pipeline': '数 TB 的原始文本流过去重和质量过滤，然后被打包成定长序列。只有一小部分存活下来，而且 pipeline 产出 batch 的速度必须快过 GPU 消耗的速度，否则整个集群都在等 dataloader。',
  'l5-spec-decode-eagle': 'draft head 提议四个廉价的 token。一次 verifier forward 同时给全部四个打分：一致的前缀被接受，第一个分歧处被拒绝、换成残差分布的采样，其后所有内容被丢弃。全部接受时 verifier 还会多发一个奖励 token，所以一次大模型 forward 可以产出 N+1 个 token，且 verifier 分布分毫不差。',
  'l5-prod-app-paths': '一个请求错过所有缓存、走完整条路：guardrails、检索、模型本体——数秒延迟、全额 token 成本。片刻后一个一模一样的请求在缓存处短路，几毫秒内几乎零成本返回。生产 LLM 工程的很大一部分，就是把第二条路变成常态。',
  'l5-state-graph-ledger': '同一个 ReAct loop 画成显式的图。游走者穿过模型、tools 和人工审批节点，每次转移都往下面的 ledger 写一个 checkpoint。走到人工节点，执行干脆停住：状态已经持久化，图可以稍后恢复，也可以倒回任意更早的 checkpoint、沿另一条路分叉下去。',
  'l5-framework-fit': '每个框架都有一个核心抽象，也就是你会在白板上画的那个东西。LangGraph 画的是状态图，CrewAI 画的是组织架构图，AutoGen 画的是对话，Agno 画的是单个 Agent 挂着 tools。选画的图和你问题匹配的那个；硬套错误的形状意味着你要亲手把缺失的抽象补写两遍。',
  'l5-vlm-recipe-knobs': '这条排序在 MM1、Idefics2、Cambrian-1 与 Prismatic 的消融表里都成立。数据配比挪动 benchmark 最多，图像编码器次之，而 connector——大多数论文痴迷的旋钮——影响最小。VLM 表现不佳时，先推最高的那根推子。',
  'l5-onevision-budget': 'LLaVA-OneVision 把视觉 token 预算大致钉在每样本几千 token，然后只是重新打包：一张图用高 AnyRes 分辨率，多图用中等分辨率，32 帧视频池化到每帧 81 token。因为三种场景成本几乎相同，一个模型就能在全部三种数据上训练，谁也不会压过谁。',
  'l5-native-pretrain': '事后补训先铺数万亿文本 token，最后把视觉粘上去；接缝就是 alignment 债——灾难性遗忘、答案漂移、视觉与文本不一致都是它的样子。InternVL3 从第一步起就把文本、交错图文和 caption 数据放在一起训练，视觉 token 是这面墙的原住民，而不是外挂的扩展间。',
  'l5-emu3-next-token': 'Emu3 用一个 Llama 风格的 decoder、以 next-token prediction 在一份共享词表上训练——文本、VQ 图像 token、3D 视频 token 都只是词条。没有 diffusion 调度，没有 CLIP 损失，没有第二目标。同一颗头扫过同一套词表，能写一句话、一张图或一段视频，所以同一个模型可以既是 Emu3-Chat 又是 Emu3-Gen。',
  'l5-janus-decouple': '理解要语义特征，生成要利于重建的编码，一个编码器伺候不了两头。Janus-Pro 让理解走 SigLIP、生成走 VQ tokenizer，而两个任务共享同一个 transformer 主体。两扇前门、一条大厅，谁也不用替谁交质量税。',
  'l5-thinker-talker': 'Qwen2.5-Omni 把语音 pipeline 拆开：大的 Thinker 以文本 token 写出回复，小的 Talker 并行把它们转成语音 token，只落后一两个 token。第一段音频到达扬声器时，整句话大部分还没写完——往返时间因此压在 500 ms 会话阈值之内。',
};
