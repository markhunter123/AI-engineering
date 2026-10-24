/** zh translations — figures-visaudio4 (9 figures) */
module.exports = {
  'v4-video-temporal': '2D+pooling 模型对每帧跑同一个图像 CNN、再沿时间平均特征，于是「向左推」和「向右推」坍缩成同一个 clip 向量。3D 卷积的核在时间和空间上同时滑动：它一次看到一小段帧窗口，能编码方向。时空 transformer 把这个想法推到底——对所有 (t, h, w) token 联合做注意力，代价是随 clip 长度平方增长。',
  'v4-vision-pipeline': '生产视觉服务是一条契约链：解码字节、检测边框、裁剪每个框、对裁剪分类，最后组装一份结构化响应。每支箭头都是一个可能无声失败的接口（坐标顺序、归一化、缩放插值），所以输出在离开服务前要过一道 Pydantic schema。检测器或分类器随便换；骨架不动。',
  'v4-vlm-projector': '每个生产 VLM 都是同样三件套。ViT 把图像切成 patch 并编码；一个小 MLP 把这些 patch embedding 投进 LLM 的 token 空间；decoder 把图像 token 与文本 token 当作一条交错序列来读——「有几辆红车？」由普通的 next-token prediction 回答。模式就位后，换 ViT、换 projector、换 LLM 都是机械操作。',
  'v4-world-rollout': '纯视频生成从 prompt 预测帧然后停止。world model 是同一个视频 DiT 在循环里跑：每段去噪完的 chunk 被追加进上下文，一个动作 token（左转、开门）条件化下一段预测，模型就变成一个可以驾驶的学习型仿真器。漂移是它的失败模式——每个被幻造出来的像素都会成为未来的输入。',
  'v4-alm-tokens': '音频语言模型保留三件套模板：音频编码器（Whisper、BEATs）把每段声音变成 embedding，projector 把它们映射进 LLM token 空间，decoder 让它们与文本 prompt 内联同读。回答「有人受伤吗」需要同时读到三个事件：那声犬吠、那声喊叫、喊叫之后的寂静。这个联合读取恰恰是纯转写 ASR pipeline 扔掉的东西。',
  'v4-voice-latency': '800 ms 的体感来自重叠，不来自任何一个特别快的组件。STT 在半句话上就开始，LLM 在部分转写上就开始，TTS 在前几个 token 出来后就开始——第一字节目标约 750 ms 落地，卡在预算之内。用户插话时，VAD 事件立刻取消 TTS 和播放：pipeline 里最快的路径就是叫停它的那条。',
  'v4-audio-watermark': 'AudioSeal 风格的水印给生成语音的每个样本加一个学习得到的、不可感知的信号。重点不是保密而是存活：经过 MP3 压缩、裁剪和重采样，检测器仍能恢复把这段音频标记为合成的载荷比特。AASIST、RawNet2 这类检测模型对付不配合的对手；水印管你自己的输出。语音克隆要两个一起上。',
  'v4-controlnet-zero': 'ControlNet 克隆预训练 U-Net 的编码器半边，给克隆喂一张姿态图，再通过初始化为零的 1x1 卷积把它的特征接回冻结的 decoder。第 0 步时旁路恰好是无操作，训练只可能比底模更好；随着 zero-conv 暖起来，姿态开始驾驭生成。LoRA 在权重本身上玩同一个把戏：冻结注意力矩阵旁边的一条低秩增量。',
  'v4-3d-multiview': '2026 年的 text-to-3D 栈把问题一分为二。多视图 diffusion 模型（Zero123、MVDream、SV3D）从一张照片或一句 prompt 生成一圈一致视角；重建步骤再用梯度下降把一团带朝向的 3D Gaussian 拟合到这些视角上。渲染 splat 只是投影加 alpha 合成——所以结果能在消费级 GPU 上跑到实时帧率。',
};
