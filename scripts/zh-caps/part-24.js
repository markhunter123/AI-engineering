/** zh translations — figures-speech2 (8 figures) */
module.exports = {
  'sp-asr-attention': 'Whisper 是 encoder-decoder 结构。encoder 把 30 秒的 log-mel 窗口变成一排音频帧，decoder 一次发一个 token，每一步都对那些帧做 cross-attention。高亮的条带显示：随着转写推进，注意力从左向右漂移——这是学出来的软对齐，而不是 CTC 模型那种固定的逐帧标签。',
  'sp-eer-crossover': '调高阈值，更多真人被拒之门外（false reject rate 上升）；调低阈值，更多冒名者混进来（false accept rate 上升）。橙色线扫过阈值区间，停在两条曲线的交点。这个交叉点就是 Equal Error Rate——说话人验证榜单上人人都要报的那个数字。',
  'sp-tts-stack': '现代 text-to-speech 分三段。frontend 把字符串变成 token，acoustic model 一列一列画出 mel spectrogram，vocoder 把 mel 变成可听的波形。端到端 flow-matching 模型把后两段糅在一起，但推理和调试一个合成语音时，这三段图景依然是你的思考框架。',
  'sp-codec-tokens': '神经音频 codec 用残差向量量化把声音离散化：第一个 codebook 抓住信号本体，后面每个 codebook 各自量化剩下的残差。把第一个 codebook 拆成语义部分（语言内容，金色）与声学残差，transformer 就能像预测词语那样预测语音 token。decoder 把每帧选中的码字逐帧求和，还原成波形。',
  'sp-vad-cascade': '检测器给每 20 ms 的帧打上语音或静音标签。句中短暂的停顿不能结束对话轮次，所以静音要持续撑满 hangover 计时器，端点检测才会触发。hangover 设太短，你会打断用户；设太长，助手永远在等。只有持续静音超过计时器，这一轮才算结束。',
  'sp-fullduplex': 'pipeline 式的 voice agent 有延迟下限，因为每一段都要等上一段。full-duplex 模型把 pipeline 压扁：它一边吃进用户的音频流，一边同时吐出自己的音频流，文本 inner monologue 只是中间产物而非必经阶段。听与说重叠——200 ms 的响应延迟就是这么来的。',
  'sp-wer-align': 'Word error rate 本质是编辑距离。识别结果与参考文本逐词对齐，每个错位记作 substitution、deletion 或 insertion。三者之和除以参考词数就是 WER。这里五个参考词里有一次 substitution（kitchen 错成 chicken）、一次 deletion（the）、一次 insertion（now），共得 60%。',
  'sp-voice-factorize': '语音克隆与转换都建立在一个分解上：把「说了什么」和「谁在说」分开。源音频被分解成 content representation 和 speaker embedding；丢掉源说话人，换上从 B 的五秒录音里取出的 reference embedding。内容与新说话人重新组合，得到 B 说出同样的话——这正是同意门禁和水印必须强制存在的原因。',
};
