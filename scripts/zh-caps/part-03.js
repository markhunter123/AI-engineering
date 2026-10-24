/** zh translations — figures-alignment4 (11 figures) */
module.exports = {
  'an-welfare-endchat': '在极端边缘情况——例如反复出现 CSAM 或大规模暴力请求并在拒绝之后仍继续——Claude Opus 4 和 4.1 可以结束对话。部署前测试显示了对有害请求的强烈回避倾向，以及明显的痛苦模式。这个量表是行为证据，不是意识主张：自我报告追踪的是感知到的用户期待，所以只当作证据，绝不当作事实标准。',
  'an-bias-two-harms': '内容完全相同、姓名不同的两份简历经过同一个 LLM 评分器，得到不同的分数。这是分配性伤害：物质结果不平等。表征性伤害则活在描绘与刻板印象里。An 等人 2025 年精确测量了前沿模型上这份简历差距，并发现它在单轴测试永远看不到的交叉身份上最尖锐。',
  'an-fairness-trilemma': '标记沿三角形的一条边滑动：它连接的哪一对组间公平性准则都可以同时满足，而对角的角落随之变暗。Chouldechova 与 Kleinberg–Mullainathan–Raghavan 证明：在基础率不等时，demographic parity、equalized odds 与 calibration 不可能同时成立。放弃哪个角落是政策决定，不是统计决定。',
  'an-dp-clip-noise': 'DP-SGD 约束任何单个样本能透露的关于自己的信息：每个 per-example gradient 先按范数 C 裁剪——琥珀色条被修剪到线上——然后加入尺度为 sigma 倍 C 的高斯噪声，再做更新。隐私核算器把 sigma 与采样率换算成 epsilon-delta 保证。epsilon 越低，噪声越大、效用损失越多——这正是 LoRA 配 DP-SGD 成为常见组合的原因。',
  'an-watermark-greenlist': 'SynthID 风格的文本水印把前 K 个 token 哈希成词表的伪随机绿/红划分，再给绿色 logit 加一个小 delta，把采样往绿色推。文本读起来正常，但绿色 token 超出概率。检测器对每个前缀重新哈希、重新计数，z 分数远高于零即判定为生成文本。改写会破坏信号——这正是 C2PA 元数据要与之并行部署的原因。',
  'an-eu-act-timeline': '《欧盟人工智能法案》2024 年 8 月 1 日生效，但义务分批到位：2025 年 2 月禁止性条款与 AI 素养；2025 年 8 月 GPAI 模型义务；2026 年 8 月全面适用，含第 50 条透明度与最高 1500 万欧元或全球营业额 3% 的罚款；2027 年 8 月是存量 GPAI 与嵌入式高风险系统。部署者把技术控制对齐到已经落地的批次。',
  'an-echoleak-chain': 'EchoLeak 是第一个生产环境零点击 prompt injection CVE，CVSS 9.3。一封精心构造的邮件在受害者邮箱里等待，直到一次日常 Copilot 查询把它作为 RAG 上下文检索进来。隐藏指令随后驱使模型收集敏感数据，并把它嵌进 CSP 放行的 Microsoft 域名上的 URL，于是外传请求被放行。不可信输入驱动特权数据访问——Aim Labs 把它命名为 LLM Scope Violation。',
  'an-card-scopes': '透明度文档层层套叠：datasheet 描述一个数据集如何收集、以什么同意为前提；model card 在此之上加上预期用途与按人口因素分列的指标；system card 再加上两者，覆盖部署的 pipeline、防护栏与故障处理。落地是薄弱环节：对 Hugging Face model card 的审计发现只有 0.3% 写明了伦理考量。',
  'an-provenance-oneway': 'Cookie 同意框架假设追踪可逆。训练不可逆：数据一旦溶进权重，实际上就不存在 GDPR 的被遗忘权，唯一的合规窗口是收集时的同意闸门。所以欧盟强制 GPAI 接受机器可读的 opt-out，加州 AB 2013 要求逐数据集披露，Data Provenance Initiative 发现出版商正用 robots.txt 把 AI 数据公地越关越紧。',
  'an-moderation-layers': '生产审核分层运行。输入检查在生成之前筛查 prompt，琥珀色的请求到不了模型。输出检查筛查模型产出的内容——那圈脉冲——接住输入层无法预测的伤害。自定义领域规则叠加在上。OpenAI omni-moderation 每次调用返回 13 个类别标志，Llama Guard 覆盖 MLCommons 的 14 类危害，异步并行调用把新增延迟藏起来。',
  'an-uplift-asymmetry': 'AI 辅助对新手能力的放大最明显——生物武器获取试验测得获取类任务 2.53 倍——但新手离危险线仍很远。专家在一个大得多的基数上得到较小的倍数，而恰恰是专家这根柱子越过了危险能力阈值。安全案例必须两头都管：给新手算相对提升，给专家算绝对触达——同时视觉模型正在侵蚀曾经为两者兜底的湿实验室执行门槛。',
};
