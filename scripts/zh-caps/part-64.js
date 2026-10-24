/** zh translations — figures-infra2 (8 figures) */
module.exports = {
  'cache-aware-router': 'Round-robin 把请求乱撒，大多数 miss 掉 cache、付全价 prefill。cache-aware router 对 prompt 前缀做哈希，把每个匹配的请求都送往已持有那些 KV block 的副本——热路径保持热，TTFT 从 prefill 瓶颈塌缩成一次 cache hit。',
  'cold-start-pipeline': '缩容到零的副本，权重不进 HBM 就答不了话。cold-start 预算是节点置备、权重下载、载入 HBM、引擎初始化之和——对 70B 模型是分钟级，对着的却是两秒 SLA。分层加载把权重从 NVMe→DRAM→HBM 流式搬运，warm pool 保持 min_workers>0——用闲置 GPU 成本换掉消失的 cold-start 长尾。',
  'model-cascade-router': 'broker 给每个请求打分——任务类型、长度、置信度——把简单的多数派发给便宜模型。只有低置信请求沿虚线车道升级到 frontier 模型。大多数流量只花几分钱；同等质量下总账单降 20–60%。隐患是便宜模型的静默漂移——只有在线质量闸门能抓住。',
  'prefill-decode-split': 'Prefill 受算力限制；decode 受内存带宽限制。把两者塞进同一块 GPU，浪费的总是当前阶段用不上的那份资源。disaggregation 按各自瓶颈建两个独立池，KV cache 经高带宽 NIXL 链路跨池传送。长 prompt 才划算；短 prompt 付不起搬运成本。',
  'batch-lane-triage': '每个新的 LLM 工作负载都分诊进三条车道。交互式保持同步、全价；半交互式进异步队列；一切扛得住 24 小时延迟的都落进 batch 车道——五折，再叠上缓存输入，压到同步成本的 ~10%。许多假装实时的任务，其实只要早上有个答案就行。',
  'semantic-cache-hit': 'L1 semantic cache 给每条 prompt 算 embedding、与历史条目比相似度。近似重复的 query（蓝）直接从缓存出——快且几乎免费。只有真正新颖的 prompt（橙）miss 到底层的冷 LLM。命中率从开放聊天的 10% 到结构化 FAQ 的 70%；前缀里的动态文本一多就塌到近零。',
  'edge-bandwidth-pipe': 'decode 每 token 都要读全部权重，所以天花板是内存带宽，不是算力。数据中心的 HBM3 近 3 TB/s，约一毫秒就能冲一遍权重，每秒吐几百 token。手机 DRAM 只有 50–90 GB/s，管道窄 30–50 倍——同一个模型不管 NPU 算力闲多少，都只能 14–25 tok/s 细水长流。',
  'load-pattern-waves': '四种负载形状抓四种故障。稳态测基线吞吐；ramp 爬坡到崩溃点；spike 突然冲高、测 autoscaling 反应多快；soak 长时间压住负载、让内存泄漏现形。所有请求都一模一样时，通用测试工具还会撒谎——真实流量需要可变输入长度和多样的前缀。',
};
