/** zh translations — figures-demo (1 figure) */
module.exports = {
  'demo-token-bucket': '每个请求从它所属 client 的桶里取一个 token。桶以固定速率补充、存量从不超过容量。容量定下 demo 接受的最大突发；补充速率定下稳态请求速率。被拒的请求拿到带 Retry-After 头的 HTTP 429，模型不会运行。',
};
