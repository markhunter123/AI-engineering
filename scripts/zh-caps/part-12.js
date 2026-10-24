/** zh translations — figures-setup (12 figures) */
module.exports = {
  's0-env-stack': '环境自底向上搭建：先是 OS 和 GPU 驱动，然后包管理器，然后语言运行时，最后是依赖这三者的 AI 库。当 import torch 失败时，沿着栈向下走——出问题的那层几乎从来不是抛错的那层。',
  's0-commit-dag': '每个 commit 指向它的父提交，所以历史是一张图，不是一份列表。分支只是提交的分叉路径：main 继续前进，实验在自己的线上生长，merge commit 把两个父提交接在一起。正因如此，你可以在完全不碰 main 的情况下试验一个危险的改动。',
  's0-gpu-dispatch': 'GPU 赢在矩阵乘法是数千个独立的乘加运算——恰是它核心并行化的对象。同样的 batch 在 CPU 上一次漏过一个块，在 GPU 上却是流式涌过——所以在 CPU 上跑 8 小时的训练，在 T4 上只要约 10 分钟。先确认车道存在：nvidia-smi，然后 torch.cuda.is_available()。',
  's0-secret-inject': '被提交的代码只写变量名；值住在被 gitignore 的 .env 文件里，运行时注入进程环境，再作为 Authorization header 发出。一旦密钥出现在源码里，它就随仓库的每一次 clone 一起出走，而泄露的密钥记在你的账单上。',
  's0-cell-order': '方括号里的数字记录的是 cell 何时运行，不是它排在哪。重跑 cell 2 会把 [2] 变成 [4]，但 cell 3 仍持有旧顺序的结果——磁盘上那个文件永远显示不出的隐藏状态。当计数不再自上而下递增，Restart and Run All 是唯一诚实的检查。',
  's0-env-isolation': '两个 pin 都正确，只是不能共用一个解释器。全局安装时，第二个 pip install 会覆盖第一个，两个项目一起坏。每个项目一个 virtual environment，各自的 site-packages 互不相通——torch 2.4 和 torch 2.1 就在同一台机器上共存而永不相见。',
  's0-image-layers': 'Dockerfile 的每条指令加一层只读层：CUDA base、Python、torch，你的代码在最上面。完成的镜像就是冻结的整个栈，所以 docker run 在你的笔记本和租来的 GPU 机器上产出同一个容器——works-on-my-machine 这个失败模式随之消失。',
  's0-lsp-roundtrip': '编辑器不懂 Python；语言服务器才懂。每次编辑都把一条 didChange 通知发上线，服务器重新检查文件，诊断结果回来画成波浪线和悬浮提示。装上 Python 与 Pylance 扩展，这个循环在每次击键时运转；不装，编辑器就只是个文本框。',
  's0-data-pipeline': '原始 CSV 从不被原地修改：它流进一个带类型、被压缩的 Parquet 工件，再流进用固定种子切出的 train、val、test 三份。因为每支箭头都是脚本、种子被钉死，任何人都能重新生成你的指标所计算的那几行精确数据——实验因此可比。',
  's0-shell-pipeline': '管道把一个命令的 stdout 交给下一个命令的 stdin，中间没有临时文件。日志流整条进入，grep 丢掉每一行不含「loss」的行，awk 只留第 8 列——训练输出的消防水管变成一串干净的数字，可以实时盯着看或画出来。',
  's0-process-fork': '在 Linux GPU 机器上一切都是从父进程 fork 出来的进程：init 生出 sshd，你的登录生出 bash，bash 生出训练任务。Ctrl-C 的 SIGINT，或对 ps aux 里找到的 pid 执行 kill，会终结叶子进程、释放它的 GPU 显存——这就是你收回一台被死任务占着的机器的方法。',
  's0-flame-hot': '你怀疑的 GPU 计算并不是瓶颈；profiler 显示这一步大部分墙钟时间耗在 DataLoader 里。每根条是调用内的累计时间、堆叠在调用者之上，所以最宽的那座塔标出该修的函数——这里是给 loader 加 worker 和 pinned memory，而不是优化模型。',
};
