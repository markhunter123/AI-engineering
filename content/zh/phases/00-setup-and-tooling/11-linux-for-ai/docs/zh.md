# Linux 赋能 AI

> 大多数 AI 运行在 Linux 上。你需要掌握足够的知识，避免束手无策。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## 学习目标

- 在命令行中浏览 Linux 文件系统并执行基本文件操作
- 使用 `chmod` 与 `chown` 管理文件权限，解决 "Permission denied" 错误
- 使用 `apt` 安装系统包，并为 AI 工作配置一台全新的 GPU 机器
- 识别 macOS 与 Linux 之间的常见差异，避免在远程机器上踩坑

## 问题

你在 macOS 或 Windows 上开发。但一旦 SSH 进入云 GPU 机器、租用 Lambda 实例或启动 EC2 机器，你就进入了 Ubuntu。终端是你唯一的界面。没有 Finder、没有 Explorer、没有 GUI。如果你无法从命令行浏览文件系统、安装包、管理进程，就会在搜索「如何在 Linux 中解压文件」时，为闲置的 GPU 小时付费。

这是一份生存指南。它只覆盖在远程 Linux 机器上做 AI 工作所需的内容，不多不少。

## 文件系统布局

Linux 把所有内容组织在单一的根目录 `/` 下。没有 `C:\` 也没有 `/Volumes`。你实际会碰到的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的家目录是 `~` 或 `/home/your-username`。几乎所有操作都在这里发生。

## 核心命令

这 15 条命令覆盖了你在远程 GPU 机器上 95% 的操作。

### 移动

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件与目录

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是永久操作，没有撤销。按回车前务必再三检查路径。

### 读取文件

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限

Linux 中每个文件都有所有者和权限位。脚本无法执行、或目录无法写入时，你就会碰到它。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见修复：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

看到 "Permission denied" 时，几乎总是权限问题。`chmod +x` 或 `sudo` 能解决大部分情况。

## 包管理（apt）

Ubuntu 使用 `apt`。这是安装系统级软件的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

你会在全新 GPU 机器上安装的常用包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户与 sudo

你通常以普通用户登录。某些操作需要 root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云 GPU 实例上，你通常是唯一用户，并且已经拥有 sudo 权限。不要凡事都用 root。只在需要时用 sudo。

## 进程与 systemd

训练卡住或想查看运行情况时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 管理服务（后台守护进程）。如果你运行推理服务器，会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间

GPU 机器的磁盘空间通常有限。模型和数据集会快速填满它。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见的空间清理方法：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你会从命令行下载模型、传输文件、调用 API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

对于大文件传输，优先使用 `rsync` 而不是 `scp`。它只传输变化的字节，并能在连接中断后恢复。

## tmux：让会话保持运行

SSH 进入远程机器后，合上笔记本会中断训练。tmux 可以防止这一点。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

长时间训练任务一定要放在 tmux 里。一定。

## WSL2：给 Windows 用户

如果你在 Windows 上，WSL2 提供了一个真正的 Linux 环境，无需双系统。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真正的 Linux 内核。本课的所有内容都可在其中使用。你的 Windows 文件在 WSL 内部位于 `/mnt/c/Users/YourName/`。

GPU 透传依赖 Windows 端安装的 NVIDIA 驱动。安装 Windows 版 NVIDIA 驱动（不是 Linux 版），CUDA 即可在 WSL2 内使用。

## 踩坑点：macOS 到 Linux

如果你从 macOS 转来，以下差异容易让你踩坑：

| macOS | Linux | Notes |
|-------|-------|-------|
| `brew install` | `sudo apt install` | Different package names sometimes. `brew install htop` vs `sudo apt install htop` works the same, but `brew install readline` vs `sudo apt install libreadline-dev` does not. |
| `open file.txt` | `xdg-open file.txt` | But you won't have a GUI on a remote box. Use `cat` or `less`. |
| `pbcopy` / `pbpaste` | Not available | Pipe to/from clipboard doesn't exist over SSH. |
| `~/.zshrc` | `~/.bashrc` | macOS defaults to zsh. Most Linux servers use bash. |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | Binaries live in different places. |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS sed needs an empty string after `-i`. Linux does not. |
| Case-insensitive filesystem | Case-sensitive filesystem | `Model.py` and `model.py` are two different files on Linux. |
| Line endings `\n` | Line endings `\n` | Same. But Windows uses `\r\n`, which breaks bash scripts. Run `dos2unix` to fix. |

## 快速参考卡

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

```figure
s0-process-fork
```

## 练习

1. SSH 进入任意 Linux 机器（或打开 WSL2），回到家目录，创建一个项目文件夹，用 `touch` 在里面创建三个空文件，然后用 `ls -la` 列出。
2. 用 apt 安装 `htop`，运行它，找出占用内存最多的进程。
3. 启动一个 tmux 会话，在里面运行 `sleep 300`，detach，列出会话，再 attach。
4. 用 `df -h` 查看可用磁盘空间，再用 `du -sh ~/.cache/*` 找出缓存中占空间的内容。
5. 用 `scp` 把文件从本机传到远程机器，再用 `rsync` 做同一次传输，比较两者的体验。

### 提示

**练习 1.** 使用 `cd ~`、`mkdir` 并 `cd` 进入新文件夹，然后运行 `touch` 加三个文件名。在 `ls -la` 输出中，检查三个文件大小为 0，且所有者为你的用户名。

**练习 2.** 用 `sudo apt install -y htop` 安装并启动，按 `F6`（或点击列名）按 `MEM%` 排序。列表最顶部的进程占用内存最多。

**练习 3.** 使用课中的 `tmux new -s train`，运行 `sleep 300`，按 `Ctrl+B` 再按 `d` detach。运行 `tmux ls` 查看会话，再执行 `tmux attach -t train`；`sleep` 必须仍在运行。

**练习 4.** 运行 `df -h`，读取包含家目录的文件系统的 `Avail` 列。然后运行 `du -sh ~/.cache/* | sort -h`，让最大的文件夹显示在最后一行。

**练习 5.** 用 `scp file.txt user@host:~/` 复制文件，再运行 `rsync -avP file.txt user@host:~/` 做同样的事。再运行一次 `rsync` 命令：这次几乎不传输数据，因为远程端的文件没有改变。
