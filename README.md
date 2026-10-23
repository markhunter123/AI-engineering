# AI Engineering from Scratch — 中英双语文档站

一个完全静态、零依赖的文档站点，用于呈现
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
课程：**542 节课程 · 20 个阶段 · English + 中文**。

作者：[Musen.Lin](https://github.com/markhunter123?tab=repositories)。

## 特性

- **中英双语** — 每一课、每个阶段页面和所有界面文案都有英文与中文两个版本
  （侧边栏、搜索、测验、翻页、页脚全覆盖）。切换语言时停留在当前页面的对应位置。
- **深色 / 浅色主题** — DeepSeek 风格的设计令牌，加载时不闪烁，
  在你手动选择之前会跟随系统偏好。
- **交互式测验** — 392 节课自带课前 / 课中 / 课后检查，附解析和分阶段计分。
- **Ctrl+K 搜索** — 按语言分索引，支持键盘导航。
- **目录 + 滚动定位** — 右侧大纲跟随滚动位置高亮。
- **数学公式与图表** — KaTeX 和 Mermaid 只在需要的页面上懒加载；
  即使资源不可达，页面也保持可读。
- **零 npm 依赖** — 生成器、服务器和站点运行时全部是纯 Node.js / 原生 JS。

## 快速开始

```bash
node build.js          # 生成 dist/（约 1128 页，约 6 秒）
node scripts/serve.mjs # 在 http://localhost:4173 提供 dist/ 静态服务
node scripts/verify.mjs# 链接图 + 配对 + JSON 检查（exit 0 = 全部通过）
```

## 项目结构

```text
build.js               静态站点生成器（扫描 → 渲染 → 搜索/目录树 → 根页/404）
config.js              源仓库路径、站点元数据、CDN 端点、阶段中文名
scripts/
  export-zh.mjs        一次性导出：git archive origin/translations i18n/zh → content/zh
  serve.mjs            零依赖静态服务器，带 SPA 式 404 回退
  verify.mjs           构建后完整性检查（约 9.5 万条链接）
vendor/lesson-markdown.js  markdown 解析器（内联引入 + 本地补丁块）
src/
  templates/           布局 / 侧边栏 / 课程 / 阶段 / 首页 / 图标
  css/                 设计令牌（DeepSeek 配色）→ 基础 → 布局 → 内容 → 组件
  js/                  主题 / 应用 / 搜索 / 测验 / 增强（KaTeX+Mermaid）
  data/ui-extra.json   上游没有的中文界面文案
content/zh/            导出的中文课程源文件（已提交）
dist/                  构建输出（已 gitignore）
```

## 刷新中文内容

中文课程文本来自上游 `translations` 分支，导出一次后即可离线构建：

```bash
node scripts/export-zh.mjs   # 从 origin/translations 重新导出（需先 fetch）
node build.js
```

## 部署

`dist/` 是纯静态文件——任何静态托管都能用。

- **域名根路径**（Vercel、Netlify）：上传 `dist/` 即可。
- **GitHub Pages 项目站**：以子路径为 base 构建：
  ```bash
  node build.js --base=/your-repo-name/
  ```
- **任意静态服务器**：`npx serve dist` 或 `python -m http.server -d dist`。

## 备注

- 本项目对源课程仓库只读，不会向其写入任何内容。
- 测验题目目前仅有英文（上游翻译分支未包含）；中文页面会在测验上方显示小提示。
- 上游注册表中的交互式 `figure` 块会渲染成带 figure key 的占位卡片，
  因为它们需要静态构建无法提供的运行时。
