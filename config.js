/**
 * Site configuration. Edit here — build.js and templates read everything.
 * No npm dependencies; plain CommonJS.
 */
'use strict';

const path = require('path');

/** Source curriculum repo (read-only; never modified by this project). */
const SRC_REPO = 'D:/project/ai-engineering-from-scratch';

/** Root-relative base for all generated URLs. '/' for domain-root deploys;
 *  use '/ai-engineer/' for a GitHub Pages project site. Overridable via --base. */
const BASE = '/';

/** Where build output goes. */
const DIST = path.join(__dirname, 'dist');

/** Exported Chinese snapshot directory (created by scripts/export-zh.mjs). */
const ZH_CONTENT = path.join(__dirname, 'content', 'zh');

/** Chinese quiz translations, mirroring the upstream quiz.json schema at
 *  phases/<phase>/<lesson>/quiz.json. Kept outside content/zh because a
 *  re-export wipes that directory. */
const ZH_QUIZ = path.join(__dirname, 'content', 'zh-quiz');

/** Chinese text overlays for lesson asset SVGs, at
 *  zh-figures/phases/<phase>/<lesson>/<stem>.json — a flat
 *  { "exact text in the svg": "中文" } map. The build rewrites <text>
 *  content and emits <stem>.zh.svg next to the English copy. Kept outside
 *  content/zh because a re-export wipes that directory. */
const ZH_FIGURES = path.join(__dirname, 'content', 'zh-figures');

const SITE = {
  name: 'AI Engineering from Scratch',
  nameZh: 'AI 工程从零开始',
  /** Short brand for the navbar corner — same in both languages. */
  brand: 'AI Engineering',
  description:
    '542 hands-on lessons across 20 phases. Build every algorithm from raw math, then run it through the production library.',
  descriptionZh:
    '20 个阶段、542 节动手课程。从原始数学开始实现每个算法，再用生产级库运行同样的操作。',
  /** Site author — navbar GitHub button and footer credit. coAuthor gets a
   *  plain-text footer credit (no link) until it has a profile URL. */
  author: 'Musen.Lin',
  coAuthor: 'LiuBin.Zhao',
  github: 'https://github.com/markhunter123/AI-engineering',
  githubLabel: 'Musen.Lin',
  /** Motto banner above the footer on every page. mottoBy: '' hides the
   *  attribution line; motto: '' hides the whole banner. */
  motto: 'Stay hungry, stay foolish.',
  mottoBy: 'Steve Jobs',
  /** Original curriculum the lesson content comes from (footer attribution). */
  source: {
    url: 'https://github.com/rohitg00/ai-engineering-from-scratch',
    label: 'rohitg00/ai-engineering-from-scratch',
  },
};

/** Enhancement endpoints for progressive enhancement. Loaded lazily by
 *  enhance.js and skipped entirely when the page does not need them.
 *  Both KaTeX (vendor/katex) and mermaid (vendor/mermaid, ESM + chunks)
 *  are self-hosted — copied to assets/ at build time — so math and
 *  diagrams render with zero network access. */
const CDN = {
  katexCss: BASE + 'assets/katex/katex.min.css',
  katexJs: BASE + 'assets/katex/katex.min.js',
  mermaid: BASE + 'assets/mermaid/mermaid.esm.min.mjs',
};

/** Curated Chinese phase titles (upstream zh translations keep English
 *  frontmatter; phase directories carry no per-language titles). */
const PHASE_ZH = {
  '00-setup-and-tooling': '环境搭建与工具链',
  '01-math-foundations': '数学基础',
  '02-ml-fundamentals': '机器学习基础',
  '03-deep-learning-core': '深度学习核心',
  '04-computer-vision': '计算机视觉',
  '05-nlp-foundations-to-advanced': '自然语言处理：从基础到进阶',
  '06-speech-and-audio': '语音与音频',
  '07-transformers-deep-dive': 'Transformer 深度解析',
  '08-generative-ai': '生成式 AI',
  '09-reinforcement-learning': '强化学习',
  '10-llms-from-scratch': '从零构建大语言模型',
  '11-llm-engineering': 'LLM 工程',
  '12-multimodal-ai': '多模态 AI',
  '13-tools-and-protocols': '工具与协议',
  '14-agent-engineering': 'Agent 工程',
  '15-autonomous-systems': '自主系统',
  '16-multi-agent-and-swarms': '多 Agent 与集群',
  '17-infrastructure-and-production': '基础设施与生产部署',
  '18-ethics-safety-alignment': '伦理、安全与对齐',
  '19-capstone-projects': '实战项目',
};

module.exports = { SRC_REPO, BASE, DIST, ZH_CONTENT, ZH_QUIZ, ZH_FIGURES, SITE, CDN, PHASE_ZH };
