# AI Engineering from Scratch — Bilingual Docs Site

A fully static, zero-dependency documentation site for the
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum: **542 lessons · 20 phases · English + 中文**.

Authored by [Musen.Lin](https://github.com/markhunter123?tab=repositories).

## Features

- **Bilingual** — every lesson, phase page, and UI string in English and Chinese
  (sidebar, search, quiz, pager, footer included). Language switch keeps your
  place on the same page.
- **Dark / light theme** — DeepSeek-style design tokens, no flash on load,
  follows your OS preference until you choose.
- **Interactive quizzes** — 392 lessons ship pre/mid/post checks with
  explanations and per-stage scoring.
- **Ctrl+K search** — per-language index with keyboard navigation.
- **TOC + scrollspy** — right-hand outline tracks your scroll position.
- **Math & diagrams** — KaTeX and Mermaid load lazily from CDN only on pages
  that need them; the page stays readable if the CDN is unreachable.
- **Zero npm dependencies** — the generator, the server, and the site runtime
  are all plain Node.js / vanilla JS.

## Quick start

```bash
node build.js          # generate dist/ (~1128 pages, ~6 s)
node scripts/serve.mjs # serve dist/ at http://localhost:4173
node scripts/verify.mjs# link-graph + pairing + JSON checks (exit 0 = clean)
```

## Project layout

```text
build.js               static site generator (scan → render → search/tree → root/404)
config.js              source repo path, site metadata, CDN endpoints, phase 中文名
scripts/
  export-zh.mjs        one-time export: git archive origin/translations i18n/zh → content/zh
  serve.mjs            zero-dep static server with SPA-style 404 fallback
  verify.mjs           post-build integrity checks (~95k links)
vendor/lesson-markdown.js  markdown parser (vendored + local patch block)
src/
  templates/           layout / sidebar / lesson / phase / home / icons
  css/                 tokens (DeepSeek palette) → base → layout → content → components
  js/                  theme / app / search / quiz / enhance (KaTeX+Mermaid)
  data/ui-extra.json   Chinese UI strings not present upstream
content/zh/            exported Chinese lesson sources (committed)
dist/                  build output (gitignored)
```

## Refreshing the Chinese content

Chinese lesson text comes from the upstream `translations` branch, exported
once so builds run offline:

```bash
node scripts/export-zh.mjs   # re-export from origin/translations (fetch first)
node build.js
```

## Deploying

`dist/` is plain static files — any static host works.

- **Domain root** (Vercel, Netlify): upload `dist/`, done.
- **GitHub Pages project site**: build with a subpath base:
  ```bash
  node build.js --base=/your-repo-name/
  ```
- **Any static server**: `npx serve dist` or `python -m http.server -d dist`.

## Notes

- The source curriculum repo is read-only for this project; nothing writes to it.
- Quiz questions are English-only (the upstream translation branch does not
  include them); Chinese pages show a small notice above the quiz.
- Interactive `figure` blocks from the upstream registry render as placeholder
  cards with the figure key, since they need a runtime not available statically.
