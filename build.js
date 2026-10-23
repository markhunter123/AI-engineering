/**
 * build.js — static site generator. Zero npm dependencies; plain Node.
 *
 *   node build.js [--base=/ai-engineer/]
 *
 * Pipeline: strings → scan 542 lessons (en + zh) → assets → render
 * (home/phase/lesson × en/zh) → search + tree JSON → root negotiation
 * page + 404 → report. Source repo is read-only.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const config = require('./config');
const md = require('./vendor/lesson-markdown');
const layout = require('./src/templates/layout');
const sidebarTpl = require('./src/templates/sidebar');
const lessonTpl = require('./src/templates/lesson');
const phaseTpl = require('./src/templates/phase');
const homeTpl = require('./src/templates/home');
const pathsTpl = require('./src/templates/paths');

const { SRC_REPO, DIST, ZH_CONTENT, SITE, CDN, PHASE_ZH } = config;

/** Cache-busting version stamped on every static asset URL (layout vurl +
 *  the CDN config below) so browsers re-fetch JS/CSS/JSON after a deploy
 *  instead of serving stale copies that predate a fix. */
const ASSET_VER = Date.now().toString(36);
layout.setAssetVer(ASSET_VER);
CDN.katexCss += '?v=' + ASSET_VER;
CDN.katexJs += '?v=' + ASSET_VER;
CDN.mermaid += '?v=' + ASSET_VER;

const LANGS = ['en', 'zh'];
const GITHUB_TREE = SITE.github + '/tree/main/';

/* ------------------------------------------------------------------ args */
let BASE = config.BASE;
for (const arg of process.argv.slice(2)) {
  const m = arg.match(/^--base=(.+)$/);
  if (m) BASE = m[1].replace(/\/?$/, '/');
}

/* --------------------------------------------------------------- strings */
const T_KEYS = [
  'Skip to content', 'Open navigation', 'Close navigation', 'Toggle theme',
  'GitHub repository', 'Search', 'Search lessons...', 'No results found',
  'to navigate', 'to open', 'to close', 'On this page',
  '← Previous', 'Next →', 'View lesson source', 'Prerequisites',
  'Quiz', 'Pre-Lesson Check', 'Mid-Lesson Check', 'Post-Lesson Quiz',
  'Retry this check', 'Score', 'Perfect score!', 'Great work!', 'Keep studying!',
  'Quiz content is currently available in English only.',
  'Interactive figure omitted in the static build',
  'Home', 'Phases', 'Lessons', 'All Phases', 'Start the Course',
  'Browse all phases', 'Free and open source curriculum',
  'How this works', 'Interactive Quizzes', 'Code Languages',
  'Loading lesson...', 'Copied!', 'Phase', 'Page not found',
  'The page you requested does not exist. Pick a language to continue:',
  'Back to Home', 'Content from', 'MIT License', 'authored by',
  'Built as a static bilingual mirror of the open curriculum.',
  'How to use this site',
  'Pick a phase below — lessons build on each other within a phase.',
  'Every lesson explains the concept, builds it from raw math or code, then runs the same operation through the production library.',
  'Take the quiz at the end of each lesson to check your understanding.',
  'Learning Paths',
];

function loadStrings() {
  const zhUi = JSON.parse(fs.readFileSync(path.join(ZH_CONTENT, 'ui.json'), 'utf8')).strings || {};
  const extras = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'src', 'data', 'ui-extra.json'), 'utf8'),
  ).zh || {};
  const strings = { en: {}, zh: {} };
  for (const key of T_KEYS) {
    strings.en[key] = key;
    strings.zh[key] = zhUi[key] || extras[key] || key;
    if (strings.zh[key] === key && extras[key]) strings.zh[key] = extras[key];
  }
  return strings;
}

function makeLookup(lang) {
  return function lookup(key) {
    return strings[lang][key] || strings.en[key] || key;
  };
}

/** The camelCase `t` object all templates consume. */
function makeT(lang) {
  const L = makeLookup(lang);
  return {
    skipToContent: L('Skip to content'),
    openNavigation: L('Open navigation'),
    closeNavigation: L('Close navigation'),
    toggleTheme: L('Toggle theme'),
    githubRepo: L('GitHub repository'),
    search: L('Search'),
    searchShort: L('Search'),
    searchPlaceholder: L('Search lessons...'),
    searchEmpty: L('No results found'),
    searchNavigate: L('to navigate'),
    searchOpen: L('to open'),
    searchClose: L('to close'),
    onThisPage: L('On this page'),
    learningMap: L('Learning Paths'),
    prev: L('← Previous'),
    next: L('Next →'),
    home: L('Home'),
    prerequisites: L('Prerequisites'),
    viewSource: L('View lesson source'),
    quizTitle: L('Quiz'),
    quizNoteEn: L('Quiz content is currently available in English only.'),
    pagerLabel: L('Lesson navigation'),
    phaseLabel: L('Phase'),
    lessons: L('Lessons'),
    allPhases: L('All Phases'),
    loading: L('Loading lesson...'),
    copied: L('Copied!'),
    footerText: 'AI Engineering',
    footerLicense: L('MIT License'),
    authoredBy: L('authored by'),
  };
}

/* ------------------------------------------------------------ scanning */
/** slug "14-agent-engineering" → "Agent Engineering" */
function slugToTitle(slug) {
  return slug
    .replace(/^\d+-/, '')
    .split('-')
    .map((w) => {
      const up = { ml: 'ML', ai: 'AI', nlp: 'NLP', llm: 'LLM', llms: 'LLMs', api: 'API' };
      return up[w.toLowerCase()] || (w.charAt(0).toUpperCase() + w.slice(1));
    })
    .join(' ');
}

function parseMeta(mdx) {
  const lines = mdx.split(/\r?\n/);
  const meta = {
    title: '',
    hook: '',
    type: '',
    languages: '',
    prerequisites: '',
    minutes: '',
    keywords: '',
  };
  let inCode = false;
  let titleSeen = false;
  const h3s = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^```/);
    if (fence) { inCode = !inCode; continue; }
    if (inCode) continue;

    if (!titleSeen) {
      const h1 = line.match(/^# (.+)/);
      if (h1) {
        const t = h1[1].trim();
        // Upstream NMT glitch guard: a title that is one short fragment
        // repeated (机器人机器人机器人…) is corruption — empty it so
        // titleFor() falls back to the English title.
        const squashed = t.replace(/\s+/g, '');
        if (/^(.{1,8}?)\1{4,}$/u.test(squashed)) meta.title = '';
        else meta.title = t;
        titleSeen = true;
        continue;
      }
    }
    if (!meta.hook && titleSeen && /^>\s/.test(line)) {
      meta.hook = line.replace(/^>\s?/, '').trim();
    }
    // Bold-label frontmatter: "**Type:** Learn". The colon sits INSIDE the
    // bold span, so the closing "**" must be consumed before the value —
    // matching "**Type:**" with a loose colon instead leaks "** " into every
    // meta chip. Accepts the "Language" singular variant some zh exports use;
    // a "--" value means "none" and hides the chip.
    const metaLine = line.match(/^\*\*(Type|Languages?|Prerequisites|Time):\*\*\s*(.+)$/);
    if (metaLine) {
      const key = metaLine[1].toLowerCase();
      const value = /^--+$/.test(metaLine[2].trim()) ? '' : metaLine[2].trim();
      if (key === 'type') meta.type = value;
      else if (key === 'languages' || key === 'language') meta.languages = value;
      else if (key === 'prerequisites') meta.prerequisites = value;
      else if (key === 'time') meta.minutes = value;
      continue;
    }
    const h3 = line.match(/^### (.+)/);
    if (h3) h3s.push(h3[1].trim());
  }
  meta.keywords = h3s.slice(0, 12).join(' ');
  return meta;
}

function scan() {
  const phasesDir = path.join(SRC_REPO, 'phases');
  const phaseSlugs = fs
    .readdirSync(phasesDir)
    .filter((d) => /^\d{2}-/.test(d))
    .sort();

  const phases = [];
  const lessons = [];
  const warnings = [];

  for (const phaseSlug of phaseSlugs) {
    const numLabel = phaseSlug.slice(0, 2);
    const phaseDir = path.join(phasesDir, phaseSlug);
    const lessonSlugs = fs
      .readdirSync(phaseDir)
      .filter((d) => /^\d{2}-/.test(d))
      .sort();

    const phase = {
      slug: phaseSlug,
      numLabel,
      title: { en: slugToTitle(phaseSlug), zh: PHASE_ZH[phaseSlug] || slugToTitle(phaseSlug) },
      hook: { en: '', zh: '' },
      lessons: [],
    };

    for (const lessonSlug of lessonSlugs) {
      const enPath = path.join(phaseDir, lessonSlug, 'docs', 'en.md');
      const zhPath = path.join(ZH_CONTENT, 'phases', phaseSlug, lessonSlug, 'docs', 'zh.md');
      const quizPath = path.join(phaseDir, lessonSlug, 'quiz.json');
      if (!fs.existsSync(enPath)) {
        warnings.push('missing en.md: ' + phaseSlug + '/' + lessonSlug);
        continue;
      }
      const enMd = fs.readFileSync(enPath, 'utf8');
      const hasZh = fs.existsSync(zhPath);
      const zhMd = hasZh ? fs.readFileSync(zhPath, 'utf8') : null;
      if (!hasZh) warnings.push('missing zh.md: ' + phaseSlug + '/' + lessonSlug);

      let quiz = null;
      if (fs.existsSync(quizPath)) {
        try {
          const parsed = JSON.parse(fs.readFileSync(quizPath, 'utf8'));
          // Two upstream shapes: {questions: [...]} and a bare top-level array.
          const rawQuestions = Array.isArray(parsed) ? parsed : (parsed.questions || []);
          const questions = rawQuestions.filter(
            (q) => q && typeof q.question === 'string' && Array.isArray(q.options) && q.options.length >= 2,
          );
          if (questions.length) quiz = { questions };
        } catch (e) {
          warnings.push('bad quiz.json: ' + phaseSlug + '/' + lessonSlug);
        }
      }

      const record = {
        phaseSlug,
        lessonSlug,
        numLabel: lessonSlug.slice(0, 2),
        meta: { en: parseMeta(enMd), zh: hasZh ? parseMeta(zhMd) : null },
        quiz,
        enMd,
        zhMd,
        hasZh,
        githubUrl: GITHUB_TREE + 'phases/' + phaseSlug + '/' + lessonSlug,
      };
      lessons.push(record);
      phase.lessons.push(record);
    }
    phases.push(phase);
  }

  // Phase hooks: first blockquote of phase README.md (en) — zh not available.
  for (const phase of phases) {
    const readme = path.join(SRC_REPO, 'phases', phase.slug, 'README.md');
    if (fs.existsSync(readme)) {
      const head = fs.readFileSync(readme, 'utf8').split(/\r?\n/);
      for (const line of head) {
        if (/^>\s/.test(line)) { phase.hook.en = line.replace(/^>\s?/, '').trim(); break; }
      }
    }
    phase.hook.zh = phase.hook.en;
  }

  // Global order + prev/next.
  lessons.forEach((lesson, i) => {
    lesson.order = i;
    lesson.prev = i > 0 ? lessons[i - 1] : null;
    lesson.next = i < lessons.length - 1 ? lessons[i + 1] : null;
  });

  return { phases, lessons, warnings };
}

/* --------------------------------------------------------- url helpers */
const lessonUrl = (lang, phaseSlug, lessonSlug) => BASE + lang + '/phases/' + phaseSlug + '/' + lessonSlug + '/';
const phaseUrl = (lang, phaseSlug) => BASE + lang + '/phases/' + phaseSlug + '/';

/** Lesson title in `lang`, falling back to English when the zh H1 is empty. */
const titleFor = (record, lang) =>
  (record.meta[lang] && record.meta[lang].title) || record.meta.en.title;

/* --------------------------------------------------- markdown pipeline */
function posixJoin(...parts) {
  const joined = path.posix.normalize(parts.join('/'));
  return joined.replace(/^\.\//, '');
}

/** Resolve one relative md target to (phase, lesson) inside the repo. */
function resolveLessonTarget(href) {
  const cleaned = href.split('#')[0].split('?')[0];
  if (!cleaned || !/\.md$/i.test(cleaned)) return null;
  if (!/docs\/(en|zh)\.md$/i.test(cleaned)) return null;
  const norm = path.posix.normalize(cleaned);
  const m = norm.match(/phases\/(\d{2}-[^/]+)\/(\d{2}-[^/]+)\/docs\/(?:en|zh)\.md$/);
  if (!m) return null;
  return { phaseSlug: m[1], lessonSlug: m[2] };
}

function preprocess(raw, lang, record) {
  const docDir = 'phases/' + record.phaseSlug + '/' + record.lessonSlug + '/docs';

  let mdx = raw;

  // Strip the bold frontmatter lines — they render as meta chips instead.
  // Also catches the "**Language:**" singular variant found in zh exports,
  // which otherwise leaks into the article body as a stray bold line.
  mdx = mdx.replace(/^\*\*(Type|Languages?|Prerequisites|Time)\*?:.*$\n?/gm, '');

  // Rewrite markdown link/image targets.
  mdx = mdx.replace(/\]\(([^)\s]+)\)/g, (full, href) => {
    if (/^(https?:|mailto:|#)/i.test(href)) return full;
    const hashIdx = href.indexOf('#');
    const hash = hashIdx >= 0 ? href.slice(hashIdx) : '';
    const target = hashIdx >= 0 ? href.slice(0, hashIdx) : href;
    if (!target) return full;

    // Lesson-to-lesson docs links → site URL.
    const lessonTarget = resolveLessonTarget(target);
    if (lessonTarget) {
      return '](' + lessonUrl(lang, lessonTarget.phaseSlug, lessonTarget.lessonSlug) + hash + ')';
    }

    // Everything else: resolve relative to docs/, then classify.
    let rel = target;
    if (/^\.\.\//.test(rel)) {
      rel = posixJoin(docDir, rel);
    } else if (/^\.\.\\/.test(rel)) {
      rel = posixJoin(docDir, rel.replace(/\\/g, '/'));
    } else {
      rel = posixJoin(docDir, rel);
    }
    rel = rel.replace(/^phases\/(\d{2}-[^/]+)\/(\d{2}-[^/]+)\/docs\/\.\.\//, 'phases/$1/$2/');

    // Lesson assets → copied static dir.
    const assetMatch = rel.match(/^(phases\/\d{2}-[^/]+\/\d{2}-[^/]+)\/(assets\/.+)$/);
    if (assetMatch) return '](' + BASE + 'static/' + assetMatch[1] + '/' + assetMatch[2] + ')';

    // Any other repo file/directory → GitHub.
    return '](' + GITHUB_TREE + rel.replace(/\/$/, '') + (target.endsWith('/') ? '/' : '') + ')';
  });

  return mdx;
}

function unescapeHtml(s) {
  return s
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}

/** TOC extracted from the rendered body — ids match the page exactly. */
function extractToc(bodyHtml) {
  const toc = [];
  for (const m of bodyHtml.matchAll(/<h([23]) id="([^"]*)">([\s\S]*?)<\/h\1>/g)) {
    const level = Number(m[1]);
    const id = m[2];
    const text = unescapeHtml(m[3].replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim();
    if (!id || !text) continue;
    if (text.toLowerCase() === 'hints') continue;
    toc.push({ id, text, level });
  }
  return toc;
}

function renderMarkdown(lang, record) {
  const raw = lang === 'zh' && record.zhMd ? record.zhMd : record.enMd;
  const L = makeLookup(lang);
  const mdx = preprocess(raw, lang, record);
  const html = md.parseMd(mdx, {
    copyLabel: lang === 'zh' ? '复制' : 'Copy',
    figureNote: L('Interactive figure omitted in the static build'),
  });
  const needsMath = html.indexOf('data-math') >= 0;
  const needsMermaid = html.indexOf('mermaid-source') >= 0;
  const needsFigures = html.indexOf('class="lesson-figure"') >= 0;
  if (needsFigures) {
    for (const m of html.matchAll(/data-figure="([^"]+)"/g)) {
      // attribute may be "key {json cfg}" — the manifest indexes bare keys
      if (m[1]) figureKeys.add(m[1].trim().split(/\s+/)[0]);
    }
  }
  // Strip the first <h1> — the page header renders the title instead.
  const body = html.replace(/<h1 id="[^"]*">[\s\S]*?<\/h1>/, '');
  return { body, needsMath, needsMermaid, needsFigures, toc: extractToc(body) };
}

/* ------------------------------------------------------------- helpers */
function quizJsonFor(record) {
  if (!record.quiz) return null;
  return JSON.stringify({ questions: record.quiz.questions }).replace(/</g, '\\u003c');
}

function sidebarHtmlFor(lang, currentPhaseSlug, currentLessons, currentLessonSlug, currentSpecial) {
  const L = makeLookup(lang);
  const phases = phasesScan.phases.map((phase) => ({
    slug: phase.slug,
    numLabel: phase.numLabel,
    title: phase.title[lang],
    count: phase.lessons.length,
  }));
  const current = phasesScan.phases.find((p) => p.slug === currentPhaseSlug);
  return sidebarTpl.render({
    phases,
    lang,
    base: BASE,
    currentPhaseSlug,
    currentSpecial: currentSpecial || null,
    pathsCount: learningPaths.length,
    currentLessons: current
      ? current.lessons.map((l) => ({
          n: l.numLabel,
          t: titleFor(l, lang),
          u: lessonUrl(lang, l.phaseSlug, l.lessonSlug),
          current: l.lessonSlug === currentLessonSlug,
        }))
      : [],
    t: {
      openNavigation: L('Open navigation'),
      closeNavigation: L('Close navigation'),
      lessons: L('Lessons'),
      learningMap: L('Learning Paths'),
    },
  });
}

function write(file, content) {
  const full = path.join(DIST, file);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content);
}

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}

/* ---------------------------------------------------------------- data */
let strings = null;
let phasesScan = null;
const figureKeys = new Set();

/* ------------------------------------------------------- learning paths */
let learningPaths = [];

/** Load role-based routes from the (read-only) source repo manifests and
 *  resolve every referenced lesson against our scanned records. */
function loadLearningPaths(lessons) {
  const dir = path.join(SRC_REPO, 'learning-paths');
  if (!fs.existsSync(dir)) return [];
  const byPath = new Map();
  for (const record of lessons) byPath.set(record.phaseSlug + '/' + record.lessonSlug, record);
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json'));
  const routes = [];
  for (const f of files) {
    let m;
    try {
      m = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    } catch (e) {
      console.log('Learning path skipped (bad JSON): ' + f);
      continue;
    }
    const flat = (m.lessons || [])
      .map((l) => {
        const parts = String(l.path || '').split('/').filter(Boolean);
        const rec = byPath.get(parts[1] + '/' + parts[2]);
        return rec ? { rec, minutes: l.minutes || 0 } : null;
      })
      .filter(Boolean);
    const stages = (m.stages || [])
      .filter((s) => Array.isArray(s.lessonPaths) && s.lessonPaths.length)
      .map((s) => ({
        id: s.id || '',
        titleEn: s.title || s.id || '',
        outcome: s.outcome || '',
        artifact: s.artifact || '',
        recs: s.lessonPaths
          .map((p) => {
            const parts = String(p).split('/').filter(Boolean);
            const rec = byPath.get(parts[1] + '/' + parts[2]);
            // Same shape as flat rows: row() reads r.rec/r.minutes.
            return rec ? { rec, minutes: 0 } : null;
          })
          .filter(Boolean),
      }));
    routes.push({ m, flat, stages });
  }
  return routes;
}

const STAGE_TITLES_ZH = {
  'common-core': '公共核心',
  'role-practice': '角色实战',
  'proof-project': '实战作品',
  'interview-readiness-evidence': '面试与就绪证据',
};

/** Localized view model of every route (zh overlay when lang is zh). */
function localizedPaths(lang) {
  const zhDir = path.join(__dirname, 'src', 'data', 'paths-zh');
  return learningPaths.map(({ m, flat, stages }) => {
    let zh = null;
    if (lang === 'zh') {
      try { zh = JSON.parse(fs.readFileSync(path.join(zhDir, m.id + '.json'), 'utf8')); } catch (_) { zh = null; }
    }
    const zhStage = (id) => (zh && Array.isArray(zh.stages) ? zh.stages.find((s) => s && s.id === id) : null);
    const pick = (enVal, zhVal) => (lang === 'zh' && zhVal ? zhVal : enVal);
    const row = (r) => ({
      n: r.rec.numLabel,
      title: titleFor(r.rec, lang),
      url: lessonUrl(lang, r.rec.phaseSlug, r.rec.lessonSlug),
      minutes: r.minutes,
    });
    return {
      id: m.id,
      title: pick(m.title, zh && zh.title),
      titleEn: lang === 'zh' ? m.title : '',
      summary: pick(m.summary, zh && zh.summary),
      decisionPrompt: pick(m.decisionPrompt, zh && zh.decisionPrompt),
      minutes: m.estimatedMinutes || 0,
      lessonCount: flat.length,
      prerequisites: (m.prerequisites || []).map((p, i) => ({
        title: pick(p.title, zh && zh.prerequisites && zh.prerequisites[i] && zh.prerequisites[i].title),
        description: pick(p.description, zh && zh.prerequisites && zh.prerequisites[i] && zh.prerequisites[i].description),
      })),
      responsibilities: (m.responsibilities || []).map((s, i) => pick(s, zh && zh.responsibilities && zh.responsibilities[i])),
      goodFitIf: (m.goodFitIf || []).map((s, i) => pick(s, zh && zh.goodFitIf && zh.goodFitIf[i])),
      portfolio: m.portfolioProof
        ? {
            title: pick(m.portfolioProof.title, zh && zh.portfolio && zh.portfolio.title),
            description: pick(m.portfolioProof.description, zh && zh.portfolio && zh.portfolio.description),
          }
        : null,
      stages: stages.map((st) => {
        const zs = zhStage(st.id);
        return {
          id: st.id,
          title: lang === 'zh' ? (STAGE_TITLES_ZH[st.id] || st.titleEn) : st.titleEn,
          outcome: pick(st.outcome, zs && zs.outcome),
          artifact: pick(st.artifact, zs && zs.artifact),
          lessons: st.recs.map(row),
        };
      }),
      lessons: flat.map(row),
    };
  });
}

/* ---------------------------------------------------------------- main */
function main() {
  const t0 = Date.now();
  strings = loadStrings();
  phasesScan = scan();

  const { phases, lessons, warnings } = phasesScan;
  const quizCount = lessons.filter((l) => l.quiz).length;
  learningPaths = loadLearningPaths(lessons);
  console.log(
    'Scanned ' + phases.length + ' phases, ' + lessons.length + ' lessons ('
      + quizCount + ' with quiz, ' + lessons.filter((l) => l.hasZh).length + ' zh)',
  );

  // 1. clean dist
  fs.rmSync(DIST, { recursive: true, force: true });
  fs.mkdirSync(DIST, { recursive: true });

  // 2. static assets: css concat, js, favicon
  const cssDir = path.join(__dirname, 'src', 'css');
  const css = ['tokens.css', 'base.css', 'layout.css', 'content.css', 'components.css']
    .map((f) => fs.readFileSync(path.join(cssDir, f), 'utf8'))
    .join('\n');
  write('assets/site/site.css', css);
  for (const f of ['theme.js', 'app.js', 'search.js', 'quiz.js', 'enhance.js']) {
    fs.copyFileSync(path.join(__dirname, 'src', 'js', f), path.join(DIST, 'assets', 'site', f));
    fs.mkdirSync(path.join(DIST, 'assets', 'site'), { recursive: true });
  }
  fs.copyFileSync(
    path.join(__dirname, 'src', 'assets', 'favicon.svg'),
    path.join(DIST, 'assets', 'site', 'favicon.svg'),
  );

  // 2b. vendored KaTeX — math must render without any CDN access.
  copyDir(path.join(__dirname, 'vendor', 'katex'), path.join(DIST, 'assets', 'katex'));

  // 2c. vendored mermaid ESM — diagram rendering must also survive offline.
  if (fs.existsSync(path.join(__dirname, 'vendor', 'mermaid'))) {
    copyDir(path.join(__dirname, 'vendor', 'mermaid'), path.join(DIST, 'assets', 'mermaid'));
  }

  // 3. lesson images (shared across languages)
  const phasesRoot = path.join(SRC_REPO, 'phases');
  let assetDirs = 0;
  for (const phase of phases) {
    for (const lesson of phase.lessons) {
      const assetsDir = path.join(phasesRoot, phase.slug, lesson.lessonSlug, 'assets');
      if (fs.existsSync(assetsDir)) {
        copyDir(assetsDir, path.join(DIST, 'static', 'phases', phase.slug, lesson.lessonSlug, 'assets'));
        assetDirs++;
      }
    }
  }

  // 4. render pages
  let pages = 0;
  for (const lang of LANGS) {
    const t = makeT(lang);
    const L = makeLookup(lang);

    // home
    const startLesson = lessons[0];
    const homeMain = homeTpl.renderMain({
      lang,
      base: BASE,
      t,
      site: SITE,
      hero: {
        kicker: L('Free and open source curriculum'),
        title: lang === 'zh' ? SITE.nameZh : SITE.name,
        sub: lang === 'zh' ? SITE.descriptionZh : SITE.description,
        ctaStart: L('Start the Course'),
        ctaBrowse: L('Browse all phases'),
        startUrl: lessonUrl(lang, startLesson.phaseSlug, startLesson.lessonSlug),
        browseUrl: BASE + lang + '/#phases',
      },
      stats: [
        { num: String(phases.length), label: L('Phases') },
        { num: String(lessons.length), label: L('Lessons') },
        { num: String(quizCount), label: L('Interactive Quizzes') },
        { num: '4', label: L('Code Languages') },
      ],
      phases: phases.map((phase) => ({
        numLabel: phase.numLabel,
        title: phase.title[lang],
        count: phase.lessons.length,
        url: phaseUrl(lang, phase.slug),
      })),
      how: {
        title: L('How this works'),
        steps: [
          L('Pick a phase below — lessons build on each other within a phase.'),
          L('Every lesson explains the concept, builds it from raw math or code, then runs the same operation through the production library.'),
          L('Take the quiz at the end of each lesson to check your understanding.'),
        ],
      },
    });
    write(path.join(lang, 'index.html'), layout.render({
      lang,
      base: BASE,
      t,
      site: SITE,
      cdn: CDN,
      title: (lang === 'zh' ? SITE.nameZh : SITE.name) + ' — ' + (lang === 'zh' ? SITE.descriptionZh : SITE.description),
      description: lang === 'zh' ? SITE.descriptionZh : SITE.description,
      urlPath: BASE + lang + '/',
      urlEn: BASE + 'en/',
      urlZh: BASE + 'zh/',
      sidebarHtml: sidebarHtmlFor(lang, null, null, null),
      mainHtml: homeMain,
      tocHtml: '',
      pageData: clientData(lang),
      needsMath: false,
      needsMermaid: false,
    }));
    pages++;

    // phase landings
    for (const phase of phases) {
      const phaseMain = phaseTpl.renderMain({
        lang,
        base: BASE,
        t,
        phase: {
          numLabel: phase.numLabel,
          title: phase.title[lang],
          hook: phase.hook[lang],
          url: phaseUrl(lang, phase.slug),
        },
        lessons: phase.lessons.map((l) => ({
          n: l.numLabel,
          title: titleFor(l, lang),
          url: lessonUrl(lang, l.phaseSlug, l.lessonSlug),
          type: l.meta.en.type,
          minutes: l.meta.en.minutes,
          hasQuiz: !!l.quiz,
        })),
      });
      write(
        path.join(lang, 'phases', phase.slug, 'index.html'),
        layout.render({
          lang,
          base: BASE,
          t,
          site: SITE,
          cdn: CDN,
          title: phase.title[lang] + ' — ' + (lang === 'zh' ? SITE.nameZh : SITE.name),
          description: phase.hook[lang] || (lang === 'zh' ? SITE.descriptionZh : SITE.description),
          urlPath: phaseUrl(lang, phase.slug),
          urlEn: phaseUrl('en', phase.slug),
          urlZh: phaseUrl('zh', phase.slug),
          sidebarHtml: sidebarHtmlFor(lang, phase.slug, phase.lessons, null),
          mainHtml: phaseMain,
          tocHtml: '',
          pageData: clientData(lang),
          needsMath: false,
          needsMermaid: false,
        }),
      );
      pages++;
    }

    // learning paths (the AI-engineer learning map, above phase 00 in the nav)
    const pathsView = localizedPaths(lang);
    const pathsTitle = lang === 'zh' ? 'AI 工程师学习地图' : 'AI Engineer Learning Map';
    const pathsDesc = lang === 'zh'
      ? '13 条面向在职 AI 工程师的角色路线，按岗位选择课程顺序，直达实战作品。'
      : 'Thirteen role-based routes through the curriculum — pick a role, follow the ordered lessons, build the portfolio.';
    write(
      path.join(lang, 'learning-paths', 'index.html'),
      layout.render({
        lang,
        base: BASE,
        t,
        site: SITE,
        cdn: CDN,
        title: pathsTitle + ' — ' + (lang === 'zh' ? SITE.nameZh : SITE.name),
        description: pathsDesc,
        urlPath: BASE + lang + '/learning-paths/',
        urlEn: BASE + 'en/learning-paths/',
        urlZh: BASE + 'zh/learning-paths/',
        sidebarHtml: sidebarHtmlFor(lang, null, null, null, 'paths'),
        mainHtml: pathsTpl.renderMain({ lang, t, paths: pathsView }),
        tocHtml: '',
        pageData: clientData(lang),
        needsMath: false,
        needsMermaid: false,
      }),
    );
    pages++;

    // lessons
    for (const record of lessons) {
      const meta = record.meta[lang] || record.meta.en;
      const title = titleFor(record, lang);
      const rendered = renderMarkdown(lang, record);
      const url = lessonUrl(lang, record.phaseSlug, record.lessonSlug);
      const t2 = makeT(lang);

      const tocHtml = lessonTpl.renderToc(rendered.toc);
      const mainHtml = lessonTpl.renderMain({
        lang,
        base: BASE,
        t: t2,
        lesson: {
          title,
          type: meta.type,
          languages: meta.languages,
          minutes: meta.minutes,
          prerequisites: meta.prerequisites,
          hook: meta.hook,
        },
        phase: {
          title: phasesScan.phases.find((p) => p.slug === record.phaseSlug).title[lang],
          url: phaseUrl(lang, record.phaseSlug),
        },
        mdHtml: rendered.body,
        toc: rendered.toc,
        prev: record.prev
          ? { title: titleFor(record.prev, lang), url: lessonUrl(lang, record.prev.phaseSlug, record.prev.lessonSlug) }
          : null,
        next: record.next
          ? { title: titleFor(record.next, lang), url: lessonUrl(lang, record.next.phaseSlug, record.next.lessonSlug) }
          : null,
        quizJson: quizJsonFor(record),
        githubUrl: record.githubUrl,
        quizNote: lang === 'zh' ? t2.quizNoteEn : '',
      });

      write(
        path.join(lang, 'phases', record.phaseSlug, record.lessonSlug, 'index.html'),
        layout.render({
          lang,
          base: BASE,
          t: t2,
          site: SITE,
          cdn: CDN,
          title: title + ' — ' + (lang === 'zh' ? SITE.nameZh : SITE.name),
          description: meta.hook || (lang === 'zh' ? SITE.descriptionZh : SITE.description),
          urlPath: url,
          urlEn: lessonUrl('en', record.phaseSlug, record.lessonSlug),
          urlZh: lessonUrl('zh', record.phaseSlug, record.lessonSlug),
          sidebarHtml: sidebarHtmlFor(lang, record.phaseSlug, null, record.lessonSlug),
          mainHtml,
          tocHtml,
          pageData: clientData(lang),
          needsMath: rendered.needsMath,
          needsMermaid: rendered.needsMermaid,
          needsFigures: rendered.needsFigures,
        }),
      );
      pages++;
    }
  }

  // 5. search + tree JSON per language
  for (const lang of LANGS) {
    const search = [];
    for (const phase of phases) {
      search.push({
        u: phaseUrl(lang, phase.slug),
        t: phase.title[lang],
        s: phase.hook[lang] || '',
        k: '',
        p: lang === 'zh' ? '阶段' : 'Phase',
        y: '',
      });
    }
    for (const record of lessons) {
      const meta = record.meta[lang] || record.meta.en;
      search.push({
        u: lessonUrl(lang, record.phaseSlug, record.lessonSlug),
        t: titleFor(record, lang),
        s: (meta.hook || '').slice(0, 140),
        k: meta.keywords,
        p: phasesScan.phases.find((p) => p.slug === record.phaseSlug).title[lang],
        y: meta.type,
      });
    }
    write(path.join('assets', 'search', lang + '.json'), JSON.stringify(search));

    const tree = {};
    for (const phase of phases) {
      tree[phase.slug] = phase.lessons.map((l) => ({
        n: l.numLabel,
        t: l.meta[lang] ? l.meta[lang].title : l.meta.en.title,
        u: lessonUrl(lang, l.phaseSlug, l.lessonSlug),
      }));
    }
    write(path.join('assets', 'tree', lang + '.json'), JSON.stringify(tree));
  }

  // 6. figure runtime: providers + lazy-load manifest (absolute URLs so the
  //    runtime's relative injection works from any page depth)
  const figDir = path.join(__dirname, 'vendor', 'figures');
  if (figureKeys.size) {
    fs.mkdirSync(path.join(DIST, 'assets', 'figures'), { recursive: true });
    const runtimeSrc = fs.readFileSync(path.join(figDir, 'lesson-figures.js'), 'utf8');
    fs.copyFileSync(path.join(figDir, 'lesson-figures.js'), path.join(DIST, 'assets', 'figures', 'lesson-figures.js'));
    const providerFiles = fs.readdirSync(figDir).filter((f) => /^figures-.*\.js$/.test(f)).sort();
    const providerSources = providerFiles.map((f) => ({
      file: f,
      src: fs.readFileSync(path.join(figDir, f), 'utf8'),
    }));
    const hash = (s) => {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = ((h * 31) + s.charCodeAt(i)) >>> 0;
      return h.toString(36);
    };
    const versions = {};
    const providerUrls = [];
    for (const p of providerSources) {
      const url = BASE + 'assets/figures/' + p.file;
      versions[url] = hash(p.src);
      providerUrls.push(url);
      fs.copyFileSync(path.join(figDir, p.file), path.join(DIST, 'assets', 'figures', p.file));
    }
    const providersByFigure = {};
    const unresolved = [];
    for (const key of [...figureKeys].sort()) {
      const quoted = new RegExp("['\"]" + key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + "['\"]");
      const providers = providerSources.filter((p) => quoted.test(p.src)).map((p) => BASE + 'assets/figures/' + p.file);
      if (providers.length) providersByFigure[key] = providers;
      else if (!quoted.test(runtimeSrc)) unresolved.push(key);
    }
    if (unresolved.length) {
      console.log('Figure keys with no provider (placeholder card kept): ' + unresolved.slice(0, 8).join(', ')
        + (unresolved.length > 8 ? ' …' : ''));
    }
    const manifest = '// Auto-generated by build.js from figure fences and provider registrations.\n'
      + '// Provider order is significant: later registrations override earlier ones.\n'
      + 'window.AIFS_FIGURE_PROVIDER_ORDER=' + JSON.stringify(providerUrls) + ';\n'
      + 'window.AIFS_FIGURE_PROVIDER_VERSIONS=' + JSON.stringify(versions) + ';\n'
      + 'window.AIFS_FIGURE_PROVIDERS=' + JSON.stringify(providersByFigure) + ';\n';
    write(path.join('assets', 'figures', 'figure-manifest.js'), manifest);
    console.log('Figures: ' + figureKeys.size + ' keys, ' + providerFiles.length + ' providers, '
      + Object.keys(providersByFigure).length + ' resolved');
  }

  // 7. root negotiation page + 404
  write('index.html', negotiationHtml());
  write('404.html', notFoundHtml());

  // 7. report
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log('Rendered ' + pages + ' pages, ' + assetDirs + ' lesson asset dirs');
  if (warnings.length) {
    console.log('Warnings (' + warnings.length + '):');
    warnings.slice(0, 10).forEach((w) => console.log('  - ' + w));
    if (warnings.length > 10) console.log('  … and ' + (warnings.length - 10) + ' more');
  }
  console.log('Build complete in ' + elapsed + 's → ' + DIST);
}

/** Client-side string subset for app/search/quiz/enhance. */
function clientData(lang) {
  const t = makeT(lang);
  const L = makeLookup(lang);
  return {
    t: {
      copied: t.copied,
      loading: t.loading,
      searchEmpty: t.searchEmpty,
      searchNavigate: t.searchNavigate,
      searchOpen: t.searchOpen,
      searchClose: t.searchClose,
      quiz: {
        stages: {
          pre: L('Pre-Lesson Check'),
          check: L('Mid-Lesson Check'),
          post: L('Post-Lesson Quiz'),
        },
        score: L('Score'),
        retry: L('Retry this check'),
        perfect: L('Perfect score!'),
        great: L('Great work!'),
        keep: L('Keep studying!'),
      },
    },
  };
}

const SHARED_CSS_LINK = '<link rel="stylesheet" href="';

function baseHtml(lang, title) {
  const boot = '(function(){try{var s=localStorage.getItem("ae-theme");'
    + 'var d=s?s==="dark":!!(window.matchMedia&&matchMedia("(prefers-color-scheme: dark)").matches);'
    + 'if(d)document.documentElement.setAttribute("data-ds-dark-theme","");'
    + 'document.documentElement.style.colorScheme=d?"dark":"light";}catch(e){}})();';
  return '<!DOCTYPE html>\n<html lang="' + (lang || 'en') + '">\n<head>\n'
    + '<meta charset="utf-8">\n'
    + '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    + '<title>' + title + '</title>\n'
    + '<link rel="icon" type="image/svg+xml" href="' + BASE + 'assets/site/favicon.svg">\n'
    + '<script>' + boot + '</scr' + 'ipt>\n'
    + SHARED_CSS_LINK + BASE + 'assets/site/site.css">\n'
    + '<script defer src="' + BASE + 'assets/site/theme.js"></scr' + 'ipt>\n'
    + '</head>\n<body>\n';
}

function negotiationHtml() {
  return baseHtml('en', SITE.name + ' · ' + SITE.nameZh)
    + '<div class="center-page"><div class="center-box">'
    + '<span class="brand-mark" style="margin:0 auto 18px;width:48px;height:48px;display:grid;place-items:center;border-radius:12px;background:#151517;color:#f9fafb">'
    + '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"><path d="M2 3h20"/><path d="M4 7h16"/><path d="M6 11h12"/><path d="M9 15h6"/><path d="M11 19h2"/></svg></span>'
    + '<h1 style="font-size:30px;font-weight:700;margin:0 0 10px">AI Engineering from Scratch</h1>'
    + '<p style="color:var(--ds-label-secondary);margin:0 0 28px">' + SITE.description + '</p>'
    + '<div class="lang-choice">'
    + '<a class="btn btn-primary" id="go-en" href="' + BASE + 'en/">English</a>'
    + '<a class="btn" id="go-zh" href="' + BASE + 'zh/">中文</a>'
    + '</div>'
    + '<noscript><p style="margin-top:16px;color:var(--ds-label-tertiary)">Pick a language above. / 请选择语言。</p></noscript>'
    + '<script>(function(){var l=(navigator.language||"en").toLowerCase();'
    + 'var pref=null;try{pref=localStorage.getItem("ae-lang")}catch(e){}'
    + 'var target=(pref||l).indexOf("zh")===0?"go-zh":"go-en";'
    + 'location.replace(document.getElementById(target).href);})();</scr' + 'ipt>'
    + '</div></div>\n</body>\n</html>\n';
}

function notFoundHtml() {
  return baseHtml('en', '404 — ' + SITE.name)
    + '<div class="center-page"><div class="center-box">'
    + '<div class="code-404">404</div>'
    + '<h1>Page not found</h1>'
    + '<p>页面不存在 · The page you requested does not exist. Pick a language to continue:</p>'
    + '<div class="lang-choice">'
    + '<a class="btn btn-primary" href="' + BASE + 'en/">English</a>'
    + '<a class="btn" href="' + BASE + 'zh/">中文</a>'
    + '</div>'
    + '</div></div>\n</body>\n</html>\n';
}

main();
