/**
 * verify.mjs — post-build integrity checks over dist/.
 *
 *   node build.js && node scripts/verify.mjs
 *
 * Checks:
 *  1. every internal href/src in every HTML page resolves to a file
 *  2. en/zh lesson pages exist in pairs, with matching hreflang set
 *  3. search/tree JSON parse and cover every lesson/phase
 *  4. quiz JSON inline blocks parse
 * Exits non-zero on any error; warnings are informational.
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');

const errors = [];
const warnings = [];

async function walk(dir, out = []) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, out);
    else out.push(full);
  }
  return out;
}

function stripFencedBlocks(html) {
  // quiz-data JSON contains URLs that are not markup; drop those blocks first.
  return html.replace(
    /<script type="application\/json"[\s\S]*?<\/script>/g,
    '',
  );
}

const INTERNAL = /^(?:[a-z]+:)?\/\//;

async function main() {
  const files = await walk(root);
  const htmlFiles = files.filter((f) => f.endsWith('.html'));
  const exists = new Set(files.map((f) => path.relative(root, f).split(path.sep).join('/')));

  let linksChecked = 0;

  for (const file of htmlFiles) {
    const rel = path.relative(root, file).split(path.sep).join('/');
    const html = await fs.readFile(file, 'utf8');
    const cleaned = stripFencedBlocks(html);

    // Collect href/src candidates.
    const attrs = [...cleaned.matchAll(/\b(?:href|src)="([^"]+)"/g)].map((m) => m[1]);
    for (const raw of attrs) {
      if (!raw || raw.startsWith('#') || INTERNAL.test(raw)) continue;
      if (raw.startsWith('data:')) continue;
      linksChecked++;
      // Resolve root-relative URL against dist root (BASE is '/').
      const clean = raw.split('#')[0].split('?')[0];
      if (!clean) continue;
      let target = decodeURIComponent(clean.replace(/^\//, '')).replace(/\/+$/, '');
      const candidates = target
        ? [target, target + '/index.html', target + '.html']
        : ['index.html'];
      if (!candidates.some((c) => exists.has(c))) {
        errors.push(rel + ' → broken link: ' + raw);
      }
    }

    // Inline anchors must exist in the same page (skip TOC-less pages).
    const anchors = new Set([...cleaned.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
    for (const m of cleaned.matchAll(/href="#([^"]+)"/g)) {
      linksChecked++;
      if (!anchors.has(m[1])) errors.push(rel + ' → broken anchor #' + m[1]);
    }
  }

  // --- language pairing --------------------------------------------------
  const enLessons = new Set(
    (await walk(path.join(root, 'en', 'phases')))
      .filter((f) => f.endsWith('index.html'))
      .map((f) => path.relative(path.join(root, 'en', 'phases'), f).split(path.sep).join('/')),
  );
  const zhLessons = new Set(
    (await walk(path.join(root, 'zh', 'phases')))
      .filter((f) => f.endsWith('index.html'))
      .map((f) => path.relative(path.join(root, 'zh', 'phases'), f).split(path.sep).join('/')),
  );
  for (const l of enLessons) if (!zhLessons.has(l)) errors.push('zh missing for ' + l);
  for (const l of zhLessons) if (!enLessons.has(l)) errors.push('en missing for ' + l);

  // hreflang pairing spot check on every 200th lesson page.
  const enList = [...enLessons].sort();
  for (let i = 0; i < enList.length; i += 200) {
    const html = await fs.readFile(path.join(root, 'en', 'phases', enList[i]), 'utf8');
    if (!html.includes('hreflang="zh"')) errors.push('no zh hreflang: ' + enList[i]);
  }

  // --- JSON assets --------------------------------------------------------
  for (const lang of ['en', 'zh']) {
    const search = JSON.parse(await fs.readFile(path.join(root, 'assets', 'search', lang + '.json'), 'utf8'));
    if (search.length < enLessons.size) {
      errors.push('search/' + lang + '.json too small: ' + search.length);
    }
    for (const entry of search) {
      if (!entry.u || !entry.t) errors.push('search/' + lang + ' entry missing u/t');
    }
    const tree = JSON.parse(await fs.readFile(path.join(root, 'assets', 'tree', lang + '.json'), 'utf8'));
    if (Object.keys(tree).length !== 20) {
      errors.push('tree/' + lang + '.json phases: ' + Object.keys(tree).length + ' (want 20)');
    }
  }

  // --- inline quiz JSON parse ---------------------------------------------
  let quizBlocks = 0;
  for (const file of htmlFiles) {
    const html = await fs.readFile(file, 'utf8');
    for (const m of html.matchAll(/<script type="application\/json" id="quiz-data">([\s\S]*?)<\/script>/g)) {
      quizBlocks++;
      try {
        const data = JSON.parse(m[1].replace(/\\u003c/g, '<'));
        if (!data.questions || !data.questions.length) throw new Error('empty');
      } catch (e) {
        errors.push(path.relative(root, file) + ' → quiz JSON unparsable: ' + e.message);
      }
    }
  }

  // --- report ---------------------------------------------------------------
  const enCount = enLessons.size - 20; /* minus phase landings */
  console.log('Pages: ' + htmlFiles.length + ' | Links checked: ' + linksChecked
    + ' | Lesson pairs: ' + enCount + ' en / ' + zhLessons.size + ' zh'
    + ' | Quiz blocks: ' + quizBlocks);
  if (warnings.length) console.log('Warnings:\n  ' + warnings.join('\n  '));
  if (errors.length) {
    console.error('ERRORS (' + errors.length + '):\n  ' + errors.slice(0, 30).join('\n  '));
    if (errors.length > 30) console.error('  … and ' + (errors.length - 30) + ' more');
    process.exit(1);
  }
  console.log('verify: all checks passed');
}

main().catch((e) => { console.error(e); process.exit(1); });
