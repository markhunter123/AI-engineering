/**
 * audit-figures.mjs — list figure keys used by dist pages that have no
 * registered provider (their placeholder card can never be replaced).
 * Run from the project root: node scripts/audit-figures.mjs
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = process.cwd();
const DIST = join(ROOT, 'dist');
const FIGDIR = join(ROOT, 'vendor', 'figures');

/* provider sources: a key resolves if its quoted form appears in a provider
 * file or in the runtime itself (mirrors build.js step 6). */
const runtimeSrc = readFileSync(join(FIGDIR, 'lesson-figures.js'), 'utf8');
const providerFiles = readdirSync(FIGDIR).filter((f) => /^figures-.*\.js$/.test(f)).sort();
const providerSources = providerFiles.map((f) => ({
  file: f,
  src: readFileSync(join(FIGDIR, f), 'utf8'),
}));

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/* walk dist for HTML files */
function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (name.endsWith('.html')) out.push(full);
  }
  return out;
}

/* decode the handful of entities the renderer escapes */
function decode(s) {
  return s
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

const usage = new Map(); // key -> Set of page paths
const pages = walk(DIST);
for (const page of pages) {
  const html = readFileSync(page, 'utf8');
  for (const m of html.matchAll(/data-figure="([^"]+)"/g)) {
    let raw = decode(m[1]).trim();
    // cfg-carrying fences: "iso-machine {\"machine\":\"x\"}" — key is token 1
    try { JSON.parse(raw); } catch (_) { raw = raw.split(/\s+/)[0]; }
    if (!raw) continue;
    if (!usage.has(raw)) usage.set(raw, new Set());
    usage.get(raw).add(page.slice(DIST.length + 1));
  }
}

const unresolved = [];
for (const key of [...usage.keys()].sort()) {
  const quoted = new RegExp("['\"]" + escapeRe(key) + "['\"]");
  const hit = providerSources.some((p) => quoted.test(p.src)) || quoted.test(runtimeSrc);
  if (!hit) unresolved.push(key);
}

console.log('pages scanned: ' + pages.length);
console.log('figure keys used in pages: ' + usage.size);
console.log('providers: ' + providerFiles.length);
console.log('unresolved keys: ' + unresolved.length);
for (const key of unresolved) {
  const sites = [...usage.get(key)];
  console.log('\n  ' + key + '  (' + sites.length + ' page' + (sites.length > 1 ? 's' : '') + ')');
  for (const s of sites.slice(0, 4)) console.log('    ' + s);
}
