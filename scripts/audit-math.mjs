/**
 * audit-math.mjs — render every TeX payload in dist pages through KaTeX
 * (strict throwOnError mode) and report any that would not compile.
 * Browser rendering uses throwOnError:false (red error text, never raw
 * TeX), so anything reported here is cosmetic-quality, not a blocker.
 * Run from the project root: node scripts/audit-math.mjs
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const katex = require(join(process.cwd(), 'vendor', 'katex', 'katex.min.js'));

function decode(s) {
  return s
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (name.endsWith('.html')) out.push(full);
  }
  return out;
}

const pages = walk(join(process.cwd(), 'dist'));
let total = 0;
const failures = [];
for (const page of pages) {
  const html = readFileSync(page, 'utf8');
  for (const m of html.matchAll(/data-math="(display|inline)"(?:\s[^>]*)?>([\s\S]*?)<\/(?:div|span)>/g)) {
    const tex = decode(m[2]).trim();
    if (!tex) continue;
    total++;
    try {
      katex.renderToString(tex, { displayMode: m[1] === 'display', throwOnError: true, strict: false });
    } catch (e) {
      failures.push({ page: page.slice(process.cwd().length + 1), mode: m[1], tex: tex.slice(0, 90), err: e.message.slice(0, 90) });
    }
  }
}

console.log('pages: ' + pages.length + ' | math payloads: ' + total + ' | strict-mode failures: ' + failures.length);
for (const f of failures.slice(0, 25)) {
  console.log('\n  ' + f.page + ' [' + f.mode + ']\n    tex: ' + f.tex + '\n    err: ' + f.err);
}
