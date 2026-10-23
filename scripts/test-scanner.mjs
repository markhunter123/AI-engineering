/** verify scanner + katex against the exact dist math payloads that failed. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const katex = require(join(process.cwd(), 'vendor', 'katex', 'katex.min.js'));

const src = readFileSync('src/js/enhance.js', 'utf8');
const m = src.match(/function escapeTextUnderscores\(tex\) \{[\s\S]*?\n  \}/);
const scanner = new Function('return (' + m[0].replace(/^function /, 'function ') + ')')();

function decode(s) {
  return s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

const cases = [
  ['dist/en/phases/19-capstone-projects/89-eval-harness-from-scratch/index.html', 'display'],
  ['dist/zh/phases/01-math-foundations/21-graph-theory/index.html', 'inline'],
  ['dist/zh/phases/01-math-foundations/11-singular-value-decomposition/index.html', 'display'],
  ['dist/zh/phases/01-math-foundations/14-norms-and-distances/index.html', 'inline'],
];

let total = 0, ok = 0;
for (const [file, mode] of cases) {
  const html = readFileSync(file, 'utf8');
  for (const mm of html.matchAll(/data-math="(?:display|inline)"(?:\s[^>]*)?>([\s\S]*?)<\/(?:div|span)>/g)) {
    const tex = decode(mm[1]).trim();
    if (!tex) continue;
    total++;
    const fixed = scanner(tex);
    try {
      katex.renderToString(fixed, { displayMode: mode === 'display', throwOnError: true, strict: false });
      ok++;
    } catch (e) {
      console.log('FAIL in ' + file + ' [' + mode + ']: ' + e.message.slice(0, 100));
      console.log('  tex was: ' + tex.slice(0, 80));
    }
  }
}
console.log('scanner+katex strict pass: ' + ok + '/' + total);
