/**
 * scan-colors.mjs — report any residual colored value (hex/rgb/hsl with
 * unequal channels, or a named CSS color used as a value) in the built
 * site assets, so the strict monochrome design can be verified.
 * Usage: node scripts/scan-colors.mjs
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';

// dist/assets/katex is vendored third-party (pinned katex@0.16.11) and is
// excluded on purpose: its JS carries internal error/default color constants
// that no lesson or page style can reach. Site-owned surfaces are all here.
const roots = ['dist/assets/site', 'dist/assets/figures', 'src/css', 'src/js', 'src/assets'];
const exts = new Set(['.css', '.js', '.svg', '.html']);
const NAMED = /\b(red|blue|green|orange|purple|yellow|teal|cyan|magenta|pink|brown|navy|maroon|olive|lime|aqua|fuchsia|silver|steelblue|tomato|gold|indigo|violet|orchid|coral|salmon|khaki|plum|turquoise|lavender|crimson)\b/g;

let problems = 0;
function walk(dir) {
  let out = [];
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) out = out.concat(walk(p));
    else if (exts.has(extname(p))) out.push(p);
  }
  return out;
}

function check(file) {
  const text = readFileSync(file, 'utf8');
  const found = [];
  for (const m of text.matchAll(/#([0-9a-fA-F]{6})\b/g)) {
    const h = m[1];
    const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
    if (r !== g || g !== b) found.push(`hex ${m[0]}`);
  }
  for (const m of text.matchAll(/#([0-9a-fA-F]{3})\b/g)) {
    const h = m[1];
    const r = parseInt(h[0] + h[0], 16), g = parseInt(h[1] + h[1], 16), b = parseInt(h[2] + h[2], 16);
    if (r !== g || g !== b) found.push(`hex ${m[0]}`);
  }
  for (const m of text.matchAll(/rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})/g)) {
    const r = +m[1], g = +m[2], b = +m[3];
    if (r !== g || g !== b) found.push(`rgb ${m[0]})`);
  }
  if (/hsl/.test(text)) found.push('hsl()');
  for (const m of text.matchAll(NAMED)) {
    // Painted-value check: the word must sit directly after a value colon
    // (CSS files) or after a paint property (fill/stroke/color/... in
    // JS/HTML/SVG). Prose ("red zone in use"), comments ("no blue tint"),
    // identifiers (var green = [1,0,1]) and label strings never paint
    // pixels, so they pass. A word followed by [ or ( is an indexed value
    // (green[i]) or a function call, not a literal.
    const idx = m.index;
    const next = text[idx + m[0].length];
    if (next === '[' || next === '(') continue;
    const prefix = text.slice(Math.max(0, idx - 48), idx);
    const re = /\.css$/i.test(file)
      ? /[:=]\s*[\s'"`]*$/
      : /(fill|stroke|color|background|border)\s*[:=]\s*[\s'"`]*$/i;
    if (re.test(prefix)) found.push(`named ${m[0]}`);
  }
  if (found.length) {
    problems += found.length;
    console.log(`${file}:`);
    for (const f of [...new Set(found)].slice(0, 6)) console.log(`   ${f}`);
  }
}

for (const root of roots) {
  try { walk(root).forEach(check); } catch { console.log(`skip missing ${root}`); }
}
console.log(problems ? `TOTAL residual colored values: ${problems}` : 'CLEAN: no colored values anywhere');
process.exit(problems ? 1 : 0);
