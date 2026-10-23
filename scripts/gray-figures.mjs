/**
 * gray-figures.mjs — convert every color literal in the vendored figure
 * runtime to its grayscale equivalent (ITU-R BT.601 luminance), so the
 * interactive figures match the site's strict black-and-white system.
 * Handles #rgb / #rrggbb (any case) and rgb()/rgba() with integer channels.
 * Alpha and everything else is preserved. Run from the project root.
 */
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const dir = 'vendor/figures';
const files = readdirSync(dir).filter((f) => f.endsWith('.js'));

function lum(r, g, b) {
  return Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
}

function hex2(v) {
  return v.toString(16).padStart(2, '0');
}

function convert(text) {
  let changed = 0;

  // #rrggbb / #RGB (only when every channel is equal -> already gray, skip)
  text = text.replace(/#([0-9a-fA-F]{6})\b/g, (m, h) => {
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    if (r === g && g === b) return m;
    changed++;
    const y = hex2(lum(r, g, b));
    return `#${y}${y}${y}`;
  });
  text = text.replace(/#([0-9a-fA-F]{3})\b/g, (m, h) => {
    const r = parseInt(h[0] + h[0], 16);
    const g = parseInt(h[1] + h[1], 16);
    const b = parseInt(h[2] + h[2], 16);
    if (r === g && g === b) return m;
    changed++;
    const y = hex2(lum(r, g, b));
    return `#${y}${y}${y}`;
  });

  // rgb(r, g, b) / rgba(r, g, b, a)
  text = text.replace(/rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(,[^)]*)?\)/g, (m, rs, gs, bs, alpha) => {
    const r = Number(rs), g = Number(gs), b = Number(bs);
    if (r === g && g === b) return m;
    changed++;
    const y = lum(r, g, b);
    return `rgba(${y}, ${y}, ${y}${alpha || ''})`;
  });

  return { text, changed };
}

let total = 0;
for (const f of files) {
  const path = join(dir, f);
  const { text, changed } = convert(readFileSync(path, 'utf8'));
  if (changed) {
    writeFileSync(path, text, 'utf8');
    total += changed;
    console.log(`${f}: ${changed} colors grayed`);
  }
}
console.log(`done: ${total} color literals converted across ${files.length} files`);
