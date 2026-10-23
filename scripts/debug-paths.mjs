/** debug-paths.mjs — one-off: find learning-path lesson refs that our scan
 *  keys (phaseSlug/lessonSlug) do not match. */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const src = 'D:/project/ai-engineering-from-scratch';
// emulate the scan's slug fields from directory names
const recs = new Set();
for (const ph of readdirSync(join(src, 'phases'))) {
  for (const ls of readdirSync(join(src, 'phases', ph))) {
    if (existsSync(join(src, 'phases', ph, ls, 'docs', 'en.md'))) recs.add(ph + '/' + ls);
  }
}
const dir = join(src, 'learning-paths');
let bad = 0;
for (const f of readdirSync(dir).filter((x) => x.endsWith('.json'))) {
  const m = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  for (const l of m.lessons || []) {
    const parts = String(l.path || '').split('/').filter(Boolean);
    const key = parts[1] + '/' + parts[2];
    if (!recs.has(key)) { bad++; console.log(`${m.id}: "${l.path}" -> key "${key}" NOT IN SCAN`); }
  }
  for (const s of m.stages || []) {
    for (const p of s.lessonPaths || []) {
      const parts = String(p).split('/').filter(Boolean);
      const key = parts[1] + '/' + parts[2];
      if (!recs.has(key)) { bad++; console.log(`${m.id} stage ${s.id}: "${p}" -> "${key}" NOT IN SCAN`); }
    }
  }
}
console.log('bad refs:', bad);
