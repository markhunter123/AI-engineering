/** probe-paths.mjs — one-off: validate every learning-path lesson reference
 * against the lessons our site actually ships, and report optionals. */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const dir = 'D:/project/ai-engineering-from-scratch/learning-paths';
const files = readdirSync(dir).filter((f) => f.endsWith('.json'));
let refs = 0, optional = 0;
const missing = [];
for (const f of files) {
  const m = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  for (const l of m.lessons || []) {
    refs++;
    if (l.required === false) optional++;
    const [ph, ls] = l.path.split('/').slice(1);
    const en = `D:/project/ai-engineering-from-scratch/${l.path}/docs/en.md`;
    if (!existsSync(en)) missing.push(`${m.id}: ${l.path}`);
    else {
      // lesson dir number must exist in our dist URL naming
      const dist = `D:/project/ai-engineer/dist/en/${l.path}/index.html`;
      if (!existsSync(dist)) missing.push(`NO-DIST ${m.id}: ${l.path}`);
    }
  }
}
console.log(`lesson refs: ${refs}, optional: ${optional}, missing: ${missing.length}`);
missing.forEach((x) => console.log('  ' + x));
