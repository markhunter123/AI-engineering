/**
 * reword-capstone.mjs — reposition wording for working AI engineers:
 * replace the exact phrase 毕业项目 with 项目 across content/zh lesson
 * markdown. Leaves literal uses like "CNCF 毕业" / "MATS 毕业生" intact
 * (they never contain the four-character phrase). Idempotent.
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const root = 'content/zh';
let files = 0, hits = 0;

function walk(dir) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e.endsWith('.md')) {
      const text = readFileSync(p, 'utf8');
      const n = text.split('毕业项目').length - 1;
      if (n) {
        writeFileSync(p, text.replaceAll('毕业项目', '项目'), 'utf8');
        files++;
        hits += n;
      }
    }
  }
}

walk(root);
console.log(`reworded ${hits} occurrences across ${files} files`);
