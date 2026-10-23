/**
 * make-chunks.mjs — build the translation chunk manifest.
 *
 *   node scripts/make-chunks.mjs
 *
 * Groups lessons into chunks of ≤6 lessons AND ≤80KB of en.md, small
 * enough for one translation agent per chunk. Writes chunks.json next to
 * this script and prints a summary.
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = 'D:/project/ai-engineering-from-scratch';
const here = path.dirname(fileURLToPath(import.meta.url));

const MAX_LESSONS = 6;
const MAX_BYTES = 80 * 1024;

async function main() {
  const phasesDir = path.join(repo, 'phases');
  const phaseSlugs = (await fs.readdir(phasesDir)).filter((d) => /^\d{2}-/.test(d)).sort();
  const chunks = [];
  let id = 0;

  for (const phase of phaseSlugs) {
    const lessons = (await fs.readdir(path.join(phasesDir, phase)))
      .filter((d) => /^\d{2}-/.test(d))
      .sort();
    let batch = [];
    let bytes = 0;
    const flush = async () => {
      if (!batch.length) return;
      const sizes = [];
      for (const lesson of batch) {
        const stat = await fs.stat(path.join(phasesDir, phase, lesson, 'docs', 'en.md'));
        sizes.push(stat.size);
      }
      chunks.push({
        id: 'c' + String(++id).padStart(3, '0'),
        phase,
        lessons: batch,
        bytes: sizes.reduce((a, b) => a + b, 0),
      });
      batch = [];
      bytes = 0;
    };
    for (const lesson of lessons) {
      const size = (await fs.stat(path.join(phasesDir, phase, lesson, 'docs', 'en.md'))).size;
      if (batch.length && (batch.length >= MAX_LESSONS || bytes + size > MAX_BYTES)) await flush();
      batch.push(lesson);
      bytes += size;
    }
    await flush();
  }

  await fs.writeFile(path.join(here, 'chunks.json'), JSON.stringify(chunks, null, 1));
  console.log('chunks: ' + chunks.length
    + ' | lessons: ' + chunks.reduce((a, c) => a + c.lessons.length, 0)
    + ' | avg chunk: ' + Math.round(chunks.reduce((a, c) => a + c.bytes, 0) / chunks.length / 1024) + 'KB');
}

main().catch((e) => { console.error(e); process.exit(1); });
