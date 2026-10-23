/**
 * Export the Chinese lesson snapshot from the curriculum repo's
 * `translations` branch into content/zh/ so builds run fully offline.
 *
 *   node scripts/export-zh.mjs
 *
 * Reads (never writes) the source repo via `git archive origin/translations`,
 * with a per-file `git show` fallback.
 */
import { execFileSync } from 'node:child_process';
import {
  mkdirSync,
  readdirSync,
  statSync,
  rmSync,
  writeFileSync,
  cpSync,
} from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const SRC_REPO = 'D:/project/ai-engineering-from-scratch';
const REF = 'origin/translations';
const TMP = join(ROOT, 'content', 'zh-tmp');
const DEST = join(ROOT, 'content', 'zh');
const EXPECT_LESSONS = 542;

function git(args) {
  return execFileSync('git', ['-C', SRC_REPO, ...args], {
    maxBuffer: 512 * 1024 * 1024,
  });
}

/** All i18n/zh lesson + ui.json paths on the ref. */
function listWanted() {
  const out = git(['ls-tree', '-r', '--name-only', REF, 'i18n/zh/']).toString();
  return out
    .split('\n')
    .filter(
      (f) =>
        /^i18n\/zh\/phases\/.+\/docs\/zh\.md$/.test(f) || f === 'i18n/zh/ui.json',
    );
}

function tryArchive() {
  try {
    mkdirSync(TMP, { recursive: true });
    // Archive the two top-level paths only — one path arg per file hits the
    // Windows command-line length limit with 543 files.
    const archive = git(['archive', REF, '--', 'i18n/zh/phases', 'i18n/zh/ui.json']);
    execFileSync('tar', ['-xf', '-', '-C', TMP], { input: archive });
    return true;
  } catch (err) {
    console.log(`git archive path failed (${err.message.slice(0, 80)}…); using git show fallback`);
    return false;
  }
}

function exportPerFile(files) {
  mkdirSync(TMP, { recursive: true });
  for (const rel of files) {
    const target = join(TMP, rel);
    mkdirSync(join(target, '..'), { recursive: true });
    writeFileSync(target, git(['show', `${REF}:${rel}`]));
  }
}

rmSync(TMP, { recursive: true, force: true });
rmSync(DEST, { recursive: true, force: true });

const wanted = listWanted();
console.log(
  `Found ${wanted.length} zh files on ${REF} (expect ${EXPECT_LESSONS} lessons + ui.json)`,
);

if (!tryArchive()) exportPerFile(wanted);

// Flatten i18n/zh/* -> content/zh/*
mkdirSync(DEST, { recursive: true });
cpSync(join(TMP, 'i18n', 'zh'), DEST, { recursive: true });

// Verify.
const lessonFiles = [];
(function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) walk(p);
    else if (entry.name === 'zh.md') lessonFiles.push(p);
  }
})(join(DEST, 'phases'));

const tiny = lessonFiles.filter((f) => statSync(f).size < 200);
console.log(`Exported ${lessonFiles.length} lessons to content/zh/phases`);
if (lessonFiles.length !== EXPECT_LESSONS || tiny.length > 0) {
  console.error(
    `FAIL: expected ${EXPECT_LESSONS} lessons, got ${lessonFiles.length}; tiny files: ${tiny.length}`,
  );
  process.exit(1);
}
if (!statSync(join(DEST, 'ui.json')).isFile()) {
  console.error('FAIL: content/zh/ui.json missing');
  process.exit(1);
}
rmSync(TMP, { recursive: true, force: true });
console.log('OK: content/zh ready (offline builds enabled).');
