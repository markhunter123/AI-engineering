/**
 * smoke-figures.mjs — mount every registered figure against a minimal DOM
 * shim and report the ones that throw. Zero deps. Run from the project root:
 *   node scripts/smoke-figures.mjs
 * The runtime logs "lesson figure \"x\" failed:" per broken figure; we count
 * them and print the distinct errors.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import vm from 'node:vm';

/* --- minimal DOM shim ----------------------------------------------------- */
function makeEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    children: [],
    attributes: {},
    style: {},
    dataset: {},
    _cls: new Set(),
    classList: null,
    style: { setProperty() {}, getPropertyValue: () => '', removeProperty() {} },
    setAttribute(k, v) {
      this.attributes[k] = String(v);
      if (k.indexOf('data-') === 0) {
        const prop = k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        this.dataset[prop] = String(v);
      }
    },
    getAttribute(k) { return this.attributes[k] == null ? null : this.attributes[k]; },
    removeAttribute(k) { delete this.attributes[k]; },
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); return c; },
    remove() {},
    insertBefore(c, ref) {
      const i = ref ? this.children.indexOf(ref) : -1;
      if (i >= 0) this.children.splice(i, 0, c);
      else this.children.push(c);
      c.parentNode = this;
      return c;
    },
    addEventListener() {}, removeEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
    contains() { return false; },
    focus() {}, blur() {},
    get textContent() { return ''; },
    set textContent(v) {},
    get innerHTML() { return ''; },
    set innerHTML(v) {},
    get firstChild() { return this.children[0] || null; },
    get lastChild() { return this.children[this.children.length - 1] || null; },
    get childNodes() { return this.children; },
    get parentNode() { return this.parentNode_ || null; },
    set parentNode(v) { this.parentNode_ = v; },
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
    querySelectorAll(sel) {
      const out = [];
      const m = /^([a-zA-Z][a-zA-Z0-9-]*)?((?:\.[a-zA-Z0-9_-]+)*)$/.exec(String(sel).trim());
      const wantTag = m && m[1] ? m[1].toUpperCase() : null;
      const wantCls = m && m[2] ? m[2].slice(1).split('.') : [];
      (function rec(node) {
        for (const c of node.children || []) {
          if (c.nodeType !== 1) continue; // skip text nodes etc.
          const tagOk = !wantTag || c.tagName === wantTag;
          const clsOk = wantCls.every((x) => c._cls.has(x));
          if (tagOk && clsOk) out.push(c);
          rec(c);
        }
      })(this);
      return out;
    },
    width: { baseVal: { value: 0 } },
    height: { baseVal: { value: 0 } },
    getBBox() { return { x: 0, y: 0, width: 0, height: 0 }; },
    getBoundingClientRect() { return { x: 0, y: 0, width: 800, height: 400, top: 0, left: 0 }; },
    createSVGRect() { return makeEl('rect'); },
    getScreenCTM() { return { inverse() { return this; }, a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }; },
  };
  el.classList = {
    add: (...cs) => cs.forEach((c) => el._cls.add(c)),
    remove: (...cs) => cs.forEach((c) => el._cls.delete(c)),
    toggle: (c) => (el._cls.has(c) ? el._cls.delete(c) : el._cls.add(c)),
    contains: (c) => el._cls.has(c),
  };
  return el;
}

const hosts = []; // filled once keys are known
const document = {
  documentElement: makeEl('html'),
  head: makeEl('head'),
  body: makeEl('body'),
  readyState: 'complete',
  createElement: (t) => makeEl(t),
  createElementNS: (ns, t) => makeEl(t),
  createTextNode: () => ({ nodeType: 3, textContent: '', children: [] }),
  querySelector: () => null,
  querySelectorAll: (sel) => (sel === '.lesson-figure[data-figure]' ? hosts : []),
  getElementById: () => null,
  addEventListener() {},
};
document.head.appendChild = function (c) { this.children.push(c); return c; };

const scriptsLoaded = [];
const sandbox = {
  document,
  console,
  window: null,
  Math, JSON, Promise, Error, Object, Array, String, Number, Boolean, Date, RegExp, isNaN, parseInt, parseFloat,
  requestAnimationFrame() { return 0; },
  cancelAnimationFrame() {},
  setTimeout() { return 0; },
  clearTimeout() {},
  setInterval() { return 0; },
  clearInterval() {},
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  navigator: { userAgent: 'node', prefersReducedMotion: false },
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  location: { href: 'http://localhost/x' },
  performance: { now: () => 0 },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
sandbox.addEventListener = function () {};
sandbox.removeEventListener = function () {};
sandbox.dispatchEvent = function () { return true; };
document.addEventListener = function () {};

function run(file) {
  const code = readFileSync(file, 'utf8');
  vm.runInContext(code, vm.createContext(sandbox), { filename: file });
}

/* --- load runtime + manifest + all providers ------------------------------ */
const figDir = 'dist/assets/figures';
run(join(figDir, 'lesson-figures.js'));
run(join(figDir, 'figure-manifest.js'));

const manifestSrc = readFileSync(join(figDir, 'figure-manifest.js'), 'utf8');
const order = JSON.parse(manifestSrc.match(/AIFS_FIGURE_PROVIDER_ORDER=(\[.*?\]);/s)[1]);
for (const url of order) {
  const f = url.replace(/^.*\/assets\/figures\//, '');
  run(join(figDir, f));
  scriptsLoaded.push(f);
}

const FIGS = sandbox.window.LESSON_FIGURES || sandbox.LESSON_FIGURES;
const keys = Object.keys(FIGS);
console.log('providers loaded:', scriptsLoaded.length, '| registered figures:', keys.length);

/* --- one fake host per REAL data-figure value found in dist pages --------- */
const { readdirSync, statSync } = await import('node:fs');
const attrValues = new Set();
function walk(dir) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e === 'index.html') {
      const html = readFileSync(p, 'utf8');
      for (const m of html.matchAll(/data-figure="([^"]+)"/g)) {
        attrValues.add(m[1].replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'));
      }
    }
  }
}
walk('dist/en');
walk('dist/zh');
console.log('distinct data-figure attrs in pages:', attrValues.size);

const warn = console.warn;
const failures = new Map();
console.warn = (...args) => {
  const err = args.find((a) => a && a.stack);
  const msg = args.map(String).join(' ');
  const m = msg.match(/lesson figure "([^"]+)" failed:?/);
  if (m) {
    if (!failures.has(m[1])) {
      failures.set(m[1], err && err.stack ? err.stack.split('\n').slice(0, 4).join(' | ') : msg);
    }
  }
};
for (const v of attrValues) {
  const host = makeEl('div');
  host.setAttribute('data-figure', v);
  hosts.push(host);
}
try {
  sandbox.window.mountLessonFigures(document);
} catch (e) {
  console.error('mountLessonFigures threw:', e.message);
}
console.warn = warn;

const mounted = hosts.filter((h) => h.dataset.lfMounted).length;
console.log('mounted OK:', mounted, '/', keys.length);
if (failures.size) {
  console.log('\nFAILED FIGURES (' + failures.size + '):');
  for (const [k, msg] of failures) console.log('  ' + k + ' — ' + String(msg).split('\n')[0].slice(0, 140));
  process.exit(1);
} else {
  console.log('all figures mount cleanly');
}
