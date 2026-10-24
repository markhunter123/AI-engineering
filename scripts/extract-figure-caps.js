#!/usr/bin/env node
/**
 * extract-figure-caps.js — pull the long English caption out of every figure.
 *
 * Captions live in two shapes across vendor/figures/*.js:
 *   1. the last argument of a local shell(host, label, sub, svg, cap) call
 *   2. inline el('div', { class: 'lf-cap' }, ['…']) arrays (no shell helper)
 * This script tokenises each file (skipping strings/comments/regex literals),
 * resolves LF.register({...}) / var FIGS = {...} key→function maps, locates
 * each function body, and pulls the caption expression. Concatenated string
 * literals are joined; anything dynamic is flagged for manual handling.
 *
 * Output (scripts/zh-caps/):
 *   _caps.json          { key: { file, line, text } }  (+ flagged entries)
 *   _src/<file>.txt     "key\t<json-string>" lines, grouped for translation
 *   _missing.txt        keys whose caption could not be extracted cleanly
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const FIGDIR = path.join(ROOT, 'vendor', 'figures');
const OUT = path.join(__dirname, 'zh-caps');

// ---------------------------------------------------------------------------
// Tokeniser: strings, words, punctuation; comments and regex literals skipped.
// ---------------------------------------------------------------------------
function tokenize(src) {
  const out = [];
  const n = src.length;
  let i = 0;
  let lastSig = null; // last significant token, for the regex-literal heuristic
  while (i < n) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { i += 2; while (i < n && !(src[i] === '*' && src[i + 1] === '/')) i++; i += 2; continue; }
    if (c === "'" || c === '"' || c === '`') {
      const q = c; const start = i; let val = ''; i++;
      while (i < n && src[i] !== q) {
        if (src[i] === '\\') {
          const e = src[i + 1];
          val += e === 'n' ? '\n' : e === 't' ? '\t' : e;
          i += 2;
        } else { val += src[i]; i++; }
      }
      i++;
      lastSig = { t: 'str', v: val, i: start };
      out.push(lastSig);
      continue;
    }
    if (/[A-Za-z_$]/.test(c)) {
      const s = i; while (i < n && /[A-Za-z0-9_$]/.test(src[i])) i++;
      lastSig = { t: 'word', v: src.slice(s, i), i: s };
      out.push(lastSig);
      continue;
    }
    if (/[0-9]/.test(c)) {
      const s = i; while (i < n && /[A-Za-z0-9._]/.test(src[i])) i++;
      lastSig = { t: 'num', v: src.slice(s, i), i: s };
      out.push(lastSig);
      continue;
    }
    if (c === '/') {
      const canRegex = !lastSig || !/^(word|num|str)$/.test(lastSig.t) && !((lastSig.t === ')' ) || (lastSig.t === ']'));
      if (canRegex) {
        let m = i + 1;
        while (m < n && src[m] !== '\n' && src[m] !== '/') { if (src[m] === '\\') m++; m++; }
        if (m < n && src[m] === '/') {
          m++;
          while (m < n && /[a-z]/.test(src[m])) m++;
          lastSig = { t: 'regex', v: '', i };
          out.push(lastSig);
          i = m;
          continue;
        }
      }
      lastSig = { t: '/', v: '/', i };
      out.push(lastSig); i++; continue;
    }
    if (/\s/.test(c)) { i++; continue; }
    lastSig = { t: c, v: c, i };
    out.push(lastSig); i++;
  }
  return out;
}

function matchBracket(tok, open, close, at) {
  if (tok[at].t !== open) return -1;
  let depth = 0;
  for (let j = at; j < tok.length; j++) {
    if (tok[j].t === open) depth++;
    else if (tok[j].t === close) { depth--; if (!depth) return j; }
  }
  return -1;
}

// split tokens between `from`..`to` (exclusive) on top-level commas
function splitArgs(tok, from, to) {
  const args = [];
  let depth = 0; let cur = [];
  for (let j = from; j < to; j++) {
    const t = tok[j];
    if ('([{'.includes(t.t)) depth++;
    else if (')]}'.includes(t.t)) depth--;
    if (t.t === ',' && depth === 0) { args.push(cur); cur = []; continue; }
    cur.push(t);
  }
  if (cur.length) args.push(cur);
  return args;
}

// classify an argument token list → { text } or { raw } when dynamic
function classify(argTok) {
  const meaningful = argTok.filter((t) => t.t !== 'str' || t.v !== '');
  const strs = meaningful.filter((t) => t.t === 'str');
  const others = meaningful.filter((t) => t.t !== 'str' && t.t !== '+');
  if (!meaningful.length) return null;
  if (!others.length) return { text: strs.map((s) => s.v).join('') };
  return { raw: argTok.map((t) => (t.t === 'str' ? JSON.stringify(t.v) : t.t === 'word' ? t.v : t.t)).join(' ') };
}

// ---------------------------------------------------------------------------
// Per-file index: registration maps + function bodies
// ---------------------------------------------------------------------------
function findObjectLiteralAfter(tok, from) {
  // at `=` … `{` → return {start, end} of the object literal
  for (let j = from; j < Math.min(from + 4, tok.length); j++) {
    if (tok[j].t === '{') {
      const end = matchBracket(tok, '{', '}', j);
      return { start: j, end };
    }
    if (tok[j].t === ';') break;
  }
  return null;
}

function parseObjectPairs(tok, start, end) {
  const pairs = [];
  let j = start + 1;
  while (j < end) {
    if (tok[j].t !== 'str' && tok[j].t !== 'word') { j++; continue; }
    const key = tok[j].v;
    let k = j + 1;
    while (k < end && tok[k].t !== ':') k++;
    if (k >= end) { j++; continue; }
    k++;
    // value tokens until top-level ',' or '}'
    let depth = 0; const val = [];
    while (k < end) {
      const t = tok[k];
      if ('([{'.includes(t.t)) depth++;
      else if (')]}'.includes(t.t)) { if (!depth) break; depth--; }
      if (t.t === ',' && !depth) break;
      val.push(t); k++;
    }
    const lead = val.find((t) => t.t === 'word');
    pairs.push({ key, val, fn: lead ? lead.v : null });
    j = k;
  }
  return pairs;
}

// find registration maps in a token stream: LF.register({...}) or LF.register(name)
function findRegistrations(tok) {
  const regs = [];
  for (let i = 0; i < tok.length - 4; i++) {
    if (tok[i].t === 'word' && tok[i].v === 'register' && tok[i + 1].t === '(') {
      const close = matchBracket(tok, '(', ')', i + 1);
      if (close < 0) continue;
      const argTok = tok.slice(i + 2, close);
      if (argTok[0] && argTok[0].t === '{') {
        const end = matchBracket(argTok, '{', '}', 0);
        regs.push(...parseObjectPairs(argTok, 0, end));
      } else if (argTok[0] && argTok[0].t === 'word') {
        regs.push({ varName: argTok[0].v });
      }
    }
  }
  return regs;
}

// find `var NAME = { … }` / `const NAME = { … }` object literals
function findVarObject(tok, name) {
  for (let i = 0; i < tok.length - 3; i++) {
    if (tok[i].t === 'word' && tok[i].v === name && tok[i + 1].t === '=') {
      const obj = findObjectLiteralAfter(tok, i + 2);
      if (obj) return parseObjectPairs(tok, obj.start, obj.end);
    }
  }
  return null;
}

// locate `function NAME(` bodies (declaration or var-assign), return body ranges
function findFunctionBodies(tok, fnName) {
  const bodies = [];
  for (let i = 0; i < tok.length - 2; i++) {
    if (tok[i].t !== 'word' || tok[i].v !== fnName) continue;
    // function NAME ( … ) { … }
    if (tok[i - 1] && tok[i - 1].t === 'word' && tok[i - 1].v === 'function' && tok[i + 1].t === '(') {
      const closeParen = matchBracket(tok, '(', ')', i + 1);
      if (closeParen > 0 && tok[closeParen + 1] && tok[closeParen + 1].t === '{') {
        const end = matchBracket(tok, '{', '}', closeParen + 1);
        if (end > 0) bodies.push({ from: closeParen + 2, to: end });
      }
    }
    // NAME = function ( … ) { … }
    if (tok[i + 1] && tok[i + 1].t === '=' && tok[i + 2] && tok[i + 2].t === 'word' && tok[i + 2].v === 'function') {
      let j = i + 3;
      if (tok[j].t === 'word') j++; // named fn expr
      if (tok[j] && tok[j].t === '(') {
        const closeParen = matchBracket(tok, '(', ')', j);
        if (closeParen > 0 && tok[closeParen + 1] && tok[closeParen + 1].t === '{') {
          const end = matchBracket(tok, '{', '}', closeParen + 1);
          if (end > 0) bodies.push({ from: closeParen + 2, to: end });
        }
      }
    }
  }
  return bodies;
}

// extract the caption token list from a function body
const NON_CAPTION_FNS = new Set(['shell', 'el', 'svgEl', 'txt', 'grp', 'pop', 'enter', 'drawIn', 'anim',
  'slider', 'select', 'toggle', 'button', 'range', 'if', 'for', 'while', 'switch', 'function', 'return',
  'setTimeout', 'setInterval', 'parseFloat', 'parseInt', 'Number', 'String', 'Math']);

// object-literal argument → caption-ish property (prompt / caption / desc…)
function capFromObjectArg(argTok) {
  if (!argTok.length || argTok[0].t !== '{') return null;
  const end = matchBracket(argTok, '{', '}', 0);
  if (end < 0) return null;
  const pairs = parseObjectPairs(argTok, 0, end);
  const preferred = ['caption', 'cap', 'description', 'desc', 'prompt', 'instruction'];
  let best = null;
  for (const p of pairs) {
    if (!preferred.includes(p.key)) continue;
    const cap = classify(p.val.filter((x) => x.t !== ','));
    if (cap && cap.text !== undefined && (!best || cap.text.length > best.length)) best = cap.text;
  }
  if (best) return { text: best };
  // no obvious key: fall back to the longest string literal in the config
  for (const t of argTok) {
    if (t.t === 'str' && t.v.length > 60 && (!best || t.v.length > best.length)) best = t.v;
  }
  return best ? { text: best } : null;
}

function capFromBody(tok, body, fileTok) {
  const candidates = [];
  // 1. every shell(…) call — the last argument is the caption
  for (let i = body.from; i < body.to; i++) {
    if (tok[i].t === 'word' && tok[i + 1] && tok[i + 1].t === '(') {
      const close = matchBracket(tok, '(', ')', i + 1);
      if (close < 0 || close > body.to) continue;
      const args = splitArgs(tok, i + 2, close);
      if (!args.length) continue;
      const last = args[args.length - 1];
      if (tok[i].v === 'shell') {
        candidates.push(last);
        // some shells take a config object (shell(host, {caption…}, …))
        for (const a of args) {
          if (a.length && a[0].t === '{') candidates.push(a);
        }
      } else if (tok[i].v === 'cap' || tok[i].v === 'caption') {
        // micro-helper style: cap('text') or cap(host, title, sub, 'text')
        const first = args[0].filter((x) => x.t !== ',');
        if (first.length) candidates.push(first);
        candidates.push(args[args.length - 1]);
      } else if (!NON_CAPTION_FNS.has(tok[i].v)) {
        // delegated local helper call whose first arg is `host` — the caption
        // travels as an argument too; prefer its last argument
        const meaningful = args[0].filter((x) => x.t !== ',');
        if (meaningful.length === 1 && meaningful[0].t === 'word' && meaningful[0].v === 'host') {
          candidates.push(last);
        }
      }
    }
  }
  // resolve candidates: plain identifiers → their var assignment; object
  // literals → prompt/caption property; strings → text. The caption is by far
  // the longest candidate, so scan everything and keep the longest text.
  let bestText = null;
  for (const argTok of candidates) {
    const meaningful = argTok.filter((x) => x.t !== ',');
    if (meaningful.length === 1 && meaningful[0].t === 'word') {
      const resolved = resolveVar(tok, body, fileTok, meaningful[0].v);
      if (resolved && resolved.text !== undefined && (!bestText || resolved.text.length > bestText.length)) {
        bestText = resolved.text;
      }
      continue;
    }
    const fromObj = capFromObjectArg(argTok);
    if (fromObj) {
      if (!bestText || fromObj.text.length > bestText.length) bestText = fromObj.text;
      continue;
    }
    const cap = classify(argTok);
    if (cap && cap.text !== undefined && (!bestText || cap.text.length > bestText.length)) {
      bestText = cap.text;
    }
  }
  if (bestText) return { text: bestText };
  // fallback: inline lf-cap element or figcaption in this body
  for (let i = body.from; i < body.to; i++) {
    if (tok[i].t === 'str' && (tok[i].v === 'lf-cap' || tok[i].v === 'figcaption')) {
      for (let j = i; j < Math.min(i + 8, body.to); j++) {
        if (tok[j].t === '[') {
          const closeB = matchBracket(tok, '[', ']', j);
          if (closeB < 0) break;
          const first = splitArgs(tok, j + 1, closeB)[0];
          if (!first) break;
          const fromObj = capFromObjectArg(first);
          if (fromObj) return fromObj;
          const meaningful = first.filter((x) => x.t !== ',');
          if (meaningful.length === 1 && meaningful[0].t === 'word') {
            const resolved = resolveVar(tok, body, fileTok, meaningful[0].v);
            if (resolved && resolved.text !== undefined) return resolved;
            break;
          }
          return classify(first);
        }
      }
    }
  }
  return null;
}

// resolve `name = <expr>` / `var name = <expr>` inside body, then file scope
function resolveVar(tok, body, fileTok, name) {
  for (const scope of [body, { from: 0, to: fileTok.length }]) {
    for (let i = scope.from; i < scope.to - 2; i++) {
      if (tok[i].t === 'word' && tok[i].v === name && tok[i + 1].t === '=') {
        // expression until top-level ';'
        let depth = 0; const expr = [];
        for (let j = i + 2; j < scope.to; j++) {
          const t = tok[j];
          if ('([{'.includes(t.t)) depth++;
          else if (')]}'.includes(t.t)) depth--;
          if (t.t === ';' && !depth) break;
          expr.push(t);
        }
        const cap = classify(expr);
        if (cap && cap.text !== undefined) return cap;
      }
    }
  }
  return null;
}

// caption stored in a factory config object: makeFigure({ …, caption: '…' })
function capFromConfigCall(valTok, fileTok) {
  if (!valTok || !valTok.length) return null;
  if (valTok[0].t !== 'word' || !valTok[1] || valTok[1].t !== '(') return null;
  const close = matchBracket(valTok, '(', ')', 1);
  if (close < 0) return null;
  const args = splitArgs(valTok, 2, close);
  for (const arg of args) {
    const m = arg.filter((x) => x.t !== ',');
    if (!m.length || m[0].t !== '{') continue;
    const end = matchBracket(m, '{', '}', 0);
    if (end < 0) continue;
    const pairs = parseObjectPairs(m, 0, end);
    for (const key of ['caption', 'cap', 'description', 'desc']) {
      const hit = pairs.find((p) => p.key === key);
      if (hit) {
        const cap = classify(hit.val.filter((x) => x.t !== ','));
        if (cap) return cap;
      }
    }
    // no obvious key: fall back to the longest string literal in the config
    let best = null;
    for (const t of m) {
      if (t.t === 'str' && t.v.length > 60 && (!best || t.v.length > best.length)) best = t.v;
    }
    if (best) return { text: best };
  }
  return null;
}

// resolve a key's caption, following one level of delegation (fn → helper call)
function resolveCap(tok, file, fnName, seen) {
  seen = seen || new Set();
  if (seen.has(fnName)) return null;
  seen.add(fnName);
  const bodies = findFunctionBodies(tok, fnName);
  for (const body of bodies) {
    const cap = capFromBody(tok, body, tok);
    if (cap) return cap;
  }
  return null;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main() {
  fs.mkdirSync(path.join(OUT, '_src'), { recursive: true });
const files = fs.readdirSync(FIGDIR).filter((f) => f.endsWith('.js')).sort();
const caps = {}; // key → { file, line, text, raw? }
const missing = [];

for (const file of files) {
  const src = fs.readFileSync(path.join(FIGDIR, file), 'utf8');
  const tok = tokenize(src);
  const lineOf = (idx) => src.slice(0, idx).split('\n').length;

  // build key list for this file
  let pairs = [];
  for (const r of findRegistrations(tok)) {
    if (r.varName) pairs.push(...(findVarObject(tok, r.varName) || []));
    else pairs.push(r);
  }
  const figsVar = findVarObject(tok, 'FIGS');
  if (figsVar) pairs.push(...figsVar);

  if (!pairs.length) continue;

  const perFile = [];
  for (const p of pairs) {
    let cap = capFromConfigCall(p.val, tok);
    if (!cap && p.fn) cap = resolveCap(tok, file, p.fn);
    if (!cap) { missing.push(file + '\t' + p.key + '\t(no caption found, fn=' + (p.fn || '?') + ')'); continue; }
    const entry = { file };
    if (cap.text !== undefined) entry.text = cap.text;
    else { entry.raw = cap.raw; }
    // later provider files override earlier ones at runtime (sorted load
    // order), so the winning caption comes from the alphabetically-last file
    const prev = caps[p.key];
    const better = !prev || file > prev.file || (file === prev.file && prev.raw && cap.text !== undefined);
    if (better) {
      const bodies = p.fn ? findFunctionBodies(tok, p.fn) : [];
      entry.line = bodies.length ? lineOf(tok[bodies[0].from].i) : 0;
      caps[p.key] = entry;
    }
    perFile.push({ key: p.key, entry });
  }

  if (perFile.length) {
    const txt = perFile.map((x) => x.key + '\t' + JSON.stringify(x.entry.text !== undefined ? x.entry.text : { RAW: x.entry.raw })).join('\n');
    fs.writeFileSync(path.join(OUT, '_src', file.replace(/\.js$/, '') + '.txt'), txt + '\n', 'utf8');
  }
}

for (const [key, e] of Object.entries(caps)) {
  if (e.raw) missing.push('(flagged-dynamic)\t' + key + '\t' + e.file + '\t' + e.raw);
}
fs.writeFileSync(path.join(OUT, '_caps.json'), JSON.stringify(caps, null, 2), 'utf8');
fs.writeFileSync(path.join(OUT, '_missing.txt'), missing.join('\n') + (missing.length ? '\n' : ''), 'utf8');

const total = Object.keys(caps).length;
const literal = Object.values(caps).filter((e) => e.text !== undefined).length;
const dyn = total - literal;
console.log('keys: ' + total + '  literal: ' + literal + '  dynamic/flagged: ' + dyn + '  missing: ' + missing.filter((m) => !m.startsWith('(flagged')).length);
}

if (require.main === module) main();
module.exports = { tokenize, matchBracket, splitArgs, classify, findRegistrations, findVarObject, findFunctionBodies, capFromBody, parseObjectPairs, resolveCap };
