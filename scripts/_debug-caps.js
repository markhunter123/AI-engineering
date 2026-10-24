'use strict';
const fs = require('fs');
const M = require('./extract-figure-caps.js');
const src = fs.readFileSync('vendor/figures/figures-multimodal2.js', 'utf8');
const tok = M.tokenize(src);
const bodies = M.findFunctionBodies(tok, 'patchNPack');
const body = bodies[0];
for (let i = Math.max(body.from, body.to - 250); i < body.to; i++) {
  if (tok[i].t === 'word' && tok[i + 1] && tok[i + 1].t === '(') {
    console.log('call:', tok[i].v, '@rel', i - body.from, '/', body.to - body.from);
  }
}
// dump the tail source
const tailStart = tok[body.to - 120] ? tok[body.to - 120].i : tok[body.from].i;
console.log('--- tail source ---');
console.log(src.slice(tailStart, tok[body.to].i + 1).slice(0, 1800));
