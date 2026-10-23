/**
 * phase.js — phase landing page: phase tag, title, hook, lesson rows.
 */
'use strict';

var ICONS = require('./icons.js');

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * @param {object} o {lang, base, t, phase {numLabel, title, hook, url},
 *   lessons [{n, title, url, type, minutes, hasQuiz}]}
 */
function renderMain(o) {
  var phase = o.phase;

  var html = '<header class="phase-hero">'
    + '<span class="phase-tag">' + esc(o.t.phaseLabel) + ' ' + esc(phase.numLabel) + '</span>'
    + '<h1>' + esc(phase.title) + '</h1>'
    + (phase.hook ? '<p class="phase-hook">' + esc(phase.hook) + '</p>' : '')
    + '</header>\n';

  html += '<section class="lesson-list" aria-label="' + esc(o.t.lessons) + '">\n';
  for (var i = 0; i < o.lessons.length; i++) {
    var lesson = o.lessons[i];
    html += '<a class="lesson-row" href="' + lesson.url + '">'
      + '<span class="lr-num">' + esc(lesson.n) + '</span>'
      + '<span class="lr-title">' + esc(lesson.title) + '</span>'
      + '<span class="lr-badges">';
    if (lesson.type) {
      html += '<span class="lr-badge type-' + esc(lesson.type.toLowerCase()) + '">' + esc(lesson.type) + '</span>';
    }
    if (lesson.minutes) {
      html += '<span class="lr-badge time">' + ICONS.clock(12) + esc(lesson.minutes) + '</span>';
    }
    if (lesson.hasQuiz) {
      html += '<span class="lr-badge">' + ICONS.helpCircle(12) + esc(o.t.quizTitle) + '</span>';
    }
    html += '</span></a>\n';
  }
  html += '</section>\n';
  return html;
}

module.exports = { renderMain: renderMain };
