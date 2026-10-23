/**
 * lesson.js — lesson page main column: breadcrumb, title header, meta chips,
 * GitHub action, rendered markdown body, quiz mount, prev/next pager, TOC.
 */
'use strict';

var ICONS = require('./icons.js');

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * @param {object} o {lang, base, t, lesson {title, type, languages, minutes,
 *   prerequisites, hook}, phase {title, url}, mdHtml, toc [{id,text,level}],
 *   prev {title, url}|null, next {title,url}|null, quizJson (string|null),
 *   githubUrl, quizNote}
 */
function renderMain(o) {
  var t = o.t;
  var base = o.base;
  var lang = o.lang;
  var home = base + lang + '/';
  var lesson = o.lesson;

  var html = '<nav class="breadcrumb" aria-label="Breadcrumb">'
    + '<a href="' + home + '">' + esc(t.home || 'Home') + '</a>'
    + '<span class="sep">/</span>'
    + '<a href="' + o.phase.url + '">' + esc(o.phase.title) + '</a>'
    + '<span class="sep">/</span>'
    + '<span class="here">' + esc(lesson.title) + '</span>'
    + '</nav>\n';

  html += '<header class="lesson-header">';
  html += '<h1 class="lesson-title">' + esc(lesson.title) + '</h1>\n';
  html += '<div class="lesson-meta">';
  if (lesson.type) {
    html += '<span class="meta-chip">' + ICONS.layers() + '<b>' + esc(lesson.type) + '</b></span>';
  }
  if (lesson.minutes) {
    html += '<span class="meta-chip">' + ICONS.clock() + esc(lesson.minutes) + '</span>';
  }
  if (lesson.languages) {
    html += '<span class="meta-chip">' + ICONS.code() + esc(lesson.languages) + '</span>';
  }
  if (lesson.prerequisites && lesson.prerequisites.toLowerCase() !== 'none') {
    html += '<span class="meta-chip">' + ICONS.listTree() + esc(t.prerequisites) + ': '
      + esc(lesson.prerequisites) + '</span>';
  }
  html += '</div>\n';
  html += '</header>\n';

  html += '<article class="content">\n' + o.mdHtml + '\n</article>\n';

  if (o.quizJson) {
    html += '<section id="quiz" class="quiz" hidden>\n'
      + '<div class="quiz-head">'
      + '<span class="quiz-title">' + ICONS.helpCircle() + esc(t.quizTitle) + '</span>'
      + (o.quizNote ? '<span class="quiz-note">' + esc(o.quizNote) + '</span>' : '')
      + '</div>\n'
      + '<div class="quiz-tabs" role="tablist"></div>\n'
      + '<div class="quiz-body"></div>\n'
      + '</section>\n'
      + '<script type="application/json" id="quiz-data">' + o.quizJson + '</scr' + 'ipt>\n';
  }

  html += '<nav class="pager" aria-label="' + esc(t.pagerLabel) + '">';
  if (o.prev) {
    html += '<a class="pager-card prev" href="' + o.prev.url + '">'
      + '<span class="pager-label">' + ICONS.arrowLeft() + esc(t.prev) + '</span>'
      + '<span class="pager-title">' + esc(o.prev.title) + '</span></a>';
  } else {
    html += '<span class="pager-card pager-empty" aria-hidden="true"></span>';
  }
  if (o.next) {
    html += '<a class="pager-card next" href="' + o.next.url + '">'
      + '<span class="pager-label">' + esc(t.next) + ICONS.arrowRight() + '</span>'
      + '<span class="pager-title">' + esc(o.next.title) + '</span></a>';
  } else {
    html += '<span class="pager-card pager-empty" aria-hidden="true"></span>';
  }
  html += '</nav>\n';

  return html;
}

/** Right-hand outline from [{id, text, level}] (h2/h3 only). */
function renderToc(toc) {
  if (!toc || !toc.length) return '';
  var html = '<ul class="toc-list">\n';
  for (var i = 0; i < toc.length; i++) {
    var item = toc[i];
    html += '<li class="' + (item.level === 3 ? 'toc-h3' : 'toc-h2') + '">'
      + '<a href="#' + esc(item.id) + '">' + esc(item.text) + '</a></li>\n';
  }
  html += '</ul>';
  return html;
}

module.exports = { renderMain: renderMain, renderToc: renderToc };
