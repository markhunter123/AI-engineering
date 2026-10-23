/**
 * sidebar.js — left navigation: 20 phase groups. The current phase renders
 * expanded with its lessons inline; other groups expand client-side from
 * assets/tree/{lang}.json (app.js). Group titles link to the phase landing
 * page for no-JS users.
 */
'use strict';

var ICONS = require('./icons.js');

/**
 * @param {Array} phases [{slug, title, count}]
 * @param {string} lang
 * @param {string} base
 * @param {string} currentPhaseSlug slug of the active phase
 * @param {Array} currentLessons [{n, t, u, current}] lessons of active phase
 * @param {object} t localized labels {openNav, closeNav, lessons}
 */
function render(opts) {
  var phases = opts.phases;
  var lang = opts.lang;
  var base = opts.base;
  var t = opts.t;

  var html = '<aside class="sidebar" id="sidebar" aria-label="'
    + esc(t.openNavigation) + '">\n';

  // Learning map — sits above phase 00 as the entry point for role-based routes.
  var isPaths = opts.currentSpecial === 'paths';
  html += '<div class="nav-group nav-special' + (isPaths ? ' current' : '') + '">\n';
  html += '<a class="nav-group-head nav-map-link" href="' + base + lang + '/learning-paths/">\n';
  html += '<span class="nav-phase-num">' + ICONS.map() + '</span>\n';
  html += '<span class="nav-group-title">' + esc(t.learningMap) + '</span>\n';
  if (opts.pathsCount) html += '<span class="nav-group-count">' + opts.pathsCount + '</span>\n';
  html += '</a>\n</div>\n';

  for (var i = 0; i < phases.length; i++) {
    var phase = phases[i];
    var isCurrent = phase.slug === opts.currentPhaseSlug;
    var count = phase.count;
    html += '<div class="nav-group' + (isCurrent ? ' current open' : '') + '">\n';
    html += '<button type="button" class="nav-group-head" data-phase="' + esc(phase.slug)
      + '" aria-expanded="' + (isCurrent ? 'true' : 'false') + '">\n';
    html += '<span class="nav-phase-num">' + esc(phase.numLabel) + '</span>\n';
    html += '<span class="nav-group-title"><a href="' + base + lang + '/phases/'
      + esc(phase.slug) + '/" tabindex="-1">' + esc(phase.title) + '</a></span>\n';
    html += '<span class="nav-group-count">' + count + '</span>\n';
    html += '<span class="nav-chevron">' + ICONS.chevronRight() + '</span>\n';
    html += '</button>\n';

    if (isCurrent && opts.currentLessons && opts.currentLessons.length) {
      html += '<ul class="nav-lessons">\n';
      for (var j = 0; j < opts.currentLessons.length; j++) {
        var lesson = opts.currentLessons[j];
        html += '<li><a href="' + lesson.u + '"'
          + (lesson.current ? ' aria-current="page"' : '') + '>'
          + '<span class="lesson-num">' + esc(lesson.n) + '</span>'
          + '<span>' + esc(lesson.t) + '</span></a></li>\n';
      }
      html += '</ul>\n';
    }
    html += '</div>\n';
  }

  html += '</aside>';
  return html;
}

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

module.exports = { render: render };
