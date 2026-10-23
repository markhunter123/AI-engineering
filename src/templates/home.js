/**
 * home.js — homepage: hero, stats, phase grid, how-to card.
 */
'use strict';

var ICONS = require('./icons.js');

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * @param {object} o {lang, base, t, site, hero {kicker, title, sub,
 *   ctaStart, ctaBrowse, startUrl}, stats [{num, label}],
 *   phases [{numLabel, title, count, url}], how {title, steps[]}}
 */
function renderMain(o) {
  var html = '<section class="home-hero">'
    + '<span class="hero-kicker">' + esc(o.hero.kicker) + '</span>'
    + '<h1>' + esc(o.hero.title) + '</h1>'
    + '<p class="hero-sub">' + esc(o.hero.sub) + '</p>'
    + '<div class="hero-cta">'
    + '<a class="btn btn-primary" href="' + o.hero.startUrl + '">' + esc(o.hero.ctaStart) + ICONS.arrowRight() + '</a>'
    + '<a class="btn" href="' + o.hero.browseUrl + '">' + ICONS.layers(15) + esc(o.hero.ctaBrowse) + '</a>'
    + '</div>\n'
    + '</section>\n';

  html += '<section class="home-stats" aria-label="Statistics">';
  for (var i = 0; i < o.stats.length; i++) {
    html += '<div class="stat-card"><div class="stat-num">' + esc(o.stats[i].num) + '</div>'
      + '<div class="stat-label">' + esc(o.stats[i].label) + '</div></div>';
  }
  html += '</section>\n';

  html += '<div class="section-head" id="phases"><h2>' + esc(o.t.allPhases) + '</h2></div>\n';
  html += '<section class="phase-grid">\n';
  for (var p = 0; p < o.phases.length; p++) {
    var phase = o.phases[p];
    html += '<a class="phase-card" href="' + phase.url + '">'
      + '<span class="pc-top"><span class="pc-num">' + esc(phase.numLabel) + '</span>'
      + '<span class="pc-title">' + esc(phase.title) + '</span></span>'
      + '<span class="pc-meta">' + phase.count + ' ' + esc(o.t.lessons) + '</span>'
      + '</a>\n';
  }
  html += '</section>\n';

  html += '<section class="home-how"><h2>' + esc(o.how.title) + '</h2><ol>';
  for (var s = 0; s < o.how.steps.length; s++) {
    html += '<li>' + esc(o.how.steps[s]) + '</li>';
  }
  html += '</ol></section>\n';

  return html;
}

module.exports = { renderMain: renderMain };
