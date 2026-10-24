/**
 * layout.js — the page shell shared by every page: head with theme boot
 * (no dark-mode flash), fixed navbar (brand, search, lang switch, theme,
 * GitHub), sidebar, main, TOC, search dialog.
 */
'use strict';

var ICONS = require('./icons.js');

/** Cache-busting version for static asset URLs. build.js stamps one value
 *  per build so browsers re-fetch JS/CSS/JSON after every deploy. */
var assetVer = '';
function setAssetVer(v) { assetVer = String(v || ''); }
function vurl(url) { return url ? url + (assetVer ? '?v=' + assetVer : '') : url; }

/** Inline boot script — runs before CSS to apply the stored theme. */
var BOOT = '(function(){try{var s=localStorage.getItem("ae-theme");'
  + 'var d=s?s==="dark":!!(window.matchMedia&&matchMedia("(prefers-color-scheme: dark)").matches);'
  + 'if(d)document.documentElement.setAttribute("data-ds-dark-theme","");'
  + 'document.documentElement.style.colorScheme=d?"dark":"light";}catch(e){}})();';

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * @param {object} o
 *   lang ('en'|'zh'), base, t (strings), site {name, github, githubLabel},
 *   title, description, urlPath (this page, root-relative), urlEn, urlZh,
 *   sidebarHtml, mainHtml, tocHtml ('' to omit), pageData (object → JSON),
 *   needsMath, needsMermaid
 */
function render(o) {
  var lang = o.lang;
  var base = o.base;
  var t = o.t;
  var other = lang === 'en' ? 'zh' : 'en';

  var pageData = Object.assign({}, o.pageData, {
    lang: lang,
    base: base,
    ver: assetVer,
    cdn: { katexCss: o.needsMath ? o.cdn.katexCss : '', katexJs: o.needsMath ? o.cdn.katexJs : '', mermaid: o.needsMermaid ? o.cdn.mermaid : '' },
  });

  var alternates = '';
  if (o.urlEn && o.urlZh) {
    alternates = '<link rel="alternate" hreflang="en" href="' + esc(o.urlEn) + '">\n'
      + '<link rel="alternate" hreflang="zh" href="' + esc(o.urlZh) + '">\n'
      + '<link rel="alternate" hreflang="x-default" href="' + esc(base) + '">\n';
  }

  return '<!DOCTYPE html>\n<html lang="' + lang + '">\n<head>\n'
    + '<meta charset="utf-8">\n'
    + '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    + '<title>' + esc(o.title) + '</title>\n'
    + (o.description ? '<meta name="description" content="' + esc(o.description) + '">\n' : '')
    + (o.urlPath ? '<link rel="canonical" href="' + esc(o.urlPath) + '">\n' : '')
    + alternates
    + '<meta property="og:title" content="' + esc(o.title) + '">\n'
    + (o.description ? '<meta property="og:description" content="' + esc(o.description) + '">\n' : '')
    + '<meta property="og:type" content="website">\n'
    + '<link rel="icon" type="image/svg+xml" href="' + esc(vurl(base + 'assets/site/favicon.svg')) + '">\n'
    + '<script>' + BOOT + '</scr' + 'ipt>\n'
    + '<link rel="stylesheet" href="' + esc(vurl(base + 'assets/site/site.css')) + '">\n'
    + '<script>window.__AE_PAGE__=' + JSON.stringify(pageData) + ';</scr' + 'ipt>\n'
    + '<script defer src="' + esc(vurl(base + 'assets/site/theme.js')) + '"></scr' + 'ipt>\n'
    + '<script defer src="' + esc(vurl(base + 'assets/site/app.js')) + '"></scr' + 'ipt>\n'
    + '<script defer src="' + esc(vurl(base + 'assets/site/search.js')) + '"></scr' + 'ipt>\n'
    + '<script defer src="' + esc(vurl(base + 'assets/site/quiz.js')) + '"></scr' + 'ipt>\n'
    + '<script defer src="' + esc(vurl(base + 'assets/site/enhance.js')) + '"></scr' + 'ipt>\n'
    + (o.needsFigures
      ? '<script defer src="' + esc(vurl(base + 'assets/figures/lesson-figures.js')) + '"></scr' + 'ipt>\n'
        + '<script defer src="' + esc(vurl(base + 'assets/figures/figure-manifest.js')) + '"></scr' + 'ipt>\n'
        + (lang === 'zh'
          ? '<script defer src="' + esc(vurl(base + 'assets/figures/zh-figure-caps.js')) + '"></scr' + 'ipt>\n'
          : '')
      : '')
    + '</head>\n<body>\n'
    + '<a class="skip-link" href="#main">' + esc(t.skipToContent) + '</a>\n'

    // -- navbar -----------------------------------------------------------
    + '<header class="navbar">\n'
    + '<button type="button" class="icon-btn menu-btn" aria-label="' + esc(t.openNavigation)
    + '" aria-expanded="false" aria-controls="sidebar">' + ICONS.menu() + '</button>\n'
    + '<a class="navbar-brand" href="' + base + lang + '/">'
    + '<span class="brand-mark">' + ICONS.brand() + '</span>'
    + '<span>' + esc(o.site.brand || (lang === 'zh' ? o.site.nameZh : o.site.name)) + '</span></a>\n'
    + '<span class="navbar-spacer"></span>\n'
    + '<button type="button" class="navbar-search" data-search-open aria-label="' + esc(t.search) + '">'
    + ICONS.search() + '<span class="search-text">' + esc(t.searchShort) + '</span>'
    + '<kbd>Ctrl K</kbd></button>\n'
    + '<nav class="lang-switch" aria-label="Language">\n'
    + '<a href="' + esc(o.urlEn || (base + 'en/')) + '"' + (lang === 'en' ? ' aria-current="true"' : '') + '>EN</a>\n'
    + '<a href="' + esc(o.urlZh || (base + 'zh/')) + '"' + (lang === 'zh' ? ' aria-current="true"' : '') + '>中文</a>\n'
    + '</nav>\n'
    + '<button type="button" class="icon-btn" id="theme-toggle" aria-label="' + esc(t.toggleTheme) + '">'
    + '<span class="icon-moon">' + ICONS.moon() + '</span>'
    + '<span class="icon-sun">' + ICONS.sun() + '</span></button>\n'
    + '<a class="icon-btn" href="' + esc(o.site.github) + '" target="_blank" rel="noopener" aria-label="'
    + esc(t.githubRepo) + '" title="' + esc(o.site.githubLabel) + '">' + ICONS.github() + '</a>\n'
    + '</header>\n'

    // -- shell ------------------------------------------------------------
    + '<div class="shell">\n'
    + (o.sidebarHtml || '')
    + '<div class="sidebar-scrim"></div>\n'
    + '<main class="main" id="main">\n'
    + (o.mainHtml || '')
    + (o.site.motto
      ? '<section class="site-motto"><p class="motto-text">“' + esc(o.site.motto) + '”</p>'
        + (o.site.mottoBy ? '<p class="motto-by">— ' + esc(o.site.mottoBy) + '</p>' : '')
        + '</section>\n'
      : '')
    + '<footer class="footer">' + esc(t.footerText)
    + ' · ' + esc(t.footerLicense)
    + ' · ' + esc(t.authoredBy) + ' '
    + (o.site.coAuthor ? esc(o.site.coAuthor) + ' · ' : '')
    + '<a href="' + esc(o.site.github) + '" target="_blank" rel="noopener">' + esc(o.site.author) + '</a>'
    + '</footer>\n'
    + '</main>\n'
    + (o.tocHtml
      ? '<nav class="toc" aria-label="' + esc(t.onThisPage) + '">'
        + '<p class="toc-title">' + esc(t.onThisPage) + '</p>' + o.tocHtml + '</nav>'
      : '')
    + '</div>\n'

    // -- search dialog ------------------------------------------------------
    + '<dialog id="search-dialog" class="search-dialog">\n'
    + '<div class="search-head">' + ICONS.search(16)
    + '<input class="search-input" type="text" placeholder="' + esc(t.searchPlaceholder)
    + '" aria-label="' + esc(t.search) + '" autocomplete="off" spellcheck="false">'
    + '<span class="search-esc">esc</span>'
    + '<button type="button" class="search-close" data-search-close aria-label="'
    + (lang === 'zh' ? '关闭搜索' : 'Close search') + '">' + ICONS.x() + '</button></div>\n'
    + '<ul class="search-results" role="listbox" aria-label="' + esc(t.search) + '"></ul>\n'
    + '<div class="search-foot">'
    + '<span><kbd>↑↓</kbd>' + esc(t.searchNavigate) + '</span>'
    + '<span><kbd>↵</kbd>' + esc(t.searchOpen) + '</span>'
    + '<span><kbd>esc</kbd>' + esc(t.searchClose) + '</span>'
    + '</div>\n'
    + '</dialog>\n'
    + '</body>\n</html>\n';
}

module.exports = { render: render, setAssetVer: setAssetVer };
