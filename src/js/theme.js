/**
 * theme.js — dark/light toggle. The boot script in <head> already applied the
 * initial theme (no flash); this only handles the button and persistence.
 * Dispatches 'ae-theme-change' so enhance.js can re-theme mermaid diagrams.
 */
(function () {
  'use strict';

  var KEY = 'ae-theme';
  var root = document.documentElement;

  function isDark() {
    return root.hasAttribute('data-ds-dark-theme');
  }

  function apply(dark) {
    if (dark) root.setAttribute('data-ds-dark-theme', '');
    else root.removeAttribute('data-ds-dark-theme');
    root.style.colorScheme = dark ? 'dark' : 'light';
  }

  function persist(dark) {
    try {
      localStorage.setItem(KEY, dark ? 'dark' : 'light');
    } catch (e) { /* private mode */ }
  }

  function toggle() {
    var dark = !isDark();
    apply(dark);
    persist(dark);
    document.dispatchEvent(new CustomEvent('ae-theme-change', { detail: { dark: dark } }));
  }

  function ready() {
    var btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', toggle);

    // Follow the OS while the visitor has not expressed a preference.
    var stored = null;
    try { stored = localStorage.getItem(KEY); } catch (e) { /* ignore */ }
    if (!stored && window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onSystem = function (e) {
        apply(e.matches);
        document.dispatchEvent(new CustomEvent('ae-theme-change', { detail: { dark: e.matches } }));
      };
      if (mq.addEventListener) mq.addEventListener('change', onSystem);
      else if (mq.addListener) mq.addListener(onSystem);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
