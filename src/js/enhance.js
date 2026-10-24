/**
 * enhance.js — progressive enhancement: KaTeX for [data-math] elements and
 * mermaid for .mermaid-source blocks, both loaded from CDN only when the
 * page needs them. On any failure the build-time fallback (styled TeX text,
 * mermaid source) stays visible. Re-themes mermaid on 'ae-theme-change'.
 * CDN endpoints arrive via window.__AE_PAGE__.cdn.
 */
(function () {
  'use strict';

  var page = window.__AE_PAGE__ || {};
  var cdn = page.cdn || {};
  var katexLoading = null;
  var mermaidLoading = null;

  function loadCss(href) {
    if (document.querySelector('link[href="' + href + '"]')) return Promise.resolve();
    // Resolve on load (or error — rendering without CSS beats not rendering);
    // callers chain .then, so a plain return here would break the pipeline.
    return new Promise(function (resolve) {
      var link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      link.onload = resolve;
      link.onerror = resolve;
      document.head.appendChild(link);
    });
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var existing = document.querySelector('script[data-enhance="' + src + '"]');
      if (existing) {
        existing.addEventListener('load', function () { resolve(); });
        existing.addEventListener('error', reject);
        if (window.katex) resolve();
        return;
      }
      var s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.dataset.enhance = src;
      s.addEventListener('load', function () { resolve(); });
      s.addEventListener('error', reject);
      document.head.appendChild(s);
    });
  }

  /* -- KaTeX --------------------------------------------------------------- */
  /* Upstream lessons occasionally write bare "_" inside \text{...}
   * (e.g. \text{out_degree}); LaTeX text mode requires \_. KaTeX refuses
   * the bare form, so escape underscores inside text-mode groups before
   * rendering. Skips already-escaped \_ and honors nested braces. */
  function escapeTextUnderscores(tex) {
    var out = '';
    var i = 0;
    var textDepth = 0; // brace depth inside an open \text{...} argument
    while (i < tex.length) {
      var ch = tex.charAt(i);
      if (textDepth > 0) {
        if (ch === '{') textDepth++;
        else if (ch === '}') textDepth--;
        else if (ch === '_' && tex.charAt(i - 1) !== '\\') ch = '\\_';
        out += ch;
        i++;
        continue;
      }
      if (ch === '\\') {
        var brace = tex.indexOf('{', i);
        var cmd = brace > i ? tex.slice(i, brace).replace(/\s+$/, '') : '';
        if (/^\\text[a-z]*$/.test(cmd)) {
          out += cmd + '{';
          textDepth = 1;
          i = brace + 1;
          continue;
        }
      }
      out += ch;
      i++;
    }
    return out;
  }

  function renderMath() {
    var nodes = document.querySelectorAll('[data-math]');
    if (!nodes.length || !cdn.katexJs) return;

    katexLoading = katexLoading || loadCss(cdn.katexCss)
      .then(function () { return loadScript(cdn.katexJs); })
      .then(function () {
        if (!window.katex) throw new Error('katex missing');
      })
      .catch(function () { /* fail soft: styled TeX stays */ });

    katexLoading.then(function () {
      Array.prototype.forEach.call(nodes, function (node) {
        var tex = escapeTextUnderscores(node.textContent);
        if (!tex) return;
        var isDisplay = node.getAttribute('data-math') === 'display';
        try {
          window.katex.render(tex, node, {
            displayMode: isDisplay,
            throwOnError: false,
            strict: false,
            output: 'html',
          });
        } catch (e) { /* keep raw TeX */ }
      });
    });
  }

  /* -- mermaid --------------------------------------------------------------- */
  function currentMermaidTheme() {
    return document.documentElement.hasAttribute('data-ds-dark-theme') ? 'dark' : 'default';
  }

  function renderMermaid() {
    var blocks = document.querySelectorAll('.mermaid-block');
    if (!blocks.length || !cdn.mermaid) return Promise.resolve();

    mermaidLoading = mermaidLoading || import(cdn.mermaid)
      .then(function (mod) {
        var mermaid = mod.default || mod;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: currentMermaidTheme(),
          fontFamily: 'inherit',
        });
        return mermaid;
      })
      .catch(function () { return null; });

    return mermaidLoading.then(function (mermaid) {
      if (!mermaid) return;
      var jobs = [];
      Array.prototype.forEach.call(blocks, function (block) {
        var source = block.querySelector('.mermaid-source');
        var target = block.querySelector('.mermaid-render');
        if (!source || !target || !source.textContent.trim()) return;
        jobs.push(
          Promise.resolve()
            .then(function () { return mermaid.render('mmd-' + Math.random().toString(36).slice(2), source.textContent); })
            .then(function (result) {
              target.innerHTML = result.svg;
              block.classList.add('rendered');
            })
            .catch(function () { /* keep source visible */ })
        );
      });
      return Promise.all(jobs);
    });
  }

  function rethemeMermaid() {
    if (!window.mermaidQueue) return;
    // Simplest robust path: re-render all diagrams with the new theme.
    renderMermaid();
  }

  /* -- lesson figures -------------------------------------------------------- */
  /* The vendored runtime (lesson-figures.js + figure-manifest.js) loads via
   * defer on pages that contain figure fences. It injects each figure's
   * provider on demand; mounts that succeed set data-lf-mounted, so the
   * static placeholder card inside can go. Unknown keys keep the card. */
  function mountFigures(attempt) {
    var hosts = document.querySelectorAll('.lesson-figure[data-figure]');
    if (!hosts.length) return;
    // The figure runtime is a later defer script; on a slow hop (proxy, weak
    // network) this idle callback can fire before it has executed. Retry
    // instead of giving up — a silent miss leaves the placeholder up forever.
    if (typeof window.AIFS_loadFigureProviders !== 'function'
      || typeof window.mountLessonFigures !== 'function') {
      if ((attempt || 0) < 50) setTimeout(function () { mountFigures((attempt || 0) + 1); }, 120);
      return;
    }
    // Most figures live in provider modules (figures-<topic>.js) that must be
    // loaded per the manifest before mountLessonFigures can find them.
    window.AIFS_loadFigureProviders(document).then(function () {
      try { window.mountLessonFigures(document); } catch (e) { /* placeholders stay */ }
      Array.prototype.forEach.call(
        document.querySelectorAll('.lesson-figure[data-lf-mounted] > .figure-placeholder'),
        function (node) { node.remove(); },
      );
    });
  }

  function idle(fn) {
    if ('requestIdleCallback' in window) requestIdleCallback(fn, { timeout: 1800 });
    else setTimeout(fn, 250);
  }

  function ready() {
    idle(function () {
      renderMath();
      renderMermaid();
      mountFigures();
    });
    document.addEventListener('ae-theme-change', rethemeMermaid);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
