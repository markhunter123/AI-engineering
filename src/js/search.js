/**
 * search.js — Ctrl/Cmd+K command-palette search over the per-language index
 * at {base}assets/search/{lang}.json. Entries: {u, t, s, k, p, y}.
 * Scored client-side; keyboard navigable; ARIA combobox roles.
 */
(function () {
  'use strict';

  var page = window.__AE_PAGE__ || {};
  var base = page.base || '/';
  var t = page.t || {};
  var index = null;
  var indexPromise = null;

  function loadIndex() {
    if (index) return Promise.resolve(index);
    if (!indexPromise) {
      var ver = page.ver ? '?v=' + page.ver : '';
      indexPromise = fetch(base + 'assets/search/' + (page.lang || 'en') + '.json' + ver)
        .then(function (r) { return r.json(); })
        .then(function (json) { index = json; return json; })
        .catch(function () { index = []; return []; });
    }
    return indexPromise;
  }

  function score(entry, tokens) {
    var total = 0;
    var title = entry.t.toLowerCase();
    var words = title.split(/[\s\-/]+/);
    for (var i = 0; i < tokens.length; i++) {
      var tok = tokens[i];
      if (!tok) continue;
      var s = 0;
      if (title.indexOf(tok) === 0) s = 12;
      else if (title.indexOf(tok) > 0) s = 10;
      if (!s) {
        for (var w = 0; w < words.length; w++) {
          if (words[w].indexOf(tok) === 0) { s = 8; break; }
        }
      }
      if (!s && entry.k && entry.k.toLowerCase().indexOf(tok) >= 0) s = 4;
      if (!s && entry.s && entry.s.toLowerCase().indexOf(tok) >= 0) s = 2;
      if (!s && entry.b) {
        var body = entry.b.toLowerCase();
        var occ = 0;
        var pos = body.indexOf(tok);
        while (pos >= 0) {
          occ++;
          if (occ >= 5) break;
          pos = body.indexOf(tok, pos + tok.length);
        }
        if (occ) s = occ >= 4 ? 3 : occ >= 2 ? 2 : 1;
      }
      if (!s && entry.p && entry.p.toLowerCase().indexOf(tok) >= 0) s = 1;
      if (!s) return 0;
      total += s;
    }
    return total;
  }

  function search(query) {
    if (!index || !query) return [];
    var tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    var hits = [];
    for (var i = 0; i < index.length; i++) {
      var sc = score(index[i], tokens);
      if (sc > 0) hits.push([sc, index[i]]);
    }
    hits.sort(function (a, b) { return b[0] - a[0]; });
    return hits.slice(0, 20).map(function (h) { return h[1]; });
  }

  function highlight(title, query) {
    var tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    var html = '';
    var lower = title.toLowerCase();
    var i = 0;
    while (i < title.length) {
      var matched = null;
      for (var k = 0; k < tokens.length; k++) {
        if (tokens[k] && lower.indexOf(tokens[k], i) === i) { matched = tokens[k]; break; }
      }
      if (matched) {
        html += '<mark>' + esc(title.slice(i, i + matched.length)) + '</mark>';
        i += matched.length;
      } else {
        html += esc(title[i]);
        i++;
      }
    }
    return html;
  }

  function snippet(entry, tokens) {
    if (!entry.b) return '';
    var lower = entry.b.toLowerCase();
    var idx = -1;
    for (var k = 0; k < tokens.length && idx < 0; k++) {
      if (tokens[k]) idx = lower.indexOf(tokens[k]);
    }
    if (idx < 0) return '';
    var start = Math.max(0, idx - 50);
    var end = Math.min(entry.b.length, idx + 90);
    return (start > 0 ? '…' : '')
      + entry.b.slice(start, end).trim()
      + (end < entry.b.length ? '…' : '');
  }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function setup(dialog) {
    var input = dialog.querySelector('.search-input');
    var list = dialog.querySelector('.search-results');
    var prevQuery = '';
    var current = -1;
    var rows = [];

    function render(results, query) {
      list.innerHTML = '';
      current = -1;
      rows = [];
      var tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
      if (!results.length) {
        var empty = document.createElement('li');
        empty.className = 'search-empty';
        empty.textContent = t.searchEmpty || 'No results.';
        list.appendChild(empty);
        return;
      }
      results.forEach(function (entry) {
        var li = document.createElement('li');
        var item = document.createElement('a');
        item.className = 'search-item';
        item.href = entry.u;
        item.setAttribute('role', 'option');
        var title = document.createElement('span');
        title.className = 's-title';
        title.innerHTML = highlight(entry.t, query);
        var meta = document.createElement('span');
        meta.className = 's-meta';
        var phase = document.createElement('span');
        phase.className = 's-phase';
        phase.textContent = entry.p + (entry.y ? ' · ' + entry.y : '');
        meta.appendChild(phase);
        item.appendChild(title);
        var snip = snippet(entry, tokens);
        if (snip) {
          var sn = document.createElement('span');
          sn.className = 's-snippet';
          sn.innerHTML = highlight(snip, query);
          item.appendChild(sn);
        }
        item.appendChild(meta);
        li.appendChild(item);
        list.appendChild(li);
        rows.push(item);
      });
    }

    function move(delta) {
      if (!rows.length) return;
      current = (current + delta + rows.length) % rows.length;
      rows.forEach(function (r, i) { r.classList.toggle('selected', i === current); });
      rows[current].scrollIntoView({ block: 'nearest' });
      input.setAttribute('aria-activedescendant', '');
    }

    function showLoading() {
      list.innerHTML = '';
      current = -1;
      rows = [];
      var loading = document.createElement('li');
      loading.className = 'search-empty';
      loading.textContent = t.searchLoading
        || (page.lang === 'zh' ? '正在加载搜索索引…' : 'Loading search index…');
      list.appendChild(loading);
    }

    function onInput() {
      var q = input.value.trim();
      if (q === prevQuery) return;
      prevQuery = q;
      if (!index) {
        showLoading();
        loadIndex().then(function () {
          if (input.value.trim() === q) render(search(q), q);
        });
        return;
      }
      render(search(q), q);
    }

    var debounce = null;
    input.addEventListener('input', function () {
      if (debounce) clearTimeout(debounce);
      debounce = setTimeout(onInput, 100);
    });
    dialog.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') {
        var target = rows[current] || list.querySelector('.search-item');
        if (target) { e.preventDefault(); window.location.href = target.href; }
      }
    });

    function open() {
      dialog.showModal();
      input.value = '';
      prevQuery = '';
      render([], '');
      input.focus();
      loadIndex();
    }

    document.querySelectorAll('[data-search-open]').forEach(function (el) {
      el.addEventListener('click', open);
    });
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        if (dialog.open) dialog.close();
        else open();
      } else if (e.key === '/' && !dialog.open) {
        var tag = (document.activeElement && document.activeElement.tagName) || '';
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && !document.activeElement.isContentEditable) {
          e.preventDefault();
          open();
        }
      }
    });
  }

  function ready() {
    var dialog = document.getElementById('search-dialog');
    if (dialog) setup(dialog);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
