/**
 * app.js — page chrome: mobile drawer, lazy sidebar expansion, code copy
 * buttons, TOC scrollspy. Reads window.__AE_PAGE__ = {lang, base, t:{...}}.
 */
(function () {
  'use strict';

  var page = window.__AE_PAGE__ || {};
  var base = page.base || '/';
  var t = page.t || {};
  var treeCache = null;

  /* -- mobile drawer ------------------------------------------------------ */
  function setupDrawer() {
    var sidebar = document.querySelector('.sidebar');
    var scrim = document.querySelector('.sidebar-scrim');
    var btn = document.querySelector('.menu-btn');
    if (!sidebar || !scrim || !btn) return;

    function setOpen(open) {
      sidebar.classList.toggle('open', open);
      scrim.classList.toggle('show', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      document.body.style.overflow = open ? 'hidden' : '';
    }

    btn.addEventListener('click', function () {
      setOpen(!sidebar.classList.contains('open'));
    });
    scrim.addEventListener('click', function () { setOpen(false); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') setOpen(false);
    });
    sidebar.addEventListener('click', function (e) {
      // Close after navigating from the drawer on small screens.
      if (e.target.closest('a') && window.innerWidth < 960) setOpen(false);
    });
  }

  /* -- sidebar: lazy expansion from tree/{lang}.json ---------------------- */
  function lessonItem(entry) {
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = entry.u;
    var num = document.createElement('span');
    num.className = 'lesson-num';
    num.textContent = entry.n;
    var label = document.createElement('span');
    label.textContent = entry.t;
    a.appendChild(num);
    a.appendChild(label);
    if (entry.current) a.setAttribute('aria-current', 'page');
    li.appendChild(a);
    return li;
  }

  function loadTree() {
    if (treeCache) return Promise.resolve(treeCache);
    var ver = page.ver ? '?v=' + page.ver : '';
    return fetch(base + 'assets/tree/' + (page.lang || 'en') + '.json' + ver)
      .then(function (r) {
        if (!r.ok) throw new Error('tree ' + r.status);
        return r.json();
      })
      .then(function (json) {
        treeCache = json;
        return json;
      })
      .catch(function () {
        treeCache = {};
        return treeCache;
      });
  }

  function setupSidebar() {
    var sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    sidebar.addEventListener('click', function (e) {
      var head = e.target.closest('.nav-group-head');
      if (!head) return;
      // Anchor heads (the learning-map entry) navigate normally; only
      // button heads toggle their group.
      if (head.tagName !== 'BUTTON') return;
      // The head embeds a plain phase link for no-JS users; with JS the
      // whole head toggles instead of navigating.
      if (e.target.closest('a')) e.preventDefault();
      var group = head.parentElement;
      var alreadyOpen = group.classList.contains('open');

      if (group.querySelector('.nav-lessons') || alreadyOpen) {
        group.classList.toggle('open', !alreadyOpen);
        head.setAttribute('aria-expanded', !alreadyOpen ? 'true' : 'false');
        return;
      }

      // First expansion: fetch the lesson tree, then open.
      head.setAttribute('aria-expanded', 'true');
      var status = document.createElement('div');
      status.className = 'nav-loading';
      status.textContent = t.loading || '…';
      group.appendChild(status);
      loadTree().then(function (tree) {
        status.remove();
        var lessons = tree[head.getAttribute('data-phase')] || [];
        var ul = document.createElement('ul');
        ul.className = 'nav-lessons';
        lessons.forEach(function (entry) { ul.appendChild(lessonItem(entry)); });
        group.appendChild(ul);
        group.classList.add('open');
      });
    });
  }

  /* -- code copy buttons ---------------------------------------------------- */
  function setupCopy() {
    document.addEventListener('click', function (e) {
      var btn = e.target.closest('.code-copy');
      if (!btn) return;
      var text = btn.getAttribute('data-code') || '';
      function done() {
        var original = btn.textContent;
        btn.textContent = t.copied || 'Copied';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1400);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (err) { /* ignore */ }
        ta.remove();
      }
    });
  }

  /* -- TOC scrollspy ----------------------------------------------------------- */
  function setupSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.toc-list a[href^="#"]'));
    if (!links.length || !('IntersectionObserver' in window)) return;
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var visible = {};

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible[entry.target.id] = true;
        else delete visible[entry.target.id];
      });
      // Highlight the first heading visible nearest the top of the viewport.
      var active = null;
      for (var i = 0; i < links.length; i++) {
        var id = links[i].getAttribute('href').slice(1);
        if (visible[id] && !active) { active = id; break; }
      }
      if (!active) return;
      links.forEach(function (a) { a.classList.remove('active'); });
      byId[active].classList.add('active');
    }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });

    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) io.observe(el);
    });
  }

  function ready() {
    setupDrawer();
    setupSidebar();
    setupCopy();
    setupSpy();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
