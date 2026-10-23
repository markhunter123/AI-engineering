/**
 * quiz.js — interactive lesson quiz. Reads inline
 * <script type="application/json" id="quiz-data"> with
 * {questions: [{stage, question, options, correct, explanation}]}.
 * Stages render as tabs (pre → check → post); answers reveal the
 * explanation; per-stage score + retry. UI labels come from
 * window.__AE_PAGE__.t.quiz.
 */
(function () {
  'use strict';

  var page = window.__AE_PAGE__ || {};
  var T = (page.t && page.t.quiz) || {};
  var STAGE_ORDER = ['pre', 'check', 'post'];

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function renderQuestion(parent, q, stageLabel, state) {
    var wrap = el('div', 'quiz-q');
    var qText = el('p', 'q-text');
    var badge = el('span', 'q-stage-badge', stageLabel);
    qText.appendChild(badge);
    qText.appendChild(document.createTextNode(q.question));
    wrap.appendChild(qText);

    var opts = el('ul', 'quiz-opts');
    var explained = el('div', 'quiz-explain');
    q.options.forEach(function (opt, i) {
      var li = el('li');
      var btn = el('button', 'quiz-opt');
      btn.type = 'button';
      var key = el('span', 'opt-key', String.fromCharCode(65 + i));
      btn.appendChild(key);
      btn.appendChild(document.createTextNode(opt));
      btn.addEventListener('click', function () {
        Array.prototype.forEach.call(opts.querySelectorAll('.quiz-opt'), function (b, j) {
          b.disabled = true;
          if (j === q.correct) b.classList.add('correct');
          if (j === i && i !== q.correct) b.classList.add('wrong');
        });
        explained.textContent = q.explanation || '';
        state.answered += 1;
        if (i === q.correct) state.correct += 1;
        updateScore();
      });
      li.appendChild(btn);
      opts.appendChild(li);
    });
    wrap.appendChild(opts);
    wrap.appendChild(explained);
    parent.appendChild(wrap);
  }

  function updateScore() {
    /* per-stage score line is refreshed by each stage's closure */
    if (updateScore.refreshers) updateScore.refreshers.forEach(function (fn) { fn(); });
  }

  function setup(root, data) {
    var stages = {};
    data.questions.forEach(function (q, qi) {
      var stage = STAGE_ORDER.indexOf(q.stage) >= 0 ? q.stage : 'check';
      if (!stages[stage]) stages[stage] = [];
      q.__i = qi;
      stages[stage].push(q);
    });
    var present = STAGE_ORDER.filter(function (s) { return stages[s] && stages[s].length; });
    if (!present.length) return;

    var tabs = root.querySelector('.quiz-tabs');
    var body = root.querySelector('.quiz-body');

    var panels = {};
    present.forEach(function (stage, idx) {
      var tab = el('button', 'quiz-tab');
      tab.type = 'button';
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
      tab.textContent = (T.stages && T.stages[stage]) || stage;
      tab.addEventListener('click', function () { select(stage); });
      tabs.appendChild(tab);
      panels[stage] = tab;
    });

    function select(stage) {
      present.forEach(function (s) {
        var tab = panels[s];
        var panel = body.querySelector('[data-stage-panel="' + s + '"]');
        var on = s === stage;
        tab.setAttribute('aria-selected', on ? 'true' : 'false');
        if (panel) panel.hidden = !on;
      });
    }

    body.innerHTML = '';
    updateScore.refreshers = [];
    present.forEach(function (stage, idx) {
      var panel = el('div');
      panel.setAttribute('data-stage-panel', stage);
      panel.hidden = idx !== 0;

      var state = { answered: 0, correct: 0 };
      stages[stage].forEach(function (q) {
        renderQuestion(panel, q, (T.stages && T.stages[stage]) || stage, state);
      });

      var scoreWrap = el('div', 'quiz-score');
      scoreWrap.hidden = true;
      var line = el('span', 'score-line');
      var retry = el('button', 'quiz-retry', T.retry || 'Retry');
      retry.type = 'button';
      scoreWrap.appendChild(line);
      scoreWrap.appendChild(retry);
      panel.appendChild(scoreWrap);

      function refresh() {
        if (state.answered === stages[stage].length) {
          scoreWrap.hidden = false;
          var ratio = state.correct / stages[stage].length;
          var good = ratio === 1;
          line.classList.toggle('good', good);
          line.innerHTML = '';
          line.appendChild(el('span', null, (T.score || 'Score') + ': '));
          var b = el('b', null, state.correct + ' / ' + stages[stage].length);
          line.appendChild(b);
          var praise = ratio === 1 ? T.perfect : (ratio >= 0.5 ? T.great : T.keep);
          if (praise) line.appendChild(el('span', null, ' · ' + praise));
        }
      }
      updateScore.refreshers.push(refresh);

      retry.addEventListener('click', function () {
        state.answered = 0;
        state.correct = 0;
        scoreWrap.hidden = true;
        var fresh = el('div');
        var current = panel;
        /* re-render questions of this stage in place */
        var quizQs = panel.querySelectorAll('.quiz-q');
        Array.prototype.forEach.call(quizQs, function (n) { n.remove(); });
        stages[stage].forEach(function (q) {
          renderQuestion(current, q, (T.stages && T.stages[stage]) || stage, state);
        });
        panel.insertBefore(fresh, scoreWrap);
        fresh.remove();
      });

      body.appendChild(panel);
    });

    root.hidden = false;
  }

  function ready() {
    var root = document.getElementById('quiz');
    var dataEl = document.getElementById('quiz-data');
    if (!root || !dataEl) return;
    var data;
    try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }
    if (!data || !data.questions || !data.questions.length) return;
    setup(root, data);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
