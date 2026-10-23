/**
 * paths.js — the AI-engineer learning map: 13 role-based routes rendered
 * from the upstream learning-paths manifests (read-only source), localized
 * via src/data/paths-zh overlays. Sits above phase 00 in the sidebar.
 */
'use strict';

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function chip(text) {
  return '<span class="path-chip">' + esc(text) + '</span>';
}

function lessonRow(l) {
  return '<li><a href="' + esc(l.url) + '">'
    + '<span class="path-lesson-num">' + esc(l.n) + '</span>'
    + '<span class="path-lesson-title">' + esc(l.title) + '</span>'
    + (l.minutes ? '<span class="path-lesson-min">' + esc(l.minutes + ' min') + '</span>' : '')
    + '</a></li>';
}

/**
 * @param {object} o {lang, t, paths: [route], pageUrl}
 */
function renderMain(o) {
  const zh = o.lang === 'zh';
  const L = (en, zhs) => (zh ? zhs : en);

  let h = '<div class="content paths-page">';
  h += '<h1>' + L('AI Engineer Learning Map', 'AI 工程师学习地图') + '</h1>';
  h += '<p class="motto">' + L(
    'Thirteen role-based routes through 542 lessons. Each is an ordered, testable path to a working portfolio — pick the role you are hiring for, or the one you want.',
    '13 条面向在职工程师的角色路线，覆盖全部 542 课。每条都是有序、可验证的路线，通往可展示的实战作品——按目标岗位选一条开始。') + '</p>';

  // quick jump
  h += '<nav class="path-jump" aria-label="' + L('Routes', '路线索引') + '">';
  for (const p of o.paths) {
    h += '<a href="#path-' + esc(p.id) + '">' + esc(p.title) + '</a>';
  }
  h += '</nav>';

  for (const p of o.paths) {
    const hours = p.minutes ? Math.round(p.minutes / 60) : null;
    h += '<section class="path-card" id="path-' + esc(p.id) + '">';
    h += '<header class="path-head">';
    h += '<h2>' + esc(p.title) + '</h2>';
    if (p.titleEn && p.titleEn !== p.title) h += '<p class="path-title-en">' + esc(p.titleEn) + '</p>';
    if (p.decisionPrompt) h += '<p class="path-decision">' + esc(p.decisionPrompt) + '</p>';
    h += '<div class="path-meta">'
      + chip(p.lessonCount + ' ' + L('lessons', '节课'))
      + (hours ? chip('~' + hours + ' ' + L('hours', '小时')) : '')
      + chip(p.stages.length > 1 ? p.stages.length + ' ' + L('stages', '个阶段') : L('single track', '单线'))
      + '</div>';
    h += '</header>';

    if (p.summary) h += '<p class="path-summary">' + esc(p.summary) + '</p>';

    if (p.responsibilities && p.responsibilities.length) {
      h += '<h3>' + L('What the role does', '这个岗位做什么') + '</h3><ul class="path-list">';
      for (const s of p.responsibilities) h += '<li>' + esc(s) + '</li>';
      h += '</ul>';
    }
    if (p.goodFitIf && p.goodFitIf.length) {
      h += '<h3>' + L('Good fit if', '适合你，如果') + '</h3><ul class="path-list">';
      for (const s of p.goodFitIf) h += '<li>' + esc(s) + '</li>';
      h += '</ul>';
    }
    if (p.prerequisites && p.prerequisites.length) {
      h += '<h3>' + L('Prerequisites', '先修要求') + '</h3><ul class="path-list path-prereq">';
      for (const s of p.prerequisites) h += '<li><strong>' + esc(s.title) + '</strong> — ' + esc(s.description) + '</li>';
      h += '</ul>';
    }
    if (p.portfolio) {
      h += '<div class="path-portfolio"><h3>' + L('Portfolio proof', '实战作品') + '</h3>'
        + '<p><strong>' + esc(p.portfolio.title) + '</strong> — ' + esc(p.portfolio.description) + '</p></div>';
    }

    if (p.stages.length > 1) {
      for (const st of p.stages) {
        h += '<h3 class="path-stage-title">' + esc(st.title) + '</h3>';
        if (st.outcome) h += '<p class="path-stage-outcome">' + esc(st.outcome) + '</p>';
        if (st.lessons.length) h += '<ol class="path-lessons">' + st.lessons.map(lessonRow).join('') + '</ol>';
        if (st.artifact) h += '<p class="path-artifact">' + esc(st.artifact) + '</p>';
      }
    } else if (p.lessons.length) {
      h += '<h3>' + L('Ordered lessons', '课程顺序') + '</h3>';
      h += '<ol class="path-lessons">' + p.lessons.map(lessonRow).join('') + '</ol>';
    }

    h += '</section>';
  }

  h += '</div>';
  return h;
}

module.exports = { renderMain };
