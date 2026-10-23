/**
 * icons.js — inline SVG icons (Lucide-style, 24px viewBox, currentColor).
 * No emoji anywhere: design rules require vector icons.
 */
'use strict';

function svg(inner, size) {
  var s = size || 24;
  return '<svg viewBox="0 0 24 24" width="' + s + '" height="' + s
    + '" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    + inner + '</svg>';
}

var ICONS = {
  brand: function () {
    // Site mascot: a lamb (counterpart to GitHub's cat). Filled silhouette in
    // the GitHub-mark spirit — scalloped wool, droopy ears, face as negative
    // space. 24-unit viewBox, currentColor, legible at 16px.
    return '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">'
      + '<g transform="translate(0 1.7)">'
      + '<ellipse cx="4.2" cy="8.6" rx="3" ry="1.3" transform="rotate(-24 4.2 8.6)"/>'
      + '<ellipse cx="19.8" cy="8.6" rx="3" ry="1.3" transform="rotate(24 19.8 8.6)"/>'
      + '<path fill-rule="evenodd" d="M12 3.5A2.57 2.57 0 0 1 16.74 5.46A2.57 2.57 0 0 1 18.7 10.2A2.57 2.57 0 0 1 16.74 14.94A2.57 2.57 0 0 1 12 16.9A2.57 2.57 0 0 1 7.26 14.94A2.57 2.57 0 0 1 5.3 10.2A2.57 2.57 0 0 1 7.26 5.46A2.57 2.57 0 0 1 12 3.5ZM12 7.4C14.1 7.4 15.1 9.2 15.1 11.3C15.1 13.9 13.8 15.2 12 15.2C10.2 15.2 8.9 13.9 8.9 11.3C8.9 9.2 9.9 7.4 12 7.4Z"/>'
      + '</g></svg>';
  },
  menu: function () { return svg('<line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/>', 20); },
  search: function (size) { return svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>', size || 16); },
  sun: function () { return svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>', 18); },
  moon: function () { return svg('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>', 18); },
  github: function () {
    // Official GitHub mark (Octicons "mark-github") — filled, 16-unit viewBox.
    return '<svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" aria-hidden="true">'
      + '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>';
  },
  chevronRight: function (size) { return svg('<path d="m9 18 6-6-6-6"/>', size || 14); },
  chevronLeft: function (size) { return svg('<path d="m15 18-6-6 6-6"/>', size || 14); },
  arrowRight: function (size) { return svg('<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>', size || 15); },
  arrowLeft: function (size) { return svg('<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>', size || 15); },
  clock: function (size) { return svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>', size || 13); },
  layers: function (size) { return svg('<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>', size || 13); },
  code: function (size) { return svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>', size || 13); },
  helpCircle: function (size) { return svg('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>', size || 16); },
  book: function (size) { return svg('<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>', size || 15); },
  target: function (size) { return svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>', size || 13); },
  listTree: function (size) { return svg('<path d="M21 12h-6"/><path d="M9 6h12"/><path d="M9 12h0"/><path d="M9 18h12"/><circle cx="5" cy="6" r="2"/><circle cx="5" cy="12" r="2"/><circle cx="5" cy="18" r="2"/>', size || 13); },
  map: function (size) { return svg('<path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/>', size || 14); },
};

module.exports = ICONS;
