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
    return svg('<path d="M2 3h20"/><path d="M4 7h16"/><path d="M6 11h12"/><path d="M9 15h6"/><path d="M11 19h2"/>', 16);
  },
  menu: function () { return svg('<line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/>', 20); },
  search: function (size) { return svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>', size || 16); },
  sun: function () { return svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>', 18); },
  moon: function () { return svg('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>', 18); },
  github: function () {
    return svg('<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>', 18);
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
