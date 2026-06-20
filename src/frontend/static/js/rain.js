/**
 * src/frontend/static/js/rain.js
 * Generates animated rain drops and appends them to #rain container.
 * Pure vanilla JS – no dependencies.
 */
(function () {
  "use strict";

  var container = document.getElementById("rain");
  if (!container) return;

  var NUM_DROPS = 35;

  for (var i = 0; i < NUM_DROPS; i++) {
    var drop = document.createElement("div");
    drop.className = "rain-drop";
    drop.style.left              = Math.random() * 100 + "%";
    drop.style.height            = (Math.random() * 70 + 40) + "px";
    drop.style.animationDuration = (Math.random() * 2 + 1.4) + "s";
    drop.style.animationDelay    = (Math.random() * 4) + "s";
    drop.style.opacity           = (Math.random() * 0.45 + 0.15).toFixed(2);
    container.appendChild(drop);
  }
})();
