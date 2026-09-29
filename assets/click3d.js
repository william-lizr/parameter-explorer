// Plotly does not send a click event for 3D surfaces, but it does send hover
// events. So: remember the last hovered point on each 3D plot, and on a plain
// click (no drag), send it as a click event. Dash then sets clickData as usual.
(function () {
  let down = null;

  function plotOf(el) {
    const gd = el.closest && el.closest(".js-plotly-plot");
    return gd && gd._fullLayout && gd._fullLayout.scene ? gd : null;
  }

  function watch(gd) {
    if (gd.__click3dWatched) return;
    gd.__click3dWatched = true;
    gd.on("plotly_hover", function (d) { gd.__lastHover = d.points; });
    gd.on("plotly_unhover", function () { gd.__lastHover = null; });
    gd.on("plotly_click", function () { gd.__lastClick = Date.now(); });
  }

  // Start watching as soon as the mouse enters a 3D plot.
  document.addEventListener("mouseover", function (e) {
    const gd = plotOf(e.target);
    if (gd && gd.on) watch(gd);
  }, true);

  document.addEventListener("mousedown", function (e) {
    down = { x: e.clientX, y: e.clientY, t: Date.now() };
  }, true);

  document.addEventListener("mouseup", function (e) {
    const gd = plotOf(e.target);
    if (!gd || !down) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const start = down.t;
    down = null;
    if (moved > 4) return;  // that was a rotation, not a click

    // Wait briefly: if Plotly sent its own click (scatter3d does), do nothing.
    setTimeout(function () {
      if ((gd.__lastClick || 0) >= start) return;
      if (gd.__lastHover && gd.__lastHover.length) {
        gd.emit("plotly_click", { points: gd.__lastHover, event: e });
      }
    }, 80);
  }, true);
})();
