// Drag-and-drop between the "unconstrained" and "constrained" lists.
//
// Dash owns the state. This script only turns a drop into a click on the
// matching hidden button, so the normal Dash callback does the real work.
//
// It uses pointer events, not native HTML5 drag. Native drag failed to start
// on the first try after a choice in a Dash 4 dropdown, and pointer events
// also work with touch.
(function () {
  const THRESHOLD = 5;  // px the pointer must move before a press becomes a drag
  let press = null;     // {chip, col, from, x, y}
  let ghost = null;
  let overZone = null;

  function zoneAt(x, y) {
    const el = document.elementFromPoint(x, y);
    return el && el.closest ? el.closest(".drop-zone") : null;
  }

  function setOver(zone) {
    if (overZone === zone) return;
    if (overZone) overZone.classList.remove("drag-over");
    overZone = zone;
    if (overZone) overZone.classList.add("drag-over");
  }

  function startGhost(chip, x, y) {
    const r = chip.getBoundingClientRect();
    ghost = chip.cloneNode(true);
    ghost.removeAttribute("id");
    ghost.querySelectorAll("[id]").forEach(function (n) { n.removeAttribute("id"); });
    Object.assign(ghost.style, {
      position: "fixed", left: r.left + "px", top: r.top + "px", width: r.width + "px",
      pointerEvents: "none", opacity: "0.85", zIndex: 9999, boxShadow: "0 4px 14px rgba(0,0,0,0.5)",
    });
    ghost.dataset.dx = x - r.left;
    ghost.dataset.dy = y - r.top;
    document.body.appendChild(ghost);
    chip.classList.add("drag-source");
    document.body.classList.add("dragging");
  }

  function finish() {
    if (ghost) ghost.remove();
    if (press) press.chip.classList.remove("drag-source");
    document.body.classList.remove("dragging");
    setOver(null);
    ghost = null;
    press = null;
  }

  document.addEventListener("pointerdown", function (e) {
    if (e.button !== 0) return;
    const chip = e.target.closest && e.target.closest("[data-col][data-from]");
    // Let the buttons, sliders and dropdowns inside a chip work as usual.
    if (!chip || e.target.closest("[data-action], [id*='mode-btn'], input, .dash-dropdown")) return;
    press = { chip: chip, col: chip.getAttribute("data-col"), from: chip.getAttribute("data-from"),
              x: e.clientX, y: e.clientY };
  });

  document.addEventListener("pointermove", function (e) {
    if (!press) return;
    if (!ghost) {
      if (Math.hypot(e.clientX - press.x, e.clientY - press.y) < THRESHOLD) return;
      startGhost(press.chip, e.clientX, e.clientY);
    }
    e.preventDefault();  // no text selection while dragging
    ghost.style.left = (e.clientX - ghost.dataset.dx) + "px";
    ghost.style.top = (e.clientY - ghost.dataset.dy) + "px";
    setOver(zoneAt(e.clientX, e.clientY));
  });

  document.addEventListener("pointerup", function (e) {
    if (!press) return;
    const dragged = !!ghost;
    const zone = dragged ? zoneAt(e.clientX, e.clientY) : null;
    const col = press.col;
    const from = press.from;
    finish();
    if (!zone) return;

    const target = zone.getAttribute("data-zone");
    let action = null;
    if (target === "constrained" && from === "free") action = "constrain";
    if (target === "free" && from === "constrained") action = "free";
    if (!action) return;

    const btns = document.querySelectorAll('[data-action="' + action + '"]');
    for (const btn of btns) {
      if (btn.getAttribute("data-btn-col") === col) {
        btn.click();
        break;
      }
    }
  });

  document.addEventListener("pointercancel", finish);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") finish(); });
})();
