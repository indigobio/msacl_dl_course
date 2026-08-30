/* ============================================================
   MSACL DL Course — shared slide behaviors
   - renders .conv-demo / .pool-demo numeric grids from matrices
     declared in the deck (window.<NAME>), so values are always
     computed, never hand-typed
   - injects the m/z-ruler footer + slide numbers
   - #<slide-id> in the URL isolates one slide (screenshots,
     presenting); arrow keys navigate in that mode
   Load this BEFORE the deck's own <script> that declares matrices.
   ============================================================ */

"use strict";

/* ---------- convolution / pooling math ---------- */
function conv2d(img, f, stride, pad) {
  const p = pad || 0, s = stride || 1;
  const padded = padImg(img, p);
  const n = padded.length, k = f.length;
  const out = [];
  for (let r = 0; r + k <= n; r += s) {
    const row = [];
    for (let c = 0; c + k <= n; c += s) {
      let sum = 0;
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) sum += padded[r + i][c + j] * f[i][j];
      row.push(sum);
    }
    out.push(row);
  }
  return out;
}
function padImg(img, p) {
  if (!p) return img;
  const n = img.length, m = n + 2 * p;
  const out = Array.from({ length: m }, () => Array(m).fill(0));
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out[r + p][c + p] = img[r][c];
  return out;
}
function maxPool(img, size) {
  const s = size || 2, out = [];
  for (let r = 0; r + s <= img.length; r += s) {
    const row = [];
    for (let c = 0; c + s <= img[0].length; c += s) {
      let m = -Infinity;
      for (let i = 0; i < s; i++) for (let j = 0; j < s; j++) m = Math.max(m, img[r + i][c + j]);
      row.push(m);
    }
    out.push(row);
  }
  return out;
}

/* ---------- grid table builders ---------- */
function fmt(v) { return v < 0 ? "−" + Math.abs(v) : String(v); }

function buildGrid(matrix, opts) {
  const o = opts || {};
  const t = document.createElement("table");
  t.className = "ngrid" + (o.small ? " small" : "");
  matrix.forEach((row, r) => {
    const tr = t.insertRow();
    row.forEach((v, c) => {
      const td = tr.insertCell();
      td.textContent = fmt(v);
      if (o.cellClass) { const cls = o.cellClass(r, c, v); if (cls) td.className = cls; }
    });
  });
  return t;
}

function block(labelHtml, gridEl) {
  const d = document.createElement("div");
  d.className = "grid-block";
  const l = document.createElement("div");
  l.className = "glabel";
  l.innerHTML = labelHtml;
  d.appendChild(l); d.appendChild(gridEl);
  return d;
}
function opEl(sym) {
  const s = document.createElement("div");
  s.className = "op"; s.textContent = sym;
  return s;
}

/* ---------- .conv-demo renderer ----------
   data-img / data-filter : names of window matrices
   data-stride (1) · data-pad (0) · data-pos "r,c" window in padded coords
   data-show N : output values revealed (row-major); last one = current
   data-quiz "i,j;i,j" : output cells rendered as "?"
   data-hidefilter · data-small · data-calc : show dot-product line
   data-imglabel / data-flabel / data-maplabel : block captions      */
function renderConvDemo(el) {
  const img0 = window[el.dataset.img || "IMG"];
  const f = window[el.dataset.filter || "F1"];
  const stride = +(el.dataset.stride || 1);
  const pad = +(el.dataset.pad || 0);
  const img = padImg(img0, pad);
  const out = conv2d(img0, f, stride, pad);
  const cols = out[0].length;
  const show = el.dataset.show !== undefined ? +el.dataset.show : out.length * cols;
  const pos = el.dataset.pos ? el.dataset.pos.split(",").map(Number) : null;
  const quiz = (el.dataset.quiz || "").split(";").filter(Boolean)
    .map(sq => sq.split(",").map(Number));
  const small = "small" in el.dataset;
  const k = f.length;

  const wrap = document.createElement("div");
  wrap.className = "grid-wrap";

  const imgGrid = buildGrid(img, {
    small,
    cellClass: (r, c) => {
      const isPad = pad && (r < pad || c < pad || r >= img.length - pad || c >= img.length - pad);
      const inWin = pos && r >= pos[0] && r < pos[0] + k && c >= pos[1] && c < pos[1] + k;
      return (inWin ? "win " : "") + (isPad ? "blankpad" : "");
    }
  });
  if (pad) imgGrid.querySelectorAll("td.blankpad, td.win.blankpad")
    .forEach(td => { td.style.color = "var(--muted)"; td.style.background = td.classList.contains("win") ? "" : "#F4F4EF"; });
  wrap.appendChild(block(el.dataset.imglabel || ("input <b>" + img.length + "×" + img.length + "</b>"), imgGrid));

  if (!("hidefilter" in el.dataset)) {
    wrap.appendChild(opEl("⊛"));
    const fGrid = buildGrid(f, { small: true, cellClass: (r, c, v) => (v > 0 ? "pos" : v < 0 ? "negv" : "") });
    wrap.appendChild(block(el.dataset.flabel || ("filter <b>" + k + "×" + k + "</b>"), fGrid));
  }

  wrap.appendChild(opEl("="));
  const mapGrid = buildGrid(out, {
    small,
    cellClass: (r, c) => {
      const idx = r * cols + c;
      if (quiz.some(q => q[0] === r && q[1] === c)) return "quiz";
      if (idx >= show) return "blank";
      return idx === show - 1 && pos ? "now" : "done";
    }
  });
  mapGrid.querySelectorAll("td.quiz").forEach(td => (td.textContent = "?"));
  wrap.appendChild(block(
    el.dataset.maplabel || ("feature map <b>" + out.length + "×" + cols + "</b> · stride " + stride),
    mapGrid));
  el.appendChild(wrap);

  if ("calc" in el.dataset && pos && show > 0) {
    const r0 = pos[0], c0 = pos[1];
    const terms = [];
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++)
      terms.push(img[r0 + i][c0 + j] + "×" + fmt(f[i][j]));
    const val = out[Math.floor((show - 1) / cols)][(show - 1) % cols];
    const line = document.createElement("div");
    line.className = "calc-line";
    line.innerHTML = terms.join(" + ").replace(/×−(\d)/g, "×(−$1)") +
      " = <b>" + fmt(val) + "</b>";
    el.appendChild(line);
  }
}

/* ---------- .pool-demo renderer ----------
   data-img : name of matrix (e.g. MAP1) · data-size (2)              */
function renderPoolDemo(el) {
  const img = window[el.dataset.img];
  const size = +(el.dataset.size || 2);
  const out = maxPool(img, size);
  const blocks = ["poolA", "poolB", "poolC", "poolD"];
  const wrap = document.createElement("div");
  wrap.className = "grid-wrap";
  const inGrid = buildGrid(img, {
    cellClass: (r, c) => blocks[(Math.floor(r / size) * (img[0].length / size) + Math.floor(c / size)) % 4]
  });
  wrap.appendChild(block(el.dataset.imglabel || "feature map <b>4×4</b>", inGrid));
  wrap.appendChild(opEl("max →"));
  const outGrid = buildGrid(out, {
    cellClass: (r, c) => blocks[(r * out[0].length + c) % 4] + " done"
  });
  wrap.appendChild(block(el.dataset.maplabel || ("pooled <b>" + out.length + "×" + out[0].length + "</b>"), outGrid));
  el.appendChild(wrap);
}

/* ---------- m/z ruler footer + slide numbers ---------- */
function injectChrome() {
  const slides = Array.from(document.querySelectorAll(".slide"));
  slides.forEach((s, i) => {
    const frac = slides.length > 1 ? i / (slides.length - 1) : 0;
    const x = 60 + frac * 1120;
    const ruler = document.createElement("div");
    ruler.className = "mz-ruler";
    let ticks = "";
    for (let t = 60; t <= 1180; t += 56)
      ticks += `<line x1="${t}" y1="30" x2="${t}" y2="${(t - 60) % 280 === 0 ? 22 : 26}" stroke="#C9C9C1" stroke-width="1"/>`;
    ruler.innerHTML =
      `<svg viewBox="0 0 1280 44">
         <line x1="40" y1="30" x2="1240" y2="30" stroke="#C9C9C1" stroke-width="1"/>
         ${ticks}
         <path d="M ${x - 7} 30 Q ${x} 8 ${x + 7} 30 Z" fill="#0E7C7B"/>
       </svg>`;
    s.appendChild(ruler);
    if (!s.classList.contains("title-slide")) {
      const n = document.createElement("div");
      n.className = "slide-no";
      n.textContent = String(i + 1).padStart(2, "0") + " / " + slides.length;
      s.appendChild(n);
    }
    if (!s.id) s.id = "s" + (i + 1);
  });
}

/* ---------- solo mode (screenshots / presenting) ---------- */
function applyHash() {
  const id = location.hash.slice(1);
  const slides = Array.from(document.querySelectorAll(".slide"));
  const target = id && document.getElementById(id);
  document.body.classList.toggle("solo", !!target);
  slides.forEach(s => (s.style.display = target ? (s === target ? "flex" : "none") : ""));
  if (target) { document.body.style.padding = "0"; window.scrollTo(0, 0); }
  else document.body.style.padding = "";
}
function soloNav(delta) {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const cur = slides.findIndex(s => "#" + s.id === location.hash);
  const next = Math.min(slides.length - 1, Math.max(0, (cur < 0 ? 0 : cur) + delta));
  location.hash = "#" + slides[next].id;
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".conv-demo").forEach(renderConvDemo);
  document.querySelectorAll(".pool-demo").forEach(renderPoolDemo);
  injectChrome();
  applyHash();
});
window.addEventListener("hashchange", applyHash);
document.addEventListener("keydown", e => {
  if (!document.body.classList.contains("solo")) return;
  if (e.key === "ArrowRight" || e.key === " ") soloNav(1);
  if (e.key === "ArrowLeft") soloNav(-1);
});
