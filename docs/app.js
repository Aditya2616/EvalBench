/* EvalBench front end. Vanilla ES module. Everything drawn comes from data/run.json. */

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

const MODE_LABEL = { hallucination: 'Hallucination', format_break: 'Format break', over_refusal: 'Over-refusal', wrong_answer: 'Wrong answer', degenerate: 'Degenerate' };
const MODES = Object.keys(MODE_LABEL);
const MODE_COLOR = { hallucination: '#ff7a3d', format_break: '#ffd23d', over_refusal: '#ff4d6d', wrong_answer: '#a99bff', degenerate: '#7fb4cf' };
// each checkpoint keeps one shape and one tint in the lanes, the map and the detail sheet
const CK_TINT = ['#7fc8ff', '#ffd23d', '#c4a8ff', '#ff9ec4', '#9be8c0'];
const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'circle'];
function glyph(c, size = 12) {
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '-6 -6 12 12'); svg.setAttribute('width', size); svg.setAttribute('height', size); svg.setAttribute('class', 'gl'); svg.setAttribute('aria-hidden', 'true');
  const sh = SHAPES[c % SHAPES.length], p = document.createElementNS(NS, sh === 'circle' ? 'circle' : 'path');
  if (sh === 'circle') { p.setAttribute('r', 4.6); } else if (sh === 'square') { p.setAttribute('d', 'M-4 -4H4V4H-4Z'); } else if (sh === 'triangle') { p.setAttribute('d', 'M0 -5.2L5 4L-5 4Z'); } else { p.setAttribute('d', 'M0 -5.6L5.6 0L0 5.6L-5.6 0Z'); }
  p.setAttribute('fill', CK_TINT[c % CK_TINT.length]); svg.append(p);
  return svg;
}
const TASKS = ['math', 'knowledge', 'format', 'truth', 'unanswerable', 'benign'];
const TASK_SHORT = { math: 'math', knowledge: 'know', format: 'format', truth: 'truth', unanswerable: 'unans', benign: 'benign', ALL: 'all' };
const RATE = 24; // replayed prompts per second at 1x
const FLIGHT = 16; // prompts of replay time (about two thirds of a second) a dot takes to fly to its place

function wilson(k, n, z = 1.96) {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

let D, NAMES, N, C, SAMPLE, ORDER, DOTS, CLUSTERS, TRY = false;

async function load() {
  const r = await fetch('data/run.json');
  if (!r.ok) throw new Error('no recorded run found');
  D = await r.json();
  try { TRY = (await (await fetch('data/config.json')).json()).try_enabled === true; } catch { TRY = false; }
}

/* ------------------------------------------------------------------ data prep */
function prepare() {
  NAMES = D.meta.checkpoints.map((c) => c.name);
  C = NAMES.length; N = D.meta.n_items; ORDER = D.order;
  SAMPLE = new Map(D.samples.map((s) => [s.c + '|' + s.i, s]));
  CLUSTERS = new Map(D.clusters.clusters.map((c) => [c.id, c]));
  const pts = new Map(D.clusters.points.map((p) => [p.ckpt + '|' + p.item, p]));
  // running totals after each replayed prompt
  const cum = NAMES.map(() => new Float64Array(N + 1));
  const win = NAMES.map(() => new Float64Array(N + 1));
  const wsum = new Float64Array(C);
  ORDER.forEach((id, k) => {
    const sc = NAMES.map((_, c) => SAMPLE.get(c + '|' + id).s);
    for (let c = 0; c < C; c++) {
      cum[c][k + 1] = cum[c][k] + (SAMPLE.get(c + '|' + id).p ? 1 : 0);
      for (let o = 0; o < C; o++) if (o !== c) wsum[c] += sc[c] > sc[o] ? 1 : sc[c] === sc[o] ? 0.5 : 0;
      win[c][k + 1] = wsum[c] / ((k + 1) * (C - 1));
    }
  });
  D._cum = cum; D._win = win;
  DOTS = [];
  ORDER.forEach((id, k) => {
    for (let c = 0; c < C; c++) {
      const s = SAMPLE.get(c + '|' + id);
      if (s.p) continue;
      const p = pts.get(NAMES[c] + '|' + id);
      if (!p) continue;
      DOTS.push({ c, id, k, spawn: k + c * 0.2, p, s, ringA: 0.9 });
    }
  });
  const share = {};
  for (const d of DOTS) { const m = d.p.mode; share[m] = share[m] || [0, 0]; share[m][1]++; if (d.p.mislabelled) share[m][0]++; }
  for (const d of DOTS) if (share[d.p.mode][0] / share[d.p.mode][1] > 0.5) d.ringA = 0.3; // an island that is ringed almost everywhere would drown in rings
}

/* ------------------------------------------------------------------ stage (race + failure map) */
const Stage = (() => {
  const canvas = $('#canvas'), ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1, L = null;
  let t = 0, playing = false, speed = 1, visible = false, started = false, last = 0, dirty = true;
  let colorMode = 'ckpt', view = 'islands', vm = 0, selected = null, axisMax = 0.6;
  const scrub = $('#scrub'), count = $('#count');
  const mono = (px, w = 400) => `${w} ${px}px "JetBrains Mono", ui-monospace, monospace`;

  function layout() {
    W = canvas.parentElement.clientWidth;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const phone = W < 560, pad = 14, laneH = 42, lanesTop = 14;
    const mapTop = lanesTop + C * laneH + 26;
    const key = W >= 720 ? 'wide' : 'sq', pre = key === 'wide' ? 'w' : '';
    const ext = D.clusters.stats.extent[key];
    const mx0 = pad + 8, mx1 = W - pad - 8, pw = mx1 - mx0;
    const need = Math.round(pw * (ext[1] / ext[0]));
    const mapH = clamp(need + (phone ? 96 : 84), phone ? 300 : 280, 620);
    H = mapTop + mapH + 12;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.height = H + 'px';
    const my0 = mapTop + 40, my1 = mapTop + mapH - 12, ph = my1 - my0;
    L = { pad, laneH, lanesTop, mapTop, mapH, mx0, mx1, my0, my1, bx0: pad, bx1: W - pad, key, pre, phone };
    const maxFinal = Math.max(...NAMES.map((_, c) => D._cum[c][N] / N));
    axisMax = Math.min(1, Math.ceil((maxFinal + 0.06) * 10) / 10);
    const sc = Math.min(pw / 2 / ext[0], ph / 2 / ext[1]), cx = (mx0 + mx1) / 2, cy = (my0 + my1) / 2 + 6;
    L.sc = sc;
    DOTS.forEach((d) => {
      d.ix = cx + d.p[pre + 'x'] * sc; d.iy = cy + d.p[pre + 'y'] * sc;
      d.rx = cx + d.p.rx * (pw / 2); d.ry = cy + d.p.ry * (ph / 2);
    });
    L.subs = D.clusters.clusters.map((c) => ({ c, x: cx + c[pre + 'x'] * sc, y: cy + c[pre + 'y'] * sc, r: c[pre + 'r'] * sc }));
    L.isl = D.clusters.islands.map((i) => ({ i, x: cx + i[pre + 'x'] * sc, y: cy + i[pre + 'y'] * sc, r: i[pre + 'r'] * sc }));
    // island labels: try above and below each island, keep the spot that covers the least of any other island or label
    const lpx = phone ? 9.5 : 10.5, minY = mapTop + 20, maxY = my1 + 8, lh = 2 * lpx + 6;
    const placed = [];
    const hit = (x, y, w, o) => { // how far the label box pokes into another island's circle
      const dx = Math.max(Math.abs(o.x - x) - w / 2, 0), dy = Math.max(Math.abs(o.y - (y - lpx + lh / 2)) - lh / 2, 0);
      return Math.max(0, o.r - Math.hypot(dx, dy));
    };
    L.lab = [];
    [...L.isl].sort((p, q) => q.i.n - p.i.n).forEach((o) => {
      const w = (MODE_LABEL[o.i.mode].length + 5) * lpx * 0.64 + 8, x = clamp(o.x, w / 2 + 4, W - w / 2 - 4);
      let best = null;
      for (const y0 of [o.y - o.r - 10, o.y + o.r + lpx + 8, o.y - o.r + 4]) {
        const y = clamp(y0, minY, maxY);
        let cost = Math.abs(y - y0) * 3;
        for (const q of L.isl) if (q !== o) cost += hit(x, y, w, q) * 2;
        for (const q of placed) if (Math.abs(q.x - x) < (q.w + w) / 2 && Math.abs(q.y - y) < lh) cost += 60;
        if (!best || cost < best.cost - 0.5) best = { y, cost };
      }
      const lb = { o, w, x, y: best.y }; placed.push(lb); L.lab.push(lb);
    });
    dirty = true;
  }

  const barX = (v) => lerp(L.bx0, L.bx1, clamp(v / axisMax));
  function rateAt(c, tt) {
    const k = Math.min(Math.floor(tt), N), f = tt - k, a = k ? D._cum[c][k] / k : 0;
    if (k >= N) return D._cum[c][N] / N;
    const b = D._cum[c][k + 1] / (k + 1);
    return lerp(a, b, easeIO(f));
  }

  function dotColor(d) {
    if (colorMode === 'tag' || view === 'raw') return MODE_COLOR[d.s.m] || '#ff7a3d';
    return CK_TINT[d.c % CK_TINT.length];
  }

  function shape(c, x, y, r) {
    switch (SHAPES[c % SHAPES.length]) {
      case 'square': ctx.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76); break;
      case 'triangle': ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.05, y + r * 0.8); ctx.lineTo(x - r * 1.05, y + r * 0.8); ctx.closePath(); break;
      case 'diamond': ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath(); break;
      default: ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 7);
    }
  }

  // a dot's path: leaves the head of its lane, drops below the lanes, then settles into its place on the map
  function pos(d, f) {
    const e = easeIO(clamp(f));
    const x0 = barX(D._cum[d.c][d.k + 1] / (d.k + 1)), y0 = L.lanesTop + d.c * L.laneH + 22;
    const tx = lerp(d.ix, d.rx, vm), ty = lerp(d.iy, d.ry, vm);
    const c1x = lerp(x0, tx, 0.3), c1y = L.mapTop - 2;
    const u = 1 - e;
    return [u * u * x0 + 2 * u * e * c1x + e * e * tx, u * u * y0 + 2 * u * e * c1y + e * e * ty];
  }

  function halo(txt, x, y, a) {
    ctx.lineJoin = 'round'; ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(10,11,14,.92)'; ctx.strokeText(txt, x, y);
    ctx.fillStyle = a; ctx.fillText(txt, x, y);
  }

  function draw() {
    if (!L) return;
    const k = Math.floor(clamp(t, 0, N));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const rates = NAMES.map((_, c) => rateAt(c, t));
    const lead = rates.indexOf(Math.max(...rates));
    // map frame and islands
    ctx.textAlign = 'left'; ctx.font = mono(10.5, 500); ctx.fillStyle = 'rgba(237,236,230,.46)';
    ctx.fillText(view === 'islands' ? (L.phone ? 'FAILURE MAP · one island per mode' : 'FAILURE MAP · one island per failure mode, mode set by the rule detectors') : (L.phone ? 'RAW t-SNE · no grouping' : 'RAW PROJECTION · t-SNE of the failure embeddings, no grouping'), L.pad, L.mapTop + 8);
    ctx.strokeStyle = 'rgba(237,236,230,.09)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L.pad, L.mapTop - 8); ctx.lineTo(W - L.pad, L.mapTop - 8); ctx.stroke();
    const landedSub = new Map(), landedIsl = new Map();
    for (const d of DOTS) if (t >= d.spawn + FLIGHT) { landedSub.set(d.p.cluster, (landedSub.get(d.p.cluster) || 0) + 1); landedIsl.set(d.p.mode, (landedIsl.get(d.p.mode) || 0) + 1); }
    const islA = 1 - vm;
    if (islA > 0.02) {
      ctx.globalAlpha = islA;
      L.isl.forEach((o) => {
        const n = landedIsl.get(o.i.mode) || 0, a = clamp(n / Math.max(4, o.i.n * 0.5));
        ctx.strokeStyle = MODE_COLOR[o.i.mode]; ctx.globalAlpha = islA * (0.22 + 0.5 * a); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      });
      ctx.globalAlpha = islA * 0.9;
      L.subs.forEach((s) => {
        const n = landedSub.get(s.c.id) || 0, a = clamp(n / Math.max(4, s.c.size * 0.5));
        ctx.strokeStyle = `rgba(237,236,230,${0.05 + 0.12 * a})`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 7); ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }
    // dots, drawn before the lanes so the lanes stay on top
    const rad = L.phone ? 2.3 : 2.9;
    const trail = [];
    for (const d of DOTS) {
      if (t < d.spawn) continue;
      const f = (t - d.spawn) / FLIGHT, [x, y] = pos(d, f);
      d._x = x; d._y = y;
      const fl = f < 1;
      if (fl) trail.push(d);
      ctx.globalAlpha = fl ? 1 : 0.86;
      ctx.fillStyle = dotColor(d);
      ctx.beginPath(); shape(d.c, x, y, fl ? rad * 1.3 : rad); ctx.fill();
      if (!fl && d.p.mislabelled) { ctx.globalAlpha = d.ringA; ctx.strokeStyle = '#edece6'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, rad + 2.5, 0, 7); ctx.stroke(); }
      if (!fl && selected && selected.c === d.c && selected.id === d.id) {
        ctx.globalAlpha = 1; ctx.strokeStyle = '#d7ff3f'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, rad + 5, 0, 7); ctx.stroke();
      }
    }
    for (const d of trail) { // faint trail behind each dot in flight
      const f = (t - d.spawn) / FLIGHT;
      for (let j = 1; j <= 4; j++) {
        const ff = f - j * 0.05; if (ff <= 0) break;
        const [x, y] = pos(d, ff);
        ctx.globalAlpha = 0.28 / j; ctx.fillStyle = dotColor(d);
        ctx.beginPath(); ctx.arc(x, y, rad * (1.25 - j * 0.15), 0, 7); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    // island labels with live counts
    if (islA > 0.02) {
      ctx.textAlign = 'center';
      L.lab.forEach((lb) => {
        const o = lb.o, n = landedIsl.get(o.i.mode) || 0, px = L.phone ? 9.5 : 10.5;
        ctx.globalAlpha = islA; ctx.font = mono(px, 600);
        halo(MODE_LABEL[o.i.mode].toUpperCase(), lb.x, lb.y, MODE_COLOR[o.i.mode]);
        ctx.font = mono(px, 500);
        halo(`${n} of ${o.i.n}`, lb.x, lb.y + px + 3, 'rgba(237,236,230,.7)');
      });
      if (!L.phone) {
        ctx.font = mono(9.5, 400);
        L.subs.forEach((s) => {
          const n = landedSub.get(s.c.id) || 0;
          if (s.r * 2 < 78 || n < s.c.size * 0.6) return;
          ctx.globalAlpha = islA * 0.9;
          halo(s.c.name.length > 26 ? s.c.name.slice(0, 25) + '…' : s.c.name, s.x, s.y + s.r + 11, 'rgba(237,236,230,.62)');
        });
      }
      ctx.globalAlpha = 1; ctx.textAlign = 'left';
    }
    if (vm > 0.02) { // legend for the raw view: dots are coloured by detector mode
      ctx.globalAlpha = vm; ctx.font = mono(10, 500); ctx.textAlign = 'left';
      let lx = L.pad, ly = L.my1 + 2;
      MODES.forEach((m) => {
        const w = MODE_LABEL[m].length * 6.4 + 22;
        if (lx + w > W - L.pad) { lx = L.pad; ly -= 14; }
        ctx.fillStyle = MODE_COLOR[m]; ctx.beginPath(); ctx.arc(lx + 4, ly - 3, 3.2, 0, 7); ctx.fill();
        ctx.fillStyle = 'rgba(237,236,230,.7)'; ctx.fillText(MODE_LABEL[m], lx + 12, ly); lx += w;
      });
      ctx.globalAlpha = 1;
    }
    // lanes (on top of the dots; text gets a halo so a passing dot never hides a number)
    NAMES.forEach((name, c) => {
      const y0 = L.lanesTop + c * L.laneH, isLead = c === lead && t > 3;
      ctx.font = mono(11, isLead ? 600 : 400); ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = CK_TINT[c % CK_TINT.length]; ctx.beginPath(); shape(c, L.bx0 + 5, y0 + 7.5, 4.2); ctx.fill();
      ctx.textAlign = 'left';
      halo(name, L.bx0 + 15, y0 + 11, isLead ? '#edece6' : 'rgba(237,236,230,.62)');
      const w = k ? D._win[c][k] : 0;
      ctx.textAlign = 'right';
      halo(`${pct(rates[c])} pass · win ${w.toFixed(2)}`, L.bx1, y0 + 11, isLead ? '#d7ff3f' : 'rgba(237,236,230,.62)');
      ctx.fillStyle = 'rgba(237,236,230,.07)';
      rr(L.bx0, y0 + 18, L.bx1 - L.bx0, 8, 4); ctx.fill();
      const ex = barX(rates[c]);
      ctx.fillStyle = isLead ? '#d7ff3f' : 'rgba(237,236,230,.38)';
      rr(L.bx0, y0 + 18, Math.max(8, ex - L.bx0), 8, 4); ctx.fill();
      if (k >= 4) { // Wilson interval on what has been replayed so far
        const [lo, hi] = wilson(D._cum[c][k], k);
        ctx.strokeStyle = 'rgba(237,236,230,.8)'; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(barX(lo), y0 + 22); ctx.lineTo(barX(hi), y0 + 22);
        ctx.moveTo(barX(lo), y0 + 19); ctx.lineTo(barX(lo), y0 + 25); ctx.moveTo(barX(hi), y0 + 19); ctx.lineTo(barX(hi), y0 + 25); ctx.stroke();
      }
      if (k < N && t > 0) { // verdict of the prompt being replayed right now
        const s = SAMPLE.get(c + '|' + ORDER[k]), fr = t - k;
        ctx.globalAlpha = 1 - fr; ctx.fillStyle = s.p ? '#d7ff3f' : '#ff7a3d';
        ctx.beginPath(); ctx.arc(Math.min(ex + 2, L.bx1 - 3), y0 + 22, 3.5, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
      }
    });
    ctx.textAlign = 'left';
    if (!landedIsl.size && t < 1) {
      ctx.font = mono(12); ctx.fillStyle = 'rgba(237,236,230,.5)'; ctx.textAlign = 'center';
      ctx.fillText('press play: failing samples will land here', W / 2, (L.my0 + L.my1) / 2); ctx.textAlign = 'left';
    }
    count.textContent = `${Math.floor(t)} / ${N}`;
    if (document.activeElement !== scrub) scrub.value = t;
  }

  function rr(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (playing) {
      t += dt * RATE * speed;
      if (t >= N + FLIGHT) { t = N + FLIGHT; setPlaying(false); }
      dirty = true;
    }
    const goal = view === 'raw' ? 1 : 0;
    if (vm !== goal) { vm = reduced ? goal : (goal > vm ? Math.min(goal, vm + dt * 1.8) : Math.max(goal, vm - dt * 1.8)); dirty = true; }
    if (dirty) { draw(); dirty = false; }
  }

  function setPlaying(p) {
    playing = p;
    $('#ico-pause').toggleAttribute('hidden', !p); $('#ico-play').toggleAttribute('hidden', p);
    $('#play').setAttribute('aria-label', p ? 'Pause' : 'Play');
  }

  function pick(x, y) {
    let best = null, bd = 22 * 22;
    for (const d of DOTS) {
      if (t < d.spawn + FLIGHT) continue;
      const dd = (d._x - x) ** 2 + (d._y - y) ** 2;
      if (dd < bd) { bd = dd; best = d; }
    }
    if (best) return { dot: best };
    if (view === 'raw') return null;
    for (const s of L.subs) if (Math.hypot(s.x - x, s.y - y) <= s.r + 3) return { cluster: s.c };
    for (const o of L.isl) if (Math.hypot(o.x - x, o.y - y) <= o.r + 3) return { island: o.i };
    return null;
  }

  function setToggle(sel, attr, val) {
    document.querySelectorAll(sel).forEach((o) => { const on = o.dataset[attr] === val; o.classList.toggle('on', on); o.setAttribute('aria-checked', on); });
  }

  function init() {
    layout();
    new ResizeObserver(() => { layout(); }).observe(canvas.parentElement);
    new IntersectionObserver((es) => {
      visible = es[0].isIntersecting; dirty = true;
      if (visible && !started) {
        started = true;
        if (reduced) { t = N + FLIGHT; draw(); } else setTimeout(() => setPlaying(true), 350);
      }
    }, { threshold: 0.25 }).observe(canvas);
    requestAnimationFrame((n) => { last = n; frame(n); });
    document.fonts && document.fonts.ready.then(() => { dirty = true; });
    $('#play').addEventListener('click', () => { if (t >= N + FLIGHT) t = 0; setPlaying(!playing); });
    $('#replay').addEventListener('click', () => { t = 0; selected = null; setPlaying(true); dirty = true; });
    scrub.max = N;
    scrub.addEventListener('input', () => { t = Math.min(+scrub.value, N); setPlaying(false); dirty = true; });
    document.querySelectorAll('[data-speed]').forEach((b) => b.addEventListener('click', () => { speed = +b.dataset.speed; setToggle('[data-speed]', 'speed', b.dataset.speed); }));
    document.querySelectorAll('[data-color]').forEach((b) => b.addEventListener('click', () => { colorMode = b.dataset.color; setToggle('[data-color]', 'color', colorMode); dirty = true; }));
    document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => { view = b.dataset.view; setToggle('[data-view]', 'view', view); dirty = true; }));
    // a click is only fired for taps, not for scroll gestures, and it fires before anything opens over the canvas
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect(), hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (!hit) { selected = null; dirty = true; return; }
      setPlaying(false);
      if (hit.dot) { selected = hit.dot; dirty = true; Drawer.openCluster(CLUSTERS.get(hit.dot.p.cluster), hit.dot.c + '|' + hit.dot.id); }
      else if (hit.cluster) Drawer.openCluster(hit.cluster);
      else Drawer.openIsland(hit.island);
    });
    window.__evalbench = {
      seek: (v) => { t = v; setPlaying(false); dirty = true; draw(); }, state: () => ({ t, playing, N, FLIGHT, view, vm }),
      view: (v) => { view = v; vm = v === 'raw' ? 1 : 0; dirty = true; draw(); setToggle('[data-view]', 'view', v); },
      pickAt: (x, y) => pick(x, y), dot: (i) => (DOTS[i] && DOTS[i]._x != null ? { x: DOTS[i]._x, y: DOTS[i]._y } : null),
      fps: () => new Promise((res) => { let n = 0; const s = performance.now(); const f = () => { n++; if (performance.now() - s < 1000) requestAnimationFrame(f); else res(n); }; requestAnimationFrame(f); }),
      play: (v) => { setPlaying(v); },
    };
  }
  return { init, redraw: () => { dirty = true; } };
})();

/* ------------------------------------------------------------------ drawer */
const Drawer = (() => {
  const root = $('#drawer'), scrim = $('#scrim'), body = $('#dr-body'), filt = $('#dr-filter');
  let opener = null, state = null, open = false;
  const wide = () => matchMedia('(min-width: 900px)').matches;
  const from = () => (wide() ? { x: '100%', y: 0 } : { y: '100%', x: 0 });

  function show() {
    if (open) return;
    open = true; opener = document.activeElement;
    root.hidden = false; scrim.hidden = false; document.body.style.overflow = 'hidden';
    if (window.gsap && !reduced) {
      gsap.fromTo(root, { ...from(), opacity: 1 }, { x: 0, y: 0, duration: 0.55, ease: 'expo.out' });
      gsap.fromTo(scrim, { opacity: 0 }, { opacity: 1, duration: 0.3 });
    }
    $('#dr-close').focus({ preventScroll: true });
  }
  function close() {
    if (!open) return;
    open = false; document.body.style.overflow = '';
    const done = () => { root.hidden = true; scrim.hidden = true; if (window.gsap) gsap.set(root, { clearProps: 'all' }); opener && opener.focus && opener.focus({ preventScroll: true }); };
    if (window.gsap && !reduced) {
      gsap.to(root, { ...from(), duration: 0.35, ease: 'power3.in', onComplete: done });
      gsap.to(scrim, { opacity: 0, duration: 0.3 });
    } else done();
  }

  function card(entry, expand) {
    const { s, c } = entry, it = D.items[s.i];
    const li = el('div', 'card' + (expand ? ' open' : ''));
    const head = el('button'); head.type = 'button'; head.setAttribute('aria-expanded', !!expand);
    const tags = el('div', 'tags');
    const ckt = el('span', 'ck'); ckt.append(glyph(c, 11), document.createTextNode(NAMES[c]));
    tags.append(ckt, el('span', '', it.task));
    tags.append(el('span', s.p ? 'ok' : 'mode', s.p ? 'pass' : MODE_LABEL[s.m] || 'fail'));
    if (entry.ring) tags.append(el('span', 'ring', 'embedding looks like another mode'));
    head.append(tags, el('div', 'pv', it.prompt));
    const more = el('div', 'more');
    const fill = () => {
      if (more.childNodes.length) return;
      more.append(el('h4', '', 'Prompt'), el('pre', '', it.prompt));
      if (it.gold) more.append(el('h4', '', 'Gold'), el('pre', '', it.gold));
      more.append(el('h4', '', 'Output'), Object.assign(el('pre', 'out'), { textContent: s.tx || '(empty output)' }));
      let v = `${s.p ? 'Pass' : 'Fail'} via ${s.sc === 'judge' ? 'rubric judge' : 'rule check'}: ${s.ev}`;
      if (s.jp != null) v += ` · judge p(yes) ${s.jp.toFixed(2)}`;
      if (s.tr) v += ' · output hit the token cap';
      more.append(el('p', '', v), el('p', '', `Source ${it.source} · ${it.license} · ${s.nt} new tokens`));
    };
    head.addEventListener('click', () => { fill(); const o = li.classList.toggle('open'); head.setAttribute('aria-expanded', o); });
    li.append(head, more);
    if (expand) fill();
    return li;
  }

  function render(expandKey) {
    const { entries, ck } = state;
    body.replaceChildren();
    const list = entries.filter((e) => ck == null || e.c === ck);
    if (!list.length) { body.append(el('p', 'empty', 'No samples for this filter.')); return; }
    let target = null;
    list.slice(0, 80).forEach((e) => { const k = e.c + '|' + e.s.i; const n = card(e, k === expandKey); if (k === expandKey) target = n; body.append(n); });
    if (list.length > 80) body.append(el('p', 'empty', `Showing 80 of ${list.length}. Filter by checkpoint to narrow it.`));
    if (target) requestAnimationFrame(() => target.scrollIntoView({ block: 'center' }));
    else body.scrollTop = 0;
  }

  function setFilters() {
    filt.replaceChildren();
    const cks = [...new Set(state.entries.map((e) => e.c))];
    filt.hidden = cks.length < 2;
    const mk = (label, val, gc) => {
      const b = el('button', '', label); b.type = 'button'; if (gc != null) b.prepend(glyph(gc, 11), ' '); b.setAttribute('aria-pressed', state.ck === val);
      b.addEventListener('click', () => { state.ck = val; filt.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); render(null); });
      filt.append(b);
    };
    mk('all', null); cks.forEach((c) => mk(`${NAMES[c]} (${state.entries.filter((e) => e.c === c).length})`, c, c));
  }

  function openList({ kicker, title, meta, entries, expand }) {
    state = { entries, ck: null };
    $('#dr-kicker').textContent = kicker; $('#dr-title').textContent = title; $('#dr-meta').textContent = meta || '';
    setFilters(); render(expand || null); show();
  }

  function entriesFor(pts) {
    const entries = pts.map((p) => ({ c: NAMES.indexOf(p.ckpt), s: SAMPLE.get(NAMES.indexOf(p.ckpt) + '|' + p.item), ring: p.mislabelled ? MODE_LABEL[p.mode] : null }));
    entries.sort((a, b) => (b.ring ? 1 : 0) - (a.ring ? 1 : 0) || a.c - b.c);
    return entries;
  }
  function openCluster(cl, expandKey) {
    const ring = cl.disagree ? ` · ${cl.disagree} ringed (embedding looks like another mode)` : '';
    openList({
      kicker: `Cluster inside ${MODE_LABEL[cl.label]}`, title: cl.name,
      meta: `${cl.size} samples · mostly ${cl.task} (${Math.round(cl.task_share * 100)}%) · mode set by the rule detectors${ring}`,
      entries: entriesFor(D.clusters.points.filter((p) => p.cluster === cl.id)), expand: expandKey,
    });
  }
  function openIsland(isl) {
    openList({
      kicker: 'Failure mode', title: MODE_LABEL[isl.mode],
      meta: `${isl.n} failing samples · ${isl.subs.length} clusters inside · mode set by the rule detectors · ${isl.disagree} ringed`,
      entries: entriesFor(D.clusters.points.filter((p) => p.mode === isl.mode)),
    });
  }

  function openSamples(kicker, title, meta, pairs) {
    openList({ kicker, title, meta, entries: pairs.map(([c, id]) => ({ c, s: SAMPLE.get(c + '|' + id) })) });
  }

  function init() {
    $('#dr-close').addEventListener('click', close);
    scrim.addEventListener('click', close);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
    // swipe the handle or header down to dismiss on phones
    let y0 = null;
    const grab = $('#grab'), head = $('.dr-head');
    [grab, head].forEach((n) => {
      n.addEventListener('pointerdown', (e) => { if (wide() || e.target.closest('button')) return; y0 = e.clientY; n.setPointerCapture(e.pointerId); });
      n.addEventListener('pointermove', (e) => { if (y0 == null) return; const dy = Math.max(0, e.clientY - y0); root.style.transform = `translateY(${dy}px)`; });
      n.addEventListener('pointerup', (e) => { if (y0 == null) return; const dy = e.clientY - y0; y0 = null; root.style.transform = ''; if (dy > 90) close(); });
      n.addEventListener('pointercancel', () => { y0 = null; root.style.transform = ''; });
    });
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab' || !open) return;
      const f = [...root.querySelectorAll('button')].filter((b) => !b.closest('[hidden]'));
      if (!f.length) return;
      const first = f[0], lastEl = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastEl.focus(); }
      else if (!e.shiftKey && document.activeElement === lastEl) { e.preventDefault(); first.focus(); }
    });
  }
  return { init, openCluster, openIsland, openSamples, close, isOpen: () => open };
})();

/* ------------------------------------------------------------------ panels */
const letter = (i) => String.fromCharCode(65 + i);

function devText() {
  const d = Object.entries(D.meta.devices || {}).filter(([k]) => !k.startsWith('judge:')).map(([, v]) => `${v.device} ${v.dtype}`);
  const u = [...new Set(d)];
  return u.length ? u.join(' / ') : 'device not recorded';
}

function renderBanner() {
  const m = D.meta;
  $('#banner-text').replaceChildren(
    ...[['Recorded run', true], [` · ${m.created} · ${m.n_items} prompts × ${C} checkpoints · suite v${m.suite_version} ${m.suite_hash} · judge ${m.judge.replace('local:', '')} · ${devText()}${m.commit ? ' · code ' + m.commit : ''}`, false]]
      .map(([t, b]) => (b ? el('b', '', t) : document.createTextNode(t))),
  );
}

function renderChips() {
  const ul = $('#cluster-chips'), cl = D.clusters;
  cl.islands.forEach((isl) => {
    const g = el('li', 'grp'), dot = el('i'); dot.style.background = MODE_COLOR[isl.mode];
    const gb = el('button', 'grpbtn'); gb.type = 'button'; gb.append(dot, `${MODE_LABEL[isl.mode]} · ${isl.n}`);
    gb.addEventListener('click', () => Drawer.openIsland(isl)); g.append(gb); ul.append(g);
    cl.clusters.filter((c) => isl.subs.includes(c.id)).sort((a, b) => b.size - a.size).forEach((c) => {
      const li = el('li'), b = el('button'); b.type = 'button';
      b.append(el('span', '', c.name), el('small', '', `n=${c.size}${c.disagree ? ' · ' + c.disagree + ' ringed' : ''}`));
      b.addEventListener('click', () => Drawer.openCluster(c));
      li.append(b); ul.append(li);
    });
  });
  const s = cl.stats, box = $('#cluster-stats');
  box.replaceChildren(el('p', 'note mono', `${s.n_failures} failing samples. The mode of each sample comes from deterministic detectors, never from the embedding. ` +
    `The embedding is compared with those modes below (k chosen by silhouette); ringed dots are samples whose mode differs from the label of their embedding cluster (${s.mislabelled}). ` +
    `Inside each island the samples are grouped by embedding and named by their top task and distinctive words. The raw view shows the unsupervised projection with no grouping.`));
  const tbl = el('table', 'rep mono'), head = el('tr');
  ['representation', 'k', 'silhouette', 'purity', 'AMI vs mode', 'AMI vs task', 'ringed'].forEach((h) => head.append(el('th', '', h)));
  tbl.append(head);
  const nm = { plain: 'plain embedding', task_centred: 'task-centred embedding', failure_text: 'failure description + centred embedding' };
  Object.entries(s.representations).forEach(([k, r]) => {
    const tr = el('tr', k === s.chosen ? 'used' : '');
    [nm[k] + (k === s.chosen ? ' (used)' : ''), r.k, r.silhouette.toFixed(3), r.purity.toFixed(3), r.ami_vs_mode.toFixed(3), r.ami_vs_task.toFixed(3), r.mislabelled].forEach((v) => tr.append(el('td', '', String(v))));
    tbl.append(tr);
  });
  const sc = el('div', 'scroll'); sc.append(tbl); box.append(sc);
}

function heat(a) { return `rgba(215,255,63,${(0.06 + 0.5 * a).toFixed(3)})`; }

function renderPass() {
  const rows = new Map(D.pass_rates.map((r) => [r.ckpt + '|' + r.task, r]));
  const tbl = el('table', 'heat'), thead = el('tr'), cols = [...TASKS, 'ALL'];
  thead.append(el('th', 'row', ''));
  cols.forEach((t) => thead.append(el('th', '', TASK_SHORT[t])));
  tbl.append(thead);
  NAMES.forEach((n) => {
    const tr = el('tr'); tr.append(el('th', 'row', n));
    cols.forEach((t) => {
      const r = rows.get(n + '|' + t), td = el('td', '', r ? Math.round(r.rate * 100) + '' : '–');
      if (r) { td.style.background = heat(r.rate); td.title = `${r.passed}/${r.n} · 95% CI ${pct(r.ci_lo, 0)} to ${pct(r.ci_hi, 0)}`; td.setAttribute('aria-label', `${n} ${t}: ${r.passed} of ${r.n}`); if (t === 'ALL') td.style.fontWeight = 600; }
      tr.append(td);
    });
    tbl.append(tr);
  });
  $('#pass-table').append(tbl);
}

function renderWin() {
  const m = D.win_matrix, tbl = el('table', 'heat'), head = el('tr');
  head.append(el('th', 'row', ''));
  NAMES.forEach((_, i) => head.append(el('th', '', letter(i))));
  tbl.append(head);
  const note = $('#win-note');
  NAMES.forEach((a, i) => {
    const tr = el('tr'); tr.append(el('th', 'row', `${letter(i)} ${a}`));
    NAMES.forEach((b) => {
      const td = el('td');
      if (a === b) { td.className = 'na'; td.textContent = '·'; }
      else {
        const c = m.cells[a][b], v = c.win_rate, btn = el('button', '', v.toFixed(2)); btn.type = 'button';
        td.style.background = v >= 0.5 ? `rgba(215,255,63,${(0.06 + (v - 0.5) * 1.4).toFixed(3)})` : `rgba(255,122,61,${(0.06 + (0.5 - v) * 1.4).toFixed(3)})`;
        btn.setAttribute('aria-label', `${a} against ${b}: ${c.wins} wins, ${c.losses} losses, ${c.ties} ties`);
        btn.addEventListener('click', () => { note.textContent = `${a} vs ${b}: ${c.wins} wins, ${c.losses} losses, ${c.ties} ties (win-rate ${v.toFixed(3)}; ${c.wins + c.losses ? pct(c.decisive, 0) + ' of decided items' : 'no decided items'}).`; });
        td.append(btn);
      }
      tr.append(td);
    });
    tbl.append(tr);
  });
  $('#win-table').append(tbl);
  const box = $('#field-win'), best = Math.max(...NAMES.map((n) => D.field_win[n].win_rate));
  NAMES.forEach((n, i) => {
    const w = D.field_win[n], row = el('div', 'fw' + (w.win_rate === best ? ' best' : ''));
    row.append(el('span', '', `${letter(i)} ${n}`));
    const tr = el('div', 'track'), fill = el('i', 'fill'), ci = el('i', 'ci');
    fill.style.width = pct(w.win_rate, 1); ci.style.left = pct(w.ci_lo, 1); ci.style.width = pct(w.ci_hi - w.ci_lo, 1);
    tr.append(fill, ci); row.append(tr, el('span', '', `${w.win_rate.toFixed(3)}`));
    row.title = `win-rate against the field, bootstrap 95% CI ${w.ci_lo.toFixed(3)} to ${w.ci_hi.toFixed(3)}`;
    box.append(row);
  });
  box.append(el('p', 'note mono', 'Against the whole field, with bootstrap 95% interval over prompts.'));
}

function renderModes() {
  const box = $('#mode-bars'), max = Math.max(...NAMES.map((n) => MODES.reduce((a, m) => a + (D.mode_table[n][m] || 0), 0)));
  NAMES.forEach((n) => {
    const row = el('div', 'mrow'), stack = el('div', 'stack'), tot = MODES.reduce((a, m) => a + (D.mode_table[n][m] || 0), 0);
    row.append(el('span', '', n), stack);
    MODES.forEach((m) => {
      const v = D.mode_table[n][m] || 0; if (!v) return;
      const seg = el('div', 'seg-bar'); seg.style.background = MODE_COLOR[m]; seg.style.width = '0%'; seg.title = `${MODE_LABEL[m]}: ${v}`;
      stack.append(seg); requestAnimationFrame(() => requestAnimationFrame(() => { seg.style.width = (v / max) * 100 + '%'; }));
    });
    row.append(el('span', 'mnums', `${tot} failures · ` + MODES.filter((m) => D.mode_table[n][m]).map((m) => `${MODE_LABEL[m].toLowerCase()} ${D.mode_table[n][m]}`).join(' · ')));
    box.append(row);
  });
  const key = el('div', 'mkey');
  MODES.forEach((m) => { const s = el('span'); const i = el('i'); i.style.background = MODE_COLOR[m]; s.append(i, MODE_LABEL[m]); key.append(s); });
  box.append(key);
  const cap = D.meta.cap_hits;
  if (cap) {
    const p = el('p', 'note mono', 'Ran to the token cap, of which loops (counted as degenerate) and plain budget exhaustion: ' +
      NAMES.map((n) => `${n} ${cap[n].hits}/${cap[n].n} (${cap[n].loops} loops, ${cap[n].budget_only} budget only, ${pct(cap[n].budget_rate)})`).join(' · ') + '.');
    box.append(p);
  }
}

function renderJudge() {
  const j = D.judge, box = $('#judge-card'), kv = el('div', 'kv');
  const add = (k, v, cls) => { kv.append(el('span', '', k), el('b', cls || '', v)); };
  add('Judge', D.meta.judge.replace('local:', ''));
  if (j.strict) {
    add(`Accuracy vs exact match (n=${j.strict.n})`, pct(j.strict.accuracy), j.strict.accuracy > 0.75 ? 'lime' : 'warn');
    add('Cohen kappa', j.strict.kappa.toFixed(2), j.strict.kappa >= 0.4 ? 'lime' : 'warn');
    add('Says yes / truly correct', `${pct(j.strict.judge_yes_rate, 0)} / ${pct(j.strict.gold_pass_rate, 0)}`);
  }
  if (j.lexicon_agreement) add(`Agreement with lexicon (n=${j.lexicon_agreement.n})`, `${pct(j.lexicon_agreement.accuracy, 0)} · k ${j.lexicon_agreement.kappa.toFixed(2)}`);
  if (j.pairwise) {
    add(`Pairwise accuracy, A/B order (n=${j.pairwise.n})`, `${pct(j.pairwise.accuracy_order1, 0)} / ${pct(j.pairwise.accuracy_order2, 0)}`);
    add('Choice flips when order swaps', pct(j.pairwise.flip_rate, 0), j.pairwise.flip_rate > 0.25 ? 'warn' : 'lime');
  }
  box.append(kv);
  const jc = D.judge_comparison;
  if (jc) {
    const tbl = el('table', 'rep mono'), head = el('tr');
    ['judge', 'acc vs exact', 'kappa', 'lexicon', 'pair acc (swap-avg)', 'flip rate'].forEach((h) => head.append(el('th', '', h)));
    tbl.append(head);
    Object.entries(jc.judges).forEach(([name, r]) => {
      const tr = el('tr', name === jc.best_by_strict_kappa ? 'used' : '');
      const cells = [name.replace('local:Qwen/', '') + (name === jc.best_by_strict_kappa ? ' (default)' : ''), r.strict ? pct(r.strict.accuracy, 1) : '–', r.strict ? r.strict.kappa.toFixed(2) : '–',
        r.lexicon_agreement ? pct(r.lexicon_agreement.accuracy, 0) : '–', r.pairwise ? pct(r.pairwise.accuracy_swap_averaged, 0) : '–', r.pairwise ? pct(r.pairwise.flip_rate, 0) : '–'];
      cells.forEach((v) => tr.append(el('td', '', v)));
      tbl.append(tr);
    });
    const sc = el('div', 'scroll'); sc.append(tbl); box.append(sc);
    box.append(el('p', 'note mono', 'Default judge = higher kappa against exact-match labels (chosen before the comparison was run).'));
  }
  const self = D.meta.checkpoints.some((c) => D.meta.judge.includes(c.model));
  const k = j.strict ? j.strict.kappa : 0;
  box.append(el('p', 'verdict', (k >= 0.6 ? 'Reasonable agreement on this task. ' : k >= 0.4 ? 'Moderate agreement. ' : 'Weak agreement: a judge this small is not trusted to grade on its own. ') +
    'Rules decide every item that has a rule; the judge only decides the open-ended truthfulness items and is shown here so its limits are visible.' + (self ? ' The judge model is also a checkpoint, so its own scores are biased.' : '')));
}

function renderTargets() {
  const tabs = $('#target-tabs'), list = $('#target-list'); let cur = 0;
  const draw = () => {
    list.replaceChildren(); const rows = D.targets[NAMES[cur]] || [];
    if (!rows.length) list.append(el('p', 'empty', 'No failures for this checkpoint.'));
    rows.forEach((r) => {
      const row = el('div', 'tgt'); row.append(el('span', 'n', String(r.count)), el('span', 'nm', MODE_LABEL[r.mode] || r.mode), el('span', 'tk', r.tasks.join(' · ')));
      const b = el('button', '', 'view'); b.type = 'button';
      b.addEventListener('click', () => {
        const pairs = D.samples.filter((s) => s.c === cur && !s.p && s.m === r.mode).map((s) => [cur, s.i]);
        Drawer.openSamples('Fine-tuning target', `${MODE_LABEL[r.mode]} · ${NAMES[cur]}`, `${pairs.length} failing samples tagged ${r.mode}`, pairs);
      });
      row.append(b); list.append(row);
    });
    tabs.querySelectorAll('button').forEach((x, i) => x.setAttribute('aria-selected', i === cur));
  };
  NAMES.forEach((n, i) => { const b = el('button', '', n); b.type = 'button'; b.setAttribute('role', 'tab'); b.addEventListener('click', () => { cur = i; draw(); }); tabs.append(b); });
  draw();
  $('#dl-targets').addEventListener('click', () => {
    const lines = Object.entries(D.targets).flatMap(([ck, rows]) => rows.map((r) => JSON.stringify({ checkpoint: ck, ...r })));
    const a = el('a'); a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'application/jsonl' })); a.download = 'targets.jsonl'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
}

function renderDiff() {
  const d = D.diff, card = $('#diff-card'); if (!d) { $('#diff-sec').hidden = true; return; }
  const ia = NAMES.indexOf(d.from), ib = NAMES.indexOf(d.to);
  $('#diff-title').replaceChildren(document.createTextNode('What post-training '), el('em', '', 'changed'));
  $('#diff-sub').textContent = `${d.from} is the 360M base model before post-training, ${d.to} is the same architecture after instruction tuning. Same ${d.n} prompts, counted per sample.`;
  const g = el('div', 'dgrid'), left = el('div'), right = el('div');
  const big = el('div', 'big');
  [['fx', d.fixed.length, 'fixed'], ['rg', d.regressed.length, 'regressed'], ['', d.still_failing.length, 'still failing']].forEach(([c, n, l]) => { const x = el('div', c); x.append(el('b', '', String(n)), el('span', '', l)); big.append(x); });
  left.append(big);
  const mk = (label, ids, pairsFn) => { const b = el('button', 'btn', label); b.type = 'button'; b.addEventListener('click', () => Drawer.openSamples('Post-training diff', label, `${ids.length} prompts`, pairsFn(ids))); return b; };
  const row = el('div', 'btnrow');
  row.append(mk('Show fixed', d.fixed.slice(0, 40), (ids) => ids.flatMap((i) => [[ia, i], [ib, i]])), mk('Show regressed', d.regressed.slice(0, 40), (ids) => ids.flatMap((i) => [[ia, i], [ib, i]])), mk('Show still failing', d.still_failing.slice(0, 40), (ids) => ids.map((i) => [ib, i])));
  left.append(row);
  const max = Math.max(...MODES.map((m) => Math.max(d.modes_from[m], d.modes_to[m])), 1);
  MODES.forEach((m) => {
    const r = el('div', 'dm'), bars = el('div', 'bars'), a = el('i', 'b0'), b = el('i', 'b1');
    a.style.width = (d.modes_from[m] / max) * 100 + '%'; b.style.width = (d.modes_to[m] / max) * 100 + '%';
    bars.append(a, b);
    const dl = d.modes_delta[m];
    r.append(el('span', '', MODE_LABEL[m]), bars, el('span', 'd ' + (dl > 0 ? 'up' : dl < 0 ? 'down' : ''), `${d.modes_from[m]} → ${d.modes_to[m]} (${dl > 0 ? '+' : ''}${dl})`));
    right.append(r);
  });
  right.append(el('p', 'note mono', `Grey: ${d.from}. Lime: ${d.to}. Failures per mode.`));
  g.append(left, right); card.append(g);
}

function renderTry() {
  if (!TRY) return;
  $('#try-sec').hidden = false;
  const box = $('#try-ckpts');
  NAMES.forEach((n, i) => { const l = el('label'), c = el('input'); c.type = 'checkbox'; c.value = n; c.checked = i === 2 || i === 1; l.append(c, n); box.append(l); });
  $('#try-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const cks = [...box.querySelectorAll('input:checked')].map((x) => x.value), prompt = $('#try-prompt').value.trim(), out = $('#try-out'), go = $('#try-go');
    if (!prompt || !cks.length) return;
    go.disabled = true; go.textContent = 'Running'; out.replaceChildren(el('p', 'note mono', 'Generating locally. The first call loads each model.'));
    try {
      const r = await fetch('api/try', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, checkpoints: cks }) });
      if (!r.ok) throw new Error(`server said ${r.status}`);
      const res = await r.json(), wrap = el('div', 'tryres');
      res.forEach((a) => {
        const d = el('div'), sg = a.signals, bits = [];
        if (sg.refusal_phrase) bits.push(`refusal phrase "${sg.refusal_phrase}"`);
        if (sg.abstention_phrase) bits.push(`abstains "${sg.abstention_phrase}"`);
        bits.push(sg.valid_json ? 'valid JSON' : 'not JSON', `${sg.words} words`, `${a.seconds.toFixed(1)}s`);
        d.append(el('b', '', a.ckpt), el('pre', '', a.output || '(empty output)'), el('small', '', bits.join(' · '))); wrap.append(d);
      });
      out.replaceChildren(wrap);
    } catch (err) { out.replaceChildren(el('p', 'note mono', 'Could not run: ' + err.message)); }
    go.disabled = false; go.textContent = 'Run';
  });
}

/* ------------------------------------------------------------------ boot */
(async () => {
  try {
    await load(); prepare();
  } catch (e) {
    $('#banner-text').textContent = 'No recorded run found. Run `evalbench run` and `evalbench export-demo` first.';
    return;
  }
  renderBanner(); renderChips(); renderPass(); renderWin(); renderModes(); renderJudge(); renderTargets(); renderDiff(); renderTry();
  Drawer.init(); Stage.init();
})();
