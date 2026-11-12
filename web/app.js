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
const CK_COLOR = ['#d7ff3f', '#8fd3ff', '#ffffff', '#ff9ec4', '#c8a7ff'];
const TASKS = ['math', 'knowledge', 'format', 'truth', 'unanswerable', 'benign'];
const TASK_SHORT = { math: 'math', knowledge: 'know', format: 'format', truth: 'truth', unanswerable: 'unans', benign: 'benign', ALL: 'all' };
const RATE = 8; // replayed prompts per second at 1x
const FLIGHT = 5; // prompts of replay time a dot takes to fly to its cluster

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
      DOTS.push({ c, id, k, spawn: k + c * 0.15, p, s });
    }
  });
}

/* ------------------------------------------------------------------ stage (race + failure map) */
const Stage = (() => {
  const canvas = $('#canvas'), ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1, L = null;
  let t = 0, playing = false, speed = 1, visible = false, started = false, last = 0, dirty = true;
  let colorMode = 'fail', selected = null, axisMax = 0.6;
  const scrub = $('#scrub'), count = $('#count');

  function layout() {
    W = canvas.parentElement.clientWidth;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pad = 14, laneH = 42, lanesTop = 14;
    const lanesBottom = lanesTop + C * laneH;
    const mapTop = lanesBottom + 26;
    const mapH = W < 560 ? Math.round(W * 0.98) : Math.min(440, Math.round(W * 0.4));
    H = mapTop + mapH + 12;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.height = H + 'px';
    const mx0 = pad + 6, mx1 = W - pad - 6, my0 = mapTop + 34, my1 = mapTop + mapH - 10;
    L = { pad, laneH, lanesTop, mapTop, mapH, mx0, mx1, my0, my1, bx0: pad, bx1: W - pad };
    const maxFinal = Math.max(...NAMES.map((_, c) => D._cum[c][N] / N));
    axisMax = Math.min(1, Math.ceil((maxFinal + 0.06) * 10) / 10);
    // map geometry from the data, independent of replay time
    const proj = (p) => [lerp(mx0 + 8, mx1 - 8, (p.x + 1) / 2), lerp(my0 + 8, my1 - 8, (p.y + 1) / 2)];
    DOTS.forEach((d) => { [d.tx, d.ty] = proj(d.p); });
    L.ell = D.clusters.clusters.map((cl) => {
      const m = D.clusters.points.filter((p) => p.cluster === cl.id).map(proj);
      const mx = m.reduce((a, q) => a + q[0], 0) / m.length, my = m.reduce((a, q) => a + q[1], 0) / m.length;
      const sx = Math.sqrt(m.reduce((a, q) => a + (q[0] - mx) ** 2, 0) / m.length), sy = Math.sqrt(m.reduce((a, q) => a + (q[1] - my) ** 2, 0) / m.length);
      return { cl, x: mx, y: my, rx: Math.max(18, sx * 2.1 + 8), ry: Math.max(15, sy * 2.1 + 8) };
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
    if (colorMode === 'tag') return MODE_COLOR[d.s.m] || '#ff7a3d';
    if (colorMode === 'ckpt') return CK_COLOR[d.c % CK_COLOR.length];
    return '#ff7a3d';
  }

  function draw() {
    if (!L) return;
    const k = Math.floor(clamp(t, 0, N));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const mono = (px, w = 400) => `${w} ${px}px "JetBrains Mono", ui-monospace, monospace`;
    const rates = NAMES.map((_, c) => rateAt(c, t));
    const lead = rates.indexOf(Math.max(...rates));
    // lanes
    NAMES.forEach((name, c) => {
      const y0 = L.lanesTop + c * L.laneH, isLead = c === lead && t > 3;
      ctx.font = mono(11, isLead ? 600 : 400);
      ctx.fillStyle = isLead ? '#edece6' : 'rgba(237,236,230,.62)'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(name, L.bx0, y0 + 11);
      const w = k ? D._win[c][k] : 0;
      ctx.textAlign = 'right'; ctx.fillStyle = isLead ? '#d7ff3f' : 'rgba(237,236,230,.62)';
      ctx.fillText(`${pct(rates[c])} pass · win ${w.toFixed(2)}`, L.bx1, y0 + 11);
      ctx.fillStyle = 'rgba(237,236,230,.07)';
      rr(L.bx0, y0 + 18, L.bx1 - L.bx0, 8, 4); ctx.fill();
      const ex = barX(rates[c]);
      ctx.fillStyle = isLead ? '#d7ff3f' : 'rgba(237,236,230,.38)';
      rr(L.bx0, y0 + 18, Math.max(8, ex - L.bx0), 8, 4); ctx.fill();
      // interval whisker (Wilson, on what has been replayed so far)
      if (k >= 4) {
        const [lo, hi] = wilson(D._cum[c][k], k);
        ctx.strokeStyle = 'rgba(237,236,230,.8)'; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(barX(lo), y0 + 22); ctx.lineTo(barX(hi), y0 + 22);
        ctx.moveTo(barX(lo), y0 + 19); ctx.lineTo(barX(lo), y0 + 25); ctx.moveTo(barX(hi), y0 + 19); ctx.lineTo(barX(hi), y0 + 25); ctx.stroke();
      }
      // verdict of the prompt being replayed right now
      if (k < N && t > 0) {
        const s = SAMPLE.get(c + '|' + ORDER[k]), fr = t - k;
        ctx.globalAlpha = 1 - fr; ctx.fillStyle = s.p ? '#d7ff3f' : '#ff7a3d';
        ctx.beginPath(); ctx.arc(Math.min(ex + 2, L.bx1 - 3), y0 + 22, 3.5, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
      }
    });
    // map frame
    ctx.textAlign = 'left'; ctx.font = mono(10.5, 500); ctx.fillStyle = 'rgba(237,236,230,.46)';
    ctx.fillText('FAILURE MAP · t-SNE of embedded failing answers', L.pad, L.mapTop + 8);
    ctx.strokeStyle = 'rgba(237,236,230,.09)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L.pad, L.mapTop - 8); ctx.lineTo(W - L.pad, L.mapTop - 8); ctx.stroke();
    // landed counts per cluster
    const landed = new Map();
    for (const d of DOTS) if (t >= d.spawn + FLIGHT) landed.set(d.p.cluster, (landed.get(d.p.cluster) || 0) + 1);
    L.ell.forEach((e) => {
      const n = landed.get(e.cl.id) || 0, a = clamp(n / Math.max(4, e.cl.size * 0.5));
      ctx.strokeStyle = `rgba(255,122,61,${0.18 + 0.45 * a})`; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.ellipse(e.x, e.y, e.rx, e.ry, 0, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      const px = W < 560 ? 9 : 10.5;
      ctx.font = mono(px, 500); ctx.textAlign = 'center';
      const lx = clamp(e.x, 52, W - 52), ly = Math.max(L.mapTop + 22, e.y - e.ry - 14);
      ctx.fillStyle = `rgba(237,236,230,${0.45 + 0.4 * a})`; ctx.fillText(e.cl.name, lx, ly);
      ctx.fillStyle = `rgba(255,122,61,${0.55 + 0.45 * a})`; ctx.font = mono(px + 1, 600); ctx.fillText(String(n), lx, ly + px + 3);
    });
    ctx.textAlign = 'left';
    // dots
    const rad = W < 560 ? 2.6 : 3;
    for (const d of DOTS) {
      if (t < d.spawn) continue;
      const f = clamp((t - d.spawn) / FLIGHT), e = easeIO(f);
      const y0 = L.lanesTop + d.c * L.laneH + 22, x0 = barX(D._cum[d.c][d.k + 1] / (d.k + 1));
      let x = lerp(x0, d.tx, e), y = lerp(y0, d.ty, e);
      const arc = Math.sin(f * Math.PI) * 26 * (d.c % 2 ? 1 : -1);
      x += arc * (1 - e) ; // gentle drift
      ctx.globalAlpha = f < 1 ? 0.95 : 0.82;
      ctx.fillStyle = dotColor(d);
      ctx.beginPath(); ctx.arc(x, y, f < 1 ? rad * 1.35 : rad, 0, 7); ctx.fill();
      if (f >= 1 && d.p.mislabelled) { ctx.globalAlpha = 1; ctx.strokeStyle = '#edece6'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, rad + 2.4, 0, 7); ctx.stroke(); }
      if (selected && selected.c === d.c && selected.id === d.id && f >= 1) {
        ctx.globalAlpha = 1; ctx.strokeStyle = '#d7ff3f'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, rad + 5, 0, 7); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (!landed.size && t < 1) {
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
      const dd = (d.tx - x) ** 2 + (d.ty - y) ** 2;
      if (dd < bd) { bd = dd; best = d; }
    }
    if (best) return { dot: best };
    for (const e of L.ell) if (((x - e.x) / (e.rx * 1.2)) ** 2 + ((y - e.y) / (e.ry * 1.2)) ** 2 <= 1) return { cluster: e.cl };
    return null;
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
    $('#replay').addEventListener('click', () => { t = 0; selected = null; setPlaying(!reduced || true); dirty = true; });
    scrub.max = N;
    scrub.addEventListener('input', () => { t = Math.min(+scrub.value, N); setPlaying(false); dirty = true; });
    document.querySelectorAll('[data-speed]').forEach((b) => b.addEventListener('click', () => {
      speed = +b.dataset.speed;
      document.querySelectorAll('[data-speed]').forEach((o) => { const on = o === b; o.classList.toggle('on', on); o.setAttribute('aria-checked', on); });
    }));
    document.querySelectorAll('[data-color]').forEach((b) => b.addEventListener('click', () => {
      colorMode = b.dataset.color;
      document.querySelectorAll('[data-color]').forEach((o) => { const on = o === b; o.classList.toggle('on', on); o.setAttribute('aria-checked', on); });
      dirty = true;
    }));
    // a click is only fired for taps, not for scroll gestures, and it fires before anything opens over the canvas
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect(), hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (!hit) { selected = null; dirty = true; return; }
      setPlaying(false);
      if (hit.dot) { selected = hit.dot; dirty = true; Drawer.openCluster(CLUSTERS.get(hit.dot.p.cluster), hit.dot.c + '|' + hit.dot.id); }
      else Drawer.openCluster(hit.cluster);
    });
    window.__evalbench = { seek: (v) => { t = v; setPlaying(false); dirty = true; draw(); }, state: () => ({ t, playing, N, FLIGHT }), pickAt: (x, y) => pick(x, y), dot: (i) => (DOTS[i] ? { x: DOTS[i].tx, y: DOTS[i].ty } : null) };
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
    tags.append(el('span', '', NAMES[c]), el('span', '', it.task));
    tags.append(el('span', s.p ? 'ok' : 'mode', s.p ? 'pass' : MODE_LABEL[s.m] || 'fail'));
    if (entry.ring) tags.append(el('span', 'ring', `tag ≠ cluster (${entry.ring})`));
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
    const mk = (label, val) => {
      const b = el('button', '', label); b.type = 'button'; b.setAttribute('aria-pressed', state.ck === val);
      b.addEventListener('click', () => { state.ck = val; filt.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); render(null); });
      filt.append(b);
    };
    mk('all', null); cks.forEach((c) => mk(`${NAMES[c]} (${state.entries.filter((e) => e.c === c).length})`, c));
  }

  function openList({ kicker, title, meta, entries, expand }) {
    state = { entries, ck: null };
    $('#dr-kicker').textContent = kicker; $('#dr-title').textContent = title; $('#dr-meta').textContent = meta || '';
    setFilters(); render(expand || null); show();
  }

  function openCluster(cl, expandKey) {
    const pts = D.clusters.points.filter((p) => p.cluster === cl.id);
    const entries = pts.map((p) => ({ c: NAMES.indexOf(p.ckpt), s: SAMPLE.get(NAMES.indexOf(p.ckpt) + '|' + p.item), ring: p.mislabelled ? MODE_LABEL[p.mode] : null }));
    entries.sort((a, b) => (b.ring ? 1 : 0) - (a.ring ? 1 : 0) || a.c - b.c);
    const votes = Object.entries(cl.votes).map(([m, n]) => `${m} ${n}`).join(', ');
    openList({
      kicker: 'Failure cluster', title: cl.name,
      meta: `${cl.size} samples · purity ${cl.purity.toFixed(2)} · tags: ${votes}${cl.terms.length ? ' · terms: ' + cl.terms.join(', ') : ''}`,
      entries, expand: expandKey,
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
  return { init, openCluster, openSamples, close, isOpen: () => open };
})();

/* ------------------------------------------------------------------ panels */
const letter = (i) => String.fromCharCode(65 + i);

function renderBanner() {
  const m = D.meta;
  $('#banner-text').replaceChildren(
    ...[['Recorded run', true], [` · ${m.created} · ${m.n_items} prompts × ${C} checkpoints · suite ${m.suite_hash} · judge ${m.judge.replace('local:', '')}${m.commit ? ' · commit ' + m.commit : ''}`, false]]
      .map(([t, b]) => (b ? el('b', '', t) : document.createTextNode(t))),
  );
}

function renderChips() {
  const ul = $('#cluster-chips');
  D.clusters.clusters.slice().sort((a, b) => b.size - a.size).forEach((cl) => {
    const li = el('li'), b = el('button'); b.type = 'button';
    b.append(el('span', '', cl.name), el('small', '', `n=${cl.size} · purity ${cl.purity.toFixed(2)}`));
    b.addEventListener('click', () => Drawer.openCluster(cl));
    li.append(b); ul.append(li);
  });
  const s = D.clusters.stats;
  $('#cluster-stats').textContent = `${s.n_failures} failing samples, k=${s.k} chosen by silhouette (${s.silhouette}). Overall purity ${s.purity}; ${s.mislabelled} samples carry a tag that disagrees with their cluster. ` +
    `Adjusted mutual information of clusters vs failure tag ${s.ami_vs_mode}, vs task ${s.ami_vs_task}; tag vs task alone ${s.ami_mode_vs_task}.`;
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
    go.disabled = true; go.textContent = 'Running'; out.replaceChildren(el('p', 'note mono', 'Generating on CPU. The first call loads each model.'));
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
