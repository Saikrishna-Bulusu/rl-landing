'use client';
import { useCallback, useEffect, useRef } from 'react';

/**
 * Lines against flight time, QGC-style, sharing one cursor and one zoom with every other chart on the page.
 *
 *   <SeriesChart t={t} lines={[{ y, color, label, dash }]} unit="deg" cursor={i} range={[i0, i1]}
 *                bands={[[i0, i1, colour]]} onHover={setCursor} onRange={setRange} />
 *
 * t: seconds (PX4's clock); y arrays match t, null for a gap; cursor and range are indices into t.
 * Drag across to zoom to that stretch, double-click to zoom back out.
 */
export default function SeriesChart({ t, lines, unit = '', cursor = null, range = null, bands = [],
                                      onHover, onRange, height = 170, yMin = null, yMax = null, t0 = 0, yLabels = null,
                                      values = true }) {
  const cv = useRef(null);
  const drag = useRef(null);
  const sel = useRef(null);

  const span = range || [0, Math.max(0, t.length - 1)];
  const L = 46, Rm = 10, T = 22, B = 20;

  const paint = useCallback(() => {
    const c = cv.current;
    if (!c || !t.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const [i0, i1] = span;
    const ta = t[i0], tb = t[i1] > ta ? t[i1] : ta + 1;
    let lo = Infinity, hi = -Infinity;
    for (const l of lines) for (let i = i0; i <= i1; i++) {
      const v = l.y[i];
      if (v != null) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (yMin != null) lo = Math.min(lo, yMin);
    if (yMax != null) hi = Math.max(hi, yMax);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = (tv) => L + (tv - ta) / (tb - ta) * (W - L - Rm);
    const Y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);
    // phase / state bands behind everything
    for (const [a, b, col] of bands) {
      if (b < i0 || a > i1) continue;
      ctx.fillStyle = col;
      ctx.fillRect(X(t[Math.max(a, i0)]), T, X(t[Math.min(b, i1)]) - X(t[Math.max(a, i0)]), H - T - B);
    }
    // grid and axes
    ctx.strokeStyle = '#26303d'; ctx.lineWidth = 1; ctx.fillStyle = '#93a1b1'; ctx.font = '10.5px ui-monospace, monospace';
    const step = nice((hi - lo) / 4);
    const ticks = yLabels ? Object.keys(yLabels).map(Number) : [];
    if (!yLabels) for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
    for (const v of ticks) {
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - Rm, Y(v)); ctx.stroke();
      ctx.fillText(yLabels ? '' : fmt(v), 4, Y(v) + 3.5);
    }
    const ts = nice((tb - ta) / 6);
    for (let v = Math.ceil((ta - t0) / ts) * ts; v <= tb - t0; v += ts) {
      ctx.beginPath(); ctx.moveTo(X(v + t0), T); ctx.lineTo(X(v + t0), H - B); ctx.stroke();
      ctx.fillText(`${fmt(v)}s`, X(v + t0) - 10, H - 5);
    }
    // the lines, with gaps where there is no data
    for (const l of lines) {
      ctx.strokeStyle = l.color; ctx.lineWidth = l.width || 1.6; ctx.setLineDash(l.dash ? [5, 4] : []);
      ctx.beginPath();
      let pen = false;
      const stride = Math.max(1, Math.floor((i1 - i0) / (W * 1.5)));
      for (let i = i0; i <= i1; i += stride) {
        const v = l.y[i];
        if (v == null) { pen = false; continue; }
        if (pen) ctx.lineTo(X(t[i]), Y(v)); else { ctx.moveTo(X(t[i]), Y(v)); pen = true; }
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // cursor, with each line's value at it
    const at = cursor != null && cursor >= i0 && cursor <= i1 ? cursor : null;
    let x = L;
    lines.forEach((l) => {
      const v = at != null && values ? l.y[at] : null;
      const txt = `${l.label}${v != null ? ` ${fmt(v)}${unit}` : ''}`;
      ctx.fillStyle = l.color; ctx.fillRect(x, 8, 10, l.dash ? 2 : 3);
      ctx.fillStyle = '#c3ccd6'; ctx.fillText(txt, x + 14, 12);
      x += ctx.measureText(txt).width + 28;
    });
    if (at != null) {
      ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(t[at]), T); ctx.lineTo(X(t[at]), H - B); ctx.stroke();
    }
    if (sel.current) {
      const [a, b] = sel.current;
      ctx.fillStyle = 'rgba(127,166,204,.18)';
      ctx.fillRect(Math.min(a, b), T, Math.abs(b - a), H - T - B);
    }
  }, [t, lines, unit, cursor, span, bands, yMin, yMax, t0, yLabels, values]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { paint(); }, [paint]);
  useEffect(() => {
    if (!cv.current || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => paint());
    ro.observe(cv.current);
    return () => ro.disconnect();
  }, [paint]);

  const idx = (e) => {
    const r = cv.current.getBoundingClientRect();
    const [i0, i1] = span;
    const ta = t[i0], tb = t[i1];
    const tv = ta + (e.clientX - r.left - L) / (r.width - L - Rm) * (tb - ta);
    let lo = i0, hi = i1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] < tv) lo = m; else hi = m; }
    return Math.max(i0, Math.min(i1, Math.abs(t[lo] - tv) < Math.abs(t[hi] - tv) ? lo : hi));
  };

  return (
    <canvas ref={cv} style={{ width: '100%', height, display: 'block', cursor: 'col-resize', touchAction: 'none' }}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); drag.current = { i: idx(e), u: e.clientX - e.currentTarget.getBoundingClientRect().left }; }}
      onPointerMove={(e) => {
        const i = idx(e);
        if (onHover) onHover(i);
        if (drag.current) { sel.current = [drag.current.u, e.clientX - e.currentTarget.getBoundingClientRect().left]; paint(); }
      }}
      onPointerUp={(e) => {
        const d = drag.current; drag.current = null; sel.current = null;
        if (!d) return;
        const i = idx(e);
        if (Math.abs(i - d.i) > 5 && onRange) onRange([Math.min(i, d.i), Math.max(i, d.i)]); else paint();
      }}
      onPointerLeave={() => { drag.current = null; sel.current = null; }}
      onDoubleClick={() => onRange && onRange(null)} />
  );
}

function nice(x) {
  if (!(x > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(x)), f = x / e;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * e;
}

function fmt(v) {
  const a = Math.abs(v);
  if (a < 1e-9) return '0';
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1).replace(/\.0$/, '') : v.toFixed(a >= 1 ? 1 : 2).replace(/\.?0+$/, '') || '0';
}
