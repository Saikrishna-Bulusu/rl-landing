'use client';
import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * The sim's floor as a map, in Gazebo metres (x east, y north; north up), with pan and zoom.
 *
 *   <FlightMap extent={[x0, x1, y0, y1]} draw={(ctx, px, k) => ...} fit={[[x, y], ...]} />
 *
 * draw gets the canvas context, px([x, y]) -> [u, v] in CSS pixels, and k = pixels per metre.
 * onPick(x, y, tol) is asked on pointer-down: return true to take the drag (onDrag(x, y) follows,
 * then onDrop()), false to pan. onClick(x, y, right, tol) fires on a click that did not drag (right: the
 * context-menu button); tol is 9 pixels in metres.
 * The basemap is /fl/basemap.jpg: Esri Wayback 2023, the sim's own floor -- internal use only.
 */
export default function FlightMap({ extent, draw, fit, onPick, onDrag, onDrop, onClick, onHover,
                                    height = 520, redraw }) {
  const box = useRef(null);
  const cv = useRef(null);
  const img = useRef(null);
  const drag = useRef(null);
  const [view, setView] = useState(null);                 // {cx, cy, k}: centre (m), pixels per metre
  const [, setLoaded] = useState(false);

  useEffect(() => {
    const im = new Image();
    im.onload = () => { img.current = im; setLoaded(true); };
    im.src = '/fl/basemap.jpg';
  }, []);

  const fitView = useCallback((pts) => {
    const el = box.current;
    if (!el || !extent || el.clientWidth < 10 || el.clientHeight < 10) return;   // not laid out yet: the observer refits
    const [x0, x1, y0, y1] = extent;
    const P = pts && pts.length ? pts : [[x0, y0], [x1, y1]];
    const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
    const w = Math.max(40, Math.max(...xs) - Math.min(...xs)), h = Math.max(40, Math.max(...ys) - Math.min(...ys));
    const k = Math.min(el.clientWidth / (w * 1.15), el.clientHeight / (h * 1.15));
    setView({ cx: (Math.max(...xs) + Math.min(...xs)) / 2, cy: (Math.max(...ys) + Math.min(...ys)) / 2, k });
  }, [extent]);

  const fitKey = fit ? JSON.stringify(fit.length ? [fit[0], fit[fit.length - 1], fit.length] : []) : '';
  useEffect(() => { fitView(fit); }, [fitKey, fitView]); // eslint-disable-line react-hooks/exhaustive-deps

  const paint = useCallback(() => {
    const c = cv.current, el = box.current;
    if (!c || !el || !view || !extent) return;
    const dpr = window.devicePixelRatio || 1;
    const W = el.clientWidth, H = el.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    const px = (p) => [W / 2 + (p[0] - view.cx) * view.k, H / 2 - (p[1] - view.cy) * view.k];
    const [x0, x1, y0, y1] = extent;
    if (img.current) {
      const [u0, v0] = px([x0, y1]);
      ctx.globalAlpha = 0.8;
      ctx.drawImage(img.current, u0, v0, (x1 - x0) * view.k, (y1 - y0) * view.k);
      ctx.globalAlpha = 1;
    }
    // a 50 m scale bar, bottom left
    const bar = 50 * view.k;
    ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(12, H - 14); ctx.lineTo(12 + bar, H - 14); ctx.stroke();
    ctx.fillStyle = '#e6edf3'; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('50 m', 16 + bar, H - 10);
    // north
    ctx.fillText('N ↑', W - 36, 18);
    if (draw) draw(ctx, px, view.k);
  }, [view, extent, draw]);

  useEffect(() => { paint(); }, [paint, redraw]);
  useEffect(() => {
    if (!box.current || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => { if (!view) fitView(fit); else paint(); });
    ro.observe(box.current);
    return () => ro.disconnect();
  }, [paint, view, fitView, fit]);

  const toM = (e) => {
    const r = cv.current.getBoundingClientRect();
    const u = e.clientX - r.left, v = e.clientY - r.top;
    return [view.cx + (u - r.width / 2) / view.k, view.cy - (v - r.height / 2) / view.k];
  };

  return (
    <div ref={box} style={{ position: 'relative', height, background: '#000', touchAction: 'none' }}>
      <canvas ref={cv} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair' }}
        onPointerDown={(e) => {
          if (!view) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          const [x, y] = toM(e);
          const took = e.button === 0 && onPick ? onPick(x, y, 9 / view.k) : false;
          drag.current = { took, u: e.clientX, v: e.clientY, cx: view.cx, cy: view.cy, moved: false };
        }}
        onPointerMove={(e) => {
          if (!view) return;
          const d = drag.current;
          if (!d) { if (onHover) onHover(...toM(e)); return; }
          if (Math.abs(e.clientX - d.u) + Math.abs(e.clientY - d.v) > 3) d.moved = true;
          if (d.took) { onDrag && onDrag(...toM(e)); return; }
          setView((v) => ({ ...v, cx: d.cx - (e.clientX - d.u) / v.k, cy: d.cy + (e.clientY - d.v) / v.k }));
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          if (d.took) { onDrop && onDrop(); if (d.moved) return; }
          if (!d.moved && e.button === 0 && onClick && !d.took) onClick(...toM(e), false, 9 / view.k);
        }}
        onContextMenu={(e) => { e.preventDefault(); if (onClick && view) onClick(...toM(e), true, 9 / view.k); }}
        onWheel={(e) => {
          if (!view) return;
          const [x, y] = toM(e);
          const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
          setView((v) => {
            const k = Math.max(0.3, Math.min(40, v.k * f));
            return { k, cx: x - (x - v.cx) * (v.k / k), cy: y - (y - v.cy) * (v.k / k) };
          });
        }} />
      <div style={{ position: 'absolute', top: 8, left: 8, display: 'flex', gap: 5 }}>
        <button className="btn sm icon" onClick={() => setView((v) => v && ({ ...v, k: Math.min(40, v.k * 1.3) }))}>+</button>
        <button className="btn sm icon" onClick={() => setView((v) => v && ({ ...v, k: Math.max(0.3, v.k / 1.3) }))}>&minus;</button>
        <button className="btn sm" onClick={() => fitView(fit)}>Fit</button>
        <button className="btn sm" onClick={() => fitView(null)}>Whole map</button>
      </div>
    </div>
  );
}
