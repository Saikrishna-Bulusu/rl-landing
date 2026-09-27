'use client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import FlightMap from '@/components/FlightMap';
import Nav from '@/components/Nav';
import SeriesChart from '@/components/SeriesChart';

// Colours: PX4's belief blue (the GeoAnchor page's "predicted"), Gazebo's truth white, fixes as in the films.
const C = { est: '#7fa6cc', truth: '#e6edf3', sp: '#d29922', ga: '#3fb950', marker: '#e3b341', gnss: '#93a1b1',
            roll: '#7fa6cc', pitch: '#f0883e', err: '#f85149', rng: '#a371f7' };

const LIVE_FIT = [[-60, -170], [300, 110]];
const phaseTint = (p) => (/climb/.test(p) ? 'rgba(127,166,204,.07)'
  : /descent|drop/.test(p) ? 'rgba(210,153,34,.09)'
  : /landing|planner|landed|rule|learner/.test(p) ? 'rgba(63,185,80,.08)' : null);

function Panel({ title, meta, right, children, className = '', style }) {
  return (
    <section className={`panel ${className}`} style={style}>
      <span className="marks" aria-hidden="true" />
      <header><h2>{title}</h2>{meta && <span className="meta">{meta}</span>}<span className="spacer" />{right}</header>
      {children}
    </section>
  );
}

/** Live rows (geo_cruise's live.jsonl, one per control step) as the same series a recorded flight has. */
function liveSeries(rows) {
  const s = { t: [], x_est: [], y_est: [], h_est: [], x_true: [], y_true: [], h_true: [], roll: [], pitch: [],
              yaw: [], tilt: [], v_h: [], v_down: [], err: [], gnss: [], agp: [] };
  for (const r of rows) {
    s.t.push(r.t); s.x_est.push(r.est[0]); s.y_est.push(r.est[1]); s.h_est.push(r.est[2]);
    s.x_true.push(r.gz ? r.gz[0] : null); s.y_true.push(r.gz ? r.gz[1] : null); s.h_true.push(r.gz ? r.gz[2] : null);
    s.roll.push(r.roll); s.pitch.push(r.pitch); s.yaw.push(r.yaw);
    s.tilt.push(Math.hypot(r.roll, r.pitch));
    s.v_h.push(Math.hypot(r.vel[0], r.vel[1])); s.v_down.push(-r.vel[2]);
    s.err.push(r.gz ? Math.hypot(r.est[0] - r.gz[0], r.est[1] - r.gz[1]) : null);
    s.gnss.push(r.gnss ? 1 : 0); s.agp.push(r.agp ? 1 : 0);
  }
  const phases = [];
  for (const r of rows) {
    if (!phases.length || phases.at(-1)[2] !== r.phase) phases.push([r.t, r.t, r.phase]);
    phases.at(-1)[1] = r.t;
  }
  return { series: s, phases, events: [], fixes: [], live: true };
}

export default function FlightLog() {
  const [flights, setFlights] = useState([]);
  const [name, setName] = useState(null);
  const [doc, setDoc] = useState(null);
  const [world, setWorld] = useState(null);
  const [cursor, setCursor] = useState(null);
  const [range, setRange] = useState(null);
  const [live, setLive] = useState(false);
  const [err, setErr] = useState(null);
  const rows = useRef([]);

  useEffect(() => {
    fetch('/fl/flights').then((r) => r.json()).then((f) => {
      setFlights(f);
      const q = new URLSearchParams(window.location.search).get('f');
      setName(q || f.find((x) => x.ulog)?.name || f[0]?.name || null);
    }).catch(() => setErr('no flight server -- run python3 dashboard/flight_server.py'));
    fetch('/fl/world').then((r) => r.json()).then(setWorld).catch(() => {});
  }, []);

  useEffect(() => {
    if (!name || live) return;
    setDoc(null); setCursor(null); setRange(null);
    fetch(`/fl/flight/${name}`).then((r) => r.json()).then((d) => (d.error ? setErr(d.error) : setDoc(d)));
  }, [name, live]);

  // Live: the flight in the air, as it is flown (a row per control step, thinned to ~10 Hz here)
  useEffect(() => {
    if (!live) return undefined;
    rows.current = [];
    setDoc(null); setRange(null);
    const es = new EventSource('/fl/live');
    let last = 0, tLast = -1;
    es.addEventListener('flight', (e) => { rows.current = []; setName(JSON.parse(e.data)); });
    es.onmessage = (e) => {
      const r = JSON.parse(e.data);
      if (r.t - tLast < 0.09) return;
      tLast = r.t;
      rows.current.push(r);
      if (performance.now() - last > 400) { last = performance.now(); setDoc(liveSeries(rows.current)); }
    };
    return () => es.close();
  }, [live]);

  const s = doc?.series;
  const t = s?.t || [];
  const t0 = t[0] || 0;
  const at = cursor ?? (t.length ? t.length - 1 : null);

  const bands = useMemo(() => {
    if (!doc) return [];
    const ix = (tv) => { let i = 0; while (i < t.length - 1 && t[i] < tv) i++; return i; };
    return doc.phases.map(([a, b, p]) => [ix(a), ix(b), phaseTint(p)]).filter((b) => b[2]);
  }, [doc]); // eslint-disable-line react-hooks/exhaustive-deps

  const phaseAt = useMemo(() => {
    if (!doc || at == null) return '';
    const tv = t[at];
    const p = doc.phases.find(([a, b]) => tv >= a && tv <= b + 0.2);
    return p ? p[2] : '';
  }, [doc, at]); // eslint-disable-line react-hooks/exhaustive-deps

  // recorded: framed on its track; live: on the route area, once (re-framing on every new row fought the zoom)
  const fit = useMemo(() => (live ? LIVE_FIT
    : s ? s.x_est.map((x, i) => [x, s.y_est[i]]).filter((p) => p[0] != null) : null), [s, live]);

  const drawMap = useCallback((ctx, px, k) => {
    if (world) {
      for (const p of world.pads) {                          // every pad, ours green, its marker ids
        const [u, v] = px(p.enu);
        ctx.fillStyle = p.ours ? '#3fb950' : '#f85149';
        ctx.fillRect(u - 3, v - 3, 6, 6);
        if (k > 1.2) { ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = '#e6edf3'; ctx.fillText(p.ids.join(','), u + 5, v - 4); }
      }
    }
    if (!s) return;
    const line = (xs, ys, col, w, dash) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash ? [4, 4] : []);
      ctx.beginPath(); let pen = false;
      for (let i = 0; i < xs.length; i += 2) {
        if (xs[i] == null) { pen = false; continue; }
        const [u, v] = px([xs[i], ys[i]]);
        if (pen) ctx.lineTo(u, v); else { ctx.moveTo(u, v); pen = true; }
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    for (const [, x, y, src] of doc.fixes || []) {
      const [u, v] = px([x, y]);
      ctx.fillStyle = src === 'marker' ? C.marker : C.ga;
      ctx.fillRect(u - 1.5, v - 1.5, 3, 3);
    }
    if (s.x_true) line(s.x_true, s.y_true, C.truth, 2);
    line(s.x_est, s.y_est, C.est, 1.6, true);
    if (at != null) {                                        // PX4's belief as an arrow along its heading, the truth as a ring
      if (s.x_true && s.x_true[at] != null) {
        const [u, v] = px([s.x_true[at], s.y_true[at]]);
        ctx.strokeStyle = C.truth; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(u, v, 8, 0, 2 * Math.PI); ctx.stroke();
      }
      const [u, v] = px([s.x_est[at], s.y_est[at]]);
      const h = ((s.yaw?.[at] ?? 0) * Math.PI) / 180, dx = Math.sin(h), dy = -Math.cos(h);
      ctx.fillStyle = C.est; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(u + 13 * dx, v + 13 * dy);
      ctx.lineTo(u - 8 * dx - 8 * dy, v - 8 * dy + 8 * dx);
      ctx.lineTo(u - 3 * dx, v - 3 * dy);
      ctx.lineTo(u - 8 * dx + 8 * dy, v - 8 * dy - 8 * dx);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }, [world, s, doc, at]);

  const val = (k, d = 1) => (s && at != null && s[k] && s[k][at] != null ? Number(s[k][at]).toFixed(d) : '--');
  const nav = !s || at == null ? '--'
    : s.gnss?.[at] ? 'GNSS' : s.agp?.[at] ? (/drop|landing|landed|rule|learner|planner/.test(phaseAt) ? 'GeoAnchor + pad marker' : 'GeoAnchor')
      : s.dr?.[at] ? 'dead reckoning' : 'not aided';

  const charts = s ? [
    { title: 'Attitude: roll and pitch', unit: '°', lines: [
      { y: s.roll, color: C.roll, label: 'roll' }, { y: s.pitch, color: C.pitch, label: 'pitch' },
      ...(s.roll_sp ? [{ y: s.roll_sp, color: C.roll, label: 'roll sp', dash: true, width: 1 },
                       { y: s.pitch_sp, color: C.pitch, label: 'pitch sp', dash: true, width: 1 }] : [])] },
    { title: 'Heading (yaw)', unit: '°', yMin: 0, yMax: 360, lines: [
      { y: s.yaw, color: C.est, label: 'yaw' }, ...(s.yaw_sp ? [{ y: s.yaw_sp, color: C.sp, label: 'setpoint', dash: true, width: 1 }] : [])] },
    { title: 'Height', unit: ' m', lines: [
      { y: s.h_est, color: C.est, label: 'PX4' }, ...(s.h_true ? [{ y: s.h_true, color: C.truth, label: 'truth', width: 1 }] : [])] },
    { title: 'Speed: cruise (across) and descent (down)', unit: ' m/s', lines: [
      { y: s.v_h, color: C.est, label: 'cruise' }, { y: s.v_down, color: C.pitch, label: 'descent' },
      ...(s.v_h_sp ? [{ y: s.v_h_sp, color: C.est, label: 'cruise sp', dash: true, width: 1 },
                      { y: s.v_down_sp, color: C.pitch, label: 'descent sp', dash: true, width: 1 }] : [])] },
    { title: "PX4's position error (against Gazebo's truth)", unit: ' m', yMin: 0,
      lines: [{ y: s.err || [], color: C.err, label: 'error' }] },
    { title: 'Tilt', unit: '°', yMin: 0, lines: [{ y: s.tilt || [], color: C.pitch, label: 'tilt' }] },
  ] : [];

  const flags = s ? [
    { y: (s.gnss || []).map((v) => (v ? 4 : null)), color: C.gnss, label: 'GNSS', width: 7 },
    { y: (s.agp || []).map((v) => (v ? 3 : null)), color: C.ga, label: 'GeoAnchor (AGP)', width: 7 },
    { y: (s.rng || []).map((v) => (v ? 2 : null)), color: C.rng, label: 'rangefinder', width: 7 },
    { y: (s.dr || []).map((v) => (v ? 1 : null)), color: C.err, label: 'dead reckoning', width: 7 },
  ] : [];

  const rep = doc?.report;
  const land = doc?.landing;

  return (
    <main>
      <div className="panel masthead" style={{ flexDirection: 'row' }}>
        <span className="marks" aria-hidden="true" />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 11, minWidth: 0 }}>
          <span className="brand">GeoAnchor</span><span className="sub">flight log &mdash; PX4 in Gazebo, GNSS denied</span>
        </div>
        <span className="spacer" />
        <Nav here="/flight/" />
        <div className="field" style={{ minWidth: 220 }}>
          <select value={name || ''} disabled={live} onChange={(e) => setName(e.target.value)}
                  style={{ height: 26, background: 'var(--bg)', color: 'var(--ink)', border: '1px solid var(--line)', padding: '0 8px' }}>
            {flights.map((f) => (
              <option key={f.name} value={f.name}>{f.name} &middot; {f.when}{f.ulog ? '' : ' (no PX4 log)'}</option>
            ))}
          </select>
        </div>
        <div className="seg">
          <button className={!live ? 'on' : ''} onClick={() => setLive(false)}>Recorded</button>
          <button className={live ? 'on' : ''} onClick={() => setLive(true)}>Live</button>
        </div>
        <div className="status"><span className={`dot ${live ? 'ok' : ''}`} />{live ? `live${name ? ` · ${name}` : ' · waiting'}` : doc?.ulog ? "PX4's log + record" : 'record only'}</div>
      </div>

      {err && <div className="banner"><span>&#9888;</span><div>{err}</div></div>}

      <div className="row">
        <Panel className="grow" title="Track" meta="white: Gazebo's truth · blue dashes: PX4's belief · green: GeoAnchor fixes · yellow: pad-marker fixes · squares: pads (ours green)">
          {world && <FlightMap extent={world.extent} draw={drawMap} fit={fit} height={460} redraw={at} />}
        </Panel>
        <Panel className="side" title="At the cursor" meta={s && at != null ? `t +${(t[at] - t0).toFixed(1)} s` : ''}>
          <div className="body" style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '7px 14px', alignItems: 'baseline' }}>
            {[['phase', phaseAt || '--'], ['navigating on', nav],
              ['roll / pitch', `${val('roll')}° / ${val('pitch')}°`], ['heading', `${val('yaw', 0)}°`],
              ['height PX4 / truth', `${val('h_est')} / ${val('h_true')} m`],
              ['cruise / descent', `${val('v_h')} / ${val('v_down')} m/s`],
              ["PX4's error", `${val('err', 2)} m`], ['tilt', `${val('tilt')}°`]].map(([k, v]) => (
              <div key={k} style={{ display: 'contents' }}>
                <span className="label">{k}</span>
                <span style={{ font: '600 17px/1.2 var(--cond)', letterSpacing: '.02em' }}>{v}</span>
              </div>
            ))}
          </div>
          {(rep || land) && (
            <div className="body" style={{ borderTop: '1px solid var(--line)' }}>
              <span className="kicker">This flight</span>
              <p className="note" style={{ margin: '6px 0 0', lineHeight: 1.6 }}>
                {rep?.denied_s != null && <>GNSS denied {rep.denied_s} s, never restored: {rep.gnss_restored_at == null ? 'yes' : 'no'}.<br /></>}
                {rep?.fix_err_m && <>GeoAnchor fixes: median {rep.fix_err_m.median} m, p90 {rep.fix_err_m.p90} m, {rep.fix_within_10m_pct}% within 10 m.<br /></>}
                {rep?.est_err_m && <>PX4 after denial: median {rep.est_err_m.median} m, p90 {rep.est_err_m.p90} m.<br /></>}
                {land && <>Landing: {land.outcome}{land.miss != null ? `, ${Number(land.miss).toFixed(2)} m from centre` : ''}.<br /></>}
                {rep?.rtf != null && <>Sim speed {rep.rtf}x real time.</>}
              </p>
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Along the flight" meta="hover: cursor on every graph and the map · drag: zoom to a stretch · double-click: all"
             right={range && <button className="btn sm on" onClick={() => setRange(null)}>Reset zoom</button>}>
        {s && (
          <div style={{ padding: '6px 10px 0' }}>
            <SeriesChart t={t} t0={t0} lines={flags} cursor={at} range={range} onHover={setCursor} onRange={setRange}
                         height={70} yMin={0.5} yMax={4.5} yLabels={{ 1: '', 2: '', 3: '', 4: '' }} values={false} />
          </div>
        )}
        <div className="grid-1px">
          {charts.map((c) => (
            <div className="chart" key={c.title}>
              <div className="head"><span className="title">{c.title}</span></div>
              <div style={{ padding: '0 6px' }}>
                <SeriesChart t={t} t0={t0} lines={c.lines} unit={c.unit} cursor={at} range={range} bands={bands}
                             onHover={setCursor} onRange={setRange} yMin={c.yMin ?? null} yMax={c.yMax ?? null} />
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <p className="foot">
        Attitude, setpoints, speeds and EKF2&rsquo;s fusion flags come from PX4&rsquo;s own log (the .ulg QGC reads), kept
        with each flight since 27 Sept; the truth, phases and fixes from the flight&rsquo;s GeoAnchor record. Shaded:
        climb blue, descent amber, landing green. The basemap is Esri Wayback imagery &mdash; internal use only.
      </p>
    </main>
  );
}
