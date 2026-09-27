'use client';
import { useCallback, useEffect, useMemo, useState } from 'react';
import FlightMap from '@/components/FlightMap';
import Nav from '@/components/Nav';

const LANDERS = [['rule', 'the rule (lands on the marker)'], ['flying learner', 'flying learner (wind)'],
                 ['none', 'none: hold 20 s over the pad at 30 m']];
const input = { height: 26, background: 'var(--bg)', color: 'var(--ink)', border: '1px solid var(--line)',
                padding: '0 8px', font: '13px var(--mono)', minWidth: 0 };

function Panel({ title, meta, right, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      <span className="marks" aria-hidden="true" />
      <header><h2>{title}</h2>{meta && <span className="meta">{meta}</span>}<span className="spacer" />{right}</header>
      {children}
    </section>
  );
}

/** The turn at each waypoint, degrees (0 = straight on). */
function turns(pts) {
  const out = [];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = [pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]];
    const b = [pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]];
    const c = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b) || 1);
    out.push((Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI);
  }
  return out;
}

export default function MissionPlanner() {
  const [world, setWorld] = useState(null);
  const [saved, setSaved] = useState([]);
  const [m, setM] = useState({ name: 'to_our_pad', route: [], pad: 'arucotag', alt: 75, speed: 7, lander: 'rule',
                               wind: 0, deny_gnss: true });
  const [grab, setGrab] = useState(null);
  const [msg, setMsg] = useState(null);

  const loadSaved = () => fetch('/fl/missions').then((r) => r.json()).then(setSaved).catch(() => {});
  useEffect(() => {
    fetch('/fl/world').then((r) => r.json()).then((w) => {
      setWorld(w);
      setM((x) => ({ ...x, route: w.route, alt: w.alt, speed: w.speed }));
    }).catch(() => setMsg(['bad', 'no flight server -- run python3 dashboard/flight_server.py']));
    loadSaved();
  }, []);

  const pad = world?.pads.find((p) => p.name === m.pad);
  const pts = useMemo(() => (world && pad ? [world.depot, ...m.route, pad.enu] : []), [world, pad, m.route]);
  const tn = useMemo(() => turns(pts), [pts]);
  const length = pts.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]), 0);

  const draw = useCallback((ctx, px, k) => {
    if (!world) return;
    ctx.font = '10.5px ui-monospace, monospace';
    for (const p of world.pads) {
      const [u, v] = px(p.enu);
      const sel = p.name === m.pad;
      ctx.fillStyle = p.ours ? '#3fb950' : '#f85149';
      ctx.fillRect(u - 4, v - 4, 8, 8);
      if (sel) { ctx.strokeStyle = '#e3b341'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(u, v, 11, 0, 2 * Math.PI); ctx.stroke(); }
      if (k > 0.9 || sel) {
        ctx.fillStyle = sel ? '#e3b341' : '#e6edf3';
        ctx.fillText(`${p.ours ? 'ours ' : ''}${p.ids.join(',')}`, u + 7, v - 6);
      }
    }
    const [du, dv] = px(world.depot);
    ctx.fillStyle = '#e6edf3'; ctx.fillRect(du - 5, dv - 5, 10, 10); ctx.fillText('depot', du + 8, dv + 4);
    if (pts.length > 1) {
      ctx.strokeStyle = '#7fa6cc'; ctx.lineWidth = 2.5; ctx.beginPath();
      pts.forEach((p, i) => { const [u, v] = px(p); if (i) ctx.lineTo(u, v); else ctx.moveTo(u, v); });
      ctx.stroke();
      for (let i = 1; i < pts.length; i++) {                  // direction chevrons mid-leg
        const a = px(pts[i - 1]), b = px(pts[i]);
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        ctx.fillStyle = '#7fa6cc'; ctx.beginPath();
        ctx.moveTo(mx + 7 * Math.cos(ang), my + 7 * Math.sin(ang));
        ctx.lineTo(mx + 6 * Math.cos(ang + 2.5), my + 6 * Math.sin(ang + 2.5));
        ctx.lineTo(mx + 6 * Math.cos(ang - 2.5), my + 6 * Math.sin(ang - 2.5));
        ctx.fill();
      }
    }
    m.route.forEach((p, i) => {
      const [u, v] = px(p);
      const sharp = tn[i] > 120;
      ctx.fillStyle = sharp ? '#d29922' : '#7fa6cc'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(u, v, 9, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#0d1117'; ctx.font = '600 11px ui-monospace, monospace';
      ctx.fillText(String(i + 1), u - (i + 1 >= 10 ? 7 : 3.5), v + 4);
      ctx.font = '10.5px ui-monospace, monospace';
    });
  }, [world, m.pad, m.route, pts, tn]);

  const near = (x, y, tol) => {
    const i = m.route.findIndex((p) => Math.hypot(p[0] - x, p[1] - y) < tol * 1.3);
    return i;
  };

  const onPick = (x, y, tol) => {
    const i = near(x, y, tol);
    if (i >= 0) { setGrab(i); return true; }
    return false;
  };
  const onDrag = (x, y) => {
    if (grab == null) return;
    setM((q) => ({ ...q, route: q.route.map((p, j) => (j === grab ? [Math.round(x), Math.round(y)] : p)) }));
  };
  const onClick = (x, y, right, tol) => {
    const i = near(x, y, tol);
    if (right) { if (i >= 0) setM((q) => ({ ...q, route: q.route.filter((_, j) => j !== i) })); return; }
    const p = world?.pads.find((q) => Math.hypot(q.enu[0] - x, q.enu[1] - y) < Math.max(4, tol));
    if (p) { setM((q) => ({ ...q, pad: p.name })); return; }
    if (m.route.length >= 12) { setMsg(['warn', 'at most 12 waypoints']); return; }
    setM((q) => ({ ...q, route: [...q.route, [Math.round(x), Math.round(y)]] }));
  };

  const post = (path, body) => fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                             body: JSON.stringify(body) }).then(async (r) => [r.ok, await r.json()]);
  const save = () => post('/fl/missions', m).then(([ok, r]) => {
    setMsg(ok ? ['ok', `saved geo/missions/${r.name}.json`] : ['bad', r.error]);
    if (ok) loadSaved();
    return ok;
  });
  const fly = () => save().then((ok) => ok && post('/fl/fly', { mission: m.name }).then(([ok2, r]) => setMsg(
    ok2 ? ['ok', `flying in Gazebo: gazebo_geo/${r.started} -- watch it on the Flight log, Live`] : ['bad', r.error])));

  const set = (k, v) => setM((q) => ({ ...q, [k]: v }));
  const cmd = `geo/fly_mission.sh geo/missions/${m.name}.json mission_${m.name}`;

  return (
    <main>
      <div className="panel masthead" style={{ flexDirection: 'row' }}>
        <span className="marks" aria-hidden="true" />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 11, minWidth: 0 }}>
          <span className="brand">GeoAnchor</span><span className="sub">mission planner &mdash; Logan, sixteen pads</span>
        </div>
        <span className="spacer" />
        <Nav here="/mission/" />
      </div>

      <div className="row">
        <Panel className="grow" title="Route"
               meta="click: add a waypoint · click a pad: fly there · drag: move · right-click: delete · wheel: zoom">
          {world && (
            <FlightMap extent={world.extent} draw={draw} fit={[[-60, -170], [300, 110]]} height={560}
                       onPick={onPick} onDrag={onDrag} onDrop={() => setGrab(null)} onClick={onClick} redraw={m} />
          )}
          <footer>
            <span className="meta">{m.route.length} waypoint{m.route.length === 1 ? '' : 's'} · {length.toFixed(0)} m ·
              ~{Math.round(length / m.speed + (m.alt / 2.5) + 60)} s to the pad at {m.speed} m/s
              {tn.some((a) => a > 120) ? ' · amber: a turn over 120°, flown slowly (~2.4 m/s)' : ''}</span>
          </footer>
        </Panel>

        <Panel className="side" title="Mission">
          <div className="body" style={{ display: 'flex', flexDirection: 'column', gap: 11 }}>
            <label className="field"><span className="label">name</span>
              <input style={input} value={m.name} onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_.-]/g, '_'))} /></label>
            <label className="field"><span className="label">land on</span>
              <select style={input} value={m.pad} onChange={(e) => set('pad', e.target.value)}>
                {world?.pads.map((p) => (
                  <option key={p.name} value={p.name}>{p.ours ? 'our pad' : p.name} &mdash; markers {p.ids.join(', ')}</option>
                ))}
              </select></label>
            {pad && !pad.ours && (
              <p className="note" style={{ margin: 0 }}>A neighbour&rsquo;s pad: the landing camera will look for ids
                {' '}{pad.ids.join(', ')} and refuse ours (2, 1, 0), as it refuses theirs on a normal flight.</p>
            )}
            <div style={{ display: 'flex', gap: 10 }}>
              <label className="field" style={{ flex: 1 }}><span className="label">cruise height, m</span>
                <input style={input} type="number" min={60} max={100} value={m.alt} onChange={(e) => set('alt', Number(e.target.value))} /></label>
              <label className="field" style={{ flex: 1 }}><span className="label">speed, m/s</span>
                <input style={input} type="number" min={2} max={10} step={0.5} value={m.speed} onChange={(e) => set('speed', Number(e.target.value))} /></label>
            </div>
            <label className="field"><span className="label">lander</span>
              <select style={input} value={m.lander} onChange={(e) => set('lander', e.target.value)}>
                {LANDERS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
              </select></label>
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
              <label className="field" style={{ flex: 1 }}><span className="label">wind at 10 m, m/s (from the west)</span>
                <input style={input} type="number" min={0} max={12} value={m.wind} onChange={(e) => set('wind', Number(e.target.value))} /></label>
              <label className="label" style={{ gap: 6, paddingBottom: 6 }}>
                <input type="checkbox" checked={m.deny_gnss} onChange={(e) => set('deny_gnss', e.target.checked)} /> deny GNSS</label>
            </div>

            <div>
              <span className="kicker">Waypoints (Gazebo metres: x east, y north)</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 6 }}>
                {m.route.map((p, i) => (
                  <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', font: '12px var(--mono)' }}>
                    <span style={{ width: 20, color: tn[i] > 120 ? 'var(--warn)' : 'var(--accent)' }}>{i + 1}</span>
                    <span style={{ flex: 1 }}>{p[0]}, {p[1]}</span>
                    <span className="meta">{tn[i] != null ? `${tn[i].toFixed(0)}° turn` : ''}</span>
                    <button className="btn sm danger" onClick={() => setM((q) => ({ ...q, route: q.route.filter((_, j) => j !== i) }))}>&times;</button>
                  </div>
                ))}
                {!m.route.length && <span className="note">straight from the depot to the pad</span>}
              </div>
              <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                <button className="btn sm" onClick={() => set('route', [])}>Clear</button>
                <button className="btn sm" onClick={() => world && set('route', world.route)}>The usual route</button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn" onClick={save}>Save</button>
              <button className="btn amber" onClick={fly} title="saves, then starts PX4 + Gazebo + GeoAnchor with this mission">
                Fly it in Gazebo</button>
            </div>
            {msg && <div className={`chip ${msg[0]}`} style={{ whiteSpace: 'normal', height: 'auto', padding: '6px 9px' }}>{msg[1]}</div>}
            <div>
              <span className="kicker">or by hand</span>
              <pre className="note" style={{ margin: '6px 0 0', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{cmd}</pre>
            </div>
          </div>
          {saved.length > 0 && (
            <div className="body" style={{ borderTop: '1px solid var(--line)' }}>
              <span className="kicker">Saved missions</span>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 7 }}>
                {saved.map((s) => (
                  <button key={s.name} className="btn sm" onClick={() => setM({ ...s, pad: s.pad.name })}>{s.name}</button>
                ))}
              </div>
            </div>
          )}
        </Panel>
      </div>

      <p className="foot">
        The route is flown at the cruise height on PX4&rsquo;s own estimate, corners flown through and slowed for;
        GNSS is denied over the suburb once GeoAnchor is steadily fused, then the pad&rsquo;s marker takes over from
        30 m. GeoAnchor needs ground it can match: the suburb matched 38&ndash;70% of frames, the forest north of the
        pad 8%, so keep the route over houses. The basemap is Esri Wayback imagery &mdash; internal use only.
      </p>
    </main>
  );
}
