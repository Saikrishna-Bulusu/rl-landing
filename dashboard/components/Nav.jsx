'use client';

/** The fork's three pages. The GeoAnchor page is the original, untouched apart from this bar. */
const PAGES = [['/', 'GeoAnchor'], ['/flight/', 'Flight log'], ['/mission/', 'Mission planner']];

export default function Nav({ here }) {
  return (
    <div className="seg" role="navigation">
      {PAGES.map(([href, label]) => (
        <a key={href} href={href} className={here === href ? 'on' : ''}
           style={{ padding: '0 11px', height: 26, display: 'flex', alignItems: 'center',
                    font: '600 12px/1 var(--cond)', letterSpacing: '.08em', textTransform: 'uppercase',
                    color: here === href ? 'var(--bg)' : 'var(--ink-dim)' }}>
          {label}
        </a>
      ))}
    </div>
  );
}
