import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', exact: true },
  { to: '/events', label: 'Events' },
  { to: '/review', label: 'Review queue' },
]

export function AppShell({ children, reviewQueueCount }) {
  return (
    <div className="app-shell">
      <nav className="rail">
        <div className="rail__brand">Gate Console</div>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.exact}
            className={({ isActive }) => `rail__nav-item${isActive ? ' active' : ''}`}
          >
            {item.label}
            {item.to === '/review' && reviewQueueCount > 0 && (
              <span className="count-pill">{reviewQueueCount}</span>
            )}
          </NavLink>
        ))}
      </nav>
      <main className="main">{children}</main>
    </div>
  )
}
