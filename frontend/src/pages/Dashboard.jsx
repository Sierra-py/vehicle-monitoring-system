import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { usePolling } from '../hooks/usePolling'
import { PlateTag, statusFromEvent } from '../components/PlateTag'
import { Badge } from '../components/Badge'

export function Dashboard() {
  const { data: occupancy, loading: occLoading, error: occError } = usePolling(
    () => api.getOccupancy(),
    5000
  )
  const { data: events, loading: evLoading } = usePolling(
    () => api.getEvents({ limit: 15 }),
    5000
  )

  return (
    <>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Live occupancy and recent gate activity</p>
      </div>

      {occError && (
        <div className="card" style={{ marginBottom: 20, borderColor: 'var(--signal-bad-dim)' }}>
          <span style={{ color: 'var(--signal-bad)', fontSize: 13 }}>
            Can't reach the API — is it running? ({occError.message})
          </span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 28 }}>
        <div className="card stat-block">
          <span className="stat-block__label">Whitelisted inside</span>
          <span className="stat-block__value" style={{ color: 'var(--signal-ok)' }}>
            {occLoading ? '—' : occupancy?.whitelisted_inside ?? 0}
          </span>
        </div>
        <div className="card stat-block">
          <span className="stat-block__label">Non-whitelisted inside</span>
          <span className="stat-block__value" style={{ color: 'var(--signal-neutral)' }}>
            {occLoading ? '—' : occupancy?.non_whitelisted_inside ?? 0}
          </span>
        </div>
        <div className="card stat-block">
          <span className="stat-block__label">Total inside</span>
          <span className="stat-block__value">
            {occLoading ? '—' : occupancy?.total_inside ?? 0}
          </span>
        </div>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Recent activity
          </span>
          <Link to="/events" style={{ fontSize: 12, color: 'var(--signal-ok)' }}>
            View all →
          </Link>
        </div>

        {evLoading && !events && <div className="loading-state">Loading…</div>}

        {events && events.length === 0 && (
          <div className="empty-state">No events yet — start the pipeline to see activity here.</div>
        )}

        {events && events.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Plate</th>
                <th>Direction</th>
                <th>Transition</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.event_id}>
                  <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {new Date(ev.event_timestamp).toLocaleTimeString()}
                  </td>
                  <td><PlateTag plateText={ev.plate_text} status={statusFromEvent(ev)} /></td>
                  <td style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>{ev.claimed_direction}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
                    {ev.vehicle_state_before} → {ev.vehicle_state_after}
                  </td>
                  <td>
                    {ev.requires_review
                      ? <Badge variant="pending">Review</Badge>
                      : <Badge variant="ok">Clear</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
