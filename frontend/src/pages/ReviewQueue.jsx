import { useState } from 'react'
import { api } from '../api/client'
import { usePolling } from '../hooks/usePolling'
import { PlateTag } from '../components/PlateTag'
import { Badge } from '../components/Badge'

export function ReviewQueue() {
  const { data: queue, loading, refetch } = usePolling(() => api.getReviewQueue(50), 5000)
  const [actingOn, setActingOn] = useState(null) // event_id currently being submitted
  const [noteDrafts, setNoteDrafts] = useState({})

  async function handleAction(eventId, status) {
    setActingOn(eventId)
    try {
      await api.reviewEvent(eventId, status, noteDrafts[eventId])
      await refetch()
    } catch (err) {
      alert(`Couldn't update this event: ${err.message}`)
    } finally {
      setActingOn(null)
    }
  }

  return (
    <>
      <div className="page-header">
        <h1>Review queue</h1>
        <p>
          Events flagged by the pipeline — ambiguous whitelist matches or invalid state
          transitions. Approving or rejecting records your decision; it never changes
          current occupancy state.
        </p>
      </div>

      {loading && !queue && <div className="loading-state">Loading…</div>}

      {queue && queue.length === 0 && (
        <div className="card">
          <div className="empty-state">Nothing pending — the queue is clear.</div>
        </div>
      )}

      {queue && queue.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {queue.map((ev) => (
            <div key={ev.event_id} className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <PlateTag plateText={ev.plate_text} status="pending" />
                  <Badge>{ev.match_type || 'no_match'}</Badge>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(ev.event_timestamp).toLocaleString()}
                  </span>
                </div>
                <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {ev.vehicle_state_before} → {ev.vehicle_state_after}
                  {ev.transition_valid === false && (
                    <span style={{ color: 'var(--signal-bad)', marginLeft: 6 }}>invalid</span>
                  )}
                </span>
              </div>

              <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: '0 0 14px 0', lineHeight: 1.5 }}>
                {ev.review_reason || 'No reason recorded.'}
              </p>

              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  type="text"
                  placeholder="Optional note"
                  value={noteDrafts[ev.event_id] || ''}
                  onChange={(e) => setNoteDrafts((prev) => ({ ...prev, [ev.event_id]: e.target.value }))}
                  style={{ flex: 1, fontFamily: 'var(--font-sans)' }}
                />
                <button
                  className="btn approve"
                  disabled={actingOn === ev.event_id}
                  onClick={() => handleAction(ev.event_id, 'approved')}
                >
                  Approve
                </button>
                <button
                  className="btn reject"
                  disabled={actingOn === ev.event_id}
                  onClick={() => handleAction(ev.event_id, 'rejected')}
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
