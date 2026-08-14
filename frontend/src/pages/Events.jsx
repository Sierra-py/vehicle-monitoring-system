import { useEffect, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { PlateTag, statusFromEvent } from '../components/PlateTag'
import { Badge } from '../components/Badge'

const PAGE_SIZE = 30

export function Events() {
  const [searchParams, setSearchParams] = useSearchParams()
  const plateFilter = searchParams.get('plate') || ''
  const [plateInput, setPlateInput] = useState(plateFilter)
  const [reviewFilter, setReviewFilter] = useState('all') // all | pending | clear
  const [events, setEvents] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [offset, setOffset] = useState(0)

  const load = useCallback(async (currentOffset = 0) => {
    setLoading(true)
    setError(null)
    try {
      const requiresReview = reviewFilter === 'all' ? undefined : reviewFilter === 'pending'
      const result = await api.getEvents({
        limit: PAGE_SIZE,
        offset: currentOffset,
        plateText: plateFilter || undefined,
        requiresReview,
      })
      setEvents(result)
    } catch (err) {
      setError(err)
    } finally {
      setLoading(false)
    }
  }, [plateFilter, reviewFilter])

  useEffect(() => {
    setOffset(0)
    load(0)
  }, [load])

  function handleSearch(e) {
    e.preventDefault()
    setSearchParams(plateInput ? { plate: plateInput } : {})
  }

  function clearFilter() {
    setPlateInput('')
    setSearchParams({})
  }

  return (
    <>
      <div className="page-header">
        <h1>Events</h1>
        <p>Full event log — filter by plate to see a single vehicle's history</p>
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 20, alignItems: 'center' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8 }}>
          <input
            type="text"
            placeholder="Search plate (e.g. MH20BY3665)"
            value={plateInput}
            onChange={(e) => setPlateInput(e.target.value.toUpperCase())}
            style={{ width: 240 }}
          />
          <button type="submit" className="btn">Search</button>
          {plateFilter && (
            <button type="button" className="btn" onClick={clearFilter}>Clear</button>
          )}
        </form>

        <select value={reviewFilter} onChange={(e) => setReviewFilter(e.target.value)}>
          <option value="all">All events</option>
          <option value="pending">Needs review</option>
          <option value="clear">Clear</option>
        </select>

        <button className="btn" onClick={() => load(offset)} style={{ marginLeft: 'auto' }}>
          Refresh
        </button>
      </div>

      {plateFilter && (
        <div style={{ marginBottom: 16, fontSize: 13, color: 'var(--text-muted)' }}>
          Showing history for <PlateTag plateText={plateFilter} status="neutral" />
        </div>
      )}

      {error && (
        <div className="card" style={{ marginBottom: 20, borderColor: 'var(--signal-bad-dim)' }}>
          <span style={{ color: 'var(--signal-bad)', fontSize: 13 }}>
            Couldn't load events ({error.message})
          </span>
        </div>
      )}

      <div className="card" style={{ padding: 0 }}>
        {loading && !events && <div className="loading-state">Loading…</div>}

        {events && events.length === 0 && (
          <div className="empty-state">
            {plateFilter ? `No events found for ${plateFilter}.` : 'No events match these filters.'}
          </div>
        )}

        {events && events.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Plate</th>
                <th>Match</th>
                <th>Direction</th>
                <th>Transition</th>
                <th>OCR conf.</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.event_id}>
                  <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {new Date(ev.event_timestamp).toLocaleString()}
                  </td>
                  <td>
                    <button
                      onClick={() => setSearchParams({ plate: ev.plate_text || '' })}
                      style={{ background: 'none', border: 'none', padding: 0, cursor: ev.plate_text ? 'pointer' : 'default' }}
                      disabled={!ev.plate_text}
                    >
                      <PlateTag plateText={ev.plate_text} status={statusFromEvent(ev)} />
                    </button>
                  </td>
                  <td><Badge>{ev.match_type || 'no_match'}</Badge></td>
                  <td style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>{ev.claimed_direction}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: ev.transition_valid === false ? 'var(--signal-bad)' : 'var(--text-muted)' }}>
                    {ev.vehicle_state_before} → {ev.vehicle_state_after}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
                    {ev.ocr_confidence != null ? ev.ocr_confidence.toFixed(2) : '—'}
                  </td>
                  <td>
                    {ev.requires_review
                      ? <Badge variant="pending">{ev.review_status}</Badge>
                      : <Badge variant="ok">Clear</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {events && events.length === PAGE_SIZE && (
        <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
          <button
            className="btn"
            onClick={() => {
              const next = offset + PAGE_SIZE
              setOffset(next)
              load(next)
            }}
          >
            Load next {PAGE_SIZE} →
          </button>
        </div>
      )}
    </>
  )
}
