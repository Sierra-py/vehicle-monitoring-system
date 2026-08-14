/**
 * The plate tag is the one visual element reused across every page —
 * dashboard feed, events table, review queue, vehicle list. Status color
 * is derived from real pipeline data (match_type / current_state), not an
 * arbitrary style choice per page.
 */
export function PlateTag({ plateText, status = 'neutral' }) {
  if (!plateText) {
    return <span className="plate-tag status-neutral">— no read —</span>
  }
  return (
    <span className={`plate-tag status-${status}`}>
      <span className="plate-tag__dot" />
      {plateText}
    </span>
  )
}

/** Maps a vehicle_events row to a plate-tag status. */
export function statusFromEvent(event) {
  if (event.match_type === 'exact' || event.match_type === 'fuzzy') return 'ok'
  if (event.requires_review) return 'pending'
  if (event.transition_valid === false) return 'bad'
  return 'neutral'
}

/** Maps a vehicle_state row to a plate-tag status. */
export function statusFromVehicle(vehicle) {
  if (vehicle.is_whitelisted) return 'ok'
  return 'neutral'
}
