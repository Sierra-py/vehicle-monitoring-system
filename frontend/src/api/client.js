const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // response wasn't JSON, fall back to statusText
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  getOccupancy: () => request('/occupancy'),

  getVehicles: (state) =>
    request(`/vehicles${state ? `?state=${state}` : ''}`),

  getEvents: ({ limit = 50, offset = 0, plateText, requiresReview } = {}) => {
    const params = new URLSearchParams({ limit, offset })
    if (plateText) params.set('plate_text', plateText)
    if (requiresReview !== undefined) params.set('requires_review', requiresReview)
    return request(`/events?${params.toString()}`)
  },

  getEvent: (eventId) => request(`/events/${eventId}`),

  getReviewQueue: (limit = 50) => request(`/review-queue?limit=${limit}`),

  reviewEvent: (eventId, status, reviewerNote) =>
    request(`/events/${eventId}/review`, {
      method: 'POST',
      body: JSON.stringify({ status, reviewer_note: reviewerNote || null }),
    }),
}
