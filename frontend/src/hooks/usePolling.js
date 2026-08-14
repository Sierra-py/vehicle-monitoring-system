import { useEffect, useRef, useState } from 'react'

/**
 * Runs fetchFn immediately, then again every intervalMs. Used on Dashboard
 * and Review Queue (per project decision — those two should feel live;
 * Events stays manual-refresh so filtering/reading isn't disrupted by rows
 * shifting mid-search).
 */
export function usePolling(fetchFn, intervalMs = 5000, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const fetchFnRef = useRef(fetchFn)
  fetchFnRef.current = fetchFn

  useEffect(() => {
    let cancelled = false
    let timer

    async function tick() {
      try {
        const result = await fetchFnRef.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError(err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    tick()
    timer = setInterval(tick, intervalMs)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, loading, refetch: () => fetchFnRef.current().then(setData) }
}
