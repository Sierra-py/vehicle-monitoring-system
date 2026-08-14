import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { Dashboard } from './pages/Dashboard'
import { Events } from './pages/Events'
import { ReviewQueue } from './pages/ReviewQueue'
import { api } from './api/client'
import { usePolling } from './hooks/usePolling'

export default function App() {
  // Polled here (not inside ReviewQueue) so the nav pill count stays live
  // even while the user is on Dashboard or Events, not just when they've
  // navigated to the Review queue page itself.
  const { data: queue } = usePolling(() => api.getReviewQueue(200), 8000)

  return (
    <BrowserRouter>
      <AppShell reviewQueueCount={queue?.length ?? 0}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/events" element={<Events />} />
          <Route path="/review" element={<ReviewQueue />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
