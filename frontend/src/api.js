import axios from 'axios'

// In production, VITE_API_URL can point to the backend root or /api.
// In development, it falls back to localhost:8000.
const normalizeApiBaseUrl = (url) => {
  const trimmed = url.replace(/\/+$/, '')
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`
}

const RAW_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

const api = axios.create({
  baseURL: normalizeApiBaseUrl(RAW_BASE),
  timeout: 30000,
})

export const predictInteraction = (drugA, drugB) =>
  api.post('/predict', { drug_a: drugA, drug_b: drugB }).then(r => r.data)

export const searchDrugs = (q, limit = 15) =>
  api.get('/drugs/search', { params: { q, limit } }).then(r => r.data)

export const listDrugs = (search = '', page = 1, pageSize = 20) =>
  api.get('/drugs', { params: { search, page, page_size: pageSize } }).then(r => r.data)

export const getDrug = (name) =>
  api.get(`/drugs/${encodeURIComponent(name)}`).then(r => r.data)

export const getHistory = (limit = 20) =>
  api.get('/history', { params: { limit } }).then(r => r.data)

export const getStats = () =>
  api.get('/stats').then(r => r.data)

/**
 * Warm up the backend (Render free-tier cold start).
 * Pings /health every 3s, up to 20 attempts (~60s max).
 * Returns true if backend is awake, false if all retries exhausted.
 */
export const warmUpBackend = async () => {
  const healthUrl = RAW_BASE.endsWith('/api')
    ? RAW_BASE.replace(/\/api$/, '/health')
    : `${RAW_BASE}/health`

  for (let i = 0; i < 20; i++) {
    try {
      const res = await axios.get(healthUrl, { timeout: 5000 })
      if (res.data?.status === 'ok') return true
    } catch {
      // Backend still waking up
    }
    if (i < 19) await new Promise(r => setTimeout(r, 3000))
  }
  return false
}

export default api
