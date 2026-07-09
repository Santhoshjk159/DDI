import axios from 'axios'

// In production, VITE_API_URL can point to the backend root or /api.
// In development, it falls back to localhost:8000.
const normalizeApiBaseUrl = (url) => {
  const trimmed = url.replace(/\/+$/, '')
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`
}

const api = axios.create({
  baseURL: normalizeApiBaseUrl(import.meta.env.VITE_API_URL || 'http://localhost:8000'),
  timeout: 20000,
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

export default api
