import axios from 'axios'

// In production (Vercel), VITE_API_URL points to Render.com backend
// In development, falls back to localhost:8000
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
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
