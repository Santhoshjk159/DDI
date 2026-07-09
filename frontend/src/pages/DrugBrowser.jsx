import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Search, ChevronLeft, ChevronRight, Pill, X } from 'lucide-react'
import { listDrugs, getDrug } from '../api'
import SeverityBadge from '../components/SeverityBadge'

function DrugModal({ drug, onClose }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDrug(drug.name).then(setDetail).catch(() => setDetail(drug)).finally(() => setLoading(false))
  }, [drug.name])

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem', backdropFilter: 'blur(8px)' }} onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.92 }}
        style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-xl)', padding: '2rem', maxWidth: 600, width: '100%', maxHeight: '80vh', overflowY: 'auto' }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 44, height: 44, background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.2))', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Pill size={22} style={{ color: 'var(--color-primary-light)' }} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 800 }}>{drug.name}</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>Drug ID: {drug.drug_id}</p>
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8, padding: '0.35rem', cursor: 'pointer', color: 'var(--color-text-muted)' }}><X size={18} /></button>
        </div>

        {/* Properties */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
          {[
            ['Molecular Weight', drug.mol_weight?.toFixed(3) + ' g/mol'],
            ['XLogP', drug.xlogp?.toFixed(3)],
            ['Exact Mass', drug.exact_mass?.toFixed(5)],
            ['TPSA', drug.tpsa?.toFixed(2) + ' Å²'],
          ].map(([k, v]) => (
            <div key={k} style={{ background: 'var(--color-surface-2)', borderRadius: 'var(--radius-md)', padding: '0.85rem' }}>
              <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.3rem' }}>{k}</p>
              <p style={{ fontWeight: 700, fontFamily: 'monospace', fontSize: '0.95rem' }}>{v ?? '—'}</p>
            </div>
          ))}
        </div>

        {/* Interactions */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--color-text-muted)' }}><div className="spinner" style={{ margin: '0 auto' }} /></div>
        ) : detail?.interactions?.length > 0 ? (
          <div>
            <p style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>
              Known Interactions ({detail.interactions.length})
            </p>
            <div style={{ maxHeight: 220, overflowY: 'auto' }}>
              {detail.interactions.map(inter => (
                <div key={inter.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.55rem 0', borderBottom: '1px solid var(--color-border)', fontSize: '0.85rem' }}>
                  <span>{inter.drug_a_name === drug.name ? inter.drug_b_name : inter.drug_a_name}</span>
                  <SeverityBadge level={inter.level} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>No known interactions loaded.</p>
        )}
      </motion.div>
    </div>
  )
}

export default function DrugBrowser() {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [data, setData] = useState({ total: 0, items: [] })
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const PAGE_SIZE = 20

  const fetchDrugs = useCallback(async () => {
    setLoading(true)
    try {
      const res = await listDrugs(search, page, PAGE_SIZE)
      setData(res)
    } catch { }
    finally { setLoading(false) }
  }, [search, page])

  useEffect(() => { fetchDrugs() }, [fetchDrugs])

  // Debounce search reset to page 1
  useEffect(() => { setPage(1) }, [search])

  const totalPages = Math.ceil(data.total / PAGE_SIZE)

  return (
    <div style={{ paddingTop: 80 }}>
      <div className="container" style={{ padding: '3rem 1.5rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '0.5rem' }}>
            <span className="gradient-text">Drug</span> Browser
          </h1>
          <p style={{ color: 'var(--color-text-muted)' }}>
            Explore {data.total.toLocaleString()} drugs with molecular properties. Click any drug for details.
          </p>
        </div>

        {/* Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', maxWidth: 500, marginBottom: '1.5rem' }}>
          <Search size={16} style={{ position: 'absolute', left: 14, color: 'var(--color-text-subtle)', pointerEvents: 'none' }} />
          <input
            className="form-input" style={{ paddingLeft: '2.5rem' }}
            placeholder="Search by drug name..."
            value={search} onChange={e => setSearch(e.target.value)}
          />
        </div>

        {/* Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Drug Name</th>
                  <th>Mol. Weight (g/mol)</th>
                  <th>XLogP</th>
                  <th>Exact Mass</th>
                  <th>TPSA (Å²)</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={5} style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}><div className="spinner" style={{ margin: '0 auto' }} /></td></tr>
                ) : data.items.length === 0 ? (
                  <tr><td colSpan={5} style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}>No drugs found.</td></tr>
                ) : data.items.map(drug => (
                  <tr key={drug.id} style={{ cursor: 'pointer' }} onClick={() => setSelected(drug)}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <Pill size={14} style={{ color: 'var(--color-primary-light)', flexShrink: 0 }} />
                        <span style={{ fontWeight: 600 }}>{drug.name}</span>
                      </div>
                    </td>
                    <td style={{ fontFamily: 'monospace' }}>{drug.mol_weight?.toFixed(3) ?? '—'}</td>
                    <td style={{ fontFamily: 'monospace' }}>{drug.xlogp?.toFixed(3) ?? '—'}</td>
                    <td style={{ fontFamily: 'monospace' }}>{drug.exact_mass?.toFixed(5) ?? '—'}</td>
                    <td style={{ fontFamily: 'monospace' }}>{drug.tpsa?.toFixed(2) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.25rem', borderTop: '1px solid var(--color-border)' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>
                Page {page} of {totalPages} · {data.total.toLocaleString()} drugs
              </span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => p - 1)} disabled={page === 1}>
                  <ChevronLeft size={16} />
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => p + 1)} disabled={page === totalPages}>
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Drug detail modal */}
        {selected && <DrugModal drug={selected} onClose={() => setSelected(null)} />}
      </div>
    </div>
  )
}
