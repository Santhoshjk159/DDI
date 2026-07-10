import { useState, useEffect, useCallback } from 'react'
import { Search, ChevronLeft, ChevronRight, Pill, X, FlaskConical, Database } from 'lucide-react'
import { Link } from 'react-router-dom'
import { listDrugs, getDrug } from '../api'
import SeverityBadge from '../components/SeverityBadge'

function DrugModal({ drug, onClose }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDrug(drug.name).then(setDetail).catch(() => setDetail(drug)).finally(() => setLoading(false))
  }, [drug.name])

  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.5)',
        zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '1.5rem',
      }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`${drug.name} details`}
    >
      <div
        style={{
          background: 'var(--color-bg)',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-xl)',
          maxWidth: 600, width: '100%',
          maxHeight: '85vh', overflowY: 'auto',
          boxShadow: 'var(--shadow-xl)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Modal header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '1.25rem 1.5rem',
          background: 'var(--color-surface)',
          borderBottom: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-xl) var(--radius-xl) 0 0',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: 38, height: 38, background: 'var(--color-primary-bg)',
              border: '1px solid var(--color-primary-muted)',
              borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Pill size={18} color="var(--color-primary)" aria-hidden="true" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.0625rem', fontWeight: 700 }}>{drug.name}</h2>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Drug ID: {drug.drug_id}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            style={{
              background: 'var(--color-bg)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-md)', padding: '0.375rem',
              cursor: 'pointer', color: 'var(--color-text-muted)',
              display: 'flex', alignItems: 'center',
            }}
          >
            <X size={17} aria-hidden="true" />
          </button>
        </div>

        <div style={{ padding: '1.5rem' }}>
          {/* Molecular Properties */}
          <p className="section-heading">Molecular Descriptors</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
            {[
              ['Molecular Weight', drug.mol_weight?.toFixed(3) + ' g/mol'],
              ['XLogP', drug.xlogp?.toFixed(3)],
              ['Exact Mass', drug.exact_mass?.toFixed(5) + ' Da'],
              ['TPSA', drug.tpsa?.toFixed(2) + ' Å²'],
            ].map(([k, v]) => (
              <div key={k} style={{
                background: 'var(--color-surface)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-md)',
                padding: '0.875rem',
              }}>
                <p style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.25rem' }}>{k}</p>
                <p style={{ fontWeight: 700, fontFamily: 'Courier New, monospace', fontSize: '0.9375rem' }}>{v ?? '—'}</p>
              </div>
            ))}
          </div>

          {/* Known Interactions */}
          <p className="section-heading">Known Interactions</p>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--color-text-muted)' }}>
              <div className="spinner" style={{ margin: '0 auto 0.75rem' }} aria-label="Loading interactions" />
              <p style={{ fontSize: '0.875rem' }}>Loading interactions...</p>
            </div>
          ) : detail?.interactions?.length > 0 ? (
            <div style={{ maxHeight: 240, overflowY: 'auto', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '0.625rem 0.875rem', textAlign: 'left', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', background: 'var(--color-surface)', borderBottom: '1px solid var(--color-border)' }}>
                      Interacting Drug
                    </th>
                    <th style={{ padding: '0.625rem 0.875rem', textAlign: 'right', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', background: 'var(--color-surface)', borderBottom: '1px solid var(--color-border)' }}>
                      Severity
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {detail.interactions.map(inter => (
                    <tr key={inter.id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                      <td style={{ padding: '0.5625rem 0.875rem', fontWeight: 500 }}>
                        {inter.drug_a_name === drug.name ? inter.drug_b_name : inter.drug_a_name}
                      </td>
                      <td style={{ padding: '0.5625rem 0.875rem', textAlign: 'right' }}>
                        <SeverityBadge level={inter.level} size="sm" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state" style={{ padding: '2rem' }}>
              <Database size={32} aria-hidden="true" />
              <p>No known interactions in database.</p>
            </div>
          )}
        </div>
      </div>
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
    } catch {}
    finally { setLoading(false) }
  }, [search, page])

  useEffect(() => { fetchDrugs() }, [fetchDrugs])
  useEffect(() => { setPage(1) }, [search])

  const totalPages = Math.ceil(data.total / PAGE_SIZE)
  const startRecord = (page - 1) * PAGE_SIZE + 1
  const endRecord = Math.min(page * PAGE_SIZE, data.total)

  return (
    <div style={{ paddingTop: 'var(--nav-height)' }}>

      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-inner">
          <nav className="breadcrumb" aria-label="Breadcrumb">
            <Link to="/">Dashboard</Link>
            <span className="breadcrumb-sep" aria-hidden="true">/</span>
            <span>Drug Database</span>
          </nav>
          <h1>Drug Compound Database</h1>
          <p>
            Browse and search {data.total.toLocaleString()} drug compounds with molecular descriptors.
            Click any row to view detailed properties and known interactions.
          </p>
        </div>
      </div>

      <div className="container page-content">

        {/* Search & filters bar */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '1rem',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
        }}>
          <div style={{ position: 'relative', flex: '1 1 300px', maxWidth: 480 }}>
            <Search
              size={15}
              aria-hidden="true"
              style={{
                position: 'absolute', left: 11, top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--color-text-subtle)',
                pointerEvents: 'none',
              }}
            />
            <input
              id="drug-search-filter"
              className="form-input"
              style={{ paddingLeft: '2.125rem' }}
              placeholder="Filter by drug name..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Filter drugs by name"
            />
          </div>

          {search && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setSearch('')}
              aria-label="Clear search filter"
            >
              <X size={13} /> Clear
            </button>
          )}

          <div style={{ marginLeft: 'auto', fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
            {data.total > 0 && !loading && (
              <span>
                Showing <strong>{startRecord}–{endRecord}</strong> of <strong>{data.total.toLocaleString()}</strong> records
              </span>
            )}
          </div>
        </div>

        {/* Data table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="table-wrapper">
            <table aria-label="Drug compound database">
              <thead>
                <tr>
                  <th scope="col"># </th>
                  <th scope="col">Drug Name</th>
                  <th scope="col">Drug ID</th>
                  <th scope="col">Mol. Weight (g/mol)</th>
                  <th scope="col">XLogP</th>
                  <th scope="col">Exact Mass (Da)</th>
                  <th scope="col">TPSA (Å²)</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
                        <div className="spinner" aria-label="Loading" />
                        <span style={{ fontSize: '0.875rem' }}>Loading records...</span>
                      </div>
                    </td>
                  </tr>
                ) : data.items.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: 'center', padding: '4rem', color: 'var(--color-text-muted)' }}>
                      <div className="empty-state">
                        <Database size={40} aria-hidden="true" />
                        <p>No drugs found matching "{search}"</p>
                      </div>
                    </td>
                  </tr>
                ) : data.items.map((drug, idx) => (
                  <tr
                    key={drug.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelected(drug)}
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && setSelected(drug)}
                    aria-label={`View details for ${drug.name}`}
                  >
                    <td style={{ color: 'var(--color-text-subtle)', fontSize: '0.8125rem' }}>
                      {(page - 1) * PAGE_SIZE + idx + 1}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Pill size={13} color="var(--color-primary)" aria-hidden="true" style={{ flexShrink: 0 }} />
                        <span style={{ fontWeight: 600 }}>{drug.name}</span>
                      </div>
                    </td>
                    <td style={{ fontFamily: 'Courier New, monospace', fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
                      {drug.drug_id ?? '—'}
                    </td>
                    <td style={{ fontFamily: 'Courier New, monospace' }}>{drug.mol_weight?.toFixed(3) ?? '—'}</td>
                    <td style={{ fontFamily: 'Courier New, monospace' }}>{drug.xlogp?.toFixed(3) ?? '—'}</td>
                    <td style={{ fontFamily: 'Courier New, monospace' }}>{drug.exact_mass?.toFixed(5) ?? '—'}</td>
                    <td style={{ fontFamily: 'Courier New, monospace' }}>{drug.tpsa?.toFixed(2) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '0.875rem 1.25rem',
              borderTop: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              flexWrap: 'wrap', gap: '0.75rem',
            }}>
              <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
                Page {page} of {totalPages}
              </span>
              <div className="pagination" role="navigation" aria-label="Pagination">
                <button
                  className="page-btn"
                  onClick={() => setPage(1)}
                  disabled={page === 1}
                  aria-label="First page"
                >
                  «
                </button>
                <button
                  className="page-btn"
                  onClick={() => setPage(p => p - 1)}
                  disabled={page === 1}
                  aria-label="Previous page"
                >
                  <ChevronLeft size={14} aria-hidden="true" />
                </button>
                {/* Page numbers */}
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const p = Math.max(1, Math.min(page - 2, totalPages - 4)) + i
                  return p <= totalPages ? (
                    <button
                      key={p}
                      className={`page-btn${p === page ? ' active' : ''}`}
                      onClick={() => setPage(p)}
                      aria-label={`Page ${p}`}
                      aria-current={p === page ? 'page' : undefined}
                    >
                      {p}
                    </button>
                  ) : null
                })}
                <button
                  className="page-btn"
                  onClick={() => setPage(p => p + 1)}
                  disabled={page === totalPages}
                  aria-label="Next page"
                >
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
                <button
                  className="page-btn"
                  onClick={() => setPage(totalPages)}
                  disabled={page === totalPages}
                  aria-label="Last page"
                >
                  »
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Drug modal */}
      {selected && <DrugModal drug={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
