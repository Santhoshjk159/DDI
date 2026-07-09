import { useState, useEffect, useRef } from 'react'
import { Search, Pill } from 'lucide-react'
import { searchDrugs } from '../api'

export default function DrugSearchInput({ label, value, onChange, placeholder = 'Search drug name...' }) {
  const [query, setQuery] = useState(value || '')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(-1)
  const debounceRef = useRef(null)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (!query || query.length < 2) { setResults([]); setOpen(false); return }
    setLoading(true)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      try {
        const data = await searchDrugs(query)
        setResults(data)
        setOpen(data.length > 0)
      } catch { setResults([]) }
      finally { setLoading(false) }
    }, 280)
  }, [query])

  // Close on outside click
  useEffect(() => {
    const handler = (e) => { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const select = (drug) => {
    setQuery(drug.name)
    onChange(drug.name)
    setOpen(false)
    setResults([])
  }

  const handleKeyDown = (e) => {
    if (!open) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx(i => Math.min(i + 1, results.length - 1)) }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setActiveIdx(i => Math.max(i - 1, 0)) }
    if (e.key === 'Enter' && activeIdx >= 0) select(results[activeIdx])
    if (e.key === 'Escape') { setOpen(false); setActiveIdx(-1) }
  }

  return (
    <div className="form-group" ref={wrapRef}>
      {label && <label className="form-label">{label}</label>}
      <div className="autocomplete-wrapper">
        <div style={{ position: 'relative' }}>
          <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-text-subtle)', pointerEvents: 'none' }} />
          <input
            className="form-input"
            style={{ paddingLeft: '2.25rem', paddingRight: loading ? '2.25rem' : '1rem' }}
            value={query}
            placeholder={placeholder}
            onChange={e => { setQuery(e.target.value); onChange('') }}
            onKeyDown={handleKeyDown}
            onFocus={() => results.length > 0 && setOpen(true)}
            autoComplete="off"
          />
          {loading && (
            <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)' }}>
              <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
            </div>
          )}
        </div>

        {open && results.length > 0 && (
          <div className="autocomplete-dropdown">
            {results.map((drug, idx) => (
              <div
                key={drug.id}
                className={`autocomplete-item${idx === activeIdx ? ' active' : ''}`}
                onMouseDown={() => select(drug)}
              >
                <Pill size={14} style={{ color: 'var(--color-primary-light)', flexShrink: 0 }} />
                <span>{drug.name}</span>
                {drug.mol_weight && (
                  <small style={{ marginLeft: 'auto' }}>MW: {drug.mol_weight?.toFixed(1)}</small>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
