import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Pill, X, CheckCircle2, FlaskConical } from 'lucide-react'
import { searchDrugs } from '../api'

/* ─────────────────────────────────────────────────────────── */
/* Utility: highlight query characters inside a drug name     */
/* ─────────────────────────────────────────────────────────── */
function HighlightMatch({ text, query }) {
  if (!query || !text) return <span>{text}</span>

  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escapedQuery})`, 'gi'))

  return (
    <span>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark
            key={i}
            style={{
              background: 'transparent',
              color: 'var(--color-primary)',
              fontWeight: 800,
              padding: 0,
            }}
          >
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  )
}

/* ─────────────────────────────────────────────────────────── */
/* Lipophilicity label from XLogP                             */
/* ─────────────────────────────────────────────────────────── */
function getLipoLabel(xlogp) {
  if (xlogp == null) return null
  if (xlogp < 0)  return { label: 'Hydrophilic', color: '#1565C0', bg: '#E3F2FD' }
  if (xlogp < 2)  return { label: 'Low Lipophilicity', color: '#00897B', bg: '#E0F2F1' }
  if (xlogp < 5)  return { label: 'Lipophilic', color: '#E65100', bg: '#FFF3E0' }
  return            { label: 'Highly Lipophilic', color: '#C62828', bg: '#FFEBEE' }
}

/* ─────────────────────────────────────────────────────────── */
/* Main Component                                             */
/* ─────────────────────────────────────────────────────────── */
export default function DrugSearchInput({
  label,
  value,
  onChange,
  placeholder = 'Search drug name...',
  id,
}) {
  const [query, setQuery] = useState(value || '')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(-1)
  const [focused, setFocused] = useState(false)
  const [selected, setSelected] = useState(false) // true once a drug is picked
  const [noResults, setNoResults] = useState(false)

  const debounceRef     = useRef(null)
  const wrapRef         = useRef(null)
  const inputRef        = useRef(null)
  const listRef         = useRef(null)
  const itemRefs        = useRef([])
  const justSelectedRef = useRef(false) // guard: skip search effect after programmatic select

  const inputId   = id || `drug-search-${label?.toLowerCase().replace(/\s+/g, '-') ?? 'input'}`
  const listboxId = `${inputId}-listbox`

  /* ── Search logic ─────────────────────────────────────── */
  useEffect(() => {
    // If query was set by select(), skip the search — user didn't type this
    if (justSelectedRef.current) {
      justSelectedRef.current = false
      return
    }

    setNoResults(false)

    if (!query || query.length < 2) {
      setResults([])
      setOpen(false)
      setLoading(false)
      clearTimeout(debounceRef.current)
      return
    }

    setLoading(true)
    clearTimeout(debounceRef.current)

    debounceRef.current = setTimeout(async () => {
      try {
        const data = await searchDrugs(query, 15)
        setResults(data)
        setOpen(true)
        setNoResults(data.length === 0)
        setActiveIdx(-1)
      } catch {
        setResults([])
        setNoResults(false)
      } finally {
        setLoading(false)
      }
    }, 200)

    return () => clearTimeout(debounceRef.current)
  }, [query])

  /* ── Close on outside click ───────────────────────────── */
  useEffect(() => {
    const handler = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false)
        setNoResults(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  /* ── Scroll active item into view ─────────────────────── */
  useEffect(() => {
    if (activeIdx >= 0 && itemRefs.current[activeIdx]) {
      itemRefs.current[activeIdx].scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [activeIdx])

  /* ── Select a drug ────────────────────────────────────── */
  const select = useCallback((drug) => {
    justSelectedRef.current = true  // tell the search effect to skip this query change
    setQuery(drug.name)
    onChange(drug.name)
    setOpen(false)
    setNoResults(false)
    setResults([])
    setActiveIdx(-1)
    setSelected(true)
    inputRef.current?.blur()
  }, [onChange])

  /* ── Clear input ──────────────────────────────────────── */
  const clear = useCallback(() => {
    setQuery('')
    onChange('')
    setResults([])
    setOpen(false)
    setNoResults(false)
    setSelected(false)
    setActiveIdx(-1)
    setTimeout(() => inputRef.current?.focus(), 50)
  }, [onChange])

  /* ── Keyboard navigation ──────────────────────────────── */
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open && results.length > 0) setOpen(true)
      setActiveIdx(i => Math.min(i + 1, results.length - 1))
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx(i => Math.max(i - 1, -1))
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      if (activeIdx >= 0 && results[activeIdx]) select(results[activeIdx])
      else if (results.length === 1) select(results[0]) // auto-pick if only one result
    }
    if (e.key === 'Escape') {
      setOpen(false)
      setActiveIdx(-1)
      inputRef.current?.blur()
    }
    if (e.key === 'Tab') {
      setOpen(false)
    }
  }

  /* ── Dynamic input border color ───────────────────────── */
  const borderColor = selected
    ? 'var(--color-minor)'
    : focused
    ? 'var(--color-primary)'
    : 'var(--color-border)'

  const boxShadow = selected
    ? '0 0 0 3px rgba(46,125,50,0.12)'
    : focused
    ? '0 0 0 3px rgba(21,101,192,0.12)'
    : 'none'

  return (
    <div className="drug-search-group" ref={wrapRef} style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>

      {/* Label */}
      {label && (
        <label
          htmlFor={inputId}
          style={{
            fontSize: '0.8125rem',
            fontWeight: 600,
            color: focused || selected ? 'var(--color-primary)' : 'var(--color-text)',
            transition: 'color 150ms ease',
            display: 'flex',
            alignItems: 'center',
            gap: '0.375rem',
          }}
        >
          <FlaskConical size={13} aria-hidden="true" style={{ color: 'var(--color-primary)' }} />
          {label}
          {selected && (
            <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.6875rem', fontWeight: 600, color: 'var(--color-minor)' }}>
              <CheckCircle2 size={11} aria-hidden="true" /> Selected
            </span>
          )}
        </label>
      )}

      {/* Autocomplete wrapper */}
      <div style={{ position: 'relative' }}>

        {/* Search icon */}
        <div style={{
          position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)',
          pointerEvents: 'none', display: 'flex', alignItems: 'center',
          transition: 'color 150ms ease',
          color: selected ? 'var(--color-minor)' : focused ? 'var(--color-primary)' : 'var(--color-text-subtle)',
          zIndex: 1,
        }} aria-hidden="true">
          {loading
            ? <div style={{ width: 15, height: 15, border: '2px solid var(--color-border)', borderTopColor: 'var(--color-primary)', borderRadius: '50%', animation: 'spin 0.6s linear infinite' }} />
            : selected
            ? <CheckCircle2 size={15} />
            : <Search size={15} />
          }
        </div>

        {/* Input */}
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          value={query}
          placeholder={placeholder}
          onChange={e => {
            setQuery(e.target.value)
            if (selected) setSelected(false)
            onChange('')
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            setFocused(true)
            if (results.length > 0) setOpen(true)
          }}
          onBlur={() => setFocused(false)}
          autoComplete="off"
          spellCheck={false}
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={open}
          aria-controls={open ? listboxId : undefined}
          aria-activedescendant={activeIdx >= 0 ? `${inputId}-opt-${activeIdx}` : undefined}
          style={{
            width: '100%',
            padding: '0.5625rem 2.5rem 0.5625rem 2.25rem',
            border: `1.5px solid ${borderColor}`,
            borderRadius: 'var(--radius-md)',
            background: selected ? '#F9FBE7' : 'var(--color-bg)',
            color: 'var(--color-text)',
            fontSize: '0.9375rem',
            fontFamily: 'Inter, sans-serif',
            outline: 'none',
            boxShadow,
            transition: 'border-color 150ms ease, box-shadow 150ms ease, background 150ms ease',
            lineHeight: 1.5,
          }}
        />

        {/* Clear × button */}
        {query && (
          <button
            type="button"
            onClick={clear}
            tabIndex={-1}
            aria-label="Clear selection"
            style={{
              position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
              background: 'var(--color-surface-2)',
              border: '1px solid var(--color-border)',
              borderRadius: '50%',
              width: 20, height: 20,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer',
              color: 'var(--color-text-muted)',
              transition: 'all 120ms ease',
              padding: 0,
              lineHeight: 1,
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'var(--color-major-bg)'; e.currentTarget.style.color = 'var(--color-major)'; e.currentTarget.style.borderColor = 'var(--color-major-border)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'var(--color-surface-2)'; e.currentTarget.style.color = 'var(--color-text-muted)'; e.currentTarget.style.borderColor = 'var(--color-border)' }}
          >
            <X size={11} aria-hidden="true" />
          </button>
        )}

        {/* Dropdown */}
        {(open && results.length > 0) || noResults ? (
          <div
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-label={`${label ?? 'Drug'} suggestions`}
            style={{
              position: 'absolute',
              top: 'calc(100% + 5px)',
              left: 0, right: 0,
              background: '#FFFFFF',
              border: '1.5px solid var(--color-primary)',
              borderRadius: 10,
              boxShadow: '0 12px 40px rgba(21,101,192,0.12), 0 4px 12px rgba(0,0,0,0.06)',
              zIndex: 9999,
              overflow: 'hidden',
              maxHeight: 360,
              overflowY: 'auto',
              animation: 'dropdownSlideIn 0.15s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          >
            {/* Results header */}
            {results.length > 0 && (
              <div style={{
                padding: '0.5rem 0.875rem 0.375rem',
                fontSize: '0.6875rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.07em',
                color: 'var(--color-text-subtle)',
                background: 'var(--color-surface)',
                borderBottom: '1px solid var(--color-border)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <span>{results.length} result{results.length !== 1 ? 's' : ''} found</span>
                <span>↑↓ navigate · Enter select · Esc close</span>
              </div>
            )}

            {/* No results */}
            {noResults && (
              <div style={{
                padding: '1.5rem 1rem',
                textAlign: 'center',
                color: 'var(--color-text-muted)',
              }}>
                <Search size={28} style={{ margin: '0 auto 0.625rem', opacity: 0.25, display: 'block' }} aria-hidden="true" />
                <p style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  No drugs found for "{query}"
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-subtle)' }}>
                  Try a different name or check spelling
                </p>
              </div>
            )}

            {/* Result items */}
            {results.map((drug, idx) => {
              const lipo = getLipoLabel(drug.xlogp)
              const isActive = idx === activeIdx
              return (
                <div
                  key={drug.id}
                  id={`${inputId}-opt-${idx}`}
                  ref={el => { itemRefs.current[idx] = el }}
                  role="option"
                  aria-selected={isActive}
                  onMouseDown={e => { e.preventDefault(); select(drug) }}
                  onMouseEnter={() => setActiveIdx(idx)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.625rem 0.875rem',
                    cursor: 'pointer',
                    borderBottom: '1px solid var(--color-border)',
                    background: isActive ? 'var(--color-primary-bg)' : 'transparent',
                    transition: 'background 80ms ease',
                    position: 'relative',
                  }}
                >
                  {/* Active indicator bar */}
                  {isActive && (
                    <div style={{
                      position: 'absolute', left: 0, top: 0, bottom: 0,
                      width: 3, background: 'var(--color-primary)',
                      borderRadius: '0 2px 2px 0',
                    }} />
                  )}

                  {/* Drug icon */}
                  <div style={{
                    width: 32, height: 32,
                    borderRadius: 8,
                    background: isActive ? 'var(--color-primary)' : 'var(--color-primary-bg)',
                    border: `1px solid ${isActive ? 'var(--color-primary)' : 'var(--color-primary-muted)'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                    transition: 'all 80ms ease',
                  }}>
                    <Pill size={14} color={isActive ? '#fff' : 'var(--color-primary)'} aria-hidden="true" />
                  </div>

                  {/* Drug info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      color: isActive ? 'var(--color-primary)' : 'var(--color-text)',
                      marginBottom: '0.125rem',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      <HighlightMatch text={drug.name} query={query} />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      {drug.mol_weight != null && (
                        <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)', fontFamily: 'monospace' }}>
                          MW: {drug.mol_weight.toFixed(1)} g/mol
                        </span>
                      )}
                      {drug.tpsa != null && (
                        <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-subtle)', fontFamily: 'monospace' }}>
                          TPSA: {drug.tpsa.toFixed(1)} Å²
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Lipophilicity tag */}
                  {lipo && (
                    <span style={{
                      fontSize: '0.625rem',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      padding: '0.1875rem 0.4375rem',
                      borderRadius: 4,
                      background: lipo.bg,
                      color: lipo.color,
                      border: `1px solid ${lipo.color}30`,
                      flexShrink: 0,
                      whiteSpace: 'nowrap',
                    }}>
                      {lipo.label}
                    </span>
                  )}
                </div>
              )
            })}

            {/* Footer hint */}
            {results.length > 0 && (
              <div style={{
                padding: '0.4375rem 0.875rem',
                fontSize: '0.6875rem',
                color: 'var(--color-text-subtle)',
                background: 'var(--color-surface)',
                borderTop: '1px solid var(--color-border)',
                display: 'flex',
                justifyContent: 'space-between',
              }}>
                <span>Showing top {results.length} matches</span>
                <span>Source: DDInter · PubChem</span>
              </div>
            )}
          </div>
        ) : null}
      </div>

      {/* Helper text */}
      <p style={{
        fontSize: '0.75rem',
        color: selected ? 'var(--color-minor)' : 'var(--color-text-subtle)',
        transition: 'color 150ms ease',
        minHeight: '1rem',
      }}>
        {selected
          ? `✓ ${query} selected — ready for prediction`
          : query.length > 0 && query.length < 2
          ? 'Type 1 more character to search...'
          : loading
          ? 'Searching database...'
          : focused && !open && query.length >= 2
          ? 'No matches — try a different name'
          : 'Type drug name or common name (e.g. Aspirin, Warfarin)'
        }
      </p>
    </div>
  )
}
