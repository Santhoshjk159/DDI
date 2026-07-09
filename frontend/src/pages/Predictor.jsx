import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, RotateCcw, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import DrugSearchInput from '../components/DrugSearchInput'
import SeverityBadge from '../components/SeverityBadge'
import { predictInteraction, getHistory } from '../api'
import { useEffect } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const SEVERITY_COLORS = { Minor: '#10b981', Moderate: '#f59e0b', Major: '#ef4444' }

function PropertiesTable({ drug }) {
  const rows = [
    ['Molecular Weight', drug.mol_weight?.toFixed(3) + ' g/mol'],
    ['XLogP',           drug.xlogp?.toFixed(3)],
    ['Exact Mass',      drug.exact_mass?.toFixed(5)],
    ['TPSA',            drug.tpsa?.toFixed(2) + ' Å²'],
  ]
  return (
    <div>
      <p style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>
        {drug.name}
      </p>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} style={{ borderBottom: '1px solid var(--color-border)' }}>
              <td style={{ padding: '0.5rem 0', color: 'var(--color-text-muted)' }}>{k}</td>
              <td style={{ padding: '0.5rem 0', textAlign: 'right', fontWeight: 600, fontFamily: 'monospace' }}>{v ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HistoryItem({ item }) {
  const colors = { Minor: 'var(--color-minor)', Moderate: 'var(--color-moderate)', Major: 'var(--color-major)' }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.65rem 0', borderBottom: '1px solid var(--color-border)' }}>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: colors[item.level], flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: '0.82rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {item.drug_a_name} × {item.drug_b_name}
        </p>
        <p style={{ fontSize: '0.74rem', color: 'var(--color-text-muted)' }}>{item.level} · {(item.confidence * 100).toFixed(0)}% conf.</p>
      </div>
    </div>
  )
}

export default function Predictor() {
  const [drugA, setDrugA] = useState('')
  const [drugB, setDrugB] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])

  useEffect(() => {
    getHistory(10).then(h => setHistory(h.items)).catch(() => {})
  }, [])

  const handlePredict = async () => {
    if (!drugA || !drugB) { setError('Please select both drugs using the search boxes.'); return }
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await predictInteraction(drugA, drugB)
      setResult(data)
      // Refresh history
      getHistory(10).then(h => setHistory(h.items)).catch(() => {})
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Prediction failed. Make sure both drugs are in the database.'
      setError(msg)
    } finally { setLoading(false) }
  }

  const handleReset = () => { setResult(null); setError(''); setDrugA(''); setDrugB('') }

  const probData = result ? [
    { name: 'Minor',    value: +(result.probabilities.Minor * 100).toFixed(1)    },
    { name: 'Moderate', value: +(result.probabilities.Moderate * 100).toFixed(1) },
    { name: 'Major',    value: +(result.probabilities.Major * 100).toFixed(1)    },
  ] : []

  return (
    <div style={{ paddingTop: 80 }}>
      <div className="container" style={{ padding: '3rem 1.5rem', maxWidth: 1200 }}>
        <div style={{ marginBottom: '2.5rem' }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '0.5rem' }}>
            DDI <span className="gradient-text">Interaction Predictor</span>
          </h1>
          <p style={{ color: 'var(--color-text-muted)' }}>
            Search for two drugs to predict their interaction severity using our trained ML model.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem', alignItems: 'start' }}>
          {/* Main panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Input card */}
            <div className="card">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                <DrugSearchInput label="Drug A" value={drugA} onChange={setDrugA} placeholder="e.g. Abacavir" />
                <DrugSearchInput label="Drug B" value={drugB} onChange={setDrugB} placeholder="e.g. Naltrexone" />
              </div>

              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius-md)', marginBottom: '1rem', color: '#ef4444', fontSize: '0.88rem' }}>
                  <AlertCircle size={16} /> {error}
                </div>
              )}

              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button className="btn btn-primary" style={{ flex: 1 }} onClick={handlePredict} disabled={loading || !drugA || !drugB}>
                  {loading ? <><div className="spinner" /> Analyzing...</> : <><ArrowRight size={18} /> Predict Interaction</>}
                </button>
                {result && (
                  <button className="btn btn-secondary" onClick={handleReset}><RotateCcw size={16} /> Reset</button>
                )}
              </div>
            </div>

            {/* Result */}
            <AnimatePresence>
              {result && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
                  {/* Severity header */}
                  <div className="card" style={{
                    background: result.level === 'Major' ? 'rgba(239,68,68,0.06)' : result.level === 'Moderate' ? 'rgba(245,158,11,0.06)' : 'rgba(16,185,129,0.06)',
                    borderColor: SEVERITY_COLORS[result.level] + '40',
                    marginBottom: '1rem',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
                      <div>
                        <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginBottom: '0.4rem' }}>
                          {result.drug_a} × {result.drug_b}
                        </p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                          <SeverityBadge level={result.level} size="lg" />
                          <span style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>
                            <strong style={{ color: SEVERITY_COLORS[result.level] }}>{(result.confidence * 100).toFixed(1)}%</strong> confidence
                          </span>
                        </div>
                      </div>
                      <CheckCircle size={32} style={{ color: SEVERITY_COLORS[result.level], opacity: 0.7 }} />
                    </div>

                    <p style={{ fontSize: '0.88rem', color: 'var(--color-text-muted)', lineHeight: 1.7, padding: '0.9rem', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-md)', borderLeft: `3px solid ${SEVERITY_COLORS[result.level]}` }}>
                      {result.warning_text}
                    </p>
                  </div>

                  {/* Charts + Properties */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    {/* Probability bar chart */}
                    <div className="card">
                      <p style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Probability Breakdown</p>
                      <ResponsiveContainer width="100%" height={150}>
                        <BarChart data={probData} layout="vertical" margin={{ left: 10 }}>
                          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} tickFormatter={v => v + '%'} />
                          <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: 'var(--color-text)' }} width={70} />
                          <Tooltip formatter={v => v + '%'} contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                          <Bar dataKey="value" radius={4}>
                            {probData.map(entry => <Cell key={entry.name} fill={SEVERITY_COLORS[entry.name]} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Drug properties */}
                    <div className="card">
                      <p style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Molecular Properties</p>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {result.drug_a_properties && <PropertiesTable drug={result.drug_a_properties} />}
                        {result.drug_b_properties && <PropertiesTable drug={result.drug_b_properties} />}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* History sidebar */}
          <div className="card" style={{ position: 'sticky', top: 90 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Clock size={16} style={{ color: 'var(--color-primary-light)' }} />
              <p style={{ fontWeight: 700, fontSize: '0.9rem' }}>Recent Predictions</p>
            </div>
            {history.length === 0 ? (
              <p style={{ fontSize: '0.82rem', color: 'var(--color-text-subtle)', textAlign: 'center', padding: '1.5rem 0' }}>No predictions yet.</p>
            ) : (
              history.map(item => <HistoryItem key={item.id} item={item} />)
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
