import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowLeftRight, RotateCcw, FlaskConical, AlertCircle,
  Clock, CheckCircle2, XCircle, Info, FileText
} from 'lucide-react'
import DrugSearchInput from '../components/DrugSearchInput'
import SeverityBadge from '../components/SeverityBadge'
import { predictInteraction, getHistory } from '../api'

const SEVERITY_COLORS = {
  Minor:    'var(--color-minor)',
  Moderate: 'var(--color-moderate)',
  Major:    'var(--color-major)',
}

const RECOMMENDATIONS = {
  Minor:    'Monitor the patient as a routine precaution. No immediate clinical action required unless symptoms develop.',
  Moderate: 'Dosage adjustment or additional clinical monitoring may be warranted. Consult a pharmacist or prescribing clinician.',
  Major:    'This combination is generally contraindicated. Seek immediate clinical guidance before concurrent administration.',
}

function PropertiesTable({ drug }) {
  const rows = [
    ['Molecular Weight', drug.mol_weight != null ? drug.mol_weight.toFixed(3) + ' g/mol' : '—'],
    ['XLogP (Lipophilicity)', drug.xlogp != null ? drug.xlogp.toFixed(3) : '—'],
    ['Exact Mass', drug.exact_mass != null ? drug.exact_mass.toFixed(5) + ' Da' : '—'],
    ['TPSA', drug.tpsa != null ? drug.tpsa.toFixed(2) + ' Å²' : '—'],
  ]
  return (
    <div>
      <p style={{
        fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: '0.06em', color: 'var(--color-primary)',
        marginBottom: '0.625rem', display: 'flex', alignItems: 'center', gap: '0.375rem'
      }}>
        <FlaskConical size={12} aria-hidden="true" /> {drug.name}
      </p>
      <table className="data-table">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HistoryRow({ item }) {
  return (
    <tr>
      <td>
        <span style={{ fontWeight: 500, fontSize: '0.8125rem' }}>
          {item.drug_a_name}
        </span>
        <span style={{ color: 'var(--color-text-subtle)', margin: '0 4px' }}>×</span>
        <span style={{ fontWeight: 500, fontSize: '0.8125rem' }}>
          {item.drug_b_name}
        </span>
      </td>
      <td><SeverityBadge level={item.level} size="sm" /></td>
      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>
        {(item.confidence * 100).toFixed(0)}%
      </td>
    </tr>
  )
}

function ProbabilityRow({ label, value, level }) {
  const pct = (value * 100).toFixed(1)
  const colors = { Minor: 'progress-minor', Moderate: 'progress-moderate', Major: 'progress-major' }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.5rem 0', borderBottom: '1px solid var(--color-border)' }}>
      <span style={{ width: 80, fontSize: '0.8125rem', color: 'var(--color-text-muted)', fontWeight: 500 }}>{label}</span>
      <div style={{ flex: 1 }}>
        <div className="progress-wrap">
          <div className={`progress-fill ${colors[level]}`} style={{ width: pct + '%' }} />
        </div>
      </div>
      <span style={{ width: 44, textAlign: 'right', fontSize: '0.8125rem', fontWeight: 700, fontFamily: 'monospace' }}>
        {pct}%
      </span>
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
  const [predTime, setPredTime] = useState(null)

  useEffect(() => {
    getHistory(10).then(h => setHistory(h.items || [])).catch(() => {})
  }, [])

  const handlePredict = async () => {
    if (!drugA || !drugB) { setError('Please select both Drug A and Drug B using the search fields.'); return }
    if (drugA === drugB)  { setError('Drug A and Drug B cannot be the same compound.'); return }
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await predictInteraction(drugA, drugB)
      setResult(data)
      setPredTime(new Date())
      getHistory(10).then(h => setHistory(h.items || [])).catch(() => {})
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Prediction failed. Ensure both drugs exist in the database.'
      setError(msg)
    } finally { setLoading(false) }
  }

  const handleSwap = () => {
    const tmp = drugA; setDrugA(drugB); setDrugB(tmp)
    setResult(null); setError('')
  }

  const handleReset = () => {
    setResult(null); setError(''); setDrugA(''); setDrugB(''); setPredTime(null)
  }

  return (
    <div style={{ paddingTop: 'var(--nav-height)' }}>

      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-inner">
          <nav className="breadcrumb" aria-label="Breadcrumb">
            <Link to="/">Dashboard</Link>
            <span className="breadcrumb-sep" aria-hidden="true">/</span>
            <span>Prediction Tool</span>
          </nav>
          <h1>Drug-Drug Interaction Predictor</h1>
          <p>Select two compounds to generate an interaction severity assessment.</p>
        </div>
      </div>

      <div className="container page-content">
        <div className="predict-layout">

          {/* Left — Main panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

            {/* Drug Selection */}
            <div className="card">
              <div className="card-header" style={{ marginBottom: '1.25rem' }}>
                <h2 style={{ fontSize: '0.9375rem', fontWeight: 700 }}>Drug Selection</h2>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <DrugSearchInput
                  label="Drug A (Primary)"
                  value={drugA}
                  onChange={setDrugA}
                  placeholder="e.g. Aspirin"
                  id="drug-a-input"
                />

                {/* Swap divider */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ flex: 1, height: 1, background: 'var(--color-border)' }} />
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={handleSwap}
                    title="Swap Drug A and Drug B"
                    aria-label="Swap Drug A and Drug B"
                    style={{ borderRadius: 20, padding: '0.3125rem 0.75rem', gap: '0.375rem' }}
                  >
                    <ArrowLeftRight size={13} />
                    <span style={{ fontSize: '0.75rem' }}>Swap</span>
                  </button>
                  <div style={{ flex: 1, height: 1, background: 'var(--color-border)' }} />
                </div>

                <DrugSearchInput
                  label="Drug B (Interacting)"
                  value={drugB}
                  onChange={setDrugB}
                  placeholder="e.g. Warfarin"
                  id="drug-b-input"
                />
              </div>

              {error && (
                <div className="alert alert-error" style={{ marginTop: '1rem' }} role="alert">
                  <AlertCircle size={16} aria-hidden="true" style={{ flexShrink: 0 }} />
                  <span>{error}</span>
                </div>
              )}

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem', borderTop: '1px solid var(--color-border)', paddingTop: '1.25rem' }}>
                <button
                  id="predict-btn"
                  className="btn btn-primary"
                  style={{ flex: 1, justifyContent: 'center' }}
                  onClick={handlePredict}
                  disabled={loading || !drugA || !drugB}
                  aria-busy={loading}
                >
                  {loading
                    ? <><div className="spinner" aria-hidden="true" /> Analyzing interaction...</>
                    : <><FlaskConical size={15} aria-hidden="true" /> Generate Interaction Report</>
                  }
                </button>
                {(drugA || drugB || result) && (
                  <button className="btn btn-secondary" onClick={handleReset} aria-label="Clear all fields">
                    <RotateCcw size={14} /> Clear
                  </button>
                )}
              </div>
            </div>

            {/* Clinical Report */}
            {result && (
              <div className="report-card anim-fade" role="region" aria-label="Prediction Report" aria-live="polite">

                {/* Report Header */}
                <div className="report-header">
                  <div>
                    <p style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-text-muted)', marginBottom: '0.25rem' }}>
                      Interaction Assessment Report
                    </p>
                    <p style={{ fontSize: '0.9375rem', fontWeight: 700, color: 'var(--color-text)' }}>
                      {result.drug_a} <span style={{ color: 'var(--color-text-subtle)', fontWeight: 400 }}>+</span> {result.drug_b}
                    </p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <p className="report-header-meta">
                      <strong>Model:</strong> Random Forest v1.0
                    </p>
                    <p className="report-header-meta">
                      <strong>Generated:</strong> {predTime?.toLocaleString() || '—'}
                    </p>
                  </div>
                </div>

                {/* Severity */}
                <div className={`report-section report-severity-${result.level.toLowerCase()}`}
                  style={{ background: result.level === 'Major' ? 'var(--color-major-bg)' : result.level === 'Moderate' ? 'var(--color-moderate-bg)' : 'var(--color-minor-bg)' }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                    <div>
                      <p style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: SEVERITY_COLORS[result.level], marginBottom: '0.5rem' }}>
                        Interaction Risk Level
                      </p>
                      <SeverityBadge level={result.level} size="lg" showLabel />
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <p style={{ fontSize: '0.75rem', color: SEVERITY_COLORS[result.level], fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.375rem' }}>
                        Model Confidence
                      </p>
                      <p style={{ fontSize: '1.75rem', fontWeight: 800, color: SEVERITY_COLORS[result.level], fontFeatureSettings: '"tnum"', lineHeight: 1 }}>
                        {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Clinical Recommendation */}
                <div className="report-section">
                  <p className="section-heading" style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                    <Info size={13} aria-hidden="true" /> Clinical Recommendation
                  </p>
                  <div style={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderLeft: `3px solid ${SEVERITY_COLORS[result.level]}`,
                    borderRadius: '0 var(--radius-md) var(--radius-md) 0',
                    padding: '0.875rem 1rem',
                    fontSize: '0.9rem',
                    lineHeight: 1.7,
                    color: 'var(--color-text)',
                  }}>
                    {result.warning_text}
                  </div>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: '0.75rem', fontStyle: 'italic' }}>
                    {RECOMMENDATIONS[result.level]}
                  </p>
                </div>

                {/* Probability Breakdown */}
                <div className="report-section">
                  <p className="section-heading">Probability Distribution</p>
                  <div>
                    <ProbabilityRow label="Minor" value={result.probabilities?.Minor ?? 0} level="Minor" />
                    <ProbabilityRow label="Moderate" value={result.probabilities?.Moderate ?? 0} level="Moderate" />
                    <ProbabilityRow label="Major" value={result.probabilities?.Major ?? 0} level="Major" />
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--color-text-subtle)', marginTop: '0.75rem' }}>
                    Probability estimates from Random Forest predict_proba(). Values sum to 100%.
                  </p>
                </div>

                {/* Molecular Properties */}
                {(result.drug_a_properties || result.drug_b_properties) && (
                  <div className="report-section">
                    <p className="section-heading">Molecular Properties</p>
                    <div className="mol-props-grid">
                      {result.drug_a_properties && <PropertiesTable drug={result.drug_a_properties} />}
                      {result.drug_b_properties && <PropertiesTable drug={result.drug_b_properties} />}
                    </div>
                  </div>
                )}

                {/* Footer note */}
                <div style={{ padding: '0.875rem 1.5rem', background: 'var(--color-surface)', borderTop: '1px solid var(--color-border)' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--color-text-subtle)', lineHeight: 1.65 }}>
                    <strong>Note:</strong> This report is generated by a machine learning model and is intended for research purposes only.
                    Clinical decisions must be validated by a qualified healthcare professional.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Right — Sidebar */}
          <div className="predict-sidebar">

            {/* Instructions */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">How to Use</h3>
              </div>
              {[
                'Select Drug A using the search field',
                'Select Drug B (the interacting drug)',
                'Click "Generate Interaction Report"',
                'Review the structured clinical report',
              ].map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                    background: 'var(--color-primary-bg)', border: '1px solid var(--color-primary-muted)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-primary)',
                  }}>
                    {i + 1}
                  </div>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', lineHeight: 1.5, paddingTop: '0.125rem' }}>{step}</p>
                </div>
              ))}
            </div>

            {/* Recent predictions */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <Clock size={13} aria-hidden="true" /> Recent Predictions
                </h3>
              </div>
              {history.length === 0 ? (
                <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-subtle)', textAlign: 'center', padding: '1.5rem 0' }}>
                  No predictions yet.
                </p>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Drug Pair</th>
                        <th>Level</th>
                        <th>Conf.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map(item => <HistoryRow key={item.id} item={item} />)}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

          </div>
        </div>
      </div>
    </div>
  )
}
