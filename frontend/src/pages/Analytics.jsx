import { useEffect, useState } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import { getStats } from '../api'
import { BarChart3, Target, TrendingUp, Layers, Activity, Database } from 'lucide-react'
import { Link } from 'react-router-dom'

const LEVEL_COLORS = {
  Minor:    '#2E7D32',
  Moderate: '#E65100',
  Major:    '#C62828',
}

const FEATURE_LABELS = {
  Drug_A_ID:              'Drug A — Identifier',
  Drug_A_MolecularWeight: 'Drug A — Mol. Weight',
  Drug_A_XLogP:           'Drug A — XLogP',
  Drug_A_ExactMass:       'Drug A — Exact Mass',
  Drug_A_TPSA:            'Drug A — TPSA',
  Drug_B_ID:              'Drug B — Identifier',
  Drug_B_MolecularWeight: 'Drug B — Mol. Weight',
  Drug_B_XLogP:           'Drug B — XLogP',
  Drug_B_ExactMass:       'Drug B — Exact Mass',
  Drug_B_TPSA:            'Drug B — TPSA',
}

function KPICard({ icon: Icon, label, value, sub, color = 'var(--color-primary)' }) {
  return (
    <div className="metric-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span className="metric-label">{label}</span>
        <div style={{
          width: 36, height: 36, borderRadius: 8,
          background: 'var(--color-surface-2)',
          border: '1px solid var(--color-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={17} color={color} aria-hidden="true" />
        </div>
      </div>
      <div className="metric-value" style={{ color }}>{value}</div>
      {sub && <p className="metric-sub">{sub}</p>}
    </div>
  )
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix) return null
  const max = Math.max(...matrix.flat())
  return (
    <div>
      <p className="section-heading">Confusion Matrix (Test Set)</p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: 4 }}>
          <thead>
            <tr>
              <th style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--color-text-muted)', padding: '0.25rem 0.5rem', textAlign: 'left' }}>Actual ↓ / Predicted →</th>
              {labels.map(l => (
                <th key={l} style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', padding: '0.375rem 0.625rem', textAlign: 'center' }}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', paddingRight: '0.75rem', verticalAlign: 'middle' }}>
                  {labels[ri]}
                </td>
                {row.map((v, ci) => {
                  const intensity = max > 0 ? v / max : 0
                  const isCorrect = ri === ci
                  const bg = isCorrect
                    ? `rgba(46,125,50,${0.1 + intensity * 0.7})`
                    : `rgba(198,40,40,${intensity * 0.35})`
                  return (
                    <td key={ci} style={{
                      background: bg,
                      borderRadius: 6,
                      padding: '0.625rem 0.875rem',
                      textAlign: 'center',
                      fontSize: '0.875rem',
                      fontWeight: 700,
                      color: isCorrect ? 'var(--color-minor)' : intensity > 0.3 ? 'var(--color-major)' : 'var(--color-text-muted)',
                      border: isCorrect ? '1px solid var(--color-minor-border)' : '1px solid transparent',
                    }}>
                      {v.toLocaleString()}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--color-text-subtle)', marginTop: '0.75rem' }}>
        Diagonal cells (green) = correctly classified. Off-diagonal (red) = misclassifications.
      </p>
    </div>
  )
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--color-bg)', border: '1px solid var(--color-border)',
      borderRadius: 'var(--radius-md)', padding: '0.625rem 0.875rem',
      fontSize: '0.8125rem', boxShadow: 'var(--shadow-lg)',
    }}>
      <p style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || 'var(--color-text)' }}>
          {p.name}: <strong>{p.value}{typeof p.value === 'number' && p.name?.includes('%') ? '' : ''}</strong>
        </p>
      ))}
    </div>
  )
}

export default function Analytics() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getStats().then(setStats).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ paddingTop: 'var(--nav-height)', minHeight: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center' }}>
        <div className="spinner" style={{ width: 32, height: 32, margin: '0 auto 1rem' }} aria-label="Loading analytics" />
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.9375rem' }}>Loading analytics data...</p>
      </div>
    </div>
  )

  if (!stats) return (
    <div style={{ paddingTop: 'var(--nav-height)', textAlign: 'center', padding: '6rem 1.5rem', color: 'var(--color-text-muted)' }}>
      <BarChart3 size={48} style={{ opacity: 0.3, margin: '0 auto 1rem' }} aria-hidden="true" />
      <p>Could not load analytics. Ensure the backend API is running.</p>
    </div>
  )

  const { model } = stats

  const featureData = model?.feature_importances
    ? Object.entries(model.feature_importances)
        .sort(([, a], [, b]) => b - a)
        .map(([k, v]) => ({ name: FEATURE_LABELS[k] || k, value: +(v * 100).toFixed(2) }))
    : []

  const levelData = (stats.level_distribution || []).map(d => ({
    name: d.level, value: d.count, color: LEVEL_COLORS[d.level] || '#888',
  }))

  const topDrugsData = (stats.top_drugs || []).slice(0, 12)

  return (
    <div style={{ paddingTop: 'var(--nav-height)' }}>

      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-inner">
          <nav className="breadcrumb" aria-label="Breadcrumb">
            <Link to="/">Dashboard</Link>
            <span className="breadcrumb-sep" aria-hidden="true">/</span>
            <span>Analytics</span>
          </nav>
          <h1>Model Performance & Analytics</h1>
          <p>Quantitative performance metrics, dataset statistics, feature importances, and classification analysis.</p>
        </div>
      </div>

      <div className="container" style={{ padding: '0 2rem 3rem' }}>

        {/* KPI Row */}
        <div className="grid-4" style={{ marginBottom: '2rem' }}>
          <KPICard
            icon={Target}
            label="Test Accuracy"
            value={model?.accuracy ? (model.accuracy * 100).toFixed(2) + '%' : '—'}
            sub={model?.cv_accuracy_mean ? `5-fold CV: ${(model.cv_accuracy_mean * 100).toFixed(1)}% ± ${(model.cv_accuracy_std * 100).toFixed(1)}%` : ''}
            color="var(--color-primary)"
          />
          <KPICard
            icon={TrendingUp}
            label="ROC-AUC (Weighted)"
            value={model?.roc_auc ? model.roc_auc.toFixed(4) : '—'}
            sub="One-vs-Rest multi-class"
            color="var(--color-secondary)"
          />
          <KPICard
            icon={Layers}
            label="Training Samples"
            value={model?.total_samples?.toLocaleString() || stats.interaction_count?.toLocaleString() || '—'}
            sub="Stratified 80/20 split"
            color="var(--color-moderate)"
          />
          <KPICard
            icon={Activity}
            label="Predictions Logged"
            value={stats.prediction_count?.toLocaleString() || '0'}
            sub="Since deployment"
            color="var(--color-minor)"
          />
        </div>

        {/* Performance Metrics Table */}
        {model?.classification_report && (
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <p className="section-heading">Per-Class Classification Report</p>
            <div className="table-wrapper">
              <table aria-label="Classification report">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(model.classification_report)
                    .filter(([k]) => !['accuracy', 'macro avg', 'weighted avg'].includes(k))
                    .map(([cls, metrics]) => (
                      <tr key={cls}>
                        <td>
                          <span style={{
                            fontWeight: 600, fontSize: '0.8125rem',
                            color: LEVEL_COLORS[cls] || 'var(--color-text)',
                          }}>{cls}</span>
                        </td>
                        <td style={{ fontFamily: 'monospace' }}>{(metrics.precision * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'monospace' }}>{(metrics.recall * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{(metrics['f1-score'] * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'monospace' }}>{metrics.support?.toLocaleString()}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Charts row 1 */}
        <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* Severity Distribution */}
          <div className="card">
            <p className="section-heading">Interaction Severity Distribution</p>
            {levelData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={levelData} dataKey="value" nameKey="name"
                      cx="50%" cy="50%" outerRadius={90} innerRadius={50} paddingAngle={2}
                    >
                      {levelData.map(entry => <Cell key={entry.name} fill={entry.color} />)}
                    </Pie>
                    <Tooltip
                      content={<ChartTooltip />}
                      formatter={(v) => v.toLocaleString()}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  {levelData.map(d => (
                    <div key={d.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8125rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, background: d.color, flexShrink: 0 }} aria-hidden="true" />
                        <span style={{ color: 'var(--color-text-muted)' }}>{d.name}</span>
                      </span>
                      <strong>{d.value.toLocaleString()}</strong>
                    </div>
                  ))}
                </div>
              </>
            ) : <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>No distribution data available.</p>}
          </div>

          {/* Feature Importances */}
          <div className="card">
            <p className="section-heading">Feature Importances (Random Forest)</p>
            {featureData.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={featureData} layout="vertical" margin={{ left: 140, right: 40 }}>
                  <XAxis
                    type="number" tick={{ fontSize: 11, fill: 'var(--color-text-muted)', fontFamily: 'Inter' }}
                    tickFormatter={v => v + '%'}
                    axisLine={{ stroke: 'var(--color-border)' }}
                    tickLine={false}
                  />
                  <YAxis
                    type="category" dataKey="name"
                    tick={{ fontSize: 11, fill: 'var(--color-text)', fontFamily: 'Inter' }}
                    width={140} tickLine={false} axisLine={false}
                  />
                  <Tooltip content={<ChartTooltip />} formatter={v => [v + '%', 'Importance']} />
                  <Bar dataKey="value" fill="var(--color-primary)" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <BarChart3 size={40} aria-hidden="true" />
                <p>No feature importance data available.</p>
              </div>
            )}
          </div>
        </div>

        {/* Charts row 2 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

          {/* Top drugs */}
          <div className="card">
            <p className="section-heading">Most Frequently Interacting Compounds</p>
            {topDrugsData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={topDrugsData} margin={{ left: 0, bottom: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 9, fill: 'var(--color-text-muted)', fontFamily: 'Inter' }}
                    angle={-35} textAnchor="end" height={70}
                    axisLine={{ stroke: 'var(--color-border)' }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: 'var(--color-text-muted)', fontFamily: 'Inter' }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip content={<ChartTooltip />} formatter={v => [v, 'Interactions']} />
                  <Bar dataKey="count" fill="var(--color-secondary)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <Database size={40} aria-hidden="true" />
                <p>No drug interaction frequency data available.</p>
              </div>
            )}
          </div>

          {/* Confusion Matrix */}
          <div className="card">
            {model?.confusion_matrix && model?.confusion_matrix_labels ? (
              <ConfusionMatrix matrix={model.confusion_matrix} labels={model.confusion_matrix_labels} />
            ) : (
              <div className="empty-state">
                <BarChart3 size={40} aria-hidden="true" />
                <p>Confusion matrix unavailable.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
