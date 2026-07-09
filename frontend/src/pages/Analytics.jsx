import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import { getStats } from '../api'
import { BarChart2, Target, TrendingUp, Layers } from 'lucide-react'

const LEVEL_COLORS = { Minor: '#10b981', Moderate: '#f59e0b', Major: '#ef4444' }
const FEATURE_LABELS = {
  Drug_A_ID:              'Drug A — ID',
  Drug_A_MolecularWeight: 'Drug A — Mol. Weight',
  Drug_A_XLogP:           'Drug A — XLogP',
  Drug_A_ExactMass:       'Drug A — Exact Mass',
  Drug_A_TPSA:            'Drug A — TPSA',
  Drug_B_ID:              'Drug B — ID',
  Drug_B_MolecularWeight: 'Drug B — Mol. Weight',
  Drug_B_XLogP:           'Drug B — XLogP',
  Drug_B_ExactMass:       'Drug B — Exact Mass',
  Drug_B_TPSA:            'Drug B — TPSA',
}

function MetricCard({ icon: Icon, label, value, sub, color }) {
  return (
    <motion.div className="metric-card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <p className="metric-label">{label}</p>
        <div style={{ width: 36, height: 36, background: color + '20', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon size={18} style={{ color }} />
        </div>
      </div>
      <p className="metric-value" style={{ color }}>{value}</p>
      {sub && <p className="metric-sub">{sub}</p>}
    </motion.div>
  )
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix) return null
  const max = Math.max(...matrix.flat())
  return (
    <div>
      <p style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>Confusion Matrix</p>
      <div style={{ display: 'grid', gridTemplateColumns: `auto repeat(${labels.length}, 1fr)`, gap: 3 }}>
        <div />
        {labels.map(l => <div key={l} style={{ textAlign: 'center', fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', padding: '0.25rem' }}>{l}</div>)}
        {matrix.map((row, ri) => [
          <div key={`r${ri}`} style={{ display: 'flex', alignItems: 'center', fontSize: '0.72rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', paddingRight: '0.5rem' }}>{labels[ri]}</div>,
          ...row.map((v, ci) => {
            const intensity = v / max
            const bg = ri === ci ? `rgba(16,185,129,${0.15 + intensity * 0.6})` : `rgba(239,68,68,${intensity * 0.4})`
            return (
              <div key={ci} style={{ background: bg, borderRadius: 6, padding: '0.65rem', textAlign: 'center', fontSize: '0.88rem', fontWeight: 700, color: 'white' }}>
                {v.toLocaleString()}
              </div>
            )
          })
        ])}
      </div>
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
    <div style={{ paddingTop: 80, minHeight: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center' }}>
        <div className="spinner" style={{ width: 36, height: 36, margin: '0 auto 1rem', borderWidth: 3 }} />
        <p style={{ color: 'var(--color-text-muted)' }}>Loading analytics...</p>
      </div>
    </div>
  )

  if (!stats) return (
    <div style={{ paddingTop: 80, textAlign: 'center', padding: '5rem 1.5rem', color: 'var(--color-text-muted)' }}>
      Could not load analytics. Make sure the backend is running.
    </div>
  )

  const { model } = stats

  // Feature importance data sorted
  const featureData = model?.feature_importances
    ? Object.entries(model.feature_importances)
        .sort(([, a], [, b]) => b - a)
        .map(([k, v]) => ({ name: FEATURE_LABELS[k] || k, value: +(v * 100).toFixed(2) }))
    : []

  // Level distribution for pie
  const levelData = (stats.level_distribution || []).map(d => ({
    name: d.level, value: d.count, color: LEVEL_COLORS[d.level] || '#888',
  }))

  // Top drugs bar chart
  const topDrugsData = (stats.top_drugs || []).slice(0, 12)

  return (
    <div style={{ paddingTop: 80 }}>
      <div className="container" style={{ padding: '3rem 1.5rem' }}>
        <div style={{ marginBottom: '2.5rem' }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '0.5rem' }}>
            <span className="gradient-text">Analytics</span> Dashboard
          </h1>
          <p style={{ color: 'var(--color-text-muted)' }}>
            Model performance metrics, dataset insights, and feature analysis.
          </p>
        </div>

        {/* Metric cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
          <MetricCard icon={Target}    label="Model Accuracy"  value={model?.accuracy ? (model.accuracy * 100).toFixed(2) + '%' : '—'}          sub={model?.cv_accuracy_mean ? `CV: ${(model.cv_accuracy_mean * 100).toFixed(1)}% ±${(model.cv_accuracy_std * 100).toFixed(1)}%` : ''} color="var(--color-primary-light)" />
          <MetricCard icon={TrendingUp} label="ROC-AUC (OvR)"  value={model?.roc_auc ? model.roc_auc.toFixed(4) : '—'}                          sub="Weighted multi-class"  color="var(--color-secondary)"      />
          <MetricCard icon={Layers}     label="Total Samples"   value={model?.total_samples?.toLocaleString() || stats.interaction_count?.toLocaleString() || '—'} sub="Training dataset"  color="var(--color-moderate)"       />
          <MetricCard icon={BarChart2}  label="Predictions Run" value={stats.prediction_count?.toLocaleString() || '0'}                            sub="Since deployment"     color="var(--color-minor)"          />
        </div>

        {/* Charts row */}
        <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
          {/* Pie chart */}
          <div className="card">
            <p style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>Severity Distribution</p>
            {levelData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={levelData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95} innerRadius={55} paddingAngle={3}>
                    {levelData.map(entry => <Cell key={entry.name} fill={entry.color} />)}
                  </Pie>
                  <Tooltip formatter={(v) => v.toLocaleString()} contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                  <Legend formatter={(v) => <span style={{ fontSize: '0.85rem', color: 'var(--color-text)' }}>{v}</span>} />
                </PieChart>
              </ResponsiveContainer>
            ) : <p style={{ color: 'var(--color-text-muted)' }}>No data</p>}
          </div>

          {/* Feature importance */}
          <div className="card">
            <p style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>Feature Importances (%)</p>
            {featureData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={featureData} layout="vertical" margin={{ left: 130 }}>
                  <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} tickFormatter={v => v + '%'} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: 'var(--color-text)' }} width={130} />
                  <Tooltip formatter={v => v + '%'} contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                  <Bar dataKey="value" fill="url(#grad)" radius={[0, 4, 4, 0]}>
                    <defs>
                      <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#7c3aed" />
                        <stop offset="100%" stopColor="#06b6d4" />
                      </linearGradient>
                    </defs>
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: 'var(--color-text-muted)' }}>Train model to see feature importances.</p>}
          </div>
        </div>

        {/* Bottom row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          {/* Top interactive drugs */}
          <div className="card">
            <p style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>Top Interacting Drugs</p>
            {topDrugsData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={topDrugsData} margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} angle={-35} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} />
                  <Tooltip contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                  <Bar dataKey="count" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: 'var(--color-text-muted)' }}>Seed the database to see top drugs.</p>}
          </div>

          {/* Confusion Matrix */}
          <div className="card">
            {model?.confusion_matrix && model?.confusion_matrix_labels ? (
              <ConfusionMatrix matrix={model.confusion_matrix} labels={model.confusion_matrix_labels} />
            ) : (
              <div className="empty-state">
                <BarChart2 size={48} />
                <p>Confusion matrix will appear after model training.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
