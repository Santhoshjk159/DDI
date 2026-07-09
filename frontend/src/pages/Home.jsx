import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, FlaskConical, Database, BarChart3,
  ShieldCheck, ArrowRight, FileText, Users
} from 'lucide-react'
import { getStats } from '../api'

const FEATURES = [
  {
    icon: FlaskConical,
    title: 'ML-Powered Severity Prediction',
    desc: 'Predict Minor, Moderate, or Major drug-drug interaction severity using a trained Random Forest Classifier validated on 27,449 drug pairs.',
  },
  {
    icon: Database,
    title: 'Comprehensive Drug Database',
    desc: 'Search and browse 1,254+ drugs with molecular descriptors: Molecular Weight, XLogP, Exact Mass, and Topological Polar Surface Area (TPSA).',
  },
  {
    icon: BarChart3,
    title: 'Model Analytics & Metrics',
    desc: 'Examine model performance indicators including test accuracy, cross-validation scores, ROC-AUC, and feature importances derived from training data.',
  },
  {
    icon: ShieldCheck,
    title: '88.9% Test Accuracy',
    desc: 'Validated using stratified 80/20 train-test split and 5-fold cross-validation. Balanced class weights applied to handle dataset imbalance.',
  },
  {
    icon: FileText,
    title: 'Clinical Decision Reports',
    desc: 'Each prediction produces a structured clinical report with severity classification, confidence score, probability breakdown, and recommended action.',
  },
  {
    icon: Activity,
    title: 'Prediction History Tracking',
    desc: 'All predictions are logged to a PostgreSQL database with timestamps, enabling audit trails and retrospective review of past queries.',
  },
]

function StatCard({ value, label, icon: Icon, color = 'var(--color-primary)' }) {
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
    </div>
  )
}

export default function Home() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [])

  const accuracy = stats?.model?.accuracy
    ? (stats.model.accuracy * 100).toFixed(1) + '%'
    : '88.9%'

  return (
    <div style={{ paddingTop: 'var(--nav-height)' }}>

      {/* Platform Banner */}
      <section style={{
        background: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        padding: '3rem 0',
      }}>
        <div className="container">
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '2rem', flexWrap: 'wrap' }}>
            <div style={{ maxWidth: 680 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span className="badge badge-blue">
                  <ShieldCheck size={11} />
                  ML-Powered · Open Dataset
                </span>
                <span className="badge badge-neutral">Research Release v1.0</span>
              </div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.75rem', lineHeight: 1.2 }}>
                Drug-Drug Interaction Prediction System
              </h1>
              <p style={{ fontSize: '1rem', color: 'var(--color-text-muted)', lineHeight: 1.75, marginBottom: '1.5rem', maxWidth: 560 }}>
                A clinical decision support platform that predicts the severity of drug-drug interactions (DDIs)
                based on molecular properties using a trained Random Forest Classifier.
                Interactions are classified as <strong>Minor</strong>, <strong>Moderate</strong>, or <strong>Major</strong>.
              </p>
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <Link to="/predict" className="btn btn-primary">
                  <FlaskConical size={15} />
                  Open Prediction Tool
                  <ArrowRight size={14} />
                </Link>
                <Link to="/about" className="btn btn-secondary">
                  <FileText size={15} />
                  Read Documentation
                </Link>
              </div>
            </div>

            {/* Quick stats */}
            <div style={{
              background: 'var(--color-bg)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.25rem 1.5rem',
              minWidth: 260,
              boxShadow: 'var(--shadow-sm)',
            }}>
              <p style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
                Platform Statistics
              </p>
              {[
                { label: 'Total Drugs', value: stats?.drug_count?.toLocaleString() || '—' },
                { label: 'Interaction Pairs', value: stats?.interaction_count?.toLocaleString() || '—' },
                { label: 'Model Accuracy', value: accuracy },
                { label: 'Predictions Run', value: stats?.prediction_count?.toLocaleString() || '0' },
              ].map(({ label, value }) => (
                <div key={label} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '0.5rem 0', borderBottom: '1px solid var(--color-border)',
                  fontSize: '0.875rem',
                }}>
                  <span style={{ color: 'var(--color-text-muted)' }}>{label}</span>
                  <strong style={{ color: 'var(--color-text)', fontFeatureSettings: '"tnum"' }}>{value}</strong>
                </div>
              ))}
              <p style={{ fontSize: '0.6875rem', color: 'var(--color-text-subtle)', marginTop: '0.875rem' }}>
                Data sourced from DDInter · PubChem
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* KPI Row */}
      {stats && (
        <section style={{ padding: '2rem 0', background: 'var(--color-bg)', borderBottom: '1px solid var(--color-border)' }}>
          <div className="container">
            <div className="grid-4">
              <StatCard icon={Database}    label="Total Drugs"        value={stats.drug_count?.toLocaleString() || '—'}             color="var(--color-primary)" />
              <StatCard icon={Activity}    label="Known Interactions"  value={stats.interaction_count?.toLocaleString() || '—'}        color="var(--color-secondary)" />
              <StatCard icon={ShieldCheck} label="Model Accuracy"      value={accuracy}                                               color="var(--color-minor)" />
              <StatCard icon={Users}       label="Predictions Logged"  value={stats.prediction_count?.toLocaleString() || '0'}         color="var(--color-moderate)" />
            </div>
          </div>
        </section>
      )}

      {/* Capabilities */}
      <section className="section" style={{ background: 'var(--color-surface)' }}>
        <div className="container">
          <div style={{ marginBottom: '2rem' }}>
            <h2 style={{ marginBottom: '0.375rem' }}>Platform Capabilities</h2>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.9375rem' }}>
              An integrated research toolkit for drug interaction analysis.
            </p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.25rem' }}>
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{
                  width: 40, height: 40,
                  borderRadius: 8,
                  background: 'var(--color-primary-bg)',
                  border: '1px solid var(--color-primary-muted)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Icon size={19} color="var(--color-primary)" aria-hidden="true" />
                </div>
                <div>
                  <h3 style={{ fontSize: '0.9375rem', marginBottom: '0.375rem' }}>{title}</h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Notice */}
      <section style={{ padding: '2.5rem 0', background: 'var(--color-bg)', borderTop: '1px solid var(--color-border)' }}>
        <div className="container">
          <div className="alert alert-warning" role="note">
            <ShieldCheck size={18} style={{ flexShrink: 0 }} aria-hidden="true" />
            <p style={{ fontSize: '0.875rem', lineHeight: 1.65, color: 'inherit' }}>
              <strong>Research Use Only:</strong> This platform is intended for educational and research purposes.
              It should not be used as a substitute for professional clinical advice, pharmacist consultation,
              or established drug reference resources. Predictions are probabilistic and based solely on molecular
              descriptors — not on patient-specific pharmacokinetic data.
            </p>
          </div>
        </div>
      </section>

    </div>
  )
}
