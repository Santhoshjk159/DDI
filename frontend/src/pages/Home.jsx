import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, FlaskConical, Database, BarChart3,
  ShieldCheck, ArrowRight, FileText, Users
} from 'lucide-react'
import { getStats } from '../api'
import ddiLogo from '../assets/icon.png'

const FEATURES = [
  {
    icon: FlaskConical,
    title: 'Interaction Severity Prediction',
    desc: 'Classify drug-drug interactions as Minor, Moderate, or Major based on molecular descriptors of both compounds.',
  },
  {
    icon: Database,
    title: 'Compound Reference Database',
    desc: 'Browse 1,254 indexed compounds with Molecular Weight, XLogP, Exact Mass, and Topological Polar Surface Area.',
  },
  {
    icon: BarChart3,
    title: 'Performance Analytics',
    desc: 'Review classification metrics including per-class precision, recall, F1-score, confusion matrix, and feature importances.',
  },
  {
    icon: ShieldCheck,
    title: 'Validated Classification Model',
    desc: 'Trained on 27,449 drug pairs with stratified cross-validation. Balanced class weights applied to handle severity class imbalance.',
  },
  {
    icon: FileText,
    title: 'Structured Assessment Reports',
    desc: 'Each prediction generates a report with severity level, confidence score, probability breakdown, and clinical recommendation.',
  },
  {
    icon: Activity,
    title: 'Prediction Audit Log',
    desc: 'All predictions are logged with timestamps for audit trails, retrospective review, and usage tracking.',
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
      <section className="home-banner-section">
        <div className="container">
          <div className="home-banner">
            <div className="home-banner-text">
              <h1 style={{ fontSize: '1.75rem', fontWeight: 800, marginBottom: '0.75rem', lineHeight: 1.25 }}>
                Drug-Drug Interaction<br />Prediction System
              </h1>
              <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-muted)', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                A clinical decision support platform for predicting the severity of drug-drug interactions.
                Enter two compounds to receive an interaction assessment classified as
                <strong> Minor</strong>, <strong>Moderate</strong>, or <strong>Major</strong>.
              </p>
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <Link to="/predict" className="btn btn-primary">
                  <FlaskConical size={15} />
                  Open Prediction Tool
                  <ArrowRight size={14} />
                </Link>
                <Link to="/about" className="btn btn-secondary">
                  <FileText size={15} />
                  Documentation
                </Link>
              </div>
            </div>

            {/* Quick stats panel */}
            <div className="home-stats-panel">
              <p style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--color-text-muted)', marginBottom: '0.875rem' }}>
                System Overview
              </p>
              {[
                { label: 'Indexed Compounds', value: stats?.drug_count?.toLocaleString() || '—' },
                { label: 'Known Interaction Pairs', value: stats?.interaction_count?.toLocaleString() || '—' },
                { label: 'Classification Accuracy', value: accuracy },
                { label: 'Predictions Completed', value: stats?.prediction_count?.toLocaleString() || '0' },
              ].map(({ label, value }) => (
                <div key={label} className="stat-row">
                  <span style={{ color: 'var(--color-text-muted)' }}>{label}</span>
                  <strong style={{ color: 'var(--color-text)', fontFeatureSettings: '"tnum"' }}>{value}</strong>
                </div>
              ))}
              <p style={{ fontSize: '0.625rem', color: 'var(--color-text-subtle)', marginTop: '0.75rem' }}>
                Sources: DDInter Database · PubChem
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* KPI Row */}
      {stats && (
        <section style={{ padding: '1.75rem 0', background: 'var(--color-bg)', borderBottom: '1px solid var(--color-border)' }}>
          <div className="container">
            <div className="grid-4">
              <StatCard icon={Database}    label="Compounds"           value={stats.drug_count?.toLocaleString() || '—'}        color="var(--color-primary)" />
              <StatCard icon={Activity}    label="Known Interactions"   value={stats.interaction_count?.toLocaleString() || '—'} color="var(--color-secondary)" />
              <StatCard icon={ShieldCheck} label="Test Accuracy"        value={accuracy}                                        color="var(--color-minor)" />
              <StatCard icon={Users}       label="Predictions Logged"   value={stats.prediction_count?.toLocaleString() || '0'} color="var(--color-moderate)" />
            </div>
          </div>
        </section>
      )}

      {/* Capabilities */}
      <section className="section" style={{ background: 'var(--color-surface)' }}>
        <div className="container">
          <div style={{ marginBottom: '1.75rem' }}>
            <h2 style={{ marginBottom: '0.375rem' }}>Platform Capabilities</h2>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.9375rem' }}>
              Core features of the drug interaction analysis system.
            </p>
          </div>
          <div className="feature-grid">
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
                  <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section style={{ padding: '2rem 0', background: 'var(--color-bg)', borderTop: '1px solid var(--color-border)' }}>
        <div className="container">
          <div className="alert alert-warning" role="note">
            <ShieldCheck size={18} style={{ flexShrink: 0 }} aria-hidden="true" />
            <p style={{ fontSize: '0.8125rem', lineHeight: 1.65, color: 'inherit' }}>
              <strong>For Research Use Only.</strong> This system is intended for educational and research purposes.
              It should not replace professional clinical judgment, pharmacist consultation,
              or established drug reference resources.
            </p>
          </div>
        </div>
      </section>

    </div>
  )
}
