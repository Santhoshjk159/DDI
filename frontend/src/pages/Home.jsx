import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, Zap, Database, BarChart2, ArrowRight, Shield, Brain, FlaskConical } from 'lucide-react'
import { getStats } from '../api'

const FEATURES = [
  { icon: Zap,          title: 'Instant Prediction',  desc: 'Get AI-powered DDI severity predictions in milliseconds using our trained Random Forest model.' },
  { icon: Database,     title: '27K+ Drug Pairs',      desc: 'Backed by a dataset of 27,449 drug interactions covering Minor, Moderate, and Major severities.' },
  { icon: BarChart2,    title: 'Full Analytics',       desc: 'Explore model performance, feature importances, and interaction distributions with rich visualizations.' },
  { icon: Shield,       title: '88%+ Accuracy',        desc: 'Validated Random Forest Classifier with 5-fold cross-validation and ROC-AUC scoring.' },
  { icon: Brain,        title: 'ML-Powered',           desc: 'Built on molecular properties: Molecular Weight, XLogP, Exact Mass, and TPSA for both drugs.' },
  { icon: FlaskConical, title: 'Drug Browser',         desc: 'Search and explore our drug database with detailed molecular properties and known interactions.' },
]

export default function Home() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div>
      {/* Hero */}
      <section style={{
        minHeight: '100vh',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        textAlign: 'center', padding: '8rem 1.5rem 4rem',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Background orbs */}
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
          <div style={{ position: 'absolute', top: '15%', left: '10%', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%)', filter: 'blur(40px)' }} />
          <div style={{ position: 'absolute', bottom: '20%', right: '10%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%)', filter: 'blur(40px)' }} />
        </div>

        <div style={{ position: 'relative', zIndex: 1, maxWidth: 800, margin: '0 auto' }}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <span className="badge badge-minor" style={{ marginBottom: '1.5rem', fontSize: '0.75rem' }}>
              <Activity size={12} /> ML-Powered · Open Source
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.1 }}
            style={{ fontSize: 'clamp(2.5rem, 6vw, 4.5rem)', fontWeight: 900, lineHeight: 1.1, marginBottom: '1.5rem' }}
          >
            Predict{' '}
            <span className="gradient-text">Drug Interactions</span>{' '}
            with AI
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.2 }}
            style={{ fontSize: 'clamp(1rem, 2.5vw, 1.2rem)', color: 'var(--color-text-muted)', maxWidth: 560, margin: '0 auto 2.5rem', lineHeight: 1.7 }}
          >
            An intelligent system that identifies Minor, Moderate, and Major drug-drug interactions
            based on molecular properties — helping clinicians make safer prescribing decisions.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.3 }}
            style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}
          >
            <Link to="/predict" className="btn btn-primary btn-lg">
              Try the Predictor <ArrowRight size={18} />
            </Link>
            <Link to="/analytics" className="btn btn-secondary btn-lg">
              View Analytics
            </Link>
          </motion.div>

          {/* Stats strip */}
          {stats && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
              style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap', marginTop: '3rem' }}
            >
              <span className="stat-pill"><strong>{stats.drug_count?.toLocaleString()}</strong> Drugs</span>
              <span className="stat-pill"><strong>{stats.interaction_count?.toLocaleString()}</strong> Interactions</span>
              <span className="stat-pill"><strong>{stats.model?.accuracy ? (stats.model.accuracy * 100).toFixed(1) + '%' : '88.9%'}</strong> Accuracy</span>
              <span className="stat-pill"><strong>{stats.prediction_count?.toLocaleString() || '0'}</strong> Predictions Made</span>
            </motion.div>
          )}
        </div>
      </section>

      {/* Features */}
      <section className="section" style={{ background: 'var(--color-surface)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <h2 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '0.75rem' }}>
              Everything you need to explore DDIs
            </h2>
            <p style={{ color: 'var(--color-text-muted)', maxWidth: 500, margin: '0 auto' }}>
              A full research toolkit built with modern ML and web technologies.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.25rem' }}>
            {FEATURES.map(({ icon: Icon, title, desc }, i) => (
              <motion.div
                key={title}
                className="card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.07, duration: 0.5 }}
                viewport={{ once: true }}
              >
                <div style={{
                  width: 44, height: 44, borderRadius: 12, marginBottom: '1rem',
                  background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.1))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Icon size={22} style={{ color: 'var(--color-primary-light)' }} />
                </div>
                <h3 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>{title}</h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section style={{ padding: '5rem 1.5rem', textAlign: 'center' }}>
        <div style={{ maxWidth: 600, margin: '0 auto' }}>
          <h2 style={{ fontSize: 'clamp(1.6rem, 3.5vw, 2.2rem)', marginBottom: '1rem' }}>
            Ready to check a drug pair?
          </h2>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '2rem' }}>
            Enter two drug names and get an instant AI-powered interaction severity prediction.
          </p>
          <Link to="/predict" className="btn btn-primary btn-lg" style={{ animation: 'pulse-glow 2.5s ease-in-out infinite' }}>
            Launch Predictor <ArrowRight size={18} />
          </Link>
        </div>
      </section>
    </div>
  )
}
