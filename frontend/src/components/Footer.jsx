import { Link } from 'react-router-dom'
import { Activity, GitBranch, Heart } from 'lucide-react'

export default function Footer() {
  return (
    <footer style={{
      background: 'var(--color-surface)',
      borderTop: '1px solid var(--color-border)',
      padding: '2.5rem 1.5rem',
      marginTop: 'auto',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: '2rem', marginBottom: '2rem' }}>
          {/* Brand */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxWidth: 300 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div style={{
                width: 32, height: 32,
                background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
                borderRadius: 8,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Activity size={16} color="white" />
              </div>
              <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700 }}>
                DDI<span style={{ color: '#a855f7' }}>Predict</span>
              </span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', lineHeight: 1.7 }}>
              AI-powered drug-drug interaction prediction using machine learning to improve patient safety.
            </p>
          </div>

          {/* Links */}
          <div style={{ display: 'flex', gap: '3rem', flexWrap: 'wrap' }}>
            <div>
              <p style={{ fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>App</p>
              {[['/', 'Home'], ['/predict', 'Predictor'], ['/drugs', 'Drug Browser']].map(([to, label]) => (
                <Link key={to} to={to} style={{ display: 'block', fontSize: '0.88rem', color: 'var(--color-text-muted)', marginBottom: '0.4rem', transition: 'color 0.15s' }}
                  onMouseEnter={e => e.target.style.color = 'white'}
                  onMouseLeave={e => e.target.style.color = 'var(--color-text-muted)'}
                >{label}</Link>
              ))}
            </div>
            <div>
              <p style={{ fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>Insights</p>
              {[['/analytics', 'Analytics'], ['/about', 'About / Methodology']].map(([to, label]) => (
                <Link key={to} to={to} style={{ display: 'block', fontSize: '0.88rem', color: 'var(--color-text-muted)', marginBottom: '0.4rem', transition: 'color 0.15s' }}
                  onMouseEnter={e => e.target.style.color = 'white'}
                  onMouseLeave={e => e.target.style.color = 'var(--color-text-muted)'}
                >{label}</Link>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: '1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <p style={{ fontSize: '0.8rem', color: 'var(--color-text-subtle)' }}>
            © {new Date().getFullYear()} DDIPredict · Built with <Heart size={12} style={{ display: 'inline', color: '#ef4444' }} /> using FastAPI, React & scikit-learn
          </p>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', transition: 'color 0.15s' }}
            onMouseEnter={e => { e.currentTarget.style.color = 'white' }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--color-text-muted)' }}
          >
          <GitBranch size={16} /> View on GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
