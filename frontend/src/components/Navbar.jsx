import { useState, useEffect, useRef } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Activity, Menu, X } from 'lucide-react'

const NAV_LINKS = [
  { to: '/',          label: 'Home' },
  { to: '/predict',   label: 'Predictor' },
  { to: '/drugs',     label: 'Drug Browser' },
  { to: '/analytics', label: 'Analytics' },
  { to: '/about',     label: 'About' },
]

export default function Navbar() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
      background: scrolled ? 'rgba(10,10,20,0.92)' : 'transparent',
      backdropFilter: scrolled ? 'blur(20px)' : 'none',
      borderBottom: scrolled ? '1px solid rgba(255,255,255,0.06)' : '1px solid transparent',
      transition: 'all 0.3s ease',
      padding: '0 1.5rem',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 64 }}>
        {/* Logo */}
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', textDecoration: 'none' }}>
          <div style={{
            width: 36, height: 36,
            background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
            borderRadius: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={20} color="white" />
          </div>
          <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: '1.15rem' }}>
            DDI<span style={{ color: '#a855f7' }}>Predict</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <div className="hide-mobile" style={{ display: 'flex', gap: '0.25rem' }}>
          {NAV_LINKS.map(link => {
            const active = location.pathname === link.to
            return (
              <Link key={link.to} to={link.to} style={{
                padding: '0.45rem 1rem',
                borderRadius: 8,
                fontSize: '0.9rem',
                fontWeight: active ? 600 : 400,
                color: active ? 'white' : 'var(--color-text-muted)',
                background: active ? 'rgba(124,58,237,0.2)' : 'transparent',
                transition: 'all 0.15s ease',
                textDecoration: 'none',
              }}
                onMouseEnter={e => { if (!active) e.target.style.color = 'white' }}
                onMouseLeave={e => { if (!active) e.target.style.color = 'var(--color-text-muted)' }}
              >
                {link.label}
              </Link>
            )
          })}
        </div>

        {/* CTA */}
        <div className="hide-mobile">
          <Link to="/predict" className="btn btn-primary btn-sm">
            Try Predictor →
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMenuOpen(o => !o)}
          style={{ display: 'none', background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}
          className="show-mobile"
          aria-label="Toggle menu"
        >
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem',
          margin: '0 0 1rem',
        }}>
          {NAV_LINKS.map(link => (
            <Link key={link.to} to={link.to} style={{
              display: 'block', padding: '0.65rem 1rem', borderRadius: 8,
              color: location.pathname === link.to ? 'white' : 'var(--color-text-muted)',
              background: location.pathname === link.to ? 'rgba(124,58,237,0.2)' : 'transparent',
              fontSize: '0.95rem', fontWeight: 500, textDecoration: 'none',
            }}>
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  )
}
