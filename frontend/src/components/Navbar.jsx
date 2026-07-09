import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Activity, Menu, X } from 'lucide-react'
import ddiLogo from '../assets/icon.png'

const NAV_LINKS = [
  { to: '/',          label: 'Dashboard' },
  { to: '/predict',   label: 'Predict' },
  { to: '/drugs',     label: 'Drug Database' },
  { to: '/analytics', label: 'Analytics' },
  { to: '/about',     label: 'Documentation' },
]

export default function Navbar() {
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar-inner">

        {/* Logo */}
        <Link to="/" className="navbar-logo" aria-label="DDIPredict home">
          <img
            src={ddiLogo}
            alt="DDI Prediction Platform Logo"
            aria-hidden="true"
            style={{
              width: 42,
              height: 42,
              objectFit: 'contain',
              flexShrink: 0,
            }}
          />
          <div>
            <div className="navbar-logo-name">
              DDI<span>Predict</span>
            </div>
            <div className="navbar-tagline">Clinical Decision Support</div>
          </div>
        </Link>

        {/* Desktop Links */}
        <div className="navbar-links hide-mobile" role="menubar">
          {NAV_LINKS.map(link => {
            const active = location.pathname === link.to
            return (
              <Link
                key={link.to}
                to={link.to}
                className={`nav-link${active ? ' active' : ''}`}
                role="menuitem"
                aria-current={active ? 'page' : undefined}
              >
                {link.label}
              </Link>
            )
          })}
        </div>

        {/* CTA */}
        <div className="navbar-cta hide-mobile">
          <Link to="/predict" className="btn btn-primary btn-sm">
            <Activity size={14} />
            Run Prediction
          </Link>
        </div>

        {/* Mobile toggle */}
        <button
          className="mobile-menu-btn show-mobile"
          onClick={() => setMenuOpen(o => !o)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="mobile-dropdown" role="menu">
          {NAV_LINKS.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={`nav-link${location.pathname === link.to ? ' active' : ''}`}
              role="menuitem"
            >
              {link.label}
            </Link>
          ))}
          <div style={{ paddingTop: '0.75rem', borderTop: '1px solid var(--color-border)', marginTop: '0.5rem' }}>
            <Link to="/predict" className="btn btn-primary btn-sm" style={{ width: '100%', justifyContent: 'center' }}>
              <Activity size={14} /> Run Prediction
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}
