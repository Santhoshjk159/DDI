import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Predictor from './pages/Predictor'
import DrugBrowser from './pages/DrugBrowser'
import Analytics from './pages/Analytics'
import About from './pages/About'
import { warmUpBackend } from './api'
import ddiLogo from './assets/icon.png'
import './index.css'

const WARMUP_TIPS = [
  'Initializing database connection...',
  'Loading prediction model into memory...',
  'Preparing drug interaction engine...',
  'Validating molecular descriptor pipeline...',
  'Almost ready — finalizing startup checks...',
]

function WarmUpOverlay() {
  const [tipIdx, setTipIdx] = useState(0)
  const [dots, setDots] = useState('')

  useEffect(() => {
    const tipTimer = setInterval(() => {
      setTipIdx(i => (i + 1) % WARMUP_TIPS.length)
    }, 6000)
    const dotTimer = setInterval(() => {
      setDots(d => d.length >= 3 ? '' : d + '.')
    }, 500)
    return () => { clearInterval(tipTimer); clearInterval(dotTimer) }
  }, [])

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 99999,
      background: '#FFFFFF',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: '1.5rem',
    }}>
      <img
        src={ddiLogo}
        alt="DDIPredict"
        style={{ width: 80, height: 80, objectFit: 'contain' }}
      />
      <div style={{ textAlign: 'center' }}>
        <h1 style={{
          fontSize: '1.375rem', fontWeight: 700,
          color: '#0F172A', marginBottom: '0.25rem',
          fontFamily: 'Inter, sans-serif',
        }}>
          DDIPredict
        </h1>
        <p style={{
          fontSize: '0.8125rem', color: '#94A3B8',
          fontFamily: 'Inter, sans-serif',
          letterSpacing: '0.04em',
        }}>
          Drug-Drug Interaction Prediction Platform
        </p>
      </div>

      {/* Progress bar */}
      <div style={{
        width: 260, height: 3,
        background: '#E2E8F0',
        borderRadius: 2,
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          background: '#1565C0',
          borderRadius: 2,
          animation: 'warmupBar 2s ease-in-out infinite',
        }} />
      </div>

      <p style={{
        fontSize: '0.8125rem',
        color: '#475569',
        fontFamily: 'Inter, sans-serif',
        maxWidth: 320,
        textAlign: 'center',
        lineHeight: 1.5,
        minHeight: '2.5rem',
      }}>
        {WARMUP_TIPS[tipIdx]}{dots}
      </p>

      <p style={{
        fontSize: '0.6875rem',
        color: '#94A3B8',
        fontFamily: 'Inter, sans-serif',
        position: 'absolute',
        bottom: '2rem',
      }}>
        The server may take up to 60 seconds on first load
      </p>
    </div>
  )
}

export default function App() {
  const [backendReady, setBackendReady] = useState(false)
  const [checkDone, setCheckDone] = useState(false)

  useEffect(() => {
    let cancelled = false
    warmUpBackend().then(ok => {
      if (!cancelled) {
        setBackendReady(ok)
        setCheckDone(true)
      }
    })
    // Also set a fallback — if 65s pass, let the user in anyway
    const fallback = setTimeout(() => {
      if (!cancelled) { setBackendReady(true); setCheckDone(true) }
    }, 65000)
    return () => { cancelled = true; clearTimeout(fallback) }
  }, [])

  if (!checkDone) return <WarmUpOverlay />

  return (
    <BrowserRouter>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/"           element={<Home />} />
            <Route path="/predict"    element={<Predictor />} />
            <Route path="/drugs"      element={<DrugBrowser />} />
            <Route path="/analytics"  element={<Analytics />} />
            <Route path="/about"      element={<About />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}
