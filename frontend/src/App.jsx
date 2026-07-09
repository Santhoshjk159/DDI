import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Predictor from './pages/Predictor'
import DrugBrowser from './pages/DrugBrowser'
import Analytics from './pages/Analytics'
import About from './pages/About'
import './index.css'

export default function App() {
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
