import { motion } from 'framer-motion'
import { GitBranch, Code2, Database, Cpu, Layers, FlaskConical, BookOpen, Microscope } from 'lucide-react'

const STACK = [
  { icon: Code2,      label: 'React 18 + Vite',      desc: 'Modern SPA with fast HMR, Framer Motion animations, Recharts visualizations' },
  { icon: Cpu,        label: 'FastAPI (Python)',       desc: 'Async REST API with auto-generated OpenAPI docs, CORS, and Pydantic validation' },
  { icon: Database,   label: 'PostgreSQL',             desc: 'Relational database storing drugs, interactions, and prediction history via SQLAlchemy ORM' },
  { icon: Layers,     label: 'scikit-learn',           desc: 'Random Forest Classifier with StandardScaler, 5-fold CV, and predict_proba confidence scores' },
]

const FEATURES_EXPLAINED = [
  { name: 'Molecular Weight', desc: 'The mass of a drug molecule in g/mol. Affects absorption, distribution, and metabolism (ADME properties).' },
  { name: 'XLogP',            desc: 'Octanol-water partition coefficient. Measures lipophilicity — how well the drug permeates cell membranes.' },
  { name: 'Exact Mass',       desc: 'The precise monoisotopic mass of the molecule. Used to identify the compound via mass spectrometry.' },
  { name: 'TPSA',             desc: 'Topological Polar Surface Area in Å². Predicts drug transport, particularly across the blood-brain barrier.' },
]

const TIMELINE = [
  { step: '01', title: 'Data Collection',   desc: 'Sourced 27,449 drug pair interactions from DDInter database with severity labels.' },
  { step: '02', title: 'Feature Engineering', desc: 'Merged molecular property data for each drug using PubChem identifiers.' },
  { step: '03', title: 'Model Training',    desc: 'Trained Random Forest (200 estimators) with balanced class weights and StandardScaler.' },
  { step: '04', title: 'API Development',   desc: 'Built FastAPI backend with async PostgreSQL, model serving, and full CRUD.' },
  { step: '05', title: 'Frontend',          desc: 'Designed premium React UI with autocomplete, real-time predictions, and analytics.' },
  { step: '06', title: 'Deployment',        desc: 'Containerized with Docker Compose. Deployable to Railway, Render, or AWS.' },
]

export default function About() {
  return (
    <div style={{ paddingTop: 80 }}>
      <div className="container" style={{ padding: '3rem 1.5rem', maxWidth: 900 }}>
        {/* Hero */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: '4rem' }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.8rem)', marginBottom: '1rem', lineHeight: 1.2 }}>
            About <span className="gradient-text">DDIPredict</span>
          </h1>
          <p style={{ fontSize: '1.05rem', color: 'var(--color-text-muted)', lineHeight: 1.8, maxWidth: 720 }}>
            DDIPredict is a machine learning application that predicts the severity of drug-drug interactions (DDIs)
            based on the molecular properties of two drugs. It was built to demonstrate the practical application of
            ML in healthcare and serves as a portfolio project showcasing full-stack development with Python, React, and PostgreSQL.
          </p>
        </motion.div>

        {/* Tech Stack */}
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Code2 size={22} style={{ color: 'var(--color-primary-light)' }} /> Tech Stack
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: '1rem' }}>
            {STACK.map(({ icon: Icon, label, desc }, i) => (
              <motion.div key={label} className="card" style={{ display: 'flex', gap: '1rem' }}
                initial={{ opacity: 0, y: 15 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }} viewport={{ once: true }}
              >
                <div style={{ width: 42, height: 42, background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.1))', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Icon size={20} style={{ color: 'var(--color-primary-light)' }} />
                </div>
                <div>
                  <p style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.3rem' }}>{label}</p>
                  <p style={{ fontSize: '0.83rem', color: 'var(--color-text-muted)', lineHeight: 1.6 }}>{desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Feature Explanations */}
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Microscope size={22} style={{ color: 'var(--color-primary-light)' }} /> Molecular Features Explained
          </h2>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
            The model uses these 4 properties for each drug (8 features total) to predict interaction severity.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {FEATURES_EXPLAINED.map(({ name, desc }) => (
              <div key={name} className="card" style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-start' }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 800, fontSize: '0.9rem', color: 'var(--color-primary-light)', minWidth: 140, paddingTop: '0.1rem' }}>{name}</span>
                <p style={{ fontSize: '0.88rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Model Performance */}
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FlaskConical size={22} style={{ color: 'var(--color-primary-light)' }} /> Model Methodology
          </h2>
          <div className="card" style={{ background: 'var(--color-surface-2)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', fontSize: '0.9rem' }}>
              <div>
                <p style={{ fontWeight: 700, marginBottom: '0.5rem' }}>Algorithm</p>
                <p style={{ color: 'var(--color-text-muted)' }}>Random Forest Classifier with 200 decision trees, balanced class weights to handle class imbalance, and StandardScaler normalization.</p>
              </div>
              <div>
                <p style={{ fontWeight: 700, marginBottom: '0.5rem' }}>Validation</p>
                <p style={{ color: 'var(--color-text-muted)' }}>5-fold Stratified Cross-Validation for robust accuracy estimation. 80/20 train-test split with stratification by severity class.</p>
              </div>
              <div>
                <p style={{ fontWeight: 700, marginBottom: '0.5rem' }}>Dataset</p>
                <p style={{ color: 'var(--color-text-muted)' }}>27,449 drug pairs from the DDInter database. Classes: Minor (594), Moderate (8,088), Major (1,317) — inherently imbalanced.</p>
              </div>
              <div>
                <p style={{ fontWeight: 700, marginBottom: '0.5rem' }}>Limitations</p>
                <p style={{ color: 'var(--color-text-muted)' }}>Predictions are based solely on molecular properties. Real clinical decisions require considering patient history, dosage, and comorbidities.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Timeline */}
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BookOpen size={22} style={{ color: 'var(--color-primary-light)' }} /> Project Journey
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {TIMELINE.map(({ step, title, desc }, i) => (
              <motion.div key={step} initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.07 }} viewport={{ once: true }}
                style={{ display: 'flex', gap: '1.25rem', paddingBottom: '1.5rem', position: 'relative' }}
              >
                {/* Line */}
                {i < TIMELINE.length - 1 && <div style={{ position: 'absolute', left: 18, top: 40, bottom: 0, width: 2, background: 'var(--color-border)' }} />}
                <div style={{ width: 36, height: 36, background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', fontWeight: 800, color: 'white', flexShrink: 0 }}>
                  {step}
                </div>
                <div style={{ paddingTop: '0.4rem' }}>
                  <p style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.2rem' }}>{title}</p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', lineHeight: 1.6 }}>{desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* GitHub CTA */}
        <div className="card" style={{ textAlign: 'center', background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.08))', borderColor: 'rgba(124,58,237,0.25)' }}>
          <GitBranch size={32} style={{ margin: '0 auto 0.75rem', color: 'var(--color-text-muted)' }} />
          <h3 style={{ fontSize: '1.15rem', marginBottom: '0.5rem' }}>View Source Code</h3>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.88rem', marginBottom: '1.25rem' }}>
            The full project is open-source — FastAPI backend, React frontend, ML pipeline, and Docker setup.
          </p>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="btn btn-primary">
            <GitBranch size={16} /> View on GitHub
          </a>
        </div>
      </div>
    </div>
  )
}
