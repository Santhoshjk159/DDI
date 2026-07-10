import {
  FlaskConical, Database, Cpu, Layers,
  FileText, Microscope, BookOpen, AlertTriangle,
  CheckCircle2, ExternalLink, GitBranch
} from 'lucide-react'
import { Link } from 'react-router-dom'

const TOC = [
  { id: 'overview',     label: 'Overview' },
  { id: 'dataset',      label: 'Dataset' },
  { id: 'features',     label: 'Feature Engineering' },
  { id: 'methodology',  label: 'Methodology' },
  { id: 'performance',  label: 'Model Performance' },
  { id: 'limitations',  label: 'Limitations' },
  { id: 'stack',        label: 'Technology Stack' },
  { id: 'references',   label: 'References' },
]

const STACK = [
  { icon: Cpu,        label: 'FastAPI 0.111', role: 'Backend API', desc: 'Async Python web framework with auto-generated OpenAPI docs, Pydantic validation, and CORS middleware.' },
  { icon: Database,   label: 'PostgreSQL 16', role: 'Database', desc: 'Relational database via SQLAlchemy async ORM (asyncpg). Stores drugs, interactions, and prediction logs.' },
  { icon: Layers,     label: 'scikit-learn 1.4', role: 'ML Engine', desc: 'Random Forest Classifier with StandardScaler preprocessing, balanced class weights, and predict_proba confidence.' },
  { icon: FlaskConical, label: 'React 19 + Vite', role: 'Frontend', desc: 'Single-page application with react-router-dom, Recharts for data visualization, and lucide-react icons.' },
]

const FEATURES_EXPLAINED = [
  { name: 'Molecular Weight (g/mol)', desc: 'The mass of a drug molecule. Directly affects absorption, distribution, metabolism, and excretion (ADME). Calculated from atomic masses.' },
  { name: 'XLogP (Lipophilicity)', desc: 'Octanol-water partition coefficient. Quantifies lipophilicity — a key determinant of cell membrane permeability, bioavailability, and CNS penetration.' },
  { name: 'Exact Mass (Da)', desc: 'Monoisotopic mass calculated from the most abundant isotopes of each element. Used in mass spectrometry-based identification and structural verification.' },
  { name: 'TPSA (Å²)', desc: 'Topological Polar Surface Area. Predictive descriptor for drug transport, oral bioavailability, and blood-brain barrier permeability. Values >140 Å² indicate poor absorption.' },
]

const PERFORMANCE = [
  { metric: 'Test Accuracy', value: '88.91%', note: '80/20 stratified split' },
  { metric: 'Cross-Validation (5-fold)', value: '~88%', note: 'Mean ± standard deviation' },
  { metric: 'Algorithm', value: 'Random Forest', note: '200 estimators' },
  { metric: 'Class Balancing', value: 'class_weight="balanced"', note: 'Handles imbalanced classes' },
  { metric: 'Preprocessing', value: 'StandardScaler', note: 'Z-score normalization' },
  { metric: 'Features', value: '10 features', note: '4 per drug × 2 + 2 IDs' },
]

const REFERENCES = [
  { authors: 'Xiang Y, et al.', year: 2022, title: 'DDInter: an online drug–drug interaction database towards improving clinical decision-making and patient safety.', journal: 'Nucleic Acids Research', doi: 'https://doi.org/10.1093/nar/gkab880' },
  { authors: 'Kim S, et al.', year: 2019, title: 'PubChem 2019 update: improved access/citation of chemical data.', journal: 'Nucleic Acids Research', doi: 'https://pubchem.ncbi.nlm.nih.gov/' },
  { authors: 'Breiman L.', year: 2001, title: 'Random forests.', journal: 'Machine Learning', doi: 'https://doi.org/10.1023/A:1010933404324' },
  { authors: 'Pedregosa F, et al.', year: 2011, title: 'Scikit-learn: Machine Learning in Python.', journal: 'Journal of Machine Learning Research', doi: 'https://jmlr.org/papers/v12/pedregosa11a.html' },
]

function SectionAnchor({ id }) {
  return <span id={id} style={{ display: 'block', marginTop: '-80px', paddingTop: '80px' }} />
}

export default function About() {
  return (
    <div style={{ paddingTop: 'var(--nav-height)' }}>

      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-inner">
          <nav className="breadcrumb" aria-label="Breadcrumb">
            <Link to="/">Dashboard</Link>
            <span className="breadcrumb-sep" aria-hidden="true">/</span>
            <span>Documentation</span>
          </nav>
          <h1>Platform Documentation</h1>
          <p>Technical reference for the DDIPredict drug-drug interaction prediction system.</p>
        </div>
      </div>

      <div className="container page-content-lg">
        <div className="about-layout">

          {/* Sidebar TOC */}
          <aside className="about-toc">
            <div className="card" style={{ padding: '1rem' }}>
              <p style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--color-text-muted)', marginBottom: '0.75rem', paddingLeft: '0.5rem' }}>
                Contents
              </p>
              <nav aria-label="Table of contents">
                {TOC.map(({ id, label }) => (
                  <a
                    key={id}
                    href={`#${id}`}
                    style={{
                      display: 'block', padding: '0.375rem 0.5rem',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.8125rem',
                      color: 'var(--color-text-muted)',
                      transition: 'all 150ms ease',
                      textDecoration: 'none',
                    }}
                    onMouseEnter={e => { e.target.style.background = 'var(--color-primary-bg)'; e.target.style.color = 'var(--color-primary)' }}
                    onMouseLeave={e => { e.target.style.background = 'transparent'; e.target.style.color = 'var(--color-text-muted)' }}
                  >
                    {label}
                  </a>
                ))}
              </nav>
            </div>
          </aside>

          {/* Main content */}
          <main>

            {/* Overview */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="overview" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <FileText size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Overview</h2>
              </div>
              <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-muted)', lineHeight: 1.8, marginBottom: '1rem' }}>
                DDIPredict is a machine learning–based clinical decision support system for predicting
                the severity of drug-drug interactions (DDIs). Given two drug compounds, the system
                analyzes their molecular descriptors using a trained Random Forest Classifier to
                classify the interaction as <strong>Minor</strong>, <strong>Moderate</strong>, or <strong>Major</strong>.
              </p>
              <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-muted)', lineHeight: 1.8 }}>
                The platform is intended as a research demonstration and educational tool. It should not be
                used as a replacement for validated clinical drug reference systems such as Micromedex,
                Lexicomp, or the British National Formulary (BNF).
              </p>
            </section>

            <div className="divider" />

            {/* Dataset */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="dataset" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <Database size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Dataset</h2>
              </div>
              <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-muted)', lineHeight: 1.75, marginBottom: '1.25rem' }}>
                The training data was sourced from the <strong>DDInter database</strong> (Xiang et al., 2022),
                a curated repository of drug-drug interaction information.
                Molecular descriptors were retrieved from the <strong>PubChem</strong> compound database.
              </p>
              <div className="card" style={{ background: 'var(--color-surface)' }}>
                <div className="grid-3">
                  {[
                    { label: 'Total Drug Pairs', value: '27,449' },
                    { label: 'Minor Interactions', value: '594' },
                    { label: 'Moderate Interactions', value: '8,088' },
                    { label: 'Major Interactions', value: '1,317' },
                    { label: 'Unique Compounds', value: '1,254' },
                    { label: 'Molecular Features', value: '10 (4×2 + 2 IDs)' },
                  ].map(({ label, value }) => (
                    <div key={label}>
                      <p style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '0.25rem' }}>{label}</p>
                      <p style={{ fontSize: '1.125rem', fontWeight: 800, color: 'var(--color-text)', fontFeatureSettings: '"tnum"' }}>{value}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="alert alert-warning" style={{ marginTop: '1rem' }}>
                <AlertTriangle size={16} style={{ flexShrink: 0 }} aria-hidden="true" />
                <p style={{ fontSize: '0.875rem', lineHeight: 1.6, color: 'inherit' }}>
                  <strong>Class Imbalance:</strong> The dataset is heavily skewed toward Moderate interactions (≈83% of pairs).
                  Balanced class weights were applied during training to mitigate this imbalance.
                </p>
              </div>
            </section>

            <div className="divider" />

            {/* Feature Engineering */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="features" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '0.5rem' }}>
                <Microscope size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Feature Engineering</h2>
              </div>
              <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-muted)', marginBottom: '1.25rem', lineHeight: 1.7 }}>
                The model uses 4 molecular descriptors for each drug in the pair (8 features total),
                plus 2 numerical drug identifiers, for a total of 10 input features.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {FEATURES_EXPLAINED.map(({ name, desc }) => (
                  <div key={name} className="card" style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-start', padding: '1rem 1.25rem' }}>
                    <span style={{
                      fontFamily: 'Courier New, monospace', fontWeight: 700, fontSize: '0.8125rem',
                      color: 'var(--color-primary)', minWidth: 190, paddingTop: '0.1rem', flexShrink: 0,
                    }}>
                      {name}
                    </span>
                    <p style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{desc}</p>
                  </div>
                ))}
              </div>
            </section>

            <div className="divider" />

            {/* Methodology */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="methodology" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <FlaskConical size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Methodology</h2>
              </div>
              <div className="card" style={{ background: 'var(--color-surface)', marginBottom: '1rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', fontSize: '0.9rem' }}>
                  {[
                    { title: 'Algorithm', body: 'Random Forest Classifier with 200 decision trees (n_estimators=200). Ensemble of decision trees with random feature subsets at each split.' },
                    { title: 'Data Split', body: 'Stratified 80/20 train-test split ensuring proportional representation of all three severity classes in both partitions.' },
                    { title: 'Cross-Validation', body: 'Stratified 5-fold cross-validation applied on the training set for robust accuracy estimation and hyperparameter validation.' },
                    { title: 'Preprocessing', body: 'StandardScaler (z-score normalization) applied to all features before training and inference. Scaler is persisted alongside the model.' },
                    { title: 'Class Weighting', body: 'class_weight="balanced" applied to penalize misclassification of minority classes (Minor, Major) proportional to their inverse frequency.' },
                    { title: 'Inference', body: 'predict_proba() used to obtain probability estimates for all three classes. The argmax class is returned as the predicted severity level.' },
                  ].map(({ title, body }) => (
                    <div key={title}>
                      <p style={{ fontWeight: 700, marginBottom: '0.375rem', fontSize: '0.875rem' }}>{title}</p>
                      <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', lineHeight: 1.65 }}>{body}</p>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <div className="divider" />

            {/* Performance */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="performance" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <BookOpen size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Model Performance</h2>
              </div>
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <table aria-label="Model performance metrics">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {PERFORMANCE.map(({ metric, value, note }) => (
                      <tr key={metric}>
                        <td style={{ fontWeight: 600 }}>{metric}</td>
                        <td style={{ fontFamily: 'Courier New, monospace', fontWeight: 700, color: 'var(--color-primary)' }}>{value}</td>
                        <td style={{ color: 'var(--color-text-muted)', fontSize: '0.8125rem' }}>{note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <div className="divider" />

            {/* Limitations */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="limitations" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <AlertTriangle size={20} color="var(--color-moderate)" aria-hidden="true" />
                <h2>Limitations</h2>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                {[
                  'Predictions are based solely on molecular descriptors (MW, XLogP, TPSA, Exact Mass) — not on pharmacokinetic, pharmacodynamic, or patient-specific parameters.',
                  'The model does not account for drug dosage, route of administration, patient comorbidities, organ function, or genetic variability (pharmacogenomics).',
                  'The training dataset is imbalanced (Moderate class dominates). Despite class weighting, rare interaction types may be less reliably predicted.',
                  'Drugs not present in the DDInter database cannot be evaluated. The system relies on stored molecular descriptors.',
                  'Machine learning predictions are probabilistic and may produce incorrect classifications — clinical validation is always required.',
                  'This system has not been clinically validated or regulatory approved (FDA, EMA, or equivalent). It is not a medical device.',
                ].map((item, i) => (
                  <div key={i} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', padding: '0.75rem 1rem', background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)' }}>
                    <AlertTriangle size={14} color="var(--color-moderate)" style={{ flexShrink: 0, marginTop: '0.1rem' }} aria-hidden="true" />
                    <p style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)', lineHeight: 1.65 }}>{item}</p>
                  </div>
                ))}
              </div>
            </section>

            <div className="divider" />

            {/* Tech Stack */}
            <section style={{ marginBottom: '3rem' }}>
              <SectionAnchor id="stack" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <Cpu size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>Technology Stack</h2>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem' }}>
                {STACK.map(({ icon: Icon, label, role, desc }) => (
                  <div key={label} className="card" style={{ display: 'flex', gap: '1rem' }}>
                    <div style={{ width: 40, height: 40, borderRadius: 8, background: 'var(--color-primary-bg)', border: '1px solid var(--color-primary-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <Icon size={19} color="var(--color-primary)" aria-hidden="true" />
                    </div>
                    <div>
                      <p style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.125rem' }}>{label}</p>
                      <p style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--color-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.375rem' }}>{role}</p>
                      <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', lineHeight: 1.6 }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <div className="divider" />

            {/* References */}
            <section style={{ marginBottom: '2rem' }}>
              <SectionAnchor id="references" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1rem' }}>
                <BookOpen size={20} color="var(--color-primary)" aria-hidden="true" />
                <h2>References</h2>
              </div>
              <ol style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {REFERENCES.map((ref, i) => (
                  <li key={i} style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)', lineHeight: 1.7 }}>
                    {ref.authors} ({ref.year}). <em>{ref.title}</em> <em style={{ fontStyle: 'italic', fontWeight: 600, color: 'var(--color-text)' }}>{ref.journal}.</em>{' '}
                    <a href={ref.doi} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--color-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                      DOI/Link <ExternalLink size={11} aria-hidden="true" />
                    </a>
                  </li>
                ))}
              </ol>
            </section>

            {/* Source code CTA */}
            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3 style={{ marginBottom: '0.25rem' }}>View Source Code</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)' }}>
                  Full project available on GitHub — FastAPI backend, React frontend, ML training pipeline.
                </p>
              </div>
              <a
                href="https://github.com/Santhoshjk159/DDI"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-outline"
              >
                <GitBranch size={15} aria-hidden="true" /> View on GitHub
              </a>
            </div>

          </main>
        </div>
      </div>
    </div>
  )
}
