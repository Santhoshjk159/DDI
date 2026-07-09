# DDIPredict — Full-Stack Drug Interaction Predictor

<div align="center">
  <h3>🔬 AI-Powered Drug-Drug Interaction Prediction</h3>
  <p>Predict Minor / Moderate / Major drug interaction severity using Machine Learning</p>

  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Accuracy-88.9%25-success" />
</div>

---

## 🌟 What is DDIPredict?

DDIPredict is a production-grade full-stack web application that predicts the **severity of drug-drug interactions (DDIs)** based on the molecular properties of two drugs. It uses a trained **Random Forest Classifier** to classify interactions as:

| Severity | Description |
|---|---|
| 🟢 **Minor** | Generally well-tolerated. Monitor as precaution. |
| 🟡 **Moderate** | May require dosage adjustment or monitoring. Consult a doctor. |
| 🔴 **Major** | Potentially life-threatening. Generally contraindicated. |

---

## 🏗️ Architecture

```
┌─────────────────────────────┐
│   React 18 + Vite Frontend  │  ← Port 5173
│   Recharts · Framer Motion  │
└────────────┬────────────────┘
             │ REST API
┌────────────▼────────────────┐
│   FastAPI Backend (Python)  │  ← Port 8000
│   SQLAlchemy · asyncpg      │
│   scikit-learn RF Model     │
└────────────┬────────────────┘
             │ async ORM
┌────────────▼────────────────┐
│       PostgreSQL 16         │  ← Port 5432
│   drugs · interactions      │
│   predictions_log           │
└─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Node.js 22+](https://nodejs.org/)

### 1. Start PostgreSQL
```bash
docker-compose up -d
```

### 2. Setup Backend
```bash
cd backend
uv sync
uv run python app/ml/train.py          # Train and save model
uv run python scripts/seed_db.py       # Seed database from CSV
uv run uvicorn app.main:app --reload   # Start API on port 8000
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev                            # Start React app on port 5173
```

Visit **http://localhost:5173** 🎉

---

## 📊 Dataset

- **Source:** DDInter Drug-Drug Interaction Database (Kaggle)
- **Pairs:** 27,449 drug interaction pairs
- **Features:** Molecular Weight, XLogP, Exact Mass, TPSA (for both drugs = 10 features)
- **Classes:** Minor (594) · Moderate (8,088) · Major (1,317)

---

## 🧠 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **88.91%** |
| CV Accuracy (5-fold) | **~88%** |
| Algorithm | Random Forest (200 trees) |
| Preprocessing | StandardScaler + balanced class weights |

---

## 📁 Project Structure

```
DDI/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── config.py        # Settings
│   │   ├── database.py      # SQLAlchemy async engine
│   │   ├── models/          # ORM models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── api/routes/      # API endpoints
│   │   └── ml/              # Training + inference
│   ├── scripts/seed_db.py   # DB seeder
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── pages/           # Home, Predictor, DrugBrowser, Analytics, About
│   │   ├── components/      # Navbar, Footer, DrugSearchInput, SeverityBadge
│   │   └── api.js           # Centralized API client
│   └── vite.config.js
├── dataset/                 # Raw + processed data
├── docker-compose.yml       # PostgreSQL container
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict DDI severity |
| `GET` | `/api/drugs` | List/search drugs |
| `GET` | `/api/drugs/search?q=` | Autocomplete |
| `GET` | `/api/drugs/{name}` | Drug detail + interactions |
| `GET` | `/api/history` | Recent predictions |
| `GET` | `/api/stats` | Dataset + model stats |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

*Built with ❤️ using FastAPI, React, scikit-learn, and PostgreSQL*
