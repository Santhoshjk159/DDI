from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.database import init_db
from app.ml.predictor import predictor
from app.api import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[START] Starting DDI Predict API...")
    await init_db()
    predictor.load()
    print("[OK] DDI Predict API is ready!")
    yield
    # Shutdown
    print("[STOP] Shutting down DDI Predict API...")


app = FastAPI(
    title="DDI Predict API",
    description=(
        "Drug-Drug Interaction Prediction API powered by Machine Learning. "
        "Predicts interaction severity (Minor / Moderate / Major) based on "
        "molecular properties using a Random Forest Classifier trained on "
        "27,449 drug pairs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API
app.include_router(api_router)


@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "DDI Predict API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
        "version": "1.0.0",
    }
