from fastapi import APIRouter
from app.api.routes import predict, drugs, history, stats

api_router = APIRouter(prefix="/api")
api_router.include_router(predict.router)
api_router.include_router(drugs.router)
api_router.include_router(history.router)
api_router.include_router(stats.router)
