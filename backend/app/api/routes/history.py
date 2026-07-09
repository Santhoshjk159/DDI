from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from app.database import get_db
from app.models.prediction import PredictionLog
from app.schemas.prediction import PredictionHistoryResponse, PredictionLogResponse

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=PredictionHistoryResponse)
async def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Fetch the most recent prediction logs."""
    count_result = await db.execute(select(func.count()).select_from(PredictionLog))
    total = count_result.scalar_one()

    result = await db.execute(
        select(PredictionLog)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
    )
    logs = result.scalars().all()

    return PredictionHistoryResponse(
        total=total,
        items=[PredictionLogResponse.model_validate(log) for log in logs],
    )
