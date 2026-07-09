from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models.drug import Drug, Interaction
from app.models.prediction import PredictionLog
from app.ml.predictor import predictor

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Global dataset statistics for the analytics dashboard."""

    # Drug count
    drug_count_result = await db.execute(select(func.count()).select_from(Drug))
    drug_count = drug_count_result.scalar_one()

    # Interaction count
    interaction_count_result = await db.execute(select(func.count()).select_from(Interaction))
    interaction_count = interaction_count_result.scalar_one()

    # Prediction log count
    pred_count_result = await db.execute(select(func.count()).select_from(PredictionLog))
    pred_count = pred_count_result.scalar_one()

    # Interaction level distribution
    level_dist_result = await db.execute(
        select(Interaction.level, func.count(Interaction.level).label("count"))
        .group_by(Interaction.level)
        .order_by(func.count(Interaction.level).desc())
    )
    level_distribution = [
        {"level": row.level, "count": row.count}
        for row in level_dist_result.all()
    ]

    # Top 15 most interactive drugs (by total interaction count)
    top_drugs_result = await db.execute(
        select(
            Interaction.drug_a_name.label("name"),
            func.count().label("count")
        )
        .group_by(Interaction.drug_a_name)
        .order_by(func.count().desc())
        .limit(15)
    )
    top_drugs = [{"name": row.name, "count": row.count} for row in top_drugs_result.all()]

    # Model meta (feature importances, accuracy etc.)
    model_meta = predictor.get_meta()

    return {
        "drug_count": drug_count,
        "interaction_count": interaction_count,
        "prediction_count": pred_count,
        "level_distribution": level_distribution,
        "top_drugs": top_drugs,
        "model": {
            "accuracy": model_meta.get("accuracy"),
            "cv_accuracy_mean": model_meta.get("cv_accuracy_mean"),
            "cv_accuracy_std": model_meta.get("cv_accuracy_std"),
            "roc_auc": model_meta.get("roc_auc"),
            "feature_importances": model_meta.get("feature_importances"),
            "confusion_matrix": model_meta.get("confusion_matrix"),
            "confusion_matrix_labels": model_meta.get("confusion_matrix_labels"),
            "classification_report": model_meta.get("classification_report"),
            "n_estimators": model_meta.get("n_estimators"),
            "total_samples": model_meta.get("total_samples"),
            "class_distribution": model_meta.get("class_distribution"),
        },
    }
