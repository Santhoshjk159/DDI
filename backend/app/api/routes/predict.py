from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.drug import Drug
from app.models.prediction import PredictionLog
from app.schemas.prediction import PredictRequest, PredictResponse, DrugProperties, ProbabilityBreakdown
from app.ml.predictor import predictor
from datetime import datetime, timezone

router = APIRouter(prefix="/predict", tags=["Predict"])


def _drug_to_props(drug: Drug) -> DrugProperties:
    return DrugProperties(
        name=drug.name,
        mol_weight=drug.mol_weight,
        xlogp=drug.xlogp,
        exact_mass=drug.exact_mass,
        tpsa=drug.tpsa,
    )


@router.post("", response_model=PredictResponse)
async def predict_interaction(
    req: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Predict the severity of interaction between two drugs.
    Accepts drug names, looks up their molecular properties from the DB,
    runs inference, and logs the prediction.
    """
    # Look up Drug A
    result_a = await db.execute(
        select(Drug).where(Drug.name.ilike(req.drug_a.strip()))
    )
    drug_a = result_a.scalar_one_or_none()
    if not drug_a:
        raise HTTPException(status_code=404, detail=f"Drug '{req.drug_a}' not found in database.")

    # Look up Drug B
    result_b = await db.execute(
        select(Drug).where(Drug.name.ilike(req.drug_b.strip()))
    )
    drug_b = result_b.scalar_one_or_none()
    if not drug_b:
        raise HTTPException(status_code=404, detail=f"Drug '{req.drug_b}' not found in database.")

    if drug_a.id == drug_b.id:
        raise HTTPException(status_code=400, detail="Cannot predict interaction of a drug with itself.")

    # Validate properties
    for drug, label in [(drug_a, req.drug_a), (drug_b, req.drug_b)]:
        if any(v is None for v in [drug.drug_id, drug.mol_weight, drug.xlogp, drug.exact_mass, drug.tpsa]):
            raise HTTPException(
                status_code=422,
                detail=f"Drug '{label}' has incomplete molecular properties."
            )

    # Run prediction
    result = predictor.predict(
        drug_a_id=drug_a.drug_id,
        drug_a_mol_weight=drug_a.mol_weight,
        drug_a_xlogp=drug_a.xlogp,
        drug_a_exact_mass=drug_a.exact_mass,
        drug_a_tpsa=drug_a.tpsa,
        drug_b_id=drug_b.drug_id,
        drug_b_mol_weight=drug_b.mol_weight,
        drug_b_xlogp=drug_b.xlogp,
        drug_b_exact_mass=drug_b.exact_mass,
        drug_b_tpsa=drug_b.tpsa,
    )

    props_a = _drug_to_props(drug_a)
    props_b = _drug_to_props(drug_b)
    now = datetime.now(timezone.utc)

    # Log to DB
    log = PredictionLog(
        drug_a_name=drug_a.name,
        drug_b_name=drug_b.name,
        level=result["level"],
        level_id=result["level_id"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        drug_a_properties=props_a.model_dump(),
        drug_b_properties=props_b.model_dump(),
    )
    db.add(log)
    await db.commit()

    return PredictResponse(
        drug_a=drug_a.name,
        drug_b=drug_b.name,
        level=result["level"],
        level_id=result["level_id"],
        confidence=result["confidence"],
        probabilities=ProbabilityBreakdown(**result["probabilities"]),
        drug_a_properties=props_a,
        drug_b_properties=props_b,
        warning_text=result["warning_text"],
        created_at=now,
    )
