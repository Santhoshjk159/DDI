from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    drug_a: str
    drug_b: str


class DrugProperties(BaseModel):
    name: str
    mol_weight: Optional[float] = None
    xlogp: Optional[float] = None
    exact_mass: Optional[float] = None
    tpsa: Optional[float] = None


class ProbabilityBreakdown(BaseModel):
    Minor: float
    Moderate: float
    Major: float


class PredictResponse(BaseModel):
    drug_a: str
    drug_b: str
    level: str              # Minor / Moderate / Major
    level_id: int           # 1 / 2 / 3
    confidence: float       # max probability
    probabilities: ProbabilityBreakdown
    drug_a_properties: Optional[DrugProperties] = None
    drug_b_properties: Optional[DrugProperties] = None
    warning_text: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PredictionLogResponse(BaseModel):
    id: int
    drug_a_name: str
    drug_b_name: str
    level: str
    level_id: int
    confidence: float
    probabilities: Optional[dict] = None
    drug_a_properties: Optional[dict] = None
    drug_b_properties: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    total: int
    items: list[PredictionLogResponse]
