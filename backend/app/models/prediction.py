from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True, index=True)
    drug_a_name = Column(String(255), nullable=False)
    drug_b_name = Column(String(255), nullable=False)
    level = Column(String(20), nullable=False)       # Minor / Moderate / Major
    level_id = Column(Integer, nullable=False)       # 1 / 2 / 3
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=True)      # {"Minor": 0.1, "Moderate": 0.8, "Major": 0.1}
    drug_a_properties = Column(JSON, nullable=True)
    drug_b_properties = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
