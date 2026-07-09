from pydantic import BaseModel
from typing import Optional


class DrugBase(BaseModel):
    name: str
    drug_id: Optional[int] = None
    mol_weight: Optional[float] = None
    xlogp: Optional[float] = None
    exact_mass: Optional[float] = None
    tpsa: Optional[float] = None


class DrugResponse(DrugBase):
    id: int

    class Config:
        from_attributes = True


class DrugSearchResult(BaseModel):
    id: int
    name: str
    drug_id: Optional[int] = None
    mol_weight: Optional[float] = None
    xlogp: Optional[float] = None
    exact_mass: Optional[float] = None
    tpsa: Optional[float] = None

    class Config:
        from_attributes = True


class DrugListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[DrugSearchResult]


class InteractionResponse(BaseModel):
    id: int
    drug_a_name: Optional[str]
    drug_b_name: Optional[str]
    level: str
    level_id: int

    class Config:
        from_attributes = True


class DrugDetailResponse(DrugResponse):
    interactions: list[InteractionResponse] = []
