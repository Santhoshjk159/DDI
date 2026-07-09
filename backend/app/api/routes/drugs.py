from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from app.database import get_db
from app.models.drug import Drug, Interaction
from app.schemas.drug import DrugListResponse, DrugSearchResult, DrugDetailResponse, InteractionResponse

router = APIRouter(prefix="/drugs", tags=["Drugs"])


@router.get("", response_model=DrugListResponse)
async def list_drugs(
    search: str = Query(default="", description="Search by drug name"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size

    query = select(Drug)
    count_query = select(func.count()).select_from(Drug)

    if search:
        like_filter = Drug.name.ilike(f"%{search}%")
        query = query.where(like_filter)
        count_query = count_query.where(like_filter)

    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    query = query.order_by(Drug.name).offset(offset).limit(page_size)
    result = await db.execute(query)
    drugs = result.scalars().all()

    return DrugListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[DrugSearchResult.model_validate(d) for d in drugs],
    )


@router.get("/search", response_model=list[DrugSearchResult])
async def search_drugs(
    q: str = Query(min_length=1),
    limit: int = Query(default=10, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Autocomplete search for drug names."""
    result = await db.execute(
        select(Drug)
        .where(Drug.name.ilike(f"%{q}%"))
        .order_by(Drug.name)
        .limit(limit)
    )
    drugs = result.scalars().all()
    return [DrugSearchResult.model_validate(d) for d in drugs]


@router.get("/{name}", response_model=DrugDetailResponse)
async def get_drug(name: str, db: AsyncSession = Depends(get_db)):
    """Get a single drug with all its known interactions."""
    result = await db.execute(
        select(Drug).where(Drug.name.ilike(name))
    )
    drug = result.scalar_one_or_none()
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug '{name}' not found")

    # Get interactions
    interactions_result = await db.execute(
        select(Interaction).where(
            or_(
                Interaction.drug_a_id == drug.drug_id,
                Interaction.drug_b_id == drug.drug_id,
            )
        ).limit(100)
    )
    interactions = interactions_result.scalars().all()

    return DrugDetailResponse(
        **DrugSearchResult.model_validate(drug).model_dump(),
        interactions=[
            InteractionResponse(
                id=i.id,
                drug_a_name=i.drug_a_name,
                drug_b_name=i.drug_b_name,
                level=i.level,
                level_id=i.level_id,
            )
            for i in interactions
        ],
    )
