"""
Database seeder — reads merged_data.csv and populates PostgreSQL.
Run: uv run python scripts/seed_db.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, text
from app.config import get_settings
from app.database import Base
from app.models.drug import Drug, Interaction
from app.models.prediction import PredictionLog  # noqa: F401

settings = get_settings()


async def seed():
    print("=" * 60)
    print("DDI Database Seeder")
    print("=" * 60)

    engine = create_async_engine(settings.database_url, echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[OK] Tables created")

    # Load CSV
    data_path = settings.merged_data_path
    print(f"\n[LOAD] Loading: {data_path}")
    df = pd.read_csv(data_path).dropna().drop_duplicates()
    print(f"   {len(df):,} rows loaded")

    async with SessionLocal() as session:
        # Check if already seeded
        existing = await session.execute(select(Drug).limit(1))
        if existing.scalar_one_or_none():
            print("\n[SKIP] Database already seeded. Skipping drug seeding.")
        else:
            # Build unique drugs from both Drug_A and Drug_B columns
            print("\n[SEED] Seeding drugs...")
            drug_records: dict[str, Drug] = {}

            cols_a = ["Drug_A", "Drug_A_ID", "Drug_A_MolecularWeight",
                      "Drug_A_XLogP", "Drug_A_ExactMass", "Drug_A_TPSA"]
            cols_b = ["Drug_B", "Drug_B_ID", "Drug_B_MolecularWeight",
                      "Drug_B_XLogP", "Drug_B_ExactMass", "Drug_B_TPSA"]

            for _, row in df[cols_a].drop_duplicates(subset=["Drug_A"]).iterrows():
                name = str(row["Drug_A"]).strip()
                if name not in drug_records:
                    drug_records[name] = Drug(
                        name=name,
                        drug_id=int(row["Drug_A_ID"]),
                        mol_weight=float(row["Drug_A_MolecularWeight"]),
                        xlogp=float(row["Drug_A_XLogP"]),
                        exact_mass=float(row["Drug_A_ExactMass"]),
                        tpsa=float(row["Drug_A_TPSA"]),
                    )

            for _, row in df[cols_b].drop_duplicates(subset=["Drug_B"]).iterrows():
                name = str(row["Drug_B"]).strip()
                if name not in drug_records:
                    drug_records[name] = Drug(
                        name=name,
                        drug_id=int(row["Drug_B_ID"]),
                        mol_weight=float(row["Drug_B_MolecularWeight"]),
                        xlogp=float(row["Drug_B_XLogP"]),
                        exact_mass=float(row["Drug_B_ExactMass"]),
                        tpsa=float(row["Drug_B_TPSA"]),
                    )

            session.add_all(list(drug_records.values()))
            await session.commit()
            print(f"   [OK] {len(drug_records):,} drugs seeded")

        # Check interactions
        existing_inter = await session.execute(select(Interaction).limit(1))
        if existing_inter.scalar_one_or_none():
            print("\n[SKIP] Interactions already seeded. Skipping.")
        else:
            print("\n[LINK] Seeding interactions (batch inserts)...")
            BATCH = 500
            interactions = []
            for _, row in df.iterrows():
                interactions.append(Interaction(
                    drug_a_id=int(row["Drug_A_ID"]),
                    drug_b_id=int(row["Drug_B_ID"]),
                    drug_a_name=str(row["Drug_A"]).strip(),
                    drug_b_name=str(row["Drug_B"]).strip(),
                    level=str(row["Level"]).strip(),
                    level_id=int(row["Level_ID"]),
                ))

                if len(interactions) >= BATCH:
                    session.add_all(interactions)
                    await session.commit()
                    interactions = []

            if interactions:
                session.add_all(interactions)
                await session.commit()

            print(f"   [OK] {len(df):,} interactions seeded")

    await engine.dispose()
    print("\n[DONE] Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed())
