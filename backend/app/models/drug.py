from sqlalchemy import Column, Integer, String, Float, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class Drug(Base):
    __tablename__ = "drugs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    drug_id = Column(Integer, nullable=True, index=True)
    mol_weight = Column(Float, nullable=True)
    xlogp = Column(Float, nullable=True)
    exact_mass = Column(Float, nullable=True)
    tpsa = Column(Float, nullable=True)

    # Relationships
    interactions_as_a = relationship(
        "Interaction",
        foreign_keys="Interaction.drug_a_id",
        back_populates="drug_a",
        lazy="select",
    )
    interactions_as_b = relationship(
        "Interaction",
        foreign_keys="Interaction.drug_b_id",
        back_populates="drug_b",
        lazy="select",
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    drug_a_id = Column(Integer, nullable=False, index=True)
    drug_b_id = Column(Integer, nullable=False, index=True)
    drug_a_name = Column(String(255), nullable=True)
    drug_b_name = Column(String(255), nullable=True)
    level = Column(String(20), nullable=False)   # Minor / Moderate / Major
    level_id = Column(Integer, nullable=False)   # 1 / 2 / 3

    drug_a = relationship(
        "Drug",
        foreign_keys=[drug_a_id],
        back_populates="interactions_as_a",
        primaryjoin="Interaction.drug_a_id == Drug.drug_id",
    )
    drug_b = relationship(
        "Drug",
        foreign_keys=[drug_b_id],
        back_populates="interactions_as_b",
        primaryjoin="Interaction.drug_b_id == Drug.drug_id",
    )
