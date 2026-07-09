"""
ML Inference engine — loads trained model and runs predictions.
Singleton pattern so model loads once at startup.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from app.config import get_settings

settings = get_settings()

FEATURES = [
    "Drug_A_ID", "Drug_A_MolecularWeight", "Drug_A_XLogP",
    "Drug_A_ExactMass", "Drug_A_TPSA",
    "Drug_B_ID", "Drug_B_MolecularWeight", "Drug_B_XLogP",
    "Drug_B_ExactMass", "Drug_B_TPSA",
]

LEVEL_MAP = {1: "Minor", 2: "Moderate", 3: "Major"}

WARNING_TEXTS = {
    "Minor": (
        "This interaction is generally considered minor. "
        "While some interaction exists, it is unlikely to cause significant clinical problems. "
        "Monitor the patient as a precaution."
    ),
    "Moderate": (
        "This interaction is moderately significant. "
        "The combination may require dosage adjustments or additional monitoring. "
        "Consult a healthcare provider before concurrent use."
    ),
    "Major": (
        "⚠️ This interaction is MAJOR and potentially life-threatening. "
        "The combination is generally contraindicated or should only be used under strict medical supervision. "
        "Seek immediate medical advice."
    ),
}


class DDIPredictor:
    _instance: Optional["DDIPredictor"] = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "DDIPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        model_path = Path(settings.model_path)
        scaler_path = Path(settings.model_path).parent / "scaler.pkl"
        meta_path = Path(settings.model_meta_path)

        if not model_path.exists():
            print(f"[INFO] Model not found at {model_path}. Auto-training now (first deploy)...")
            try:
                from app.ml.train import train_and_save
                train_and_save(
                    model_path=str(model_path),
                    scaler_path=str(scaler_path),
                    meta_path=str(meta_path),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Auto-training failed: {e}. "
                    "Ensure dataset CSV files are present at the configured paths."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        self._loaded = True
        print(f"[OK] Model loaded from {model_path}")

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        drug_a_id: int,
        drug_a_mol_weight: float,
        drug_a_xlogp: float,
        drug_a_exact_mass: float,
        drug_a_tpsa: float,
        drug_b_id: int,
        drug_b_mol_weight: float,
        drug_b_xlogp: float,
        drug_b_exact_mass: float,
        drug_b_tpsa: float,
    ) -> dict:
        if not self._loaded:
            self.load()

        features = np.array([[
            drug_a_id, drug_a_mol_weight, drug_a_xlogp, drug_a_exact_mass, drug_a_tpsa,
            drug_b_id, drug_b_mol_weight, drug_b_xlogp, drug_b_exact_mass, drug_b_tpsa,
        ]])
        features_scaled = self.scaler.transform(features)
        level_id = int(self.model.predict(features_scaled)[0])
        proba = self.model.predict_proba(features_scaled)[0]

        # Map classes to probabilities (model classes: [1, 2, 3])
        classes = self.model.classes_
        proba_dict = {LEVEL_MAP[int(c)]: float(p) for c, p in zip(classes, proba)}
        confidence = float(max(proba))
        level = LEVEL_MAP[level_id]

        return {
            "level": level,
            "level_id": level_id,
            "confidence": confidence,
            "probabilities": proba_dict,
            "warning_text": WARNING_TEXTS[level],
        }

    def get_meta(self) -> dict:
        return self.meta


# Global predictor instance
predictor = DDIPredictor.get_instance()
