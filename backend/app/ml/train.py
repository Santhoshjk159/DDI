"""
ML Training Pipeline for DDI Prediction.
Trains a Random Forest Classifier with enhanced metrics and saves artifacts.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURES = [
    "Drug_A_ID", "Drug_A_MolecularWeight", "Drug_A_XLogP",
    "Drug_A_ExactMass", "Drug_A_TPSA",
    "Drug_B_ID", "Drug_B_MolecularWeight", "Drug_B_XLogP",
    "Drug_B_ExactMass", "Drug_B_TPSA",
]
TARGET = "Level_ID"
LABEL_MAP = {1: "Minor", 2: "Moderate", 3: "Major"}


def train_and_save(
    data_path: str = "../dataset/train_data/train_set.csv",
    model_path: str = "model_artifacts/rf_model.pkl",
    scaler_path: str = "model_artifacts/scaler.pkl",
    meta_path: str = "model_artifacts/model_meta.json",
):
    print("=" * 60)
    print("DDI Model Training Pipeline")
    print("=" * 60)

    # Load data
    print(f"\n[1/5] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna().drop_duplicates()
    print(f"      Loaded {len(df):,} rows | Features: {FEATURES}")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Scale features
    print("\n[2/5] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train model
    print("\n[3/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n[4/5] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Minor", "Moderate", "Major"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)

    # ROC AUC (multi-class OvR)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    print(f"      Test Accuracy:   {accuracy * 100:.2f}%")
    print(f"      CV Accuracy:     {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    if roc_auc:
        print(f"      ROC-AUC (OvR):  {roc_auc:.4f}")

    feature_importances = {
        feat: float(imp)
        for feat, imp in zip(FEATURES, model.feature_importances_)
    }

    # Save artifacts
    print("\n[5/5] Saving model artifacts...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "accuracy": float(accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "roc_auc": float(roc_auc) if roc_auc else None,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Minor", "Moderate", "Major"],
        "feature_importances": feature_importances,
        "features": FEATURES,
        "target": TARGET,
        "label_map": LABEL_MAP,
        "n_estimators": 200,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_samples": int(len(df)),
        "class_distribution": {
            LABEL_MAP[int(k)]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved  -> {model_path}")
    print(f"  Scaler saved -> {scaler_path}")
    print(f"  Meta saved   -> {meta_path}")
    print("\nTraining complete!")
    return model, scaler, meta


if __name__ == "__main__":
    train_and_save()
