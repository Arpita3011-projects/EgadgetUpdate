"""
preprocess.py
─────────────
Loads student_data.csv, cleans it, scales features,
applies SMOTE to balance classes, and saves the scaler.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


LABEL_MAP = {'Low': 0, 'Moderate': 1, 'High': 2, 'Severe': 3}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def preprocess(filepath=None, apply_smote=True):
    """
    Returns
    -------
    X_out        : np.ndarray  – (possibly SMOTE-balanced) feature matrix
    y_out        : np.ndarray  – corresponding labels (0-3)
    feature_names: list[str]   – column names matching X_out columns
    """

    # ── Resolve data path ────────────────────────────────────────────
    if filepath is None:
        base = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base, '..', 'data', 'student_data.csv')

    # ── 1. Load ──────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[preprocess] Loaded {len(df)} records from '{filepath}'")

    # ── 2. Handle missing values ─────────────────────────────────────
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── 3. Encode target label ───────────────────────────────────────
    df['addiction_label'] = df['addiction_label'].map(LABEL_MAP)

    # ── 4. Split features / target ───────────────────────────────────
    X = df.drop('addiction_label', axis=1)
    y = df['addiction_label'].astype(int)
    feature_names = X.columns.tolist()

    # ── 5. Scale features to [0, 1] ──────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 6. Save scaler & feature names for inference ─────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    print(f"[preprocess] Scaler saved -> models/scaler.pkl")

    # ── 7. SMOTE – balance class distribution ────────────────────────
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_out, y_out = smote.fit_resample(X_scaled, y)
        dist = pd.Series(y_out).map(LABEL_NAMES).value_counts().to_dict()
        print(f"[preprocess] After SMOTE: {dist}")
    else:
        X_out, y_out = X_scaled, y.values
        print("[preprocess] SMOTE skipped.")

    return X_out, y_out, feature_names


if __name__ == '__main__':
    X, y, features = preprocess()
    print(f"[preprocess] Done – X shape: {X.shape}")
