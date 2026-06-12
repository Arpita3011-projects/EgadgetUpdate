"""
train_model.py
──────────────
Trains Random Forest on preprocessed + SMOTE-balanced data.
Compares performance against Logistic Regression, Decision Tree, and SVM.
Saves the best model (Random Forest) to models/.
"""

import os
import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess, LABEL_NAMES


CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        print(f"  AUC-ROC  : {auc:.4f}")
    except Exception:
        pass
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    return acc


def train():
    # ── Data ────────────────────────────────────────────────────────
    X, y, feature_names = preprocess(apply_smote=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[train] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Primary model: Random Forest ────────────────────────────────
    print("\n[train] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest (Primary)", rf, X_test, y_test)

    # ── Comparison models ────────────────────────────────────────────
    comparison_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'SVM'                : SVC(probability=True, kernel='rbf', random_state=42),
    }

    print("\n[train] Comparison models ...")
    results = {'Random Forest': accuracy_score(y_test, rf.predict(X_test))}
    for name, model in comparison_models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)

    print("\n" + '=' * 50)
    print("  SUMMARY - Accuracy")
    print('=' * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = '#' * int(acc * 30)
        print(f"  {name:<22} {acc:.4f}  {bar}")

    # ── Save Random Forest ───────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(rf, model_path)
    print(f"\n[train] Model saved -> {model_path}")

    return rf, feature_names


if __name__ == '__main__':
    train()
