"""
train_model.py
──────────────
Trains Random Forest with hyperparameter tuning on preprocessed data.
Compares performance against Logistic Regression, Decision Tree, and SVM.
Saves the best model, metadata, and evaluation artifacts to models/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_config import get_logger
from model_evaluation import compute_metrics, run_full_evaluation
from preprocess import preprocess

logger = get_logger('train')

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']


def evaluate_model(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Fit-free evaluation; returns metric dict and prints summary."""
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        pass

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info(
        '%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f',
        name, metrics['accuracy'], metrics['precision'],
        metrics['recall'], metrics['f1_score'],
    )
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc') is not None:
        print(f"  AUC-ROC   : {metrics['roc_auc']:.4f}")
    return metrics


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """
    Hyperparameter tuning via GridSearchCV.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_cv_score)
    """
    logger.info('Starting Random Forest hyperparameter tuning')
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    base_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    search = GridSearchCV(
        base_rf,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info(
        'Best CV F1-macro=%.4f params=%s',
        search.best_score_, search.best_params_,
    )
    print(f"\n[train] Best CV F1-macro : {search.best_score_:.4f}")
    print(f"[train] Best parameters  : {search.best_params_}")
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def save_model_metadata(
    metadata: Dict[str, Any],
    models_dir: str,
) -> str:
    """Persist model metadata JSON."""
    path = os.path.join(models_dir, 'model_metadata.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    logger.info('Model metadata saved -> %s', path)
    return path


def train() -> Tuple[RandomForestClassifier, List[str]]:
    """Full training pipeline: preprocess, tune, evaluate, save."""
    X, y, feature_names = preprocess(apply_smote=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    training_samples = len(X_train)
    logger.info('Train size=%d Test size=%d', len(X_train), len(X_test))
    print(f"\n[train] Train size: {training_samples}  |  Test size: {len(X_test)}")

    print("\n[train] Tuning Random Forest (GridSearchCV) ...")
    rf, best_params, best_cv_score = tune_random_forest(X_train, y_train)

    rf_metrics = evaluate_model('Random Forest (Tuned)', rf, X_test, y_test)

    comparison_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    }

    print("\n[train] Comparison models ...")
    comparison_results: Dict[str, Dict[str, float]] = {
        'Random Forest': {
            'accuracy': rf_metrics['accuracy'],
            'precision': rf_metrics['precision'],
            'recall': rf_metrics['recall'],
            'f1_score': rf_metrics['f1_score'],
        },
    }

    for name, model in comparison_models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        comparison_results[name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
        }

    print("\n" + '=' * 50)
    print('  SUMMARY - Accuracy')
    print('=' * 50)
    for name, vals in sorted(
        comparison_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True,
    ):
        acc = vals['accuracy']
        bar = '#' * int(acc * 30)
        print(f"  {name:<22} {acc:.4f}  {bar}")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    tuning_results = {
        'best_parameters': best_params,
        'best_cv_f1_macro': round(best_cv_score, 4),
        'search_method': 'GridSearchCV',
        'scoring': 'f1_macro',
    }
    tuning_path = os.path.join(models_dir, 'tuning_results.json')
    with open(tuning_path, 'w', encoding='utf-8') as fh:
        json.dump(tuning_results, fh, indent=2)
    logger.info('Tuning results saved -> %s', tuning_path)

    metadata = {
        'algorithm': 'Random Forest (GridSearchCV tuned)',
        'accuracy': str(rf_metrics['accuracy']),
        'precision': str(rf_metrics['precision']),
        'recall': str(rf_metrics['recall']),
        'f1_score': str(rf_metrics['f1_score']),
        'roc_auc': str(rf_metrics.get('roc_auc', '')),
        'training_samples': str(training_samples),
        'features': feature_names,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_parameters': best_params,
        'best_cv_score': str(round(best_cv_score, 4)),
    }
    save_model_metadata(metadata, models_dir)

    run_full_evaluation(
        rf, X_test, y_test, feature_names,
        model_comparison=comparison_results,
        models_dir=models_dir,
    )

    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(rf, model_path)
    logger.info('Model saved -> %s', model_path)
    print(f"\n[train] Model saved -> {model_path}")

    return rf, feature_names


if __name__ == '__main__':
    train()
