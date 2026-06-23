"""
model_evaluation.py
───────────────────
Model performance analytics: confusion matrix, ROC curves,
feature importance, and multi-model comparison charts/metrics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from logging_config import get_logger

logger = get_logger('model_evaluation')

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']
EVAL_DIR_NAME = 'evaluation'


def _models_dir(base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    return os.path.abspath(base_dir)


def _eval_dir(models_dir: Optional[str] = None) -> str:
    path = os.path.join(_models_dir(models_dir), EVAL_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and optional ROC-AUC."""
    metrics: Dict[str, Any] = {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'classification_report': classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = round(
                float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')),
                4,
            )
        except Exception as exc:
            logger.warning('ROC-AUC computation failed: %s', exc)
            metrics['roc_auc'] = None
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> str:
    """Save confusion matrix heatmap and return path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Confusion matrix saved -> %s', save_path)
    return save_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> str:
    """Plot one-vs-rest ROC curves for each class."""
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, linewidth=2, label=f'{name}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-class ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('ROC curves saved -> %s', save_path)
    return save_path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    save_path: str,
    top_n: int = 10,
) -> str:
    """Plot Random Forest feature importances sorted descending."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [feature_names[i] for i in order[::-1]]
    vals = importances[order[::-1]]
    ax.barh(names, vals, color='#3498db', edgecolor='white')
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Feature importance chart saved -> %s', save_path)
    return save_path


def plot_model_comparison(
    comparison: Dict[str, Dict[str, float]],
    save_path: str,
) -> str:
    """Grouped bar chart comparing model metrics."""
    models = list(comparison.keys())
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    for i, metric in enumerate(metric_keys):
        vals = [comparison[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=metric.replace('_', ' ').title(), color=colors[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Model comparison chart saved -> %s', save_path)
    return save_path


def run_full_evaluation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_comparison: Optional[Dict[str, Dict[str, float]]] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate all evaluation artifacts and persist JSON + PNG files.

    Returns
    -------
    dict
        Evaluation results including metrics and chart paths.
    """
    eval_dir = _eval_dir(models_dir)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    primary_metrics = compute_metrics(y_test, y_pred, y_prob)
    chart_paths = {
        'confusion_matrix': plot_confusion_matrix(
            y_test, y_pred, os.path.join(eval_dir, 'confusion_matrix.png')
        ),
        'roc_curves': plot_roc_curves(
            y_test, y_prob, os.path.join(eval_dir, 'roc_curves.png')
        ),
        'feature_importance': plot_feature_importance(
            model, feature_names, os.path.join(eval_dir, 'feature_importance.png')
        ),
    }

    if model_comparison:
        chart_paths['model_comparison'] = plot_model_comparison(
            model_comparison, os.path.join(eval_dir, 'model_comparison.png')
        )

    results = {
        'primary_model': primary_metrics,
        'model_comparison': model_comparison or {},
        'chart_paths': chart_paths,
    }

    results_path = os.path.join(_models_dir(models_dir), 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    logger.info('Evaluation results saved -> %s', results_path)
    return results


def load_evaluation_results(models_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load persisted evaluation results JSON."""
    path = os.path.join(_models_dir(models_dir), 'evaluation_results.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_model_metadata(models_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load models/model_metadata.json."""
    path = os.path.join(_models_dir(models_dir), 'model_metadata.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def get_evaluation_chart_path(chart_name: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Return absolute path to a saved evaluation chart if it exists."""
    path = os.path.join(_eval_dir(models_dir), f'{chart_name}.png')
    return path if os.path.isfile(path) else None
