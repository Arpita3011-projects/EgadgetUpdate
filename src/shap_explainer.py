"""
shap_explainer.py
─────────────────
Uses SHAP (TreeExplainer) to explain individual predictions and
generate global feature importance visualizations.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_config import get_logger

logger = get_logger('shap')

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']

# Module-level explainer cache keyed by model object id
_EXPLAINER_CACHE: Dict[int, shap.TreeExplainer] = {}


def load_model_and_scaler(
    models_dir: Optional[str] = None,
) -> Tuple[Any, Any, List[str]]:
    """Load trained model, scaler, and feature names from disk."""
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    logger.info('Loading model artifacts from %s', models_dir)
    model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    return model, scaler, feature_names


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Return a cached TreeExplainer for the given model instance.

    Reuses explainer objects to avoid expensive repeated initialization.
    """
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        logger.info('Creating new SHAP TreeExplainer')
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]


def _extract_class_shap(
    shap_values: Union[List[np.ndarray], np.ndarray],
    predicted_class: int,
) -> np.ndarray:
    """Extract SHAP values for a single sample and predicted class."""
    if isinstance(shap_values, list):
        return np.array(shap_values[predicted_class][0])
    if len(shap_values.shape) == 3:
        return np.array(shap_values[0, :, predicted_class])
    return np.array(shap_values[0])


def _get_base_value(explainer: shap.TreeExplainer, predicted_class: int) -> float:
    """Return expected/base value for the predicted class."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        return float(ev[predicted_class])
    return float(ev)


def explain_single(
    input_scaled: np.ndarray,
    feature_names: List[str],
    model: Optional[Any] = None,
    explainer: Optional[shap.TreeExplainer] = None,
) -> Tuple[int, Dict[str, float], Union[List[np.ndarray], np.ndarray]]:
    """
    Explain a single scaled input.

    Parameters
    ----------
    input_scaled : np.ndarray
        Shape (1, n_features), already scaled.
    feature_names : list[str]
        Feature column names.
    model : sklearn estimator, optional
        If None, loads from disk.
    explainer : shap.TreeExplainer, optional
        Reuse a cached explainer when provided.

    Returns
    -------
    tuple
        (predicted_class, {feature: shap_value}, raw_shap_values)
    """
    if model is None:
        model, _, _ = load_model_and_scaler()

    if explainer is None:
        explainer = get_tree_explainer(model)

    logger.info('Computing SHAP values for single prediction')
    shap_values = explainer.shap_values(input_scaled)
    predicted_class = int(model.predict(input_scaled)[0])
    sv = _extract_class_shap(shap_values, predicted_class)

    impact = dict(sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    ))

    logger.debug('SHAP prediction: %s', CLASS_NAMES[predicted_class])
    return predicted_class, impact, shap_values


def plot_waterfall(
    input_scaled: np.ndarray,
    feature_names: List[str],
    model: Optional[Any] = None,
    save_path: Optional[str] = None,
    explainer: Optional[shap.TreeExplainer] = None,
    max_display: int = 10,
) -> str:
    """
    Generate an official SHAP waterfall plot and save as PNG.

    Uses ``shap.plots.waterfall`` (SHAP >= 0.44) with fallback to
    ``shap.waterfall_plot`` for older versions.
    """
    if model is None:
        model, _, _ = load_model_and_scaler()

    if explainer is None:
        explainer = get_tree_explainer(model)

    shap_values = explainer.shap_values(input_scaled)
    pred_class = int(model.predict(input_scaled)[0])
    sv = _extract_class_shap(shap_values, pred_class)
    base_value = _get_base_value(explainer, pred_class)

    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=input_scaled[0],
        feature_names=feature_names,
    )

    if save_path is None:
        reports_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'reports',
        )
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'shap_waterfall.png')

    logger.info('Generating SHAP waterfall plot -> %s', save_path)
    plt.figure(figsize=(10, 6))
    try:
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
    except (AttributeError, TypeError):
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('SHAP waterfall saved -> %s', save_path)
    return save_path


def plot_global_importance(
    X_sample: np.ndarray,
    feature_names: List[str],
    model: Optional[Any] = None,
    save_path: Optional[str] = None,
    explainer: Optional[shap.TreeExplainer] = None,
) -> str:
    """
    Plot mean |SHAP| global feature importance from a sample batch.

    Parameters
    ----------
    X_sample : np.ndarray
        Scaled feature matrix subset (e.g. 200 rows).
    """
    if model is None:
        model, _, _ = load_model_and_scaler()

    if explainer is None:
        explainer = get_tree_explainer(model)

    logger.info('Computing global SHAP importance (%d samples)', len(X_sample))
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    order = np.argsort(mean_abs)[::-1][:10]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color='#3498db',
    )
    ax.set_title('Global Feature Importance (mean |SHAP|)', fontsize=13)
    ax.set_xlabel('Mean |SHAP Value|')
    plt.tight_layout()

    if save_path is None:
        reports_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'reports',
        )
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'shap_global_importance.png')

    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info('Global importance saved -> %s', save_path)
    return save_path


def clear_explainer_cache() -> None:
    """Clear module-level SHAP explainer cache."""
    _EXPLAINER_CACHE.clear()


if __name__ == '__main__':
    from preprocess import preprocess

    model, scaler, feature_names = load_model_and_scaler()
    X, y, _ = preprocess(apply_smote=False)
    explainer = get_tree_explainer(model)

    explain_single(X[:1], feature_names, model, explainer=explainer)
    plot_waterfall(X[:1], feature_names, model, explainer=explainer)
    plot_global_importance(X[:200], feature_names, model, explainer=explainer)
