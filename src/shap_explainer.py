"""
shap_explainer.py
─────────────────
Uses SHAP (TreeExplainer) to explain individual predictions and
generate global feature importance visualizations.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']


def load_model_and_scaler():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    model         = joblib.load(os.path.join(base, 'random_forest_model.pkl'))
    scaler        = joblib.load(os.path.join(base, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(base, 'feature_names.pkl'))
    return model, scaler, feature_names


# ── Individual explanation ────────────────────────────────────────────────────

def explain_single(input_scaled, feature_names, model=None):
    """
    input_scaled : np.ndarray shape (1, n_features) – already scaled
    Returns dict: {feature: shap_value} for the predicted class.
    """
    if model is None:
        model, _, _ = load_model_and_scaler()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)   # list of arrays, one per class

    predicted_class = model.predict(input_scaled)[0]
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class][0]
    else:
        if len(shap_values.shape) == 3:
            sv = shap_values[0, :, predicted_class]
        else:
            sv = shap_values[0]

    impact = dict(sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True
    ))

    print(f"\n[SHAP] Prediction: {CLASS_NAMES[predicted_class]}")
    print("[SHAP] Top feature contributions:")
    for feat, val in list(impact.items())[:6]:
        arrow = '+' if val > 0 else '-'
        print(f"  {arrow} {feat:<35} {val:+.4f}")

    return predicted_class, impact, shap_values


# ── Waterfall chart for one student ──────────────────────────────────────────

def plot_waterfall(input_scaled, feature_names, model=None, save_path=None):
    """Saves a waterfall plot and returns the file path."""
    if model is None:
        model, _, _ = load_model_and_scaler()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    pred_class  = model.predict(input_scaled)[0]

    if isinstance(shap_values, list):
        sv = shap_values[pred_class][0]
    else:
        if len(shap_values.shape) == 3:
            sv = shap_values[0, :, pred_class]
        else:
            sv = shap_values[0]
    sorted_idx = np.argsort(np.abs(sv))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv[sorted_idx]]
    ax.barh([feature_names[i] for i in sorted_idx], sv[sorted_idx], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'SHAP Waterfall – Predicted: {CLASS_NAMES[pred_class]}', fontsize=13)
    ax.set_xlabel('SHAP Value (impact on prediction)')
    plt.tight_layout()

    if save_path is None:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'shap_waterfall.png')

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] Waterfall saved -> {save_path}")
    return save_path


# ── Global feature importance (summary plot) ─────────────────────────────────

def plot_global_importance(X_sample, feature_names, model=None, save_path=None):
    """
    X_sample : np.ndarray, scaled, subset of training data (e.g. 200 rows)
    """
    if model is None:
        model, _, _ = load_model_and_scaler()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP across all classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        if len(shap_values.shape) == 3:
            mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
    order    = np.argsort(mean_abs)[::-1][:10]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in order[::-1]], mean_abs[order[::-1]],
            color='#3498db')
    ax.set_title('Global Feature Importance (mean |SHAP|)', fontsize=13)
    ax.set_xlabel('Mean |SHAP Value|')
    plt.tight_layout()

    if save_path is None:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'shap_global_importance.png')

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] Global importance saved -> {save_path}")
    return save_path


if __name__ == '__main__':
    from preprocess import preprocess
    model, scaler, feature_names = load_model_and_scaler()
    X, y, _ = preprocess(apply_smote=False)

    # explain first student
    explain_single(X[:1], feature_names, model)
    plot_waterfall(X[:1], feature_names, model)
    plot_global_importance(X[:200], feature_names, model)
