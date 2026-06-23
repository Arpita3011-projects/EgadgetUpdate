import sys
from pathlib import Path
import numpy as np
import warnings
import pytest
from sklearn.base import InconsistentVersionWarning

# ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shap_explainer import explain_single, load_model_and_scaler
from form_column_mapper import MODEL_FEATURES


def make_candidate(kind: str) -> dict:
    """Return a candidate feature dict for 'low', 'moderate', or 'severe'."""
    if kind == 'low':
        return {
            "daily_screen_time_hours": 1,
            "num_social_media_platforms": 0,
            "late_night_usage": 0,
            "gpa": 9.0,
            "missed_classes_per_month": 0,
            "sleep_hours": 8.0,
            "sleep_disturbances": 0,
            "physical_activity_hours": 7.0,
            "stress_level": 1,
            "social_interaction_quality": 5,
        }
    if kind == 'moderate':
        return {
            "daily_screen_time_hours": 6,
            "num_social_media_platforms": 3,
            "late_night_usage": 0,
            "gpa": 7.0,
            "missed_classes_per_month": 2,
            "sleep_hours": 6.0,
            "sleep_disturbances": 0,
            "physical_activity_hours": 3.0,
            "stress_level": 5,
            "social_interaction_quality": 3,
        }
    # severe
    return {
        "daily_screen_time_hours": 14,
        "num_social_media_platforms": 8,
        "late_night_usage": 1,
        "gpa": 4.0,
        "missed_classes_per_month": 10,
        "sleep_hours": 3.0,
        "sleep_disturbances": 1,
        "physical_activity_hours": 0.0,
        "stress_level": 10,
        "social_interaction_quality": 1,
    }


def sv_for_class(explainer, shap_values, pred_class):
    if isinstance(shap_values, list):
        sv = shap_values[pred_class][0]
    else:
        if len(shap_values.shape) == 3:
            sv = shap_values[0, :, pred_class]
        else:
            sv = shap_values[0]
    return np.array(sv)


def test_explain_single_properties():
    # silence sklearn version and scaler feature-name warnings for test clarity
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")
    # load model + scaler + feature names
    model, scaler, feature_names = load_model_and_scaler()

    assert len(feature_names) == len(MODEL_FEATURES) == 10

    classes = ['low', 'moderate', 'severe']
    found = {}

    for kind in classes:
        cand = make_candidate(kind)
        input_arr = np.array([[cand.get(f, 0) for f in MODEL_FEATURES]])
        input_scaled = scaler.transform(input_arr)

        pred_class, shap_impact, shap_values = explain_single(input_scaled, feature_names, model)

        # (1) shap_impact is a dict with exactly len(feature_names) keys matching feature_names
        assert isinstance(shap_impact, dict)
        assert set(shap_impact.keys()) == set(feature_names)
        assert len(shap_impact) == len(feature_names)

        found[kind] = (pred_class, shap_impact, shap_values, input_scaled)

    # identify severe candidate result
    severe_pred, severe_impact, severe_shap_values, severe_scaled = found['severe']

    # (2) for Severe student, the top-SHAP feature has a positive value
    top_feat = next(iter(severe_impact.items()))
    top_val = top_feat[1]
    assert top_val > 0, f"Expected top SHAP for severe to be positive, got {top_val}"

    # (3) sum of SHAP values ~ (predicted_class_prob - base_value)
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(severe_scaled)
    sv = sv_for_class(explainer, shap_vals, severe_pred)

    # get expected/base value for the predicted class
    base = explainer.expected_value[severe_pred] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
    prob = float(model.predict_proba(severe_scaled)[0][severe_pred])
    sum_shap = float(np.sum(sv))

    # Allow a small tolerance
    assert abs(sum_shap - (prob - base)) < 1e-2
