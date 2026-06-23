import os
import sys
import numpy as np
import pytest

# Ensure the src directory is in the path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shap_explainer import explain_single, plot_waterfall, load_model_and_scaler

@pytest.fixture
def model_artifacts():
    model, scaler, feature_names = load_model_and_scaler()
    return model, scaler, feature_names

def test_explain_single_returns_all_features(model_artifacts):
    model, scaler, feature_names = model_artifacts
    
    # Standard profile input
    raw_input = np.array([[5.0, 3, 0, 7.0, 2, 6.5, 0, 2.0, 5, 3]])
    input_scaled = scaler.transform(raw_input)
    
    pred_class, shap_impact_dict, shap_values = explain_single(input_scaled, feature_names, model)
    
    # (1) Test explain_single() with a known input returns a dict with all 10 feature names as keys
    assert isinstance(shap_impact_dict, dict)
    assert len(shap_impact_dict) == 10
    for name in feature_names:
        assert name in shap_impact_dict

def test_explain_single_severe_risk_top_feature(model_artifacts):
    model, scaler, feature_names = model_artifacts
    
    # Extreme severe-risk profile input
    severe_raw_input = np.array([[15.0, 8, 1, 4.0, 15, 3.0, 1, 0.0, 10, 1]])
    input_scaled = scaler.transform(severe_raw_input)
    
    pred_class, shap_impact_dict, shap_values = explain_single(input_scaled, feature_names, model)
    
    # Verify it predicts severe risk (class 3)
    assert pred_class == 3
    
    # Identify the key with highest absolute SHAP impact
    top_feature = max(shap_impact_dict, key=lambda k: abs(shap_impact_dict[k]))
    
    # (2) Test the top feature is one of: daily_screen_time_hours, stress_level, or missed_classes_per_month
    expected_top_features = {'daily_screen_time_hours', 'stress_level', 'missed_classes_per_month'}
    assert top_feature in expected_top_features

def test_plot_waterfall_creates_file(model_artifacts, tmp_path):
    model, scaler, feature_names = model_artifacts
    
    raw_input = np.array([[5.0, 3, 0, 7.0, 2, 6.5, 0, 2.0, 5, 3]])
    input_scaled = scaler.transform(raw_input)
    
    # Temporary save path using pytest's tmp_path fixture
    save_path = str(tmp_path / "shap_waterfall_test.png")
    
    # (3) Test plot_waterfall() creates a file at the returned path
    returned_path = plot_waterfall(input_scaled, feature_names, model, save_path=save_path)
    
    assert returned_path == save_path
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
