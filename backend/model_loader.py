import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Any, List, Dict

# Resolve paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_model_artifacts() -> Tuple[Any, Any, List[str]]:
    """
    Loads the trained Random Forest model, the StandardScaler, 
    and the list of feature names used during training.
    
    Returns:
        Tuple[Any, Any, List[str]]: (model, scaler, feature_names)
    """
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError(
            f"Model artifacts missing in {MODELS_DIR}. "
            "Please ensure the training script has been executed."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    return model, scaler, feature_names

def predict_single(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs prediction for a single student profile using the loaded model artifacts.
    """
    try:
        # Load model, scaler, and feature names
        model, scaler, feature_names = load_model_artifacts()

        # Convert features_dict into a pandas DataFrame and reorder columns
        df = pd.DataFrame([features_dict])
        df = df[feature_names]

        # Scale data and perform inference
        X_scaled = scaler.transform(df)
        risk_index = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]

        # Label mapping as per requirements
        risk_label_map = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
        
        # Calculate results
        confidence = float(np.max(probabilities) * 100)

        return {
            "risk_index": risk_index,
            "risk_label": risk_label_map[risk_index],
            "confidence": confidence,
            "probabilities": {
                "Low": float(probabilities[0]),
                "Moderate": float(probabilities[1]),
                "High": float(probabilities[2]),
                "Severe": float(probabilities[3])
            }
        }
    except Exception as e:
        raise RuntimeError(f"Failed to perform single prediction: {str(e)}")