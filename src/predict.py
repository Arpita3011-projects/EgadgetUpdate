"""
predict.py
──────────
Load the saved Random Forest model and make predictions
for individual students or a batch CSV.

Usage (command line):
    python predict.py                        # interactive prompt
    python predict.py --csv path/to/file.csv # batch prediction
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_config import get_logger
from recommendation import get_recommendations, calculate_addiction_score

logger = get_logger('predict')

CLASS_NAMES   = ['Low', 'Moderate', 'High', 'Severe']
RISK_EMOJI    = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴'}


def load_artifacts():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    logger.info('Loading model artifacts from %s', base)
    model         = joblib.load(os.path.join(base, 'random_forest_model.pkl'))
    scaler        = joblib.load(os.path.join(base, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(base, 'feature_names.pkl'))
    return model, scaler, feature_names


def predict_single(input_dict, model, scaler, feature_names, verbose=True):
    """
    input_dict : {feature_name: value, ...}
    Returns (risk_label, risk_index, addiction_score, probabilities)
    """
    row = np.array([[input_dict[f] for f in feature_names]])
    row_scaled = scaler.transform(row)

    risk_idx    = model.predict(row_scaled)[0]
    probs       = model.predict_proba(row_scaled)[0]
    risk_label  = CLASS_NAMES[risk_idx]
    add_score   = calculate_addiction_score(probs)

    if verbose:
        print(f"\n{'─'*45}")
        print(f"  {RISK_EMOJI[risk_idx]}  Risk Level     : {risk_label}")
        print(f"  📊  Addiction Score : {add_score} / 100")
        print(f"  🎯  Confidence      : {max(probs)*100:.1f}%")
        print(f"{'─'*45}")
        for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
            bar = '█' * int(p * 20)
            print(f"  {name:<10} {p*100:5.1f}%  {bar}")

    return risk_label, risk_idx, add_score, probs


def predict_batch(csv_path, model, scaler, feature_names):
    """Predict risk for every row in a CSV and save results."""
    df = pd.read_csv(csv_path)

    # Keep only known features
    X = df[feature_names].values
    X_scaled = scaler.transform(X)

    risk_indices = model.predict(X_scaled)
    probs        = model.predict_proba(X_scaled)

    df['predicted_risk']    = [CLASS_NAMES[i] for i in risk_indices]
    df['addiction_score']   = [calculate_addiction_score(p) for p in probs]

    out_path = csv_path.replace('.csv', '_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"[predict] Batch done – saved to {out_path}")
    print(df[['predicted_risk', 'addiction_score']].value_counts().head(10))
    return df


def interactive_predict(model, scaler, feature_names):
    print("\n=== Interactive Student Risk Predictor ===")
    prompts = {
        'daily_screen_time_hours'   : ("Daily screen time (hours, e.g. 6.5) : ", float),
        'num_social_media_platforms': ("Social media platforms (e.g. 3)     : ", int),
        'late_night_usage'          : ("Late night usage? 1=Yes 0=No        : ", int),
        'gpa'                       : ("GPA (e.g. 7.5)                      : ", float),
        'missed_classes_per_month'  : ("Missed classes/month (e.g. 2)       : ", int),
        'sleep_hours'               : ("Sleep hours/night (e.g. 6.0)        : ", float),
        'sleep_disturbances'        : ("Sleep disturbances? 1=Yes 0=No      : ", int),
        'physical_activity_hours'   : ("Physical activity hrs/week (e.g. 3) : ", float),
        'stress_level'              : ("Stress level 1-10 (e.g. 6)          : ", int),
        'social_interaction_quality': ("Social quality 1-5 (e.g. 3)         : ", int),
    }
    data = {}
    for feature in feature_names:
        prompt_text, cast = prompts.get(feature, (f"{feature}: ", float))
        while True:
            try:
                data[feature] = cast(input(prompt_text))
                break
            except ValueError:
                print("  ⚠  Please enter a valid number.")

    risk_label, risk_idx, score, probs = predict_single(
        data, model, scaler, feature_names, verbose=True
    )
    tips = get_recommendations(risk_idx, verbose=True)
    return risk_label, score, tips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E-Gadget Addiction Predictor')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV for batch prediction')
    args = parser.parse_args()

    model, scaler, feature_names = load_artifacts()

    if args.csv:
        predict_batch(args.csv, model, scaler, feature_names)
    else:
        interactive_predict(model, scaler, feature_names)
