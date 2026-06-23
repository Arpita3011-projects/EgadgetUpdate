"""
dashboard.py
────────────
Streamlit web dashboard for the E-Gadget Addiction Prediction System.

Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import sys
import datetime
from typing import Any, List, Tuple, Optional

import requests

API_BASE_URL = "http://127.0.0.1:8000"

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def api_request(method: str, endpoint: str, json=None, files=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(
            method,
            url,
            json=json,
            files=files,
            timeout=30
        )
        response.raise_for_status()
        return response
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(APP_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from logging_config import get_logger, setup_logging
from model_evaluation import (
    get_evaluation_chart_path,
    load_evaluation_results,
    load_model_metadata,
)
from recommendation import get_recommendations, calculate_addiction_score, get_alert_message
from report_generator import generate_pdf_report
from shap_explainer import explain_single, get_tree_explainer, plot_waterfall

setup_logging()
logger = get_logger('dashboard')

st.set_page_config(
    page_title="E-Gadget Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-low      { color: #27ae60; font-size: 1.8rem; font-weight: 700; }
    .risk-moderate { color: #f39c12; font-size: 1.8rem; font-weight: 700; }
    .risk-high     { color: #e67e22; font-size: 1.8rem; font-weight: 700; }
    .risk-severe   { color: #c0392b; font-size: 1.8rem; font-weight: 700; }
    .score-box { background:#f0f2f6; border-radius:10px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts() -> Tuple[Any, Any, List[str]]:
    """Load and cache model, scaler, and feature names."""
    models_dir = os.path.join(ROOT_DIR, 'models')
    logger.info('Loading model artifacts')
    model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    return model, scaler, feature_names


@st.cache_resource
def load_shap_explainer(_model: Any) -> Any:
    """Cache SHAP TreeExplainer for the loaded model."""
    return get_tree_explainer(_model)


@st.cache_data(show_spinner=False)
def cached_shap_explanation(
    input_tuple: Tuple[float, ...],
    feature_names_tuple: Tuple[str, ...],
) -> Tuple[int, dict]:
    """Cache SHAP explanation by input feature values."""
    model, _, feature_names = load_artifacts()
    explainer = load_shap_explainer(model)
    input_arr = np.array([list(input_tuple)])
    _, scaler, _ = load_artifacts()
    input_scaled = scaler.transform(input_arr)
    pred_class, impact, _ = explain_single(
        input_scaled, list(feature_names_tuple), model, explainer=explainer,
    )
    return pred_class, impact


CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']
RISK_COLORS = {'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#e67e22', 'Severe': '#c0392b'}
RISK_CSS = {
    'Low': 'risk-low', 'Moderate': 'risk-moderate',
    'High': 'risk-high', 'Severe': 'risk-severe',
}
RISK_EMOJI = {'Low': '🟢', 'Moderate': '🟡', 'High': '🟠', 'Severe': '🔴'}

st.title("📱 Electronic Gadget Addiction Risk Predictor")
st.caption("AI-powered early detection system for students · KLE College of Engineering")
st.markdown("---")

try:
    model, scaler, feature_names = load_artifacts()
    shap_explainer = load_shap_explainer(model)
    model_ready = True
except Exception as e:
    model_ready = False
    logger.error('Model load failed: %s', e)
    st.error(f"⚠️  Model not found. Please run `python src/train_model.py` first.\n\n`{e}`")

with st.sidebar:
    st.header("📋 Student Profile")
    student_name = st.text_input("Student Name (optional)", value="")

    st.subheader("📱 Usage Behaviour")
    screen_time = st.slider("Daily Screen Time (hours)", 1.0, 16.0, 5.0, 0.5)
    social_media = st.slider("Social Media Platforms", 1, 8, 3, 1)
    late_night = st.selectbox("Late Night Usage?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

    st.subheader("🎓 Academic Indicators")
    gpa = st.slider("GPA", 4.0, 10.0, 7.0, 0.1)
    missed_classes = st.slider("Missed Classes / Month", 0, 15, 2, 1)

    st.subheader("😴 Sleep & Physical Health")
    sleep_hours = st.slider("Sleep Hours / Night", 3.0, 9.0, 6.5, 0.5)
    sleep_disturbances = st.selectbox("Sleep Disturbances?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 2.0, 0.5)

    st.subheader("🧠 Psychological Parameters")
    stress_level = st.slider("Stress Level (1 – 10)", 1, 10, 5)
    social_quality = st.slider("Social Interaction Quality (1 – 5)", 1, 5, 3)

    predict_btn = st.button("🔍 Predict Risk Level", width='stretch', disabled=not model_ready)

if predict_btn and model_ready:
    health = api_request("GET", "/health")

    if health:
        st.success("Backend Connected")
    else:
        st.warning("Backend Not Reachable")

    logger.info('Prediction requested for student: %s', student_name or 'Anonymous')
    input_values = [
        screen_time, social_media, late_night, gpa, missed_classes,
        sleep_hours, sleep_disturbances, physical_activity,
        stress_level, social_quality,
    ]

    # Payload creation (kept as per instruction)
    payload = {
        "daily_screen_time_hours": screen_time,
        "num_social_media_platforms": social_media,
        "late_night_usage": late_night,
        "gpa": gpa,
        "missed_classes_per_month": missed_classes,
        "sleep_hours": sleep_hours,
        "sleep_disturbances": sleep_disturbances,
        "physical_activity_hours": physical_activity,
        "stress_level": stress_level,
        "social_interaction_quality": social_quality,
        "student_name": student_name if student_name else None,
    }
    input_arr = np.array([input_values])
    input_scaled = scaler.transform(input_arr)
    response = api_request("POST", "/predict", json=payload)

    if not response:
        st.stop()

    res_data = response.json()["data"]

    risk_idx = res_data["risk_index"]

    probs = np.array([
        res_data["probabilities"]["Low"],
        res_data["probabilities"]["Moderate"],
        res_data["probabilities"]["High"],
        res_data["probabilities"]["Severe"]
    ])
    risk_label = CLASS_NAMES[risk_idx]
    add_score = calculate_addiction_score(probs)
    confidence = float(max(probs)) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Level", f"{RISK_EMOJI[risk_label]}  {risk_label}")
    c2.metric("Addiction Score", f"{add_score} / 100")
    c3.metric("Confidence", f"{confidence:.1f}%")
    c4.metric("Student", student_name if student_name else "—")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Risk Level Probabilities")
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_colors = [RISK_COLORS[c] for c in CLASS_NAMES]
        ax.barh(CLASS_NAMES, probs * 100, color=bar_colors)
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)
        for i, v in enumerate(probs * 100):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("🔍 Feature Input Summary")
        summary_df = pd.DataFrame({'Feature': feature_names, 'Value': input_values})
        st.dataframe(summary_df, width='stretch', hide_index=True)

    st.markdown("---")

    st.subheader("💡 Personalised Recommendations")
    tips = get_recommendations(risk_idx, verbose=False)
    tip_cols = st.columns(2)
    for idx, tip in enumerate(tips):
        tip_cols[idx % 2].info(tip)

    alert = get_alert_message(risk_idx)
    if alert:
        st.error(alert)

    st.markdown("---")
    st.subheader("🎯 Addiction Risk Score Gauge")
    fig2, ax2 = plt.subplots(figsize=(8, 1.2))
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax2.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    ax2.axvline(add_score, color='black', linewidth=3)
    ax2.text(add_score, 0.5, f" {add_score}", va='center', fontsize=11, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlabel('Addiction Risk Score (0 = No Risk, 100 = Severe)')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("🔍 Prediction Diagnostics (SHAP Waterfall)")

    os.makedirs(os.path.join(ROOT_DIR, 'reports'), exist_ok=True)
    with st.spinner("Calculating SHAP feature impact..."):
        _, shap_impact_dict = cached_shap_explanation(
            tuple(input_values), tuple(feature_names),
        )
        waterfall_path = plot_waterfall(
            input_scaled, feature_names, model,
            explainer=shap_explainer,
        )

    top_feature = max(shap_impact_dict, key=lambda k: abs(shap_impact_dict[k]))
    st.image(
        waterfall_path,
        caption="SHAP waterfall plot showing feature contributions to the prediction",
        width='stretch',
    )
    shap_tips = get_recommendations(risk_idx, top_feature=top_feature, verbose=False)

    st.markdown("---")
    st.subheader("📥 Export Assessment Report")

    with st.spinner("Generating PDF report..."):
        pdf_path = generate_pdf_report(
            student_name=student_name,
            risk_label=risk_label,
            score=add_score,
            tips=shap_tips,
            shap_impact=shap_impact_dict,
            waterfall_img=waterfall_path,
            confidence_pct=confidence,
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    safe_student_name = (student_name or "student").replace(" ", "_")
    report_filename = f"{safe_student_name}_{datetime.date.today().strftime('%Y-%m-%d')}_report.pdf"

    st.download_button(
        label="📄 Download Assessment PDF Report",
        data=pdf_bytes,
        file_name=report_filename,
        mime="application/pdf",
        use_container_width=True,
    )

else:
    st.subheader("Welcome 👋")
    st.markdown("""
    This system uses a **Random Forest** machine learning model to predict
    electronic gadget addiction risk in students across **four levels**:
    🟢 Low · 🟡 Moderate · 🟠 High · 🔴 Severe

    **How to use:**
    1. Fill in the student profile in the **left sidebar**
    2. Click **Predict Risk Level**
    3. View the risk classification, addiction score, and personalised recommendations
    """)

    st.info("👈 Fill in the student details in the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Algorithm", "Random Forest")
    col2.metric("Risk Categories", "4 Levels")
    col3.metric("XAI", "SHAP Integrated")

    # Model metadata display
    metadata = load_model_metadata(os.path.join(ROOT_DIR, 'models'))
    if metadata:
        st.markdown("---")
        st.subheader("📋 Model Metadata")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", metadata.get('accuracy', 'N/A'))
        mc2.metric("Precision", metadata.get('precision', 'N/A'))
        mc3.metric("Recall", metadata.get('recall', 'N/A'))
        mc4.metric("F1-Score", metadata.get('f1_score', 'N/A'))
        st.caption(
            f"Algorithm: {metadata.get('algorithm', 'Random Forest')} · "
            f"Trained: {metadata.get('date_trained', 'N/A')} · "
            f"Samples: {metadata.get('training_samples', 'N/A')}"
        )
        if metadata.get('best_parameters'):
            with st.expander("Best Hyperparameters"):
                st.json(metadata['best_parameters'])

    # Model Performance Analytics
    st.markdown("---")
    st.subheader("📈 Model Performance Analytics")

    eval_results = load_evaluation_results(os.path.join(ROOT_DIR, 'models'))
    if eval_results is None:
        st.info("Run `python src/train_model.py` to generate evaluation charts and metrics.")
    else:
        primary = eval_results.get('primary_model', {})
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        ec1.metric("Accuracy", primary.get('accuracy', 'N/A'))
        ec2.metric("Precision", primary.get('precision', 'N/A'))
        ec3.metric("Recall", primary.get('recall', 'N/A'))
        ec4.metric("F1-Score", primary.get('f1_score', 'N/A'))
        ec5.metric("ROC-AUC", primary.get('roc_auc', 'N/A'))

        chart_a, chart_b = st.columns(2)
        models_dir = os.path.join(ROOT_DIR, 'models')

        cm_path = get_evaluation_chart_path('confusion_matrix', models_dir)
        if cm_path:
            with chart_a:
                st.markdown("**Confusion Matrix**")
                st.image(cm_path, use_container_width=True)

        roc_path = get_evaluation_chart_path('roc_curves', models_dir)
        if roc_path:
            with chart_b:
                st.markdown("**ROC Curves (Multi-class)**")
                st.image(roc_path, use_container_width=True)

        chart_c, chart_d = st.columns(2)
        fi_path = get_evaluation_chart_path('feature_importance', models_dir)
        if fi_path:
            with chart_c:
                st.markdown("**Feature Importance (Random Forest)**")
                st.image(fi_path, use_container_width=True)

        cmp_path = get_evaluation_chart_path('model_comparison', models_dir)
        if cmp_path:
            with chart_d:
                st.markdown("**Model Comparison**")
                st.image(cmp_path, use_container_width=True)

        comparison = eval_results.get('model_comparison', {})
        if comparison:
            st.markdown("**Model Comparison Table**")
            cmp_df = pd.DataFrame(comparison).T
            cmp_df.index.name = 'Model'
            cmp_df = cmp_df[['accuracy', 'precision', 'recall', 'f1_score']]
            st.dataframe(cmp_df.style.format('{:.4f}'), use_container_width=True)

        report = primary.get('classification_report', {})
        if report:
            with st.expander("Classification Report (per class)"):
                rows = []
                for cls in CLASS_NAMES:
                    if cls in report:
                        rows.append({
                            'Class': cls,
                            'Precision': report[cls].get('precision', 0),
                            'Recall': report[cls].get('recall', 0),
                            'F1-Score': report[cls].get('f1-score', 0),
                            'Support': report[cls].get('support', 0),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Project by:** Arpita Pradhane · Mouneshwar Sutar · "
        "Parasuram Sanadi · Pratiksha Pawar  |  "
        "KLE College of Engineering and Technology, Chikodi · 2025-26"
    )
