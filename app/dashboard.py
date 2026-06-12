"""
dashboard.py
────────────
Streamlit web dashboard for the E-Gadget Addiction Prediction System.

Run:
    streamlit run app/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(APP_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from recommendation import get_recommendations, calculate_addiction_score, get_alert_message
from shap_explainer import explain_single, plot_waterfall

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Gadget Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-low      { color: #27ae60; font-size: 1.8rem; font-weight: 700; }
    .risk-moderate { color: #f39c12; font-size: 1.8rem; font-weight: 700; }
    .risk-high     { color: #e67e22; font-size: 1.8rem; font-weight: 700; }
    .risk-severe   { color: #c0392b; font-size: 1.8rem; font-weight: 700; }
    .score-box { background:#f0f2f6; border-radius:10px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Artefact loading (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir    = os.path.join(ROOT_DIR, 'models')
    model         = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    scaler        = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    return model, scaler, feature_names

CLASS_NAMES  = ['Low', 'Moderate', 'High', 'Severe']
RISK_COLORS  = {'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#e67e22', 'Severe': '#c0392b'}
RISK_CSS     = {'Low': 'risk-low', 'Moderate': 'risk-moderate',
                'High': 'risk-high', 'Severe': 'risk-severe'}
RISK_EMOJI   = {'Low': '🟢', 'Moderate': '🟡', 'High': '🟠', 'Severe': '🔴'}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📱 Electronic Gadget Addiction Risk Predictor")
st.caption("AI-powered early detection system for students · KLE College of Engineering")
st.markdown("---")

# ── Try loading model ─────────────────────────────────────────────────────────
try:
    model, scaler, feature_names = load_artifacts()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"⚠️  Model not found. Please run `python src/train_model.py` first.\n\n`{e}`")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Student Profile")
    student_name = st.text_input("Student Name (optional)", value="")

    st.subheader("📱 Usage Behaviour")
    screen_time  = st.slider("Daily Screen Time (hours)", 1.0, 16.0, 5.0, 0.5)
    social_media = st.slider("Social Media Platforms",    1,    8,   3,   1)
    late_night   = st.selectbox("Late Night Usage?",
                                options=[0, 1],
                                format_func=lambda x: "Yes" if x else "No")

    st.subheader("🎓 Academic Indicators")
    gpa             = st.slider("GPA",                    4.0, 10.0, 7.0, 0.1)
    missed_classes  = st.slider("Missed Classes / Month", 0,   15,   2,   1)

    st.subheader("😴 Sleep & Physical Health")
    sleep_hours        = st.slider("Sleep Hours / Night",         3.0, 9.0, 6.5, 0.5)
    sleep_disturbances = st.selectbox("Sleep Disturbances?",
                                      options=[0, 1],
                                      format_func=lambda x: "Yes" if x else "No")
    physical_activity  = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 2.0, 0.5)

    st.subheader("🧠 Psychological Parameters")
    stress_level     = st.slider("Stress Level (1 – 10)", 1, 10, 5)
    social_quality   = st.slider("Social Interaction Quality (1 – 5)", 1, 5, 3)

    predict_btn = st.button("🔍 Predict Risk Level", use_container_width=True,
                            disabled=not model_ready)

# ── Prediction flow ───────────────────────────────────────────────────────────
if predict_btn and model_ready:
    input_values = [screen_time, social_media, late_night, gpa, missed_classes,
                    sleep_hours, sleep_disturbances, physical_activity,
                    stress_level, social_quality]
    input_arr    = np.array([input_values])
    input_scaled = scaler.transform(input_arr)

    risk_idx   = int(model.predict(input_scaled)[0])
    probs      = model.predict_proba(input_scaled)[0]
    risk_label = CLASS_NAMES[risk_idx]
    add_score  = calculate_addiction_score(probs)
    confidence = float(max(probs)) * 100

    # ── Top metrics row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Level",      f"{RISK_EMOJI[risk_label]}  {risk_label}")
    c2.metric("Addiction Score", f"{add_score} / 100")
    c3.metric("Confidence",      f"{confidence:.1f}%")
    c4.metric("Student",         student_name if student_name else "—")

    st.markdown("---")

    # ── Probability bar chart ─────────────────────────────────────────
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
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Value'  : input_values
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────────────
    st.subheader("💡 Personalised Recommendations")
    tips = get_recommendations(risk_idx, verbose=False)
    tip_cols = st.columns(2)
    for idx, tip in enumerate(tips):
        tip_cols[idx % 2].info(tip)

    # ── Institutional alert ───────────────────────────────────────────
    alert = get_alert_message(risk_idx)
    if alert:
        st.error(alert)

    # ── Score gauge ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Addiction Risk Score Gauge")
    fig2, ax2 = plt.subplots(figsize=(8, 1.2))
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax2.imshow(gradient, aspect='auto', cmap='RdYlGn_r',
               extent=[0, 100, 0, 1])
    ax2.axvline(add_score, color='black', linewidth=3)
    ax2.text(add_score, 0.5, f" {add_score}", va='center',
             fontsize=11, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlabel('Addiction Risk Score (0 = No Risk, 100 = Severe)')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── SHAP Explanation & Diagnostics ────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Prediction Diagnostics (SHAP Waterfall)")
    
    os.makedirs("reports", exist_ok=True)
    with st.spinner("Calculating SHAP feature impact..."):
        _, shap_impact_dict, _ = explain_single(input_scaled, feature_names, model)
        waterfall_path = plot_waterfall(input_scaled, feature_names, model)
    
    # Identify the key with highest absolute SHAP impact
    top_feature = max(shap_impact_dict, key=lambda k: abs(shap_impact_dict[k]))
    
    # Display the waterfall plot
    st.image(waterfall_path, caption="SHAP waterfall plot showing feature contributions to the prediction", use_container_width=True)
    
    # Pass top_feature to get_recommendations:
    shap_tips = get_recommendations(risk_idx, top_feature=top_feature, verbose=False)

else:
    # ── Landing page ──────────────────────────────────────────────────
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
    col1.metric("ML Algorithm",   "Random Forest")
    col2.metric("Risk Categories", "4 Levels")
    col3.metric("XAI",            "SHAP Integrated")

    st.markdown("---")
    st.markdown("**Project by:** Arpita Pradhane · Mouneshwar Sutar · "
                "Parasuram Sanadi · Pratiksha Pawar  |  "
                "KLE College of Engineering and Technology, Chikodi · 2025-26")
