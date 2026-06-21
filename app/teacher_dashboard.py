"""
teacher_dashboard.py  —  E-Gadget Addiction Prediction System
═════════════════════════════════════════════════════════════
Full teacher workflow:
  1. Upload Google Form CSV / Excel  (or use built-in sample)
  2. Auto column mapping  →  batch predictions for every student
  3. Class analytics dashboard
  4. Per-student drill-down
  5. Alerts for High / Severe students
  6. Download  Excel · Class PDF · Student Cards PDF
  7. Google Form Builder  →  copy-paste form questions

Run:
    streamlit run app/teacher_dashboard.py
"""

# ── stdlib & third-party ──────────────────────────────────────────────────────
import os, sys, io, datetime, textwrap, re, tempfile
from typing import Any, List, Tuple, Optional

import requests

API_BASE_URL = "http://127.0.0.1:8000"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

# ── path wiring ───────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))
SRC_DIR  = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from batch_predictor        import predict_from_dataframe, read_uploaded_file, \
                                   export_results_to_bytes, load_artifacts
from batch_report_generator import generate_class_report, generate_student_cards
from form_column_mapper     import MODEL_FEATURES, DEFAULT_MAP
from recommendation         import get_recommendations, get_alert_message
from shap_explainer         import explain_single, get_tree_explainer, plot_waterfall
from logging_config         import get_logger, setup_logging

setup_logging()
logger = get_logger('teacher_dashboard')

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']
RISK_COLORS = {'Low':'#27ae60','Moderate':'#f39c12','High':'#e67e22','Severe':'#c0392b'}
RISK_EMOJI  = {'Low':'🟢','Moderate':'🟡','High':'🟠','Severe':'🔴'}
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
DATA_DIR    = os.path.join(ROOT_DIR, 'data')
SAMPLE_CSV  = os.path.join(DATA_DIR, 'sample_google_form_responses.csv')

RISK_ROW_COLORS = {
    'Low': 'background-color:#c8f7c5',
    'Moderate': 'background-color:#fff3cd',
    'High': 'background-color:#ffd6a5',
    'Severe': 'background-color:#ffcdd2',
}


def _highlight_risk_row(row: pd.Series) -> List[str]:
    color = RISK_ROW_COLORS.get(row.get('predicted_risk', ''), '')
    return [color] * len(row)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teacher Dashboard · Gadget Addiction",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e3a5f; }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
[data-testid="stSidebar"] hr { border-color: #2c5282; }
.metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 16px 20px; text-align: center;
    border-left: 5px solid #3498db;
    margin-bottom: 8px;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #2c3e50; }
.metric-card .lbl { font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase; }
.risk-badge {
    display:inline-block; padding: 4px 12px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; color: white;
}
.alert-row { background:#fff5f5; border-left:5px solid #e74c3c;
             padding:10px 14px; border-radius:6px; margin:6px 0; }
.tip-card  { background:#f0f7ff; border-left:4px solid #3498db;
             padding:8px 12px; border-radius:6px; margin:4px 0; font-size:0.9rem; }
.form-question { background:#f8f9fa; border:1px solid #dee2e6;
                 border-radius:8px; padding:12px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── session state init ────────────────────────────────────────────────────────
for key, default in [
    ('results',   None), ('summary',  None),
    ('alerts',    []),   ('col_map',  {}),
    ('raw_df',    None), ('filename', ''),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── model availability check ──────────────────────────────────────────────────
@st.cache_resource
def _load_model_artifacts() -> Tuple[Any, Any, List[str]]:
    return load_artifacts(os.path.join(ROOT_DIR, 'models'))


@st.cache_resource
def _load_shap_explainer(_model: Any) -> Any:
    return get_tree_explainer(_model)


try:
    _model, _scaler, _feature_names = _load_model_artifacts()
    _shap_explainer = _load_shap_explainer(_model)
    MODEL_READY = True
except Exception as _me:
    MODEL_READY = False
    logger.error('Model not ready: %s', _me)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏫 Teacher Dashboard")
    st.markdown("**E-Gadget Addiction Predictor**")
    st.markdown("*KLE College of Engineering*")
    st.markdown("---")

    if not MODEL_READY:
        st.error("⚠️ Model not trained yet!\nRun:\n`python src/train_model.py`")
        st.markdown("---")

    PAGES = [
        "🏠 Home",
        "📤 Upload & Predict",
        "📊 Class Analytics",
        "🧑‍🎓 Per-Student View",
        "🚨 Alerts",
        "📥 Download Reports",
        "📝 Google Form Builder",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")
    if st.session_state.results is not None:
        res  = st.session_state.results
        summ = st.session_state.summary
        st.markdown(f"**📂 File:** {st.session_state.filename}")
        st.markdown(f"**👥 Students:** {summ['total_students']}")
        st.markdown(f"**⚠️ Alerts:** {len(st.session_state.alerts)}")
        st.markdown(f"**📊 Avg Score:** {summ['avg_addiction_score']}/100")
    else:
        st.markdown("*No data loaded yet.*")
        st.markdown("Upload a file on the **Upload & Predict** page.")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: mini risk badge HTML
# ══════════════════════════════════════════════════════════════════════════════
def risk_badge(label):
    c = RISK_COLORS.get(label, '#888')
    return f'<span class="risk-badge" style="background:{c}">{RISK_EMOJI.get(label,"")} {label}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏫 Teacher Dashboard — Gadget Addiction Prediction System")
    st.markdown("*AI-powered early detection system · Powered by Random Forest + SHAP*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.info("**📤 Upload & Predict**\nUpload your Google Form CSV/Excel and run batch predictions in one click.")
    c2.info("**📊 Class Analytics**\nView risk distribution charts, department comparisons, and key stats.")
    c3.info("**🧑‍🎓 Per-Student View**\nDrill into any student's risk score, profile, and personalised tips.")
    c4.info("**🚨 Alerts**\nInstant list of all High and Severe risk students needing attention.")

    st.markdown("---")
    st.subheader("🔄 How It Works")

    steps = [
        ("1️⃣", "Create Google Form", "Use the **Google Form Builder** tab to get ready-made question text."),
        ("2️⃣", "Collect Responses",  "Share the form link with your students and wait for responses."),
        ("3️⃣", "Download CSV",       "Go to Google Forms → Responses → Download as CSV."),
        ("4️⃣", "Upload Here",        "Upload on the **Upload & Predict** page — predictions run instantly."),
        ("5️⃣", "View Results",       "Explore analytics, drill into students, review alerts, download reports."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"### {num}")
        col.markdown(f"**{title}**")
        col.caption(desc)

    st.markdown("---")
    if not MODEL_READY:
        st.error("⚠️ **Model not trained.** Run `python src/train_model.py` before uploading data.")
    else:
        st.success("✅ **Model is ready.** Go to **Upload & Predict** to get started.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload & Predict":
    st.title("📤 Upload & Predict")
    st.markdown("Upload your Google Form CSV or Excel export. The system will auto-detect columns and predict risk for every student.")

    if not MODEL_READY:
        st.error("⚠️ Model not trained. Run `python src/train_model.py` first.")
        st.stop()

    st.markdown("---")

    # ── Upload widget ─────────────────────────────────────────────────────
    tab_upload, tab_sample = st.tabs(["📁 Upload Your File", "🧪 Use Sample File"])

    raw_df   = None
    filename = ""

    with tab_upload:
        st.markdown("#### Drop your Google Form CSV or Excel file here")
        uploaded = st.file_uploader(
            label="Choose file",
            type=["csv", "xlsx", "xls"],
            help="Download from Google Forms → Responses tab → ⋮ menu → Download responses (.csv)",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                raw_df   = read_uploaded_file(uploaded)
                filename = uploaded.name
                st.success(f"✅ **{filename}** loaded — {len(raw_df)} rows, {len(raw_df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    with tab_sample:
        st.markdown("#### Try with a built-in 40-student sample (mimics Google Form output)")
        if os.path.exists(SAMPLE_CSV):
            if st.button("📂 Load Sample File", width='stretch'):
                raw_df   = pd.read_csv(SAMPLE_CSV)
                filename = "sample_google_form_responses.csv"
                st.success(f"✅ Sample file loaded — {len(raw_df)} students")
        else:
            st.warning("Sample file not found. Run `python data/generate_google_form_sample.py` to create it.")

    # ── Preview + column mapping ──────────────────────────────────────────
    if raw_df is not None:
        st.markdown("---")

        col_prev, col_map_view = st.columns([3, 2])

        with col_prev:
            st.subheader("📋 File Preview")
            st.dataframe(raw_df.head(5), use_container_width=True)
            st.caption(f"{len(raw_df)} students · {len(raw_df.columns)} columns")

        with col_map_view:
            st.subheader("🗺️ Auto Column Mapping")
            from form_column_mapper import build_column_map
            auto_map = build_column_map(list(raw_df.columns))
            if auto_map:
                map_df = pd.DataFrame([
                    {"Form Column": k[:55]+"…" if len(k)>55 else k, "→ Feature": v}
                    for k, v in auto_map.items()
                ])
                st.dataframe(map_df, use_container_width=True, hide_index=True)
                unmapped = [c for c in raw_df.columns if c not in auto_map]
                if unmapped:
                    with st.expander(f"ℹ️ {len(unmapped)} ignored columns"):
                        st.write(unmapped)
            else:
                st.warning("⚠️ No columns auto-matched! See the Google Form Builder tab for exact question text.")

        # ── RUN button ────────────────────────────────────────────────────
        st.markdown("---")
        btn_col, info_col = st.columns([1, 3])
        with btn_col:
            run = st.button("🚀 Run Predictions for All Students",
                            use_container_width=True, type="primary")
        with info_col:
            st.markdown(f"Will predict risk for all **{len(raw_df)} students** using Random Forest.")

        if run:
            with st.spinner(f"Running predictions for {len(raw_df)} students…"):
                try:
                    logger.info('Batch prediction started for %d rows', len(raw_df))
                    out = predict_from_dataframe(raw_df)

                    # Fallbacks for robustness if the API is not reachable
                    api_json = {"summary": out['summary'], "alerts": out['alerts']}
                    api_results_df = out['results_df']

                    # Convert raw_df to CSV in memory using io.BytesIO()
                    csv_buffer = io.BytesIO()
                    raw_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    # Send the file to FastAPI endpoint POST /predict-batch using api_request()
                    files = {'file': ('batch.csv', csv_buffer, 'text/csv')}
                    api_out = api_request("POST", "/predict-batch", files=files)

                    # Display success or warning messages depending on the request outcome
                    if api_out:
                        st.success("Batch API Connected")

                        api_json = api_out.json()
                        api_results_df = pd.DataFrame(api_json["results"])

                        st.write("API Data Source: FastAPI /predict-batch")
                    else:
                        st.warning("Batch API Not Reachable")

                        st.write("API Data Source: Local predict_from_dataframe fallback")

                    st.session_state.results = api_results_df
                    st.session_state.summary = api_json["summary"]
                    st.session_state.alerts = api_json["alerts"]

                    st.info(
                        f"Teacher Dashboard now using API results: "
                        f"{len(st.session_state.results)} students"
                    )
                    st.write(
                        "Data Source:",
                        "FastAPI /predict-batch"
                        if api_out
                        else "Local predict_from_dataframe fallback"
                    )
                    st.session_state.col_map  = out['column_map']
                    st.session_state.raw_df   = raw_df
                    st.session_state.filename = filename

                    summ = out['summary']
                    dist = summ['risk_distribution']

                    st.success(f"✅ Done! Predicted risk for **{summ['total_students']} students**.")

                    if out['missing_features']:
                        st.warning(f"⚠️ Missing features (set to neutral): `{out['missing_features']}`")

                    # Quick result cards
                    st.markdown("#### Results Summary")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Total",    summ['total_students'])
                    k2.metric("🟢 Low",   dist.get('Low',0),      delta=None)
                    k3.metric("🟡 Moderate", dist.get('Moderate',0))
                    k4.metric("🟠 High",  dist.get('High',0))
                    k5.metric("🔴 Severe",dist.get('Severe',0),
                               delta=f"{summ['high_severe_count']} need attention" if summ['high_severe_count'] else None,
                               delta_color="inverse")

                    st.info("👈 Navigate to **Class Analytics**, **Per-Student View**, or **Alerts** in the sidebar.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CLASS ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Class Analytics":
    st.title("📊 Class Analytics")

    if st.session_state.results is None:
        st.info("👈 Upload a file first on the **Upload & Predict** page.")
        st.stop()

    res  = st.session_state.results
    summ = st.session_state.summary

    # ── KPI row ───────────────────────────────────────────────────────────
    st.subheader("Class Overview")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Students",   summ['total_students'])
    k2.metric("Avg Risk Score",   f"{summ['avg_addiction_score']}/100")
    k3.metric("Need Attention 🚨",summ['high_severe_count'])
    k4.metric("Avg Screen Time",  f"{summ.get('avg_screen_time','—')} h/day")
    k5.metric("Avg Sleep",        f"{summ.get('avg_sleep_hours','—')} h/night")
    k6.metric("Avg GPA",          summ.get('avg_gpa','—'))

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    cl, cr = st.columns(2)

    with cl:
        st.subheader("Risk Level Distribution")
        dist   = summ['risk_distribution']
        labels = [l for l in CLASS_NAMES if dist.get(l,0)>0]
        sizes  = [dist[l] for l in labels]
        colors = [RISK_COLORS[l] for l in labels]
        fig, ax = plt.subplots(figsize=(5,4))
        wedges, texts, autos = ax.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor':'white','linewidth':2})
        for a in autos: a.set_fontsize(10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with cr:
        st.subheader("Addiction Score Distribution")
        scores = res['addiction_score'].tolist()
        fig, ax = plt.subplots(figsize=(5,4))
        n, bins, patches = ax.hist(scores, bins=20, edgecolor='white')
        for patch, left in zip(patches, bins[:-1]):
            if   left < 33: patch.set_facecolor('#27ae60')
            elif left < 55: patch.set_facecolor('#f39c12')
            elif left < 75: patch.set_facecolor('#e67e22')
            else:           patch.set_facecolor('#c0392b')
        ax.set_xlabel('Addiction Risk Score'); ax.set_ylabel('Students')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Counts bar chart ──────────────────────────────────────────────────
    st.subheader("Risk Level Counts")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    counts = [summ['risk_distribution'].get(l,0) for l in CLASS_NAMES]
    bars   = ax.barh(CLASS_NAMES, counts, color=[RISK_COLORS[l] for l in CLASS_NAMES], edgecolor='white')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of Students')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Feature averages by risk level ────────────────────────────────────
    feat_cols = [f for f in MODEL_FEATURES if f in res.columns]
    if feat_cols:
        st.subheader("Average Feature Values by Risk Level")
        agg = res.groupby('predicted_risk')[feat_cols].mean().round(2)
        agg = agg.reindex([l for l in CLASS_NAMES if l in agg.index])
        st.dataframe(
            agg.style.background_gradient(cmap='RdYlGn_r', axis=None),
            use_container_width=True
        )

    # ── Department breakdown ──────────────────────────────────────────────
    if 'department' in res.columns:
        st.markdown("---")
        st.subheader("Department-wise Average Addiction Score")
        dept = (res.groupby('department')['addiction_score']
                   .agg(['mean','count']).round(2)
                   .sort_values('mean', ascending=False))
        dept.columns = ['Avg Score','Students']

        fig, ax = plt.subplots(figsize=(8, max(3, len(dept)*0.5)))
        sc = dept['Avg Score'].tolist()
        dc = ['#c0392b' if s>=66 else '#e67e22' if s>=44 else '#f39c12' if s>=22 else '#27ae60' for s in sc]
        ax.barh(dept.index[::-1], sc[::-1], color=dc[::-1], edgecolor='white')
        for i,(d,s) in enumerate(zip(dept.index[::-1], sc[::-1])):
            ax.text(s+0.3, i, f'{s}', va='center', fontsize=9)
        ax.set_xlabel('Average Addiction Score')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.dataframe(dept, use_container_width=True)

    # ── Top 10 highest risk students ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 10 Highest Risk Students")
    top10 = res.nlargest(10, 'addiction_score')
    display_top = [c for c in
        ['student_name', 'student_id', 'department', 'semester',
         'predicted_risk', 'addiction_score', 'confidence_pct']
        if c in top10.columns]
    st.dataframe(
        top10[display_top].style.apply(_highlight_risk_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [
        str(r.get('student_name', f'S{i}'))[:15]
        for i, r in top10.iterrows()
    ]
    scores = top10['addiction_score'].tolist()
    colors = [RISK_COLORS.get(r, '#888') for r in top10['predicted_risk']]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white')
    ax.set_xlabel('Addiction Risk Score')
    ax.set_title('Top 10 Risk Scores')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Department-wise risk distribution ─────────────────────────────────
    if 'department' in res.columns:
        st.markdown("---")
        st.subheader("Department-wise Risk Distribution")
        dept_risk = pd.crosstab(
            res['department'], res['predicted_risk'],
        ).reindex(columns=CLASS_NAMES, fill_value=0)

        fig, ax = plt.subplots(figsize=(9, max(3, len(dept_risk) * 0.45)))
        bottom = np.zeros(len(dept_risk))
        for level in CLASS_NAMES:
            if level in dept_risk.columns:
                vals = dept_risk[level].values
                ax.barh(dept_risk.index, vals, left=bottom, label=level,
                        color=RISK_COLORS[level], edgecolor='white')
                bottom += vals
        ax.set_xlabel('Number of Students')
        ax.set_title('Risk Distribution by Department')
        ax.legend(title='Risk', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Department risk heatmap
        st.subheader("Department Risk Heatmap")
        fig, ax = plt.subplots(figsize=(8, max(3, len(dept_risk) * 0.5)))
        sns.heatmap(dept_risk, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Department vs Risk Level')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Semester-wise risk distribution ───────────────────────────────────
    if 'semester' in res.columns:
        st.markdown("---")
        st.subheader("Semester-wise Risk Distribution")
        sem_risk = pd.crosstab(
            res['semester'].astype(str), res['predicted_risk'],
        ).reindex(columns=CLASS_NAMES, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(len(sem_risk))
        for level in CLASS_NAMES:
            if level in sem_risk.columns:
                vals = sem_risk[level].values
                ax.bar(sem_risk.index, vals, bottom=bottom, label=level,
                       color=RISK_COLORS[level], edgecolor='white')
                bottom += vals
        ax.set_xlabel('Semester')
        ax.set_ylabel('Students')
        ax.set_title('Risk Distribution by Semester')
        ax.legend(title='Risk', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Risk trend: average score by semester
        st.subheader("Risk Trend — Average Score by Semester")
        sem_avg = res.groupby('semester')['addiction_score'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sem_avg.index.astype(str), sem_avg.values, marker='o',
                linewidth=2, color='#3498db')
        ax.fill_between(range(len(sem_avg)), sem_avg.values, alpha=0.15, color='#3498db')
        ax.set_xticks(range(len(sem_avg)))
        ax.set_xticklabels(sem_avg.index.astype(str))
        ax.set_xlabel('Semester')
        ax.set_ylabel('Avg Addiction Score')
        ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Full table ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 All Students — Results Table")

    display_cols = [c for c in
        ['student_name','student_id','department','semester',
         'predicted_risk','addiction_score','confidence_pct'] + feat_cols[:5]
        if c in res.columns]

    st.dataframe(
        res[display_cols].style.apply(_highlight_risk_row, axis=1),
        use_container_width=True, height=420
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PER-STUDENT VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧑‍🎓 Per-Student View":
    st.title("🧑‍🎓 Per-Student Drill-Down")

    if st.session_state.results is None:
        st.info("👈 Upload a file first on the **Upload & Predict** page.")
        st.stop()

    res = st.session_state.results

    # ── Filter + search ───────────────────────────────────────────────────
    fc1, fc2 = st.columns([2,1])
    with fc2:
        flt = st.multiselect("Filter by Risk", CLASS_NAMES, default=CLASS_NAMES)
    filtered = res[res['predicted_risk'].isin(flt)].reset_index(drop=True)

    with fc1:
        if 'student_name' in filtered.columns:
            options = [f"{i+1}. {filtered.at[i,'student_name']} "
                       f"({filtered.at[i,'predicted_risk']})"
                       for i in range(len(filtered))]
        else:
            options = [f"Student {i+1} ({filtered.at[i,'predicted_risk']})"
                       for i in range(len(filtered))]
        if not options:
            st.warning("No students match the filter.")
            st.stop()
        sel = st.selectbox("Select Student", options)

    idx = options.index(sel)
    row = filtered.iloc[idx]

    risk  = row.get('predicted_risk','—')
    score = row.get('addiction_score', 0)
    conf  = row.get('confidence_pct',  0)
    ridx  = int(row.get('risk_index',  0))
    color = RISK_COLORS.get(risk,'#888')

    st.markdown("---")

    # Identity row
    id1,id2,id3,id4 = st.columns(4)
    id1.metric("Name",       row.get('student_name','—'))
    id2.metric("Student ID", row.get('student_id',  '—'))
    id3.metric("Department", row.get('department',  '—'))
    id4.metric("Semester",   row.get('semester',    '—'))

    # Risk banner
    st.markdown(
        f'<div style="background:{color};color:white;padding:16px 20px;'
        f'border-radius:10px;font-size:1.3rem;font-weight:700;margin:12px 0;">'
        f'{RISK_EMOJI.get(risk,"")}  Risk Level: <b>{risk}</b>'
        f' &nbsp;|&nbsp; Addiction Score: <b>{score}/100</b>'
        f' &nbsp;|&nbsp; Confidence: <b>{conf}%</b></div>',
        unsafe_allow_html=True
    )

    # ── Probability bars + feature profile ───────────────────────────────
    pc, fc = st.columns(2)

    with pc:
        st.subheader("Risk Probabilities")
        probs = [row.get('prob_low',0), row.get('prob_moderate',0),
                 row.get('prob_high',0), row.get('prob_severe',0)]
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.barh(CLASS_NAMES, probs,
                       color=[RISK_COLORS[c] for c in CLASS_NAMES], edgecolor='white')
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0,105)
        for b,v in zip(bars,probs):
            ax.text(v+1, b.get_y()+b.get_height()/2, f'{v}%', va='center', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with fc:
        st.subheader("Student Profile")
        labels = {
            'daily_screen_time_hours'   :'📱 Screen Time (hrs/day)',
            'num_social_media_platforms':'📲 Social Media Platforms',
            'late_night_usage'          :'🌙 Late Night Usage',
            'gpa'                       :'📚 GPA',
            'missed_classes_per_month'  :'🏫 Missed Classes/Month',
            'sleep_hours'               :'😴 Sleep Hours/Night',
            'sleep_disturbances'        :'💤 Sleep Disturbances',
            'physical_activity_hours'   :'🏃 Physical Activity (hrs/week)',
            'stress_level'              :'🧠 Stress Level (1-10)',
            'social_interaction_quality':'🤝 Social Quality (1-5)',
        }
        feat_df = pd.DataFrame([
            {'Feature': lbl,
             'Value': ('Yes' if row.get(k)==1 else 'No'
                       if k in ('late_night_usage','sleep_disturbances')
                       else str(row.get(k,'—')))}
            for k,lbl in labels.items() if k in row.index
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=300)

    # ── Score gauge ───────────────────────────────────────────────────────
    st.subheader("🎯 Addiction Risk Score Gauge")
    fig2, ax2 = plt.subplots(figsize=(10,1.2))
    ax2.imshow(np.linspace(0,1,300).reshape(1,-1), aspect='auto',
               cmap='RdYlGn_r', extent=[0,100,0,1])
    ax2.axvline(score, color='black', linewidth=3)
    pos = min(score+1, 90)
    ax2.text(pos, 0.5, f' {score}', va='center', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlabel('0 = No Risk ◄──────────────────────────────► 100 = Severe Risk')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── SHAP waterfall (uses existing model/scaler if available)
    with st.spinner("Calculating SHAP waterfall..."):
        try:
            model, scaler, feature_names = _load_model_artifacts()
            explainer = _load_shap_explainer(model)

            input_values = [row.get(feat, 0) for feat in MODEL_FEATURES]
            input_arr = np.array([input_values])
            input_scaled = scaler.transform(input_arr)

            _, shap_impact_dict, _ = explain_single(
                input_scaled, feature_names, model, explainer=explainer,
            )

            raw_name = row.get('student_name') or f'student_{idx+1}'
            safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', str(raw_name))
            tmpdir = tempfile.gettempdir()
            save_path = os.path.join(
                tmpdir,
                f"shap_waterfall_{safe_name}_{datetime.date.today().isoformat()}.png",
            )

            waterfall_path = plot_waterfall(
                input_scaled, feature_names, model,
                save_path=save_path, explainer=explainer,
            )
            st.image(waterfall_path, caption=f"SHAP waterfall — {raw_name}", width='stretch')
        except Exception as e:
            logger.warning('SHAP waterfall failed: %s', e)
            st.warning(f"SHAP waterfall unavailable: {e}")

    # ── Recommendations ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Personalised Recommendations")
    tips = get_recommendations(ridx, verbose=False)
    tc1, tc2 = st.columns(2)
    for i, tip in enumerate(tips):
        (tc1 if i%2==0 else tc2).markdown(
            f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    alert_msg = get_alert_message(ridx)
    if alert_msg:
        st.error(alert_msg)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.title("🚨 Students Requiring Immediate Attention")

    if st.session_state.results is None:
        st.info("👈 Upload a file first on the **Upload & Predict** page.")
        st.stop()

    alerts = st.session_state.alerts

    if not alerts:
        st.success("✅ Great news! No students are in the High or Severe risk category.")
        st.stop()

    st.error(f"⚠️ **{len(alerts)} student(s)** are classified as High or Severe risk and need attention.")

    flt = st.multiselect("Show:", ['High','Severe'], default=['High','Severe'])
    shown = [a for a in alerts if a['risk'] in flt]

    st.markdown("---")

    for i, a in enumerate(shown, 1):
        risk  = a['risk']
        color = RISK_COLORS.get(risk,'#888')
        with st.expander(
            f"{'🟠' if risk=='High' else '🔴'}  {i}. {a['name']}  "
            f"— {risk}  |  Score: {a['score']}/100  "
            f"|  ID: {a['student_id']}  |  Dept: {a['department']}",
            expanded=(i<=3)
        ):
            ca,cb,cc = st.columns(3)
            ca.metric("Risk Level",      f"{RISK_EMOJI.get(risk,'')} {risk}")
            cb.metric("Addiction Score", f"{a['score']}/100")
            cc.metric("Student ID",      a['student_id'])

            st.markdown(
                f'<div class="alert-row"><b>Required Action:</b> {a.get("alert_message","")}</div>',
                unsafe_allow_html=True)

            tips = get_recommendations(a['risk_index'], verbose=False)
            st.markdown("**Recommended Interventions:**")
            for tip in tips:
                st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DOWNLOAD REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Download Reports":
    st.title("📥 Download Reports")

    if st.session_state.results is None:
        st.info("👈 Upload a file first on the **Upload & Predict** page.")
        st.stop()

    res    = st.session_state.results
    summ   = st.session_state.summary
    alerts = st.session_state.alerts
    today  = datetime.date.today()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    # ── Excel ─────────────────────────────────────────────────────────────
    with c1:
        st.subheader("📊 Excel Results")
        st.markdown("Colour-coded spreadsheet with all student predictions, scores, and top recommendations.")
        st.markdown(" ")
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Building Excel…"):
                xlsx_bytes = export_results_to_bytes(res)
            st.download_button(
                "⬇️ Download Excel",
                data=xlsx_bytes,
                file_name=f"gadget_results_{today}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # ── Class PDF ─────────────────────────────────────────────────────────
    with c2:
        st.subheader("📄 Class Report PDF")
        st.markdown("Teacher report with pie charts, histogram, dept breakdown, top-risk table, and full alert list.")
        if st.button("Generate Class PDF", use_container_width=True):
            with st.spinner("Generating class report PDF…"):
                os.makedirs(REPORTS_DIR, exist_ok=True)
                path = generate_class_report(summ, res, alerts, REPORTS_DIR)
                with open(path,'rb') as f:
                    pdf_bytes = f.read()
            st.download_button(
                "⬇️ Download Class Report",
                data=pdf_bytes,
                file_name=f"class_report_{today}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # ── Student cards PDF ─────────────────────────────────────────────────
    with c3:
        st.subheader("🧑‍🎓 Student Cards PDF")
        st.markdown("One page per student — risk level, score, feature profile, personalised recommendations.")
        max_s = st.number_input("Max students", 1, len(res), min(len(res),100))
        if st.button("Generate Student Cards", use_container_width=True):
            with st.spinner("Generating student cards…"):
                os.makedirs(REPORTS_DIR, exist_ok=True)
                path = generate_student_cards(res, REPORTS_DIR, max_students=int(max_s))
                with open(path,'rb') as f:
                    pdf_bytes = f.read()
            st.download_button(
                "⬇️ Download Student Cards",
                data=pdf_bytes,
                file_name=f"student_cards_{today}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: GOOGLE FORM BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Google Form Builder":
    st.title("📝 Google Form Builder")
    st.markdown("Everything you need to create the Google Form and collect student data for this system.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📋 Question Bank",
        "🔗 Step-by-Step Setup",
        "🧩 CSV Template"
    ])

    # ── TAB 1: Question Bank ──────────────────────────────────────────────
    with tab1:
        st.subheader("Copy-paste these questions into your Google Form")
        st.info("These **exact question texts** allow the system to auto-detect your columns. Copy them precisely.")

        questions = [
            {
                "section": "👤 Student Identity",
                "type": "Short answer",
                "question": "What is your full name?",
                "note": "Text — used for student identification"
            },
            {
                "section": "👤 Student Identity",
                "type": "Short answer",
                "question": "What is your student ID / USN?",
                "note": "e.g. 2KD23CS012"
            },
            {
                "section": "👤 Student Identity",
                "type": "Short answer",
                "question": "What is your department / branch?",
                "note": "e.g. CSE, ECE, MECH"
            },
            {
                "section": "👤 Student Identity",
                "type": "Multiple choice",
                "question": "What is your semester?",
                "note": "Options: 1, 2, 3, 4, 5, 6, 7, 8"
            },
            {
                "section": "📱 Usage Behaviour",
                "type": "Linear scale (1–16)",
                "question": "How many hours per day do you use electronic gadgets (phone/tablet/laptop/gaming)?",
                "note": "1 = 1 hour · 16 = 16+ hours"
            },
            {
                "section": "📱 Usage Behaviour",
                "type": "Multiple choice / Short answer",
                "question": "How many social media platforms do you actively use?",
                "note": "e.g. 0, 1, 2, 3, 4, 5+"
            },
            {
                "section": "📱 Usage Behaviour",
                "type": "Multiple choice",
                "question": "Do you use gadgets late at night (after 10 PM)?",
                "note": "Options: Yes / No"
            },
            {
                "section": "🎓 Academic Indicators",
                "type": "Short answer (number)",
                "question": "What is your current GPA / CGPA?",
                "note": "e.g. 7.5 (scale of 10)"
            },
            {
                "section": "🎓 Academic Indicators",
                "type": "Short answer (number)",
                "question": "How many classes have you missed in the past month?",
                "note": "e.g. 0, 2, 5"
            },
            {
                "section": "😴 Sleep & Health",
                "type": "Linear scale (3–9)",
                "question": "How many hours do you sleep per night on average?",
                "note": "3 = 3 hours · 9 = 9 hours"
            },
            {
                "section": "😴 Sleep & Health",
                "type": "Multiple choice",
                "question": "Do you experience sleep disturbances (difficulty sleeping, waking at night)?",
                "note": "Options: Yes / No"
            },
            {
                "section": "😴 Sleep & Health",
                "type": "Short answer (number)",
                "question": "How many hours per week do you spend on physical activity / exercise?",
                "note": "e.g. 0, 3, 7"
            },
            {
                "section": "🧠 Psychological",
                "type": "Linear scale (1–10)",
                "question": "On a scale of 1–10, how would you rate your current stress level?",
                "note": "1 = Very Low · 10 = Extremely High"
            },
            {
                "section": "🧠 Psychological",
                "type": "Linear scale (1–5)",
                "question": "On a scale of 1–5, how would you rate the quality of your social interactions?",
                "note": "1 = Very Poor · 5 = Excellent"
            },
        ]

        sections = {}
        for q in questions:
            sections.setdefault(q['section'], []).append(q)

        for section, qs in sections.items():
            st.markdown(f"### {section}")
            for q in qs:
                with st.container():
                    qcol, tcol = st.columns([4,1])
                    with qcol:
                        st.markdown(
                            f'<div class="form-question">'
                            f'<b>Q:</b> {q["question"]}<br>'
                            f'<small style="color:#7f8c8d">📝 Type: {q["type"]} &nbsp;|&nbsp; 💡 {q["note"]}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with tcol:
                        st.code(q['question'][:40]+'…' if len(q['question'])>40 else q['question'],
                                language=None)

        st.markdown("---")
        st.subheader("📋 All Questions (plain text — copy this whole block)")
        all_q = "\n\n".join(
            f"Section: {q['section']}\nQuestion: {q['question']}\nType: {q['type']}"
            for q in questions
        )
        st.code(all_q, language="text")

    # ── TAB 2: Step-by-step ───────────────────────────────────────────────
    with tab2:
        st.subheader("Step-by-Step: Create the Google Form")

        steps = {
            "Step 1 — Create Form": """
1. Go to [forms.google.com](https://forms.google.com)
2. Click the **+** (Blank form)
3. Title: **Student Digital Wellness Survey**
4. Description: *"This survey helps identify digital wellness risks. All responses are confidential."*
""",
            "Step 2 — Add Sections": """
Add these 4 section headers in your form to organise questions:
- **Student Information**
- **Gadget Usage Behaviour**
- **Academic & Sleep Health**
- **Psychological Well-being**
""",
            "Step 3 — Add Questions": """
1. Copy each question from the **Question Bank** tab
2. Paste into Google Forms as a new question
3. Set the question type (Short answer / Linear scale / Multiple choice)
4. For Yes/No questions → use **Multiple choice** with options: Yes, No
5. Mark all questions as **Required** (toggle at bottom of each question)
""",
            "Step 4 — Share with Students": """
1. Click the **Send** button (top right)
2. Click the 🔗 link icon
3. Check **Shorten URL**
4. Share the link with students via WhatsApp, Email, or the classroom portal
""",
            "Step 5 — Download Responses as CSV": """
1. Open your Google Form
2. Click the **Responses** tab
3. Click the **Google Sheets** icon → creates a linked spreadsheet
4. In Google Sheets: **File → Download → Comma-separated values (.csv)**
   — OR —
   In Google Forms Responses tab: click the ⋮ menu → **Download responses (.csv)**
5. Save the file
""",
            "Step 6 — Upload Here": """
1. Come back to this dashboard
2. Go to **Upload & Predict** in the sidebar
3. Upload the downloaded CSV file
4. Click **Run Predictions** → Done! ✅
"""
        }

        for title, content in steps.items():
            with st.expander(title, expanded=True):
                st.markdown(content)

    # ── TAB 3: CSV Template ───────────────────────────────────────────────
    with tab3:
        st.subheader("Download a blank CSV template")
        st.markdown(
            "If you prefer to collect data manually (or use Excel), "
            "download this template. Fill in one row per student and upload it."
        )

        template_cols = [
            "What is your full name?",
            "What is your student ID / USN?",
            "What is your department / branch?",
            "What is your semester?",
            "How many hours per day do you use electronic gadgets (phone/tablet/laptop/gaming)?",
            "How many social media platforms do you actively use?",
            "Do you use gadgets late at night (after 10 PM)?",
            "What is your current GPA / CGPA?",
            "How many classes have you missed in the past month?",
            "How many hours do you sleep per night on average?",
            "Do you experience sleep disturbances (difficulty sleeping, waking at night)?",
            "How many hours per week do you spend on physical activity / exercise?",
            "On a scale of 1–10, how would you rate your current stress level?",
            "On a scale of 1–5, how would you rate the quality of your social interactions?",
        ]
        example_rows = [
            ["Arpita Pradhane","2KD23CS012","CSE","5",8,"3","Yes",6.5,2,5,"Yes",1,7,3],
            ["Mouneshwar Sutar","2KD23CS044","CSE","5",4,"2","No",8.2,0,7,"No",4,4,4],
            ["[Student Name]","[USN]","[Dept]","[Sem]",
             "[1-16]","[0-8]","Yes/No","[4-10]","[0-15]","[3-9]","Yes/No","[0-10]","[1-10]","[1-5]"],
        ]
        template_df = pd.DataFrame(example_rows, columns=template_cols)

        st.dataframe(template_df, use_container_width=True)

        # Download as CSV
        csv_bytes = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Blank CSV Template",
            data=csv_bytes,
            file_name="student_survey_template.csv",
            mime="text/csv",
            use_container_width=False
        )

        st.markdown("---")
        st.subheader("📊 Sample Filled Response (what Google Forms CSV looks like)")
        sample_path = os.path.join(DATA_DIR, 'sample_google_form_responses.csv')
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            st.dataframe(sample_df.head(5), use_container_width=True)
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Sample (40 students)",
                data=csv_sample,
                file_name="sample_google_form_responses.csv",
                mime="text/csv",
            )
