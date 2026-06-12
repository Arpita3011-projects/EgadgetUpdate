# 📱 E-Gadget Addiction Prediction System

AI-driven prediction of electronic gadget addiction risk in students using
**Random Forest + SMOTE + SHAP Explainable AI + Streamlit Dashboard**.

> **Project by:** Arpita Pradhane · Mouneshwar Sutar · Parasuram Sanadi · Pratiksha Pawar  
> **Guide:** Mrs. Ashwini Pharalad  
> KLE College of Engineering and Technology, Chikodi · 2025-26

---

## 🆕 Teacher Dashboard — Google Form → Batch Prediction

Full workflow: Google Form → CSV Download → Upload to Dashboard → Batch Predictions → PDF/Excel Reports

```bash
streamlit run app/teacher_dashboard.py
```

---

## 🗂 Project Structure

```
e_gadget_addiction/
├── data/
│   ├── generate_sample_data.py   ← generates student_data.csv
│   └── student_data.csv          ← raw data (600 students)
├── notebooks/
│   └── exploration.ipynb         ← EDA, charts, SHAP plots
├── src/
│   ├── preprocess.py             ← clean, scale, SMOTE-balance
│   ├── train_model.py            ← train & compare ML models
│   ├── predict.py                ← single / batch prediction CLI
│   ├── shap_explainer.py         ← SHAP explanations & plots
│   ├── recommendation.py         ← personalised tips engine
│   └── report_generator.py       ← PDF report generation
├── models/                       ← auto-created after training
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── app/
│   ├── dashboard.py             ← Single student predictor
│   └── teacher_dashboard.py     ← 🆕 Google Form batch prediction              ← Streamlit web app
├── reports/                      ← PDF & chart outputs (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Install dependencies
```bash
# (recommended) create a virtual environment first
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Step 2 — Generate data & train the model
```bash
# Generate sample CSV (skip if you have real survey data)
python data/generate_sample_data.py

# Train the Random Forest model
python src/train_model.py
```

### Step 3 — Launch the dashboard
```bash
streamlit run app/dashboard.py
```
Open **http://localhost:8501** in your browser — done! 🎉

---

## 🧪 Other Commands

| Task | Command |
|------|---------|
| Interactive CLI prediction | `python src/predict.py` |
| Batch prediction from CSV | `python src/predict.py --csv path/to/file.csv` |
| Generate SHAP charts only | `python src/shap_explainer.py` |
| Open Jupyter notebook | `jupyter notebook notebooks/exploration.ipynb` |

---

## 🏗 System Architecture

```
Student Input (Web Form / CLI)
        │
        ▼
 [MinMaxScaler — Normalise]
        │
        ▼
 [Random Forest Model]  ←── Trained on SMOTE-balanced data
        │
        ├── Risk Level:  Low / Moderate / High / Severe
        ├── Addiction Risk Score: 0–100
        └── Class Probabilities
              │
              ▼
       [SHAP Explainer]
              │ Top contributing feature
              ▼
    [Recommendation Engine]
              │ Personalised tips + alerts
              ▼
    [Streamlit Dashboard + PDF Report]
```

---

## 📊 Model Comparison

| Algorithm | Trained & compared |
|---|---|
| **Random Forest** | ✅ Primary model (saved) |
| Logistic Regression | ✅ Benchmark |
| Decision Tree | ✅ Benchmark |
| SVM | ✅ Benchmark |

---

## 📦 Key Libraries

| Library | Purpose |
|---|---|
| `scikit-learn` | ML algorithms, metrics |
| `imbalanced-learn` | SMOTE class balancing |
| `shap` | Explainable AI |
| `streamlit` | Web dashboard |
| `fpdf2` | PDF report generation |
| `matplotlib / seaborn` | Visualisations |
