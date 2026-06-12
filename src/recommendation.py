"""
recommendation.py
─────────────────
Rule-based recommendation engine.
Maps risk level (0-3) → personalized intervention tips.
Also computes the continuous Addiction Risk Score (0–100).
"""

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']
RISK_EMOJI  = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴'}

# ── Static tip bank ───────────────────────────────────────────────────────────

_TIPS = {
    0: [  # Low
        "✅ You're maintaining healthy digital habits — keep it up!",
        "📚 Continue balancing screen time with offline activities.",
        "😴 Your sleep schedule looks healthy — maintain the routine.",
        "🏃 Stay physically active; it's a great buffer against stress.",
        "🤝 Keep nurturing face-to-face social connections.",
    ],
    1: [  # Moderate
        "⚠️  Set daily screen time limits — aim for ≤ 4 hours (recreational).",
        "📵 Introduce 'no-phone' windows: during meals and 30 min before bed.",
        "🏃 Add 30 minutes of physical activity to your daily routine.",
        "📖 Schedule device-free study blocks for better focus.",
        "🔔 Use app timers to track and cap social media use.",
        "😴 Target 7–8 hours of sleep; charge your phone outside the bedroom.",
    ],
    2: [  # High
        "🚨 Implement a structured digital detox — cut recreational screen time by 50%.",
        "🛌 Sleep by 10 PM; avoid all screens at least 1 hour before bed.",
        "👨‍🏫 Speak with your faculty advisor about academic support options.",
        "📅 Use parental controls or app limits: max 30 min/day on social media.",
        "🤝 Actively increase face-to-face interactions — join a club or sports team.",
        "🧘 Practice daily mindfulness or breathing exercises to manage stress.",
        "📓 Keep a daily journal of gadget usage to build self-awareness.",
    ],
    3: [  # Severe
        "🆘 Critical level — please seek counseling support immediately.",
        "👪 Involve parents/guardians in creating a supervised digital wellness plan.",
        "📵 Consider a complete digital detox (1–2 weeks) under professional guidance.",
        "🏥 Consult a mental health professional about digital dependency.",
        "⏰ Set hard device limits using parental controls or digital-wellbeing apps.",
        "📞 Reach out to your institution's student wellness center today.",
        "📋 Work with a counselor to build a gradual screen-time reduction schedule.",
    ],
}

# ── Feature-specific micro-tips (triggered by top SHAP driver) ───────────────

_FEATURE_TIPS = {
    'daily_screen_time_hours'   : "📱 Your screen time is the biggest driver — try the 20-20-20 rule.",
    'sleep_hours'               : "😴 Poor sleep is amplifying your risk — prioritize sleep hygiene.",
    'stress_level'              : "🧘 High stress is contributing — explore meditation or yoga.",
    'gpa'                       : "📉 Academic performance is linked — seek tutoring or study support.",
    'missed_classes_per_month'  : "🏫 Class attendance is suffering — set phone-free class goals.",
    'physical_activity_hours'   : "🏃 Low activity worsens risk — even a 15-min walk helps.",
    'social_interaction_quality': "🤝 Invest time in real-world friendships to reduce digital reliance.",
    'late_night_usage'          : "🌙 Late-night usage disrupts your brain — set a hard phone curfew.",
}


def get_recommendations(risk_idx: int, top_feature: str = None, verbose: bool = False):
    """
    Parameters
    ----------
    risk_idx    : int  0=Low, 1=Moderate, 2=High, 3=Severe
    top_feature : str  optional – name of the feature with highest SHAP impact
    verbose     : bool – print tips to console

    Returns
    -------
    tips : list[str]
    """
    tips = list(_TIPS.get(risk_idx, []))

    # Inject personalised micro-tip if we know the top driver
    if top_feature and top_feature in _FEATURE_TIPS:
        tips.insert(1, _FEATURE_TIPS[top_feature])

    if verbose:
        label = CLASS_NAMES[risk_idx]
        emoji = RISK_EMOJI[risk_idx]
        print(f"\n{emoji}  Recommendations for {label} Risk Student:")
        for t in tips:
            print(f"   {t}")

    return tips


def calculate_addiction_score(probabilities) -> float:
    """
    Convert class probability vector → continuous score in [0, 100].
    Weights: Low=0, Moderate=33, High=66, Severe=100
    """
    weights = [0.0, 33.0, 66.0, 100.0]
    score   = sum(p * w for p, w in zip(probabilities, weights))
    return round(score, 1)


def get_alert_message(risk_idx: int) -> str | None:
    """Returns an institutional alert string for High/Severe, else None."""
    alerts = {
        2: "⚠️  ALERT: This student is at HIGH risk. Please notify the faculty advisor.",
        3: "🆘  CRITICAL: This student is at SEVERE risk. Escalate to counselor and parents immediately.",
    }
    return alerts.get(risk_idx)


if __name__ == '__main__':
    for i in range(4):
        get_recommendations(i, verbose=True)
