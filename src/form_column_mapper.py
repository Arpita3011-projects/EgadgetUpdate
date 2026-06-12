"""
form_column_mapper.py
─────────────────────
Maps Google Form response column headers → model feature names.

When a teacher downloads responses from Google Forms, the CSV columns
look like full question text, e.g.:
  "How many hours per day do you use your phone/gadget?"

This module normalises those messy headers into clean feature names
the model understands.

HOW TO USE:
    1. If your Google Form questions match the DEFAULT_MAP below → zero config.
    2. If your questions are different → edit DEFAULT_MAP to match your form.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT MAP  →  { exact_google_form_column_header : model_feature_name }
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAP = {
    # ── Usage behaviour ──────────────────────────────────────────────────────
    "How many hours per day do you use electronic gadgets (phone/tablet/laptop/gaming)?":
        "daily_screen_time_hours",

    "How many social media platforms do you actively use?":
        "num_social_media_platforms",

    "Do you use gadgets late at night (after 10 PM)?":
        "late_night_usage",

    # ── Academic indicators ───────────────────────────────────────────────────
    "What is your current GPA / CGPA?":
        "gpa",

    "How many classes have you missed in the past month?":
        "missed_classes_per_month",

    # ── Sleep & health ────────────────────────────────────────────────────────
    "How many hours do you sleep per night on average?":
        "sleep_hours",

    "Do you experience sleep disturbances (difficulty sleeping, waking at night)?":
        "sleep_disturbances",

    "How many hours per week do you spend on physical activity / exercise?":
        "physical_activity_hours",

    # ── Psychological ─────────────────────────────────────────────────────────
    "On a scale of 1–10, how would you rate your current stress level?":
        "stress_level",

    "On a scale of 1–5, how would you rate the quality of your social interactions?":
        "social_interaction_quality",

    # ── Student identity (not model features, used for display) ───────────────
    "What is your full name?":
        "student_name",

    "What is your student ID / USN?":
        "student_id",

    "What is your department / branch?":
        "department",

    "What is your semester?":
        "semester",
}

# ─────────────────────────────────────────────────────────────────────────────
#  YES/NO → 1/0 for binary questions
# ─────────────────────────────────────────────────────────────────────────────
YES_NO_FEATURES = {"late_night_usage", "sleep_disturbances"}

YES_VALUES = {"yes", "y", "1", "true", "always", "often", "yeah"}
NO_VALUES  = {"no",  "n", "0", "false", "never",  "rarely", "nope"}


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL FEATURES (must match training data column order)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "daily_screen_time_hours",
    "num_social_media_platforms",
    "late_night_usage",
    "gpa",
    "missed_classes_per_month",
    "sleep_hours",
    "sleep_disturbances",
    "physical_activity_hours",
    "stress_level",
    "social_interaction_quality",
]

# Identity columns (informational only — not fed to model)
IDENTITY_COLUMNS = ["student_name", "student_id", "department", "semester"]


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT helper: fuzzy column matching
# ─────────────────────────────────────────────────────────────────────────────
import re

_KEYWORD_MAP = {
    "daily_screen_time_hours"    : ["screen time", "hours.*gadget", "gadget.*hours",
                                    "phone.*day", "daily.*use", "screen.*hour"],
    "num_social_media_platforms" : ["social media", "platforms", "apps.*use"],
    "late_night_usage"           : ["late night", "after.*pm", "night.*use"],
    "gpa"                        : ["gpa", "cgpa", "grade", "marks"],
    "missed_classes_per_month"   : ["missed.*class", "absent", "bunked"],
    "sleep_hours"                : ["sleep.*hour", "hours.*sleep", "sleep per night"],
    "sleep_disturbances"         : ["sleep disturbance", "difficulty sleep", "waking"],
    "physical_activity_hours"    : ["physical activity", "exercise", "sport", "workout"],
    "stress_level"               : ["stress", "anxiety"],
    "social_interaction_quality" : ["social interaction", "social quality", "friends"],
    "student_name"               : ["name", "full name"],
    "student_id"                 : ["student id", "usn", "roll number", "reg"],
    "department"                 : ["department", "branch", "stream"],
    "semester"                   : ["semester", "sem", "year"],
}


def _fuzzy_match(column_header: str) -> str | None:
    """Try to match a column header to a feature name using keywords."""
    h = column_header.lower().strip()
    for feature, patterns in _KEYWORD_MAP.items():
        for pattern in patterns:
            if re.search(pattern, h):
                return feature
    return None


def build_column_map(df_columns: list, custom_map: dict = None) -> dict:
    """
    Given a list of DataFrame column names (from Google Form CSV),
    return {df_column: model_feature_name} for all matched columns.

    Priority:
        1. custom_map (user-provided)
        2. DEFAULT_MAP (exact match)
        3. _fuzzy_match (keyword heuristics)
    """
    effective_map = {**DEFAULT_MAP, **(custom_map or {})}
    result = {}

    for col in df_columns:
        col_stripped = col.strip()

        # 1. exact
        if col_stripped in effective_map:
            result[col] = effective_map[col_stripped]
            continue

        # 2. fuzzy
        matched = _fuzzy_match(col_stripped)
        if matched:
            result[col] = matched

    return result


def convert_yes_no(value):
    """Convert Yes/No text responses to 1/0."""
    if isinstance(value, (int, float)):
        return int(bool(value))
    s = str(value).lower().strip()
    if s in YES_VALUES:
        return 1
    if s in NO_VALUES:
        return 0
    return 0  # default safe value


def normalise_dataframe(df, column_map: dict):
    """
    Rename and clean a Google Form DataFrame using column_map.
    Returns (features_df, identity_df)
    """
    import pandas as pd

    df = df.copy()
    df.rename(columns=column_map, inplace=True)

    # Convert yes/no columns
    for col in YES_NO_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(convert_yes_no)

    # Convert numeric columns
    numeric_features = [f for f in MODEL_FEATURES if f not in YES_NO_FEATURES]
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split identity vs model features
    identity_cols  = [c for c in IDENTITY_COLUMNS if c in df.columns]
    feature_cols   = [c for c in MODEL_FEATURES   if c in df.columns]

    identity_df = df[identity_cols].copy() if identity_cols else pd.DataFrame(index=df.index)
    features_df = df[feature_cols].copy()

    # Impute missing values with median
    for col in features_df.columns:
        if features_df[col].isnull().any():
            features_df[col].fillna(features_df[col].median(), inplace=True)

    return features_df, identity_df
