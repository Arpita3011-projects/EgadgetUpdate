from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    daily_screen_time_hours: float = Field(..., ge=1, le=16)
    num_social_media_platforms: int = Field(..., ge=1, le=8)
    late_night_usage: int = Field(..., ge=0, le=1)
    gpa: float = Field(..., ge=4, le=10)
    missed_classes_per_month: int = Field(..., ge=0, le=15)
    sleep_hours: float = Field(..., ge=3, le=9)
    sleep_disturbances: int = Field(..., ge=0, le=1)
    physical_activity_hours: float = Field(..., ge=0, le=10)
    stress_level: int = Field(..., ge=1, le=10)
    social_interaction_quality: int = Field(..., ge=1, le=5)
    student_name: str | None = None

class PredictOutput(BaseModel):
    risk_label: str
    risk_index: int
    addiction_score: float
    confidence: float
    probabilities: dict
    shap_values: dict
    recommendations: list[str]