from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from backend.schemas import PredictInput
from backend.model_loader import predict_single, get_shap_explanation

app = FastAPI(
    title="E-Gadget Addiction Prediction API",
    description="Backend API for predicting student gadget addiction risk"
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint for API verification."""
    return {"message": "Welcome to the E-Gadget Addiction Prediction API"}

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "prediction-api"}

@app.post("/predict")
async def predict_risk(data: PredictInput) -> Dict[str, Any]:
    """
    Run prediction for a student profile using the loaded model.
    """
    try:
        # Convert request body to dictionary (Pydantic v2)
        features_dict = data.model_dump()
        
        # Call the prediction logic
        result = predict_single(features_dict)
        shap_result = get_shap_explanation(features_dict)
        
        return {
            "status": "success",
            "data": {
                **result,
                "shap": shap_result
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )