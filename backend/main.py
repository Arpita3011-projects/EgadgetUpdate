from fastapi import FastAPI, HTTPException
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict, Any
from backend.schemas import PredictInput
from backend.model_loader import predict_single, get_shap_explanation
from src.batch_predictor import (
    read_uploaded_file,
    predict_from_dataframe
)

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

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accept CSV or XLSX file, run batch predictions for all students, 
    and return the class summary.
    """
    try:
        # Read file contents into memory
        contents = await file.read()
        data_stream = io.BytesIO(contents)
        # Mock the .name attribute for compatibility with batch_predictor's extension check
        data_stream.name = file.filename
        
        # Convert to DataFrame and run batch prediction
        raw_df = read_uploaded_file(data_stream)
        result = predict_from_dataframe(raw_df)
        
        return {
            "status": "success",
            "summary": result["summary"],
            "total_students": len(result["results_df"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )