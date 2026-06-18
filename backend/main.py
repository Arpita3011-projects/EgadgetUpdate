import io
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import Dict, Any
from backend.schemas import PredictInput
from backend.model_loader import predict_single, get_shap_explanation
from src.batch_predictor import (
    read_uploaded_file,
    predict_from_dataframe
)
from src.recommendation import calculate_addiction_score
from src.report_generator import generate_pdf_report
from src.batch_report_generator import generate_class_report
from src.batch_report_generator import (
    generate_class_report,
    generate_student_cards
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

@app.post("/generate-class-report")
async def generate_class_report_endpoint(file: UploadFile = File(...)):
    """
    Accept CSV or XLSX file, run batch predictions for all students, 
    and return a class-level analytics PDF report.
    """
    try:
        # Read file contents into memory
        contents = await file.read()
        data_stream = io.BytesIO(contents)
        # Mock the .name attribute for compatibility with extension check
        data_stream.name = file.filename
        
        # Convert to DataFrame and run batch prediction
        raw_df = read_uploaded_file(data_stream)
        result = predict_from_dataframe(raw_df)
        print(type(result["alerts"]))
        print(result["alerts"])
        
        # Create reports directory if it doesn't exist
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate the class report PDF
        pdf_path = generate_class_report(
            summary=result["summary"],
            results_df=result["results_df"],
            alerts=result["alerts"],
            reports_dir=reports_dir
        )
        
        return FileResponse(
            path=pdf_path,
            filename=os.path.basename(pdf_path),
            media_type='application/pdf'
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

@app.post("/generate-student-report")
async def generate_student_report(data: PredictInput):
    """
    Generates a personalised PDF report for a single student.
    """
    try:
        # Convert request body to dictionary
        features_dict = data.model_dump()
        
        # Run predictions and SHAP explanations
        result = predict_single(features_dict)
        shap_result = get_shap_explanation(features_dict)
        
        # Calculate addiction score for the report
        addiction_score = calculate_addiction_score(list(result["probabilities"].values()))
        
        # Placeholder recommendations as requested
        tips = [
            "Reduce screen time",
            "Maintain healthy sleep habits",
            "Increase physical activity"
        ]
        
        # Generate PDF report using requirements mapping
        pdf_path = generate_pdf_report(
            student_name="Anonymous",
            risk_label=result["risk_label"],
            score=addiction_score,
            tips=tips,
            shap_impact=shap_result["feature_importance"],
            confidence_pct=result["confidence"]
        )
        
        return FileResponse(
            path=pdf_path,
            filename=os.path.basename(pdf_path),
            media_type='application/pdf'
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

@app.post("/generate-student-cards")
async def generate_student_cards_endpoint(file: UploadFile = File(...), max_students: int = 200):
    """
    Accept CSV or XLSX file, run batch predictions, and return 
    a PDF with individual student report cards.
    """
    try:
        # Read file contents
        contents = await file.read()
        data_stream = io.BytesIO(contents)
        data_stream.name = file.filename
        
        # Run prediction
        raw_df = read_uploaded_file(data_stream)
        result = predict_from_dataframe(raw_df)
        
        # Setup directory
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate PDF
        pdf_path = generate_student_cards(
            results_df=result["results_df"],
            reports_dir=reports_dir,
            max_students=max_students
        )
        
        return FileResponse(
            path=pdf_path,
            filename=os.path.basename(pdf_path),
            media_type='application/pdf'
        )
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
    and return the full results required for the Teacher Dashboard.
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
            "alerts": result["alerts"],
            "results": result["results_df"].to_dict(orient="records"),
            "total_students": len(result["results_df"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )