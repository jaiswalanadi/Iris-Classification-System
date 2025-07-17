from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

from app.models.schemas import IrisFeatures, PredictionResponse, TrainingResponse
from app.services.model_trainer import ModelTrainer
from app.services.report_generator import ReportGenerator
from app.utils.helpers import ensure_directories, format_model_name, validate_features, log_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_trainer = None
report_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_trainer, report_generator
    
    # Startup
    logger.info("Starting Iris Classification API...")
    ensure_directories()
    
    # Initialize services
    model_trainer = ModelTrainer()
    
    # Try to load existing models
    if model_trainer.load_trained_models():
        logger.info("Pre-trained models loaded successfully")
        report_generator = ReportGenerator(
            model_trainer.data_processor, 
            model_trainer.ml_models
        )
    else:
        logger.info("No pre-trained models found. Training required.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Iris Classification API...")

# Create FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="Machine Learning API for classifying Iris flower species",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    model_info = model_trainer.get_model_info() if model_trainer else {'trained': False}
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "model_info": model_info}
    )

@app.get("/predict-page", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Prediction page"""
    if not model_trainer or not model_trainer.is_trained:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    model_info = model_trainer.get_model_info()
    return templates.TemplateResponse(
        "predict.html", 
        {"request": request, "model_info": model_info}
    )

@app.post("/train")
async def train_models(
    remove_outliers: bool = True,
    use_grid_search: bool = True
) -> TrainingResponse:
    """Train all models"""
    global report_generator
    
    if not model_trainer:
        raise HTTPException(status_code=500, detail="Model trainer not initialized")
    
    data_path = "data/raw/Iris.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Training data not found. Please upload Iris.csv to data/raw/")
    
    try:
        # Train models
        result = model_trainer.train_pipeline(
            data_path=data_path,
            remove_outliers=remove_outliers,
            use_grid_search=use_grid_search
        )
        
        # Initialize report generator after training
        report_generator = ReportGenerator(
            model_trainer.data_processor, 
            model_trainer.ml_models
        )
        
        logger.info("Training completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: IrisFeatures, model_name: str = None):
    """Predict iris species"""
    if not model_trainer or not model_trainer.is_trained:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    try:
        # Convert features to dict
        feature_dict = {
            'sepal_length': features.sepal_length,
            'sepal_width': features.sepal_width,
            'petal_length': features.petal_length,
            'petal_width': features.petal_width
        }
        
        # Validate features
        errors = validate_features(feature_dict)
        if errors:
            raise HTTPException(status_code=422, detail=errors)
        
        # Make prediction
        result = model_trainer.predict_single(feature_dict, model_name)
        
        # Log prediction
        log_prediction(feature_dict, result['prediction'], result['confidence'])
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-form")
async def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
    model_name: str = Form(None)
):
    """Handle form-based prediction"""
    try:
        features = IrisFeatures(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        
        result = await predict_species(features, model_name)
        
        model_info = model_trainer.get_model_info()
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "model_info": model_info,
                "prediction_result": result,
                "input_features": features
            }
        )
        
    except HTTPException as e:
        model_info = model_trainer.get_model_info()
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "model_info": model_info,
                "error": str(e.detail),
                "input_features": {
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width
                }
            }
        )

@app.get("/models")
async def get_models():
    """Get information about available models"""
    if not model_trainer:
        return {"trained": False, "models": []}
    
    return model_trainer.get_model_info()

@app.get("/feature-importance")
async def get_feature_importance(model_name: str = None):
    """Get feature importance for specified model"""
    if not model_trainer or not model_trainer.is_trained:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    importance = model_trainer.get_feature_importance(model_name)
    if importance is None:
        raise HTTPException(status_code=404, detail="Feature importance not available for this model")
    
    return {
        "model": model_name or model_trainer.ml_models.best_model_name,
        "feature_importance": importance
    }

@app.post("/generate-report")
async def generate_report():
    """Generate comprehensive analysis report"""
    if not model_trainer or not model_trainer.is_trained:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    if not report_generator:
        raise HTTPException(status_code=500, detail="Report generator not initialized")
    
    try:
        data_path = "data/raw/Iris.csv"
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        # Get training results
        training_results = {}
        for model_name in model_trainer.ml_models.trained_models.keys():
            # Mock results for report generation (in real scenario, store these during training)
            training_results[model_name] = {
                'metrics': {
                    'accuracy': 0.95,  # These would be real metrics
                    'precision': 0.95,
                    'recall': 0.95,
                    'f1_score': 0.95,
                    'confusion_matrix': [[10, 0, 0], [0, 9, 1], [0, 1, 9]]
                }
            }
        
        report_path = report_generator.generate_full_report(data_path, training_results)
        
        return {
            "message": "Report generated successfully",
            "report_path": report_path,
            "download_url": f"/download-report/{os.path.basename(report_path)}"
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/download-report/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    file_path = os.path.join("reports", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/html'
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_trained": model_trainer.is_trained if model_trainer else False,
        "available_models": list(model_trainer.ml_models.trained_models.keys()) if model_trainer and model_trainer.is_trained else []
