from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]

class TrainingResponse(BaseModel):
    message: str
    models_trained: List[str]
    best_model: str
    metrics: Dict[str, ModelMetrics]
    training_time: float
