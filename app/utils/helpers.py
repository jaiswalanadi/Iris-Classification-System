import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'data/raw',
        'data/processed', 
        'models/trained',
        'reports',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def save_json(data: Dict[Any, Any], filepath: str):
    """Save data as JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON saved: {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON {filepath}: {str(e)}")
        raise

def load_json(filepath: str) -> Dict[Any, Any]:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON loaded: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON {filepath}: {str(e)}")
        raise

def format_model_name(model_name: str) -> str:
    """Format model name for display"""
    name_mapping = {
        'random_forest': 'Random Forest',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'logistic_regression': 'Logistic Regression'
    }
    return name_mapping.get(model_name, model_name.title())

def validate_features(features: Dict[str, float]) -> Dict[str, str]:
    """Validate input features and return error messages"""
    errors = {}
    
    required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for feature in required_features:
        if feature not in features:
            errors[feature] = f"{feature} is required"
        elif not isinstance(features[feature], (int, float)):
            errors[feature] = f"{feature} must be a number"
        elif features[feature] < 0 or features[feature] > 10:
            errors[feature] = f"{feature} must be between 0 and 10"
    
    return errors

def log_prediction(features: Dict[str, float], prediction: str, confidence: float):
    """Log prediction for monitoring"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': prediction,
        'confidence': confidence
    }
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Append to daily log file
    log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        # Load existing logs or create empty list
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log
        logs.append(log_data)
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'timestamp': datetime.now().isoformat()
    }
