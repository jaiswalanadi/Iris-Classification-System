import time
import logging
from app.services.data_processor import DataProcessor
from app.models.ml_models import IrisMLModels
from app.models.schemas import ModelMetrics, TrainingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ml_models = IrisMLModels()
        self.is_trained = False
    
    def train_pipeline(self, data_path: str, remove_outliers: bool = True, 
                      use_grid_search: bool = True) -> TrainingResponse:
        """Complete training pipeline"""
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading data...")
            df = self.data_processor.load_data(data_path)
            
            # Remove outliers if requested
            if remove_outliers:
                logger.info("Removing outliers...")
                df = self.data_processor.remove_outliers(df)
            
            # Preprocess data
            logger.info("Preprocessing data...")
            X, y = self.data_processor.preprocess_data(df, fit_transforms=True)
            
            # Split data
            logger.info("Splitting data...")
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
            
            # Train models
            logger.info("Training models...")
            results = self.ml_models.train_models(
                X_train, y_train, X_test, y_test, 
                use_grid_search=use_grid_search
            )
            
            # Save models
            logger.info("Saving models...")
            self.ml_models.save_models()
            
            # Prepare response
            training_time = time.time() - start_time
            self.is_trained = True
            
            # Convert results to schema format
            metrics_dict = {}
            for model_name, result in results.items():
                metrics = result['metrics']
                metrics_dict[model_name] = ModelMetrics(
                    accuracy=metrics['accuracy'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1_score=metrics['f1_score'],
                    confusion_matrix=metrics['confusion_matrix']
                )
            
            response = TrainingResponse(
                message="Training completed successfully",
                models_trained=list(results.keys()),
                best_model=self.ml_models.best_model_name,
                metrics=metrics_dict,
                training_time=training_time
            )
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def load_trained_models(self) -> bool:
        """Load previously trained models"""
        try:
            success = self.ml_models.load_models()
            if success:
                self.is_trained = True
                logger.info("Models loaded successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def predict_single(self, features: dict, model_name: str = None):
        """Make prediction for single sample"""
        if not self.is_trained:
            raise ValueError("Models not trained. Please train models first.")
        
        # Prepare features for prediction
        scaled_features = self.data_processor.prepare_single_prediction(features)
        
        # Make prediction
        result = self.ml_models.predict_single(scaled_features, model_name)
        
        # Convert encoded prediction to species name
        species_mapping = self.data_processor.get_species_mapping()
        predicted_species = species_mapping.get(result['prediction'], 'Unknown')
        
        # Convert probabilities to species names
        species_probabilities = {}
        if result['probabilities'] is not None:
            for i, prob in enumerate(result['probabilities']):
                species = species_mapping.get(i, f'Class_{i}')
                species_probabilities[species] = float(prob)
        
        return {
            'prediction': predicted_species,
            'confidence': result['confidence'],
            'probabilities': species_probabilities,
            'model_used': model_name or self.ml_models.best_model_name
        }
    
    def get_model_info(self) -> dict:
        """Get information about trained models"""
        if not self.is_trained:
            return {'trained': False, 'models': []}
        
        return {
            'trained': True,
            'models': list(self.ml_models.trained_models.keys()),
            'best_model': self.ml_models.best_model_name,
            'species_mapping': self.data_processor.get_species_mapping()
        }
    
    def get_feature_importance(self, model_name: str = None) -> dict:
        """Get feature importance for specified model"""
        if not self.is_trained:
            return None
        
        importance = self.ml_models.get_feature_importance(model_name)
        if importance is None:
            return None
        
        feature_names = self.data_processor.feature_names
        return dict(zip(feature_names, importance.tolist()))
