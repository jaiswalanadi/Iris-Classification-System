from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisMLModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear']
            }
        }
        
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None
    
    def train_models(self, X_train, y_train, X_test, y_test, use_grid_search=True):
        """Train all models with optional hyperparameter tuning"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if use_grid_search and name in self.param_grids:
                # Use GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    model, 
                    self.param_grids[name], 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            else:
                # Train with default parameters
                best_model = model
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Store results
            results[name] = {
                'model': best_model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.trained_models[name] = best_model
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        
        # Determine best model
        self._select_best_model(results)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate various metrics for model evaluation"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _select_best_model(self, results):
        """Select the best model based on accuracy and cross-validation score"""
        best_score = 0
        for name, result in results.items():
            # Weighted score considering both accuracy and CV performance
            score = (result['metrics']['accuracy'] * 0.7 + 
                    result['metrics']['cv_mean'] * 0.3)
            
            if score > best_score:
                best_score = score
                self.best_model_name = name
                self.best_model = result['model']
        
        logger.info(f"Best model selected: {self.best_model_name}")
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found or not trained")
        
        prediction = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return prediction, probabilities
    
    def predict_single(self, features, model_name=None):
        """Predict single sample"""
        prediction, probabilities = self.predict(features, model_name)
        
        result = {
            'prediction': int(prediction[0]),
            'probabilities': probabilities[0] if probabilities is not None else None,
            'confidence': float(np.max(probabilities[0])) if probabilities is not None else None
        }
        
        return result
    
    def save_models(self, save_dir='models/trained'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f'{name}_model.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Model {name} saved to {model_path}")
        
        # Save best model info
        best_model_info = {
            'best_model_name': self.best_model_name,
            'model_names': list(self.trained_models.keys())
        }
        
        info_path = os.path.join(save_dir, 'model_info.joblib')
        joblib.dump(best_model_info, info_path)
    
    def load_models(self, load_dir='models/trained'):
        """Load all trained models"""
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Model directory not found: {load_dir}")
        
        # Load model info
        info_path = os.path.join(load_dir, 'model_info.joblib')
        if os.path.exists(info_path):
            model_info = joblib.load(info_path)
            self.best_model_name = model_info['best_model_name']
            
            # Load individual models
            for name in model_info['model_names']:
                model_path = os.path.join(load_dir, f'{name}_model.joblib')
                if os.path.exists(model_path):
                    self.trained_models[name] = joblib.load(model_path)
                    logger.info(f"Model {name} loaded from {model_path}")
            
            # Set best model
            if self.best_model_name in self.trained_models:
                self.best_model = self.trained_models[self.best_model_name]
        
        return len(self.trained_models) > 0
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance for tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.trained_models.get(model_name)
        if model is None:
            return None
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
