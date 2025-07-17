import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.target_name = 'Species'
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the Iris dataset from CSV file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame) -> dict:
        """Perform basic data exploration"""
        exploration = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'basic_stats': df.describe().to_dict(),
            'species_counts': df[self.target_name].value_counts().to_dict() if self.target_name in df.columns else {}
        }
        return exploration
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        
        for column in self.feature_names:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_clean[(df_clean[column] < lower_bound) | 
                                  (df_clean[column] > upper_bound)].index
                df_clean = df_clean.drop(outliers)
        
        logger.info(f"Outliers removed. Original shape: {df.shape}, New shape: {df_clean.shape}")
        return df_clean.reset_index(drop=True)
    
    def preprocess_data(self, df: pd.DataFrame, fit_transforms: bool = True) -> tuple:
        """Preprocess the data for machine learning"""
        # Separate features and target
        X = df[self.feature_names].copy()
        y = df[self.target_name].copy() if self.target_name in df.columns else None
        
        # Scale features
        if fit_transforms:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Encode labels
        if y is not None:
            if fit_transforms:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled, None
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)
    
    def get_species_mapping(self) -> dict:
        """Get the mapping between encoded labels and species names"""
        if hasattr(self.label_encoder, 'classes_'):
            return {i: species for i, species in enumerate(self.label_encoder.classes_)}
        return {}
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> list:
        """Convert encoded labels back to species names"""
        return self.label_encoder.inverse_transform(encoded_labels).tolist()
    
    def prepare_single_prediction(self, features: dict) -> np.ndarray:
        """Prepare single sample for prediction"""
        feature_array = np.array([[
            features['sepal_length'],
            features['sepal_width'], 
            features['petal_length'],
            features['petal_width']
        ]])
        
        # Scale the features using fitted scaler
        scaled_features = self.scaler.transform(feature_array)
        return scaled_features
