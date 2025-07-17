import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.data_processor import DataProcessor
from app.models.ml_models import IrisMLModels
from app.services.model_trainer import ModelTrainer

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        # Create sample data
        self.sample_data = pd.DataFrame({
            'SepalLengthCm': [5.1, 4.9, 4.7, 4.6, 5.0],
            'SepalWidthCm': [3.5, 3.0, 3.2, 3.1, 3.6],
            'PetalLengthCm': [1.4, 1.4, 1.3, 1.5, 1.4],
            'PetalWidthCm': [0.2, 0.2, 0.2, 0.2, 0.2],
            'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa']
        })
    
    def test_explore_data(self):
        """Test data exploration functionality"""
        exploration = self.processor.explore_data(self.sample_data)
        
        self.assertEqual(exploration['shape'], (5, 5))
        self.assertIn('SepalLengthCm', exploration['columns'])
        self.assertEqual(exploration['missing_values']['SepalLengthCm'], 0)
        self.assertEqual(exploration['species_counts']['Iris-setosa'], 5)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        X, y = self.processor.preprocess_data(self.sample_data, fit_transforms=True)
        
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(X.shape[1], 4)
        self.assertEqual(len(y), 5)
        self.assertTrue(all(y == 0))  # All samples are setosa (encoded as 0)
    
    def test_prepare_single_prediction(self):
        """Test single prediction preparation"""
        # First fit the scaler with sample data
        self.processor.preprocess_data(self.sample_data, fit_transforms=True)
        
        features = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        
        scaled_features = self.processor.prepare_single_prediction(features)
        
        self.assertEqual(scaled_features.shape, (1, 4))
        self.assertIsInstance(scaled_features, np.ndarray)

class TestMLModels(unittest.TestCase):
    def setUp(self):
        self.ml_models = IrisMLModels()
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_classes=3,
            n_informative=4,
            n_redundant=0,
            random_state=42
        )
        self.X_train = X[:80]
        self.X_test = X[80:]
        self.y_train = y[:80]
        self.y_test = y[80:]
    
    def test_train_models(self):
        """Test model training"""
        results = self.ml_models.train_models(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            use_grid_search=False  # Disable for faster testing
        )
        
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(self.ml_models.best_model_name)
        self.assertIsNotNone(self.ml_models.best_model)
        
        # Check if all models have required metrics
        for model_name, result in results.items():
            self.assertIn('metrics', result)
            self.assertIn('accuracy', result['metrics'])
            self.assertIn('precision', result['metrics'])
            self.assertIn('recall', result['metrics'])
            self.assertIn('f1_score', result['metrics'])
    
    def test_predict_single(self):
        """Test single prediction"""
        # Train models first
        self.ml_models.train_models(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            use_grid_search=False
        )
        
        # Make prediction
        sample = self.X_test[0:1]
        result = self.ml_models.predict_single(sample)
        
        self.assertIn('prediction', result)
        self.assertIn('probabilities', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['prediction'], (int, np.integer))
        self.assertIsInstance(result['confidence'], (float, np.floating))

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer()
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame({
            'SepalLengthCm': np.random.normal(5.8, 0.8, 150),
            'SepalWidthCm': np.random.normal(3.0, 0.4, 150),
            'PetalLengthCm': np.random.normal(3.8, 1.8, 150),
            'PetalWidthCm': np.random.normal(1.2, 0.8, 150),
            'Species': ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
        })
        
        # Create test data directory
        os.makedirs('test_data', exist_ok=True)
        self.test_file_path = 'test_data/test_iris.csv'
        self.test_data.to_csv(self.test_file_path, index=False)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists('test_data'):
            os.rmdir('test_data')
    
    def test_predict_single_before_training(self):
        """Test prediction before training should raise error"""
        features = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        
        with self.assertRaises(ValueError):
            self.trainer.predict_single(features)
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        info = self.trainer.get_model_info()
        
        self.assertIn('trained', info)
        self.assertIn('models', info)
        self.assertFalse(info['trained'])  # Should be False initially

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        # Create test data
        self.test_data = pd.DataFrame({
            'SepalLengthCm': [5.1, 4.9, 6.2, 5.9, 6.9, 6.7],
            'SepalWidthCm': [3.5, 3.0, 2.9, 3.0, 3.1, 3.1],
            'PetalLengthCm': [1.4, 1.4, 4.3, 4.2, 5.4, 5.6],
            'PetalWidthCm': [0.2, 0.2, 1.3, 1.3, 2.1, 2.4],
            'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 
                       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica']
        })
        
        # Save test data
        os.makedirs('test_data', exist_ok=True)
        self.test_file_path = 'test_data/integration_test.csv'
        self.test_data.to_csv(self.test_file_path, index=False)
    
    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists('test_data'):
            os.rmdir('test_data')
    
    def test_complete_pipeline(self):
        """Test the complete training and prediction pipeline"""
        trainer = ModelTrainer()
        
        # This would normally train models, but we'll test with minimal data
        # In a real scenario, you'd need more data for proper training
        
        # Test data loading
        data_processor = DataProcessor()
        df = data_processor.load_data(self.test_file_path)
        
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 5)
        
        # Test preprocessing
        X, y = data_processor.preprocess_data(df, fit_transforms=True)
        
        self.assertEqual(X.shape[1], 4)
        self.assertEqual(len(y), 6)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMLModels))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
