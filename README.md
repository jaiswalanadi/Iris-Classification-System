# Iris Classification System

A comprehensive machine learning system for classifying Iris flower species using advanced ML algorithms with a modern web interface.

## ✨ Features

- **Multiple ML Models**: Random Forest, SVM, KNN, Logistic Regression
- **Interactive Web Interface**: Modern, responsive UI for predictions
- **Comprehensive Analysis**: Detailed reports with visualizations
- **Real-time Predictions**: Instant species classification with confidence scores
- **Automated Training**: Hyperparameter tuning with GridSearchCV
- **Performance Metrics**: Cross-validation, confusion matrices, feature importance
- **RESTful API**: Complete API documentation with FastAPI

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Windows OS with VS Code (as per requirements)

### Installation

1. **Clone and Setup Project Structure**
   ```cmd
   # Run the setup commands from the artifacts above
   ```

2. **Place Your Data**
   ```cmd
   # Copy your Iris.csv file to data/raw/Iris.csv
   copy path\to\your\Iris.csv data\raw\Iris.csv
   ```

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```cmd
   python -m uvicorn app.main:app --reload
   ```

5. **Access the System**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative API Docs: http://localhost:8000/redoc

## 📋 Usage Guide

### Web Interface

1. **Home Page** (`/`)
   - System status overview
   - Quick action buttons
   - Model information display

2. **Training Models**
   - Click "Train Models" on the home page
   - Wait for training completion (may take a few minutes)
   - View training results and best model selection

3. **Making Predictions** (`/predict-page`)
   - Enter flower measurements (sepal/petal length and width)
   - Select model (optional - uses best model by default)
   - Get instant species prediction with confidence scores
   - View probability distribution across all species

4. **Generate Reports**
   - Click "Generate Report" after training models
   - Download comprehensive HTML report with visualizations

### API Usage

#### Train Models
```python
import requests

response = requests.post("http://localhost:8000/train", json={
    "remove_outliers": True,
    "use_grid_search": True
})
print(response.json())
```

#### Make Predictions
```python
response = requests.post("http://localhost:8000/predict", json={
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
})
print(response.json())
```

#### Get Model Information
```python
response = requests.get("http://localhost:8000/models")
print(response.json())
```

## 🏗️ Architecture Overview

### Backend Services
- **DataProcessor**: Data loading, preprocessing, outlier removal
- **MLModels**: Model training, prediction, evaluation
- **ModelTrainer**: Complete training pipeline management
- **ReportGenerator**: Comprehensive analysis reports

### Frontend Components
- **Modern Web UI**: Responsive design with interactive forms
- **Real-time Feedback**: Loading states, success/error modals
- **Visualization**: Charts and graphs for data analysis

### API Layer
- **FastAPI Framework**: Automatic API documentation
- **Pydantic Validation**: Request/response schema validation
- **Error Handling**: Comprehensive error messages

## 📊 Machine Learning Pipeline

### Data Processing
1. **Data Loading**: CSV file processing with validation
2. **Exploratory Analysis**: Statistical summaries and distributions
3. **Outlier Detection**: IQR method for data cleaning
4. **Feature Scaling**: StandardScaler normalization
5. **Train-Test Split**: Stratified 80-20 split

### Model Training
1. **Algorithm Selection**: 4 different ML algorithms
2. **Hyperparameter Tuning**: GridSearchCV optimization
3. **Cross-Validation**: 5-fold CV for robust evaluation
4. **Model Comparison**: Comprehensive metrics analysis
5. **Best Model Selection**: Automated based on performance

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Sensitivity for each species
- **F1-Score**: Balanced precision-recall metric
- **Confusion Matrix**: Detailed classification analysis
- **Feature Importance**: Model interpretability

## 🔧 Configuration

### Model Parameters
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **SVM**: C, gamma, kernel selection
- **KNN**: n_neighbors, weights, algorithm
- **Logistic Regression**: C, solver optimization

### Data Processing Options
- **Outlier Removal**: Configurable IQR-based filtering
- **Feature Scaling**: StandardScaler with fit/transform
- **Test Size**: Adjustable train-test split ratio

## 📈 Model Performance

### Expected Results
- **Accuracy**: 95-100% (Iris dataset is highly separable)
- **Training Time**: 1-3 minutes depending on grid search
- **Prediction Speed**: < 100ms per prediction
- **Memory Usage**: < 50MB for trained models

### Species Classification
- **Iris Setosa**: Easily distinguishable, 100% accuracy expected
- **Iris Versicolor**: Moderate separation, 90-95% accuracy
- **Iris Virginica**: Similar to Versicolor, 90-95% accuracy

## 🚀 Deployment

### Local Development
```cmd
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

#### Vercel (Recommended)
1. Install Vercel CLI: `npm i -g vercel`
2. Deploy: `vercel --prod`
3. Configuration included in `vercel.json`

#### Railway
1. Connect GitHub repository
2. Set Python buildpack
3. Configure environment variables

#### VPS/Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🧪 Testing

### Run Tests
```cmd
python -m pytest tests/ -v
```

### Test Coverage
```cmd
python tests/test_models.py
```

### Manual Testing
1. **API Endpoints**: Use `/docs` for interactive testing
2. **Web Interface**: Test all user flows
3. **Error Handling**: Test invalid inputs and edge cases

## 📁 Project Structure

```
iris-classification/
├── app/                    # Main application
│   ├── models/            # ML models and schemas
│   ├── services/          # Business logic services
│   ├── utils/             # Utility functions
│   └── main.py            # FastAPI application
├── data/                  # Data storage
│   ├── raw/               # Original datasets
│   └── processed/         # Processed data
├── models/                # Trained model storage
├── reports/               # Generated reports
├── static/                # Static web assets
├── templates/             # HTML templates
├── tests/                 # Test files
└── requirements.txt       # Dependencies
```

## 🛠️ Core Technologies

- **Backend**: FastAPI, Python 3.10
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Vercel, Railway, VPS compatible

## 📝 API Documentation

### Endpoints Overview

- `GET /` - Home page
- `GET /predict-page` - Prediction interface
- `POST /train` - Train models
- `POST /predict` - Make predictions
- `POST /predict-form` - Form-based prediction
- `GET /models` - Model information
- `GET /feature-importance` - Feature importance data
- `POST /generate-report` - Generate analysis report
- `GET /health` - System health check

### Response Formats

All API responses follow consistent JSON schemas with proper error handling and validation.

## 🤝 Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Error Handling**: Implement comprehensive error handling

## 📄 License

This project is created for educational and demonstration purposes. Feel free to use and modify as needed.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Not Found**: Place Iris.csv in `data/raw/` directory
3. **Training Fails**: Check data format and file permissions
4. **Predictions Fail**: Ensure models are trained first
