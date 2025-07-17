import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
import base64
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, data_processor, ml_models):
        self.data_processor = data_processor
        self.ml_models = ml_models
        self.report_dir = 'reports'
        os.makedirs(self.report_dir, exist_ok=True)
    
    def generate_full_report(self, data_path: str, training_results: dict) -> str:
        """Generate comprehensive HTML report"""
        # Load data for visualization
        df = self.data_processor.load_data(data_path)
        
        # Generate all visualizations
        plots = {
            'data_distribution': self._create_data_distribution_plot(df),
            'correlation_heatmap': self._create_correlation_heatmap(df),
            'species_comparison': self._create_species_comparison_plot(df),
            'model_performance': self._create_model_performance_plot(training_results),
            'confusion_matrices': self._create_confusion_matrices(training_results),
            'feature_importance': self._create_feature_importance_plot()
        }
        
        # Generate HTML report
        html_content = self._generate_html_report(df, training_results, plots)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"iris_classification_report_{timestamp}.html"
        report_path = os.path.join(self.report_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def _create_data_distribution_plot(self, df: pd.DataFrame) -> str:
        """Create data distribution plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        )
        
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for feature, pos in zip(features, positions):
            for species in df['Species'].unique():
                species_data = df[df['Species'] == species][feature]
                fig.add_trace(
                    go.Histogram(
                        x=species_data,
                        name=f"{species}",
                        opacity=0.7,
                        showlegend=(pos == (1,1))
                    ),
                    row=pos[0], col=pos[1]
                )
        
        fig.update_layout(
            title="Feature Distributions by Species",
            height=600,
            barmode='overlay'
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size":10}
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_species_comparison_plot(self, df: pd.DataFrame) -> str:
        """Create species comparison scatter plots"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Sepal Measurements', 'Petal Measurements']
        )
        
        # Sepal plot
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            fig.add_trace(
                go.Scatter(
                    x=species_data['SepalLengthCm'],
                    y=species_data['SepalWidthCm'],
                    mode='markers',
                    name=species,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Petal plot
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            fig.add_trace(
                go.Scatter(
                    x=species_data['PetalLengthCm'],
                    y=species_data['PetalWidthCm'],
                    mode='markers',
                    name=species,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Length (cm)", row=1, col=1)
        fig.update_yaxes(title_text="Width (cm)", row=1, col=1)
        fig.update_xaxes(title_text="Length (cm)", row=1, col=2)
        fig.update_yaxes(title_text="Width (cm)", row=1, col=2)
        
        fig.update_layout(
            title="Species Comparison",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_model_performance_plot(self, training_results: dict) -> str:
        """Create model performance comparison"""
        models = list(training_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [training_results[model]['metrics'][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=models,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_confusion_matrices(self, training_results: dict) -> str:
        """Create confusion matrices for all models"""
        models = list(training_results.keys())
        n_models = len(models)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=models,
            specs=[[{'type': 'heatmap'}]*cols for _ in range(rows)]
        )
        
        species_names = self.data_processor.get_species_mapping()
        labels = [species_names.get(i, f'Class_{i}') for i in range(3)]
        
        for idx, model in enumerate(models):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            cm = np.array(training_results[model]['metrics']['confusion_matrix'])
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=(idx == 0),
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size":12}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Confusion Matrices",
            height=600
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_feature_importance_plot(self) -> str:
        """Create feature importance plot for tree-based models"""
        feature_names = self.data_processor.feature_names
        
        fig = go.Figure()
        
        # Get feature importance for different models
        for model_name in self.ml_models.trained_models.keys():
            importance = self.ml_models.get_feature_importance(model_name)
            if importance is not None:
                fig.add_trace(go.Bar(
                    name=model_name.title(),
                    x=feature_names,
                    y=importance,
                    text=[f"{v:.3f}" for v in importance],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Feature Importance by Model",
            xaxis_title="Features",
            yaxis_title="Importance",
            barmode='group',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_html_report(self, df: pd.DataFrame, training_results: dict, plots: dict) -> str:
        """Generate complete HTML report"""
        
        # Calculate data exploration stats
        exploration = self.data_processor.explore_data(df)
        
        # Get best model info
        best_model = self.ml_models.best_model_name
        best_metrics = training_results[best_model]['metrics']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Classification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
            display: inline-block;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .best-model {{
            background-color: #e8f5e8;
            border-left: 5px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
        }}
        .plot-container {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌸 Iris Classification Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📊 Dataset Overview</h2>
        <div class="section">
            <div class="metric-card">
                <div class="metric-value">{exploration['shape'][0]}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exploration['shape'][1]}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(exploration['species_counts'])}</div>
                <div class="metric-label">Species</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(exploration['missing_values'].values())}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            
            <h3>Species Distribution</h3>
            <table>
                <tr><th>Species</th><th>Count</th><th>Percentage</th></tr>"""
        
        total_samples = sum(exploration['species_counts'].values())
        for species, count in exploration['species_counts'].items():
            percentage = (count / total_samples) * 100
            html_content += f"<tr><td>{species}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += f"""
            </table>
        </div>
        
        <h2>🤖 Model Performance</h2>
        <div class="best-model">
            <h3>🏆 Best Model: {best_model.title()}</h3>
            <div class="metric-card">
                <div class="metric-value">{best_metrics['accuracy']:.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics['precision']:.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics['recall']:.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics['f1_score']:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
        </div>
        
        <h3>All Models Comparison</h3>
        <table>
            <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>"""
        
        for model_name, results in training_results.items():
            metrics = results['metrics']
            html_content += f"""
            <tr>
                <td>{model_name.title()}</td>
                <td>{metrics['accuracy']:.3f}</td>
                <td>{metrics['precision']:.3f}</td>
                <td>{metrics['recall']:.3f}</td>
                <td>{metrics['f1_score']:.3f}</td>
            </tr>"""
        
        html_content += f"""
        </table>
        
        <h2>📈 Visualizations</h2>
        
        <div class="section">
            <h3>Data Distribution</h3>
            <div class="plot-container">{plots['data_distribution']}</div>
        </div>
        
        <div class="section">
            <h3>Feature Correlation</h3>
            <div class="plot-container">{plots['correlation_heatmap']}</div>
        </div>
        
        <div class="section">
            <h3>Species Comparison</h3>
            <div class="plot-container">{plots['species_comparison']}</div>
        </div>
        
        <div class="section">
            <h3>Model Performance</h3>
            <div class="plot-container">{plots['model_performance']}</div>
        </div>
        
        <div class="section">
            <h3>Confusion Matrices</h3>
            <div class="plot-container">{plots['confusion_matrices']}</div>
        </div>
        
        <div class="section">
            <h3>Feature Importance</h3>
            <div class="plot-container">{plots['feature_importance']}</div>
        </div>
        
        <h2>📝 Key Insights</h2>
        <div class="section">
            <ul>
                <li><strong>Best Performing Model:</strong> {best_model.title()} achieved {best_metrics['accuracy']:.1%} accuracy</li>
                <li><strong>Dataset Quality:</strong> {total_samples} samples with {sum(exploration['missing_values'].values())} missing values</li>
                <li><strong>Class Balance:</strong> Dataset is balanced with equal distribution across species</li>
                <li><strong>Feature Separability:</strong> Petal measurements show better species separation than sepal measurements</li>
                <li><strong>Model Consistency:</strong> Multiple models achieved high performance, indicating good data quality</li>
            </ul>
        </div>
        
        <h2>🔬 Technical Details</h2>
        <div class="section">
            <h3>Data Preprocessing</h3>
            <ul>
                <li>Outlier removal using IQR method</li>
                <li>Feature standardization using StandardScaler</li>
                <li>Label encoding for species classification</li>
                <li>80-20 train-test split with stratification</li>
            </ul>
            
            <h3>Model Training</h3>
            <ul>
                <li>Hyperparameter tuning using GridSearchCV</li>
                <li>5-fold cross-validation for model evaluation</li>
                <li>Multiple algorithms tested: Random Forest, SVM, KNN, Logistic Regression</li>
                <li>Performance metrics: Accuracy, Precision, Recall, F1-Score</li>
            </ul>
        </div>
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
            <p>Generated by Iris Classification ML Pipeline | {datetime.now().year}</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html_content
