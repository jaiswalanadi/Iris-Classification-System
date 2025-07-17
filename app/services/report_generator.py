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
        correlation_matrix = numeric_df.corr
