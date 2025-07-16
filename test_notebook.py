#!/usr/bin/env python3
"""
Test script for Hospital_Financial_Intelligence_Presentation.ipynb
"""

import unittest
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
import warnings
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        # Navigate to project root
        cls.project_root = Path(__file__).parent.resolve()
        cls.notebook_path = cls.project_root / 'notebooks' / 'Hospital_Financial_Intelligence_Presentation.ipynb'
        
        # Add project root to path
        if str(cls.project_root) not in sys.path:
            sys.path.insert(0, str(cls.project_root))
        
        # Change to project root
        os.chdir(cls.project_root)
        
        # Load notebook
        with open(cls.notebook_path) as f:
            cls.notebook = nbformat.read(f, as_version=4)
            
        # Setup execution environment
        cls.ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Suppress warnings
        warnings.filterwarnings('ignore')

    def test_environment_setup(self):
        """Test environment setup cells."""
        try:
            # Test imports
            from src.config import Config, get_config
            from src.ingest import HospitalDataLoader
            from src.eda import HospitalFinancialEDA
            from src.features import FeatureEngineering
            from src.modeling import ModelTrainer
            from src.financial_metrics import FinancialMetricsCalculator
            
            self.assertTrue(True, "Core imports successful")
        except ImportError as e:
            self.fail(f"Import error: {e}")
            
        # Test directory structure
        required_dirs = [
            './data/raw', 
            './data/processed', 
            './data/features', 
            './data/features_enhanced',
            './models',
            './reports'
        ]
        for dir_path in required_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Directory {dir_path} missing")
            
        # Test configuration
        try:
            config = get_config()
            self.assertIsNotNone(config, "Configuration should not be None")
        except Exception as e:
            self.fail(f"Configuration error: {e}")

    def test_data_processing(self):
        """Test data processing cells."""
        try:
            # Test data loading
            data_dir = Path('./data/processed')
            self.assertTrue(data_dir.exists(), "Processed data directory should exist")
            
            # Check for processed files
            processed_files = list(data_dir.glob('*.parquet'))
            self.assertGreater(len(processed_files), 0, "Should have processed data files")
            
            # Test sample data loading
            if processed_files:
                df = pd.read_parquet(processed_files[0])
                self.assertIsInstance(df, pd.DataFrame, "Should load as DataFrame")
                self.assertGreater(len(df), 0, "DataFrame should not be empty")
        except Exception as e:
            self.fail(f"Data processing error: {e}")

    def test_eda_functionality(self):
        """Test EDA cells."""
        try:
            from src.eda import HospitalFinancialEDA
            from src.config import get_config
            
            config = get_config()
            eda_platform = HospitalFinancialEDA(config)
            
            # Test EDA initialization
            self.assertIsNotNone(eda_platform, "EDA platform should initialize")
            
            # Check visualization outputs
            vis_dir = Path('visuals/eda_charts')
            self.assertTrue(vis_dir.exists(), "Visualization directory should exist")
            
            # Check for EDA outputs
            charts = list(vis_dir.glob('*.html'))
            self.assertGreater(len(charts), 0, "Should have generated charts")
        except Exception as e:
            self.fail(f"EDA error: {e}")

    def test_feature_engineering(self):
        """Test feature engineering cells."""
        try:
            features_dir = Path('./data/features_enhanced')
            self.assertTrue(features_dir.exists(), "Enhanced features directory should exist")
            
            # Check for feature files
            feature_files = list(features_dir.glob('*.parquet'))
            self.assertGreater(len(feature_files), 0, "Should have feature files")
            
            # Test feature data
            if feature_files:
                df = pd.read_parquet(feature_files[0])
                self.assertIsInstance(df, pd.DataFrame, "Should load as DataFrame")
                self.assertGreater(len(df.columns), 100, "Should have >100 features")
        except Exception as e:
            self.fail(f"Feature engineering error: {e}")

    def test_model_evaluation(self):
        """Test model evaluation cells."""
        try:
            models_dir = Path('./models')
            self.assertTrue(models_dir.exists(), "Models directory should exist")
            
            # Check for model artifacts
            model_files = list(models_dir.glob('**/model.pkl'))
            self.assertGreater(len(model_files), 0, "Should have model files")
            
            # Check for evaluation reports
            eval_reports = list(Path('./reports').glob('model_evaluation_*.json'))
            self.assertGreater(len(eval_reports), 0, "Should have evaluation reports")
        except Exception as e:
            self.fail(f"Model evaluation error: {e}")

    def test_dashboard_components(self):
        """Test dashboard visualization cells."""
        try:
            dashboard_file = Path('./streamlit_dashboard_modern.py')
            self.assertTrue(dashboard_file.exists(), "Dashboard file should exist")
            
            # Check for visualization outputs
            vis_dir = Path('./visuals')
            self.assertTrue(vis_dir.exists(), "Visualization directory should exist")
            
            # Check for required dashboard data
            self.assertTrue(Path('./data/features_enhanced').exists(), "Enhanced features required for dashboard")
            self.assertTrue(Path('./models').exists(), "Models required for dashboard")
            self.assertTrue(Path('./reports').exists(), "Reports required for dashboard")
        except Exception as e:
            self.fail(f"Dashboard error: {e}")

    def test_full_notebook_execution(self):
        """Test full notebook execution."""
        try:
            # Execute notebook
            self.ep.preprocess(self.notebook, {'metadata': {'path': str(self.project_root)}})
            self.assertTrue(True, "Notebook executed successfully")
        except Exception as e:
            self.fail(f"Notebook execution error: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 