"""
Hospital Financial AI - Production-ready financial distress prediction system.

This package provides a complete pipeline for:
- Hospital financial data ingestion and preprocessing
- Feature engineering for financial health indicators
- Machine learning model training and evaluation
- SHAP-based explainability analysis
- LLM-powered executive summary generation
- Interactive Streamlit dashboard

Author: Senior Data Scientist
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Senior Data Scientist"
__email__ = "data@hospital-analysis.com"

# Package metadata
PACKAGE_NAME = "hospital-financial-ai"
DESCRIPTION = "Production-ready hospital financial distress prediction and analysis system"

# Import core modules for easy access (commenting out until modules are created)
from .ingest import HospitalDataIngester
# from .preprocess import HospitalDataPreprocessor
# from .features import FinancialFeatureEngineer
# from .model import HospitalRiskPredictor
# from .explain import ModelExplainer
# from .llm_assist import FinancialLLMAssistant

__all__ = [
    "HospitalDataIngester",
    # "HospitalDataPreprocessor", 
    # "FinancialFeatureEngineer",
    # "HospitalRiskPredictor",
    # "ModelExplainer",
    # "FinancialLLMAssistant",
]