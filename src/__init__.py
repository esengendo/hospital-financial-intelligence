"""
Hospital Financial Intelligence - Healthcare Analytics Platform

A production-ready hospital financial distress prediction and analysis system
focused on California hospital data with modern MLOps and explainable AI capabilities.
"""

VERSION = "1.0.0"
DESCRIPTION = "Production-ready hospital financial distress prediction and analysis system"

# Import core modules for easy access (commenting out until modules are created)
from .ingest import HospitalDataLoader
from .config import Config, get_config
# from .preprocess import HospitalDataPreprocessor
# from .features import FinancialFeatureEngineer
# from .model import HospitalRiskPredictor
# from .explain import ModelExplainer
# from .llm_assist import FinancialLLMAssistant

__all__ = [
    "HospitalDataLoader",
    "Config",
    "get_config",
    # "HospitalDataPreprocessor", 
    # "FinancialFeatureEngineer",
    # "HospitalRiskPredictor",
    # "ModelExplainer",
    # "FinancialLLMAssistant",
    "VERSION",
    "DESCRIPTION",
]