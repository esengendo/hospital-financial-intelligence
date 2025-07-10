"""
Hospital Financial Intelligence - Configuration Management

Centralized configuration for Docker deployment and environment setup.
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Optional


class Config:
    """Configuration management for hospital financial intelligence platform."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize configuration."""
        self.base_dir = base_dir or Path.cwd()
        
        # Reproducibility settings
        self.random_seed = int(os.getenv('RANDOM_SEED', '42'))
        self._set_random_seeds()
        
        # Data directories
        self.raw_data_dir = self._get_path_env('RAW_DATA_DIR', 'data/raw')
        self.processed_data_dir = self._get_path_env('PROCESSED_DATA_DIR', 'data/processed')
        
        # Output directories
        self.reports_dir = self._get_path_env('REPORTS_DIR', 'reports')
        self.visuals_dir = self._get_path_env('VISUALS_DIR', 'visuals')
        self.models_dir = self._get_path_env('MODELS_DIR', 'models')
        
        # Ensure directories exist
        self._create_directories()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
    
    def _get_path_env(self, env_var: str, default: str) -> Path:
        """Get path from environment variable or use default."""
        path_str = os.getenv(env_var, default)
        path = Path(path_str)
        
        # If path is absolute, use as-is; otherwise make relative to base_dir
        if path.is_absolute():
            return path
        else:
            return self.base_dir / path
    
    def _create_directories(self):
        """Create all configured directories."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.visuals_dir,
            self.models_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def eda_charts_dir(self) -> Path:
        """Get EDA charts subdirectory."""
        return self.visuals_dir / 'eda_charts'
    
    @property
    def shap_outputs_dir(self) -> Path:
        """Get SHAP outputs subdirectory."""
        return self.visuals_dir / 'shap_outputs'
    
    def get_data_file_pattern(self) -> str:
        """Get pattern for processed data files."""
        return "*.parquet"
    
    def validate_environment(self) -> tuple[bool, list[str]]:
        """Validate environment configuration."""
        issues = []
        
        # Check source directory
        src_dir = self.base_dir / 'src'
        if not src_dir.exists():
            issues.append(f"Source directory not found: {src_dir}")
        
        # Check processed data directory
        if not self.processed_data_dir.exists():
            issues.append(f"Processed data directory not found: {self.processed_data_dir}")
        else:
            data_files = list(self.processed_data_dir.glob(self.get_data_file_pattern()))
            if not data_files:
                issues.append(f"No processed data files found in: {self.processed_data_dir}")
        
        return len(issues) == 0, issues
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""Hospital Financial Intelligence Configuration:
  Base Directory: {self.base_dir}
  Raw Data: {self.raw_data_dir}
  Processed Data: {self.processed_data_dir}
  Reports: {self.reports_dir}
  Visuals: {self.visuals_dir}
  Models: {self.models_dir}"""

    # LLM Integration Configuration
    LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_MODEL_NAME_PROD = "meta-llama/Llama-3.3-70B-Instruct"
    LLM_USE_QUANTIZATION = True
    LLM_QUANTIZATION_BITS = 4
    LLM_MAX_NEW_TOKENS = 512
    LLM_TEMPERATURE = 0.7
    LLM_TOP_P = 0.9
    LLM_DO_SAMPLE = True
    LLM_CACHE_DIR = "./models/llm_cache"
    LLM_BATCH_SIZE = 4
    
    # LLM Output Configuration
    LLM_OUTPUT_FORMATS = ["markdown", "html", "json", "text"]
    LLM_REPORTS_DIR = "./reports/llm_generated"
    LLM_ENABLE_STREAMING = False


# Global configuration instance
_config: Optional[Config] = None


def get_config(base_dir: Optional[Path] = None) -> Config:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config(base_dir)
    return _config


def reset_config():
    """Reset global configuration."""
    global _config
    _config = None 