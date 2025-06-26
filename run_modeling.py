#!/usr/bin/env uv run python
"""
Hospital Financial Distress Prediction - Model Training Pipeline

Production-ready modeling pipeline with:
- Full feature dataset (no sampling)
- Proper train/test/validation splits
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive evaluation and model persistence

Usage with UV:
    uv run python run_modeling.py --mode quick      # 5-10 minutes
    uv run python run_modeling.py --mode balanced   # 15-20 minutes (default)
    uv run python run_modeling.py --mode thorough   # 30+ minutes
    uv run python run_modeling.py --no-tuning       # 2-3 minutes (skip tuning)
"""

import logging
import pandas as pd
from pathlib import Path
from src.modeling import ModelTrainer
from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_feature_data() -> pd.DataFrame:
    """Load all engineered features from parquet files."""
    logger.info("🏥 Loading engineered feature data...")
    
    config = get_config()
    features_dir = config.base_dir / "data" / "features"
    
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load all feature files
    feature_files = sorted(features_dir.glob("features_*.parquet"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    
    datasets = []
    for file_path in feature_files:
        year_data = pd.read_parquet(file_path)
        datasets.append(year_data)
        logger.info(f"   ✓ {file_path.name}: {len(year_data)} records, {len(year_data.columns)} features")
    
    # Combine all years
    combined_data = pd.concat(datasets, ignore_index=True)
    
    logger.info(f"✅ Feature data loaded:")
    logger.info(f"   📊 Total records: {len(combined_data):,}")
    logger.info(f"   📅 Year range: {combined_data['year'].min()} - {combined_data['year'].max()}")
    logger.info(f"   📈 Features: {len(combined_data.columns) - 2}")  # Exclude oshpd_id, year
    logger.info(f"   🏥 Unique hospitals: {combined_data['oshpd_id'].nunique()}")
    
    return combined_data

def run_modeling_pipeline(tune_hyperparams: bool = True, mode: str = 'balanced'):
    """
    Execute the complete modeling pipeline with speed optimization.
    
    Args:
        tune_hyperparams: Whether to perform hyperparameter tuning
        mode: Speed mode - 'quick' (5-10 min), 'balanced' (15-20 min), 'thorough' (30+ min)
    """
    logger.info("🚀 Starting Hospital Financial Distress Prediction Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load feature data
        data = load_feature_data()
        
        # Step 2: Initialize model trainer
        trainer = ModelTrainer(data)
        
        # Step 3: Create target variable
        trainer.create_target_variable(
            target_metric='operating_margin',
            window=2,
            threshold=-5.0  # -5% operating margin for 2 consecutive years
        )
        
        # Step 4: Split data into train/test/validation
        trainer.split_data(
            train_end_year=2019,    # Training: 2003-2019 (includes start of distress)
            test_end_year=2020,     # Test: 2020
            val_start_year=2021     # Validation: 2021-2023
        )
        
        # Step 5: Hyperparameter tuning (optional)
        if tune_hyperparams:
            logger.info(f"🔧 Running {mode} hyperparameter tuning...")
            trainer.tune_hyperparameters(mode=mode, use_smote=True)
        
        # Step 6: Train final model
        trainer.train_model(use_tuned_params=tune_hyperparams, use_smote=True)
        
        # Step 7: Evaluate model
        results = trainer.evaluate_model()
        
        # Step 8: Generate feature importance plots
        trainer.plot_feature_importance()
        
        # Step 9: Save model
        model_dir = trainer.save_model()
        
        # Step 10: Summary report
        logger.info("📊 MODELING PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info("🏆 FINAL RESULTS:")
        
        for split_name, metrics in results.items():
            logger.info(f"📊 {split_name.upper()}:")
            logger.info(f"   🎯 ROC-AUC: {metrics['roc_auc']:.3f}")
            logger.info(f"   📊 PR-AUC: {metrics['pr_auc']:.3f}")
            logger.info(f"   🎯 F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"   🎯 F2-Score: {metrics['f2_score']:.3f} (recall-focused)")
            logger.info(f"   📋 Support: {metrics['support']} cases")
        
        logger.info(f"💾 Model saved to: {model_dir}")
        
        return results, model_dir
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hospital Financial Distress Prediction Pipeline")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--mode", choices=['quick', 'balanced', 'thorough'], default='balanced',
                       help="Speed mode: quick (5-10min), balanced (15-20min), thorough (30+min)")
    
    args = parser.parse_args()
    
    run_modeling_pipeline(
        tune_hyperparams=not args.no_tuning,
        mode=args.mode
    ) 