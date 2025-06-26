#!/usr/bin/env uv run python
"""
Enhanced Modeling Pipeline with Time-Series Features
===================================================

Trains and evaluates XGBoost model using enhanced feature set (147 features)
including the new time-series volatility, trend, and stability indicators.

Usage:
    uv run python run_enhanced_modeling.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.modeling import ModelTrainer
from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_enhanced_features() -> pd.DataFrame:
    """Load all enhanced feature files."""
    feature_dir = Path("data/features_enhanced")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Enhanced feature directory not found: {feature_dir}")
    
    feature_files = list(feature_dir.glob("features_enhanced_*.parquet"))
    if not feature_files:
        raise FileNotFoundError("No enhanced feature files found")
    
    logger.info(f"Loading {len(feature_files)} enhanced feature files...")
    
    all_features = []
    for file_path in sorted(feature_files):
        try:
            year_data = pd.read_parquet(file_path)
            all_features.append(year_data)
            logger.info(f"✅ Loaded {len(year_data)} records from {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
    
    if not all_features:
        raise RuntimeError("No valid enhanced feature files could be loaded")
    
    combined_data = pd.concat(all_features, ignore_index=True)
    logger.info(f"📊 Enhanced dataset: {len(combined_data)} records, {len(combined_data.columns)} features")
    return combined_data

def compare_feature_sets() -> None:
    """Compare original vs enhanced feature sets."""
    logger.info("📊 Comparing original vs enhanced feature sets...")
    
    # Load original features for comparison
    original_files = list(Path("data/features").glob("features_*.parquet"))
    if original_files:
        original_sample = pd.read_parquet(original_files[0])
        original_features = len(original_sample.columns)
    else:
        original_features = 33  # Known from previous runs
    
    # Load enhanced features
    enhanced_files = list(Path("data/features_enhanced").glob("features_enhanced_*.parquet"))
    if enhanced_files:
        enhanced_sample = pd.read_parquet(enhanced_files[0])
        enhanced_features = len(enhanced_sample.columns)
        
        # Categorize enhanced features
        new_features = [col for col in enhanced_sample.columns if col not in ['oshpd_id', 'year']]
        
        feature_categories = {
            'Core Financial Ratios': [col for col in new_features if not any(x in col for x in ['_rolling_', '_volatility_', '_cv_', '_trend_', '_momentum_', '_stability_', '_percentile_', '_bottom_10', '_dev_from_'])],
            'Rolling Averages': [col for col in new_features if '_rolling_' in col],
            'Volatility Measures': [col for col in new_features if '_volatility_' in col or '_cv_' in col],
            'Trend Analysis': [col for col in new_features if '_trend_' in col],
            'Momentum Indicators': [col for col in new_features if '_momentum_' in col],
            'Stability Scores': [col for col in new_features if '_stability_' in col],
            'Industry Percentiles': [col for col in new_features if '_percentile_' in col or '_bottom_10' in col],
            'Deviations': [col for col in new_features if '_dev_from_' in col]
        }
        
        print(f"\n📊 FEATURE SET COMPARISON:")
        print(f"Original Features: {original_features}")
        print(f"Enhanced Features: {enhanced_features}")
        print(f"New Features Added: {enhanced_features - original_features}")
        print(f"\n🎯 Enhanced Feature Categories:")
        for category, features in feature_categories.items():
            print(f"  {category:20}: {len(features):3d} features")

def train_enhanced_model(data: pd.DataFrame) -> Tuple[Dict, str]:
    """Train XGBoost model with enhanced features."""
    logger.info("🤖 Training enhanced XGBoost model...")
    
    # Initialize ModelTrainer with enhanced features
    trainer = ModelTrainer(data)
    
    # Create target variable
    logger.info("🎯 Creating target variable...")
    imbalance_ratio = trainer.create_target_variable()
    
    # Create train/test splits
    logger.info("📊 Creating time-based data splits...")
    trainer.split_data()
    
    print(f"\n📊 DATA SPLITS:")
    print(f"Training:   {len(trainer.X_train):4d} records ({trainer.y_train.sum():3d} distressed, {(1-trainer.y_train).sum():3d} healthy)")
    print(f"Test:       {len(trainer.X_test):4d} records ({trainer.y_test.sum():3d} distressed, {(1-trainer.y_test).sum():3d} healthy)")
    print(f"Validation: {len(trainer.X_val):4d} records ({trainer.y_val.sum():3d} distressed, {(1-trainer.y_val).sum():3d} healthy)")
    
    # Tune hyperparameters
    logger.info("🔧 Tuning hyperparameters...")
    trainer.tune_hyperparameters(mode='balanced')
    
    # Train model
    logger.info("🔬 Training XGBoost with enhanced feature set...")
    trainer.train_model()
    
    # Evaluate model
    logger.info("📈 Evaluating enhanced model performance...")
    results = trainer.evaluate_model()
    
    # Generate predictions for all sets
    train_pred = trainer.best_model.predict_proba(trainer.X_train_scaled)[:, 1]
    test_pred = trainer.best_model.predict_proba(trainer.X_test_scaled)[:, 1]
    val_pred = trainer.best_model.predict_proba(trainer.X_val_scaled)[:, 1]
    
    # Calculate comprehensive metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    enhanced_results = {
        'train_roc_auc': roc_auc_score(trainer.y_train, train_pred),
        'test_roc_auc': roc_auc_score(trainer.y_test, test_pred),
        'val_roc_auc': roc_auc_score(trainer.y_val, val_pred),
        'train_pr_auc': average_precision_score(trainer.y_train, train_pred),
        'test_pr_auc': average_precision_score(trainer.y_test, test_pred),
        'val_pr_auc': average_precision_score(trainer.y_val, val_pred),
        'best_params': trainer.best_model.get_params(),
        'feature_count': len(trainer.X_train.columns)
    }
    
    # Save enhanced model
    enhanced_model_path = trainer.save_model("enhanced_xgboost_model")
    
    logger.info(f"✅ Enhanced model saved to {enhanced_model_path}")
    
    return enhanced_results, str(enhanced_model_path)

def generate_enhanced_evaluation(data: pd.DataFrame, model_path: str) -> None:
    """Generate comprehensive evaluation with enhanced features."""
    logger.info("📊 Generating enhanced model evaluation...")
    
    # Load model
    import joblib
    from pathlib import Path
    model_file = Path(model_path) / "model.pkl"
    model = joblib.load(model_file)
    
    # Prepare data for evaluation
    trainer = ModelTrainer(data)
    trainer.create_target_variable()
    trainer.split_data()
    
    # Generate SHAP analysis with enhanced features
    logger.info("🔍 Generating SHAP analysis for enhanced features...")
    
    import shap
    import matplotlib.pyplot as plt
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    config = get_config()
    shap_values = explainer.shap_values(trainer.X_test_scaled.sample(min(500, len(trainer.X_test_scaled)), random_state=config.random_seed))
    
    # Enhanced feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': trainer.X_test_scaled.columns,
        'importance': model.feature_importances_,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    
    # Categorize features for analysis
    feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
    
    # Save enhanced visualizations
    vis_dir = Path("visuals/enhanced_model_evaluation")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Enhanced SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, trainer.X_test_scaled, max_display=20, show=False)
    plt.title('Enhanced Model: SHAP Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / 'enhanced_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature category analysis
    category_importance = feature_importance.groupby('category').agg({
        'shap_importance': ['sum', 'mean', 'count']
    }).round(3)
    category_importance.columns = ['total_importance', 'avg_importance', 'feature_count']
    category_importance = category_importance.sort_values('total_importance', ascending=False)
    
    print(f"\n🎯 ENHANCED FEATURE CATEGORY ANALYSIS:")
    print(category_importance)
    
    # 3. Top features comparison
    print(f"\n🏆 TOP 15 ENHANCED FEATURES (by SHAP importance):")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{row.name+1:2d}. {row['feature']:40} ({row['category']}) - SHAP: {row['shap_importance']:.3f}")
    
    # Save detailed results
    results_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)
    
    feature_importance.to_csv(results_dir / "enhanced_feature_importance.csv", index=False)
    category_importance.to_csv(results_dir / "enhanced_category_importance.csv")
    
    logger.info(f"📄 Enhanced evaluation results saved to {results_dir}/")

def categorize_feature(feature_name: str) -> str:
    """Categorize a feature based on its name."""
    if any(x in feature_name for x in ['_rolling_', '_dev_from_']):
        return 'Rolling Averages'
    elif any(x in feature_name for x in ['_volatility_', '_cv_']):
        return 'Volatility Measures'
    elif '_trend_' in feature_name:
        return 'Trend Analysis'
    elif '_momentum_' in feature_name:
        return 'Momentum Indicators'
    elif '_stability_' in feature_name:
        return 'Stability Scores'
    elif any(x in feature_name for x in ['_percentile_', '_bottom_10']):
        return 'Industry Percentiles'
    elif feature_name.startswith('z_'):
        return 'Altman Z-Score'
    elif '_yoy_change' in feature_name:
        return 'YoY Changes'
    else:
        return 'Core Financial Ratios'

def main():
    """Main execution function."""
    logger.info("🚀 Starting Enhanced Modeling Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load enhanced features
        logger.info("📂 Step 1: Loading enhanced feature dataset...")
        data = load_enhanced_features()
        
        # Step 2: Compare feature sets
        logger.info("📊 Step 2: Analyzing feature enhancements...")
        compare_feature_sets()
        
        # Step 3: Train enhanced model
        logger.info("🤖 Step 3: Training enhanced XGBoost model...")
        results, model_path = train_enhanced_model(data)
        
        # Step 4: Print performance comparison
        print(f"\n🎯 ENHANCED MODEL PERFORMANCE:")
        print(f"ROC-AUC (Test):  {results['test_roc_auc']:.3f}")
        print(f"PR-AUC (Test):   {results['test_pr_auc']:.3f}")
        print(f"ROC-AUC (Val):   {results['val_roc_auc']:.3f}")
        print(f"PR-AUC (Val):    {results['val_pr_auc']:.3f}")
        print(f"Features Used:   {results['feature_count']}")
        
        # Step 5: Generate enhanced evaluation
        logger.info("📊 Step 5: Generating enhanced evaluation...")
        generate_enhanced_evaluation(data, model_path)
        
        logger.info("✅ Enhanced Modeling Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        # Print next steps
        print(f"\n🎯 RESULTS SUMMARY:")
        print(f"✅ Enhanced model trained with {results['feature_count']} features")
        print(f"✅ Performance metrics calculated and saved")
        print(f"✅ SHAP analysis generated for feature importance")
        print(f"✅ Model saved to: {model_path}")
        
        print(f"\n📊 NEXT STEPS:")
        print("1. Compare with baseline model performance")
        print("2. Analyze enhanced feature importance patterns")
        print("3. Validate improved time-series feature performance")
        print("4. Prepare for deployment with enhanced feature set")
        
    except Exception as e:
        logger.error(f"❌ Enhanced modeling pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 