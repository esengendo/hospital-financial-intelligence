#!/usr/bin/env uv run python
"""
Enhanced Feature Engineering Pipeline
====================================

Regenerates all feature files with enhanced time-series features based on our research findings.
This script applies our successful working features plus the 114 enhanced time-series indicators.

Usage:
    uv run python run_enhanced_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
sys.path.append('.')

from enhanced_time_series_features import EnhancedTimeSeriesFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_existing_features() -> pd.DataFrame:
    """Load all existing feature files."""
    feature_dir = Path("data/features")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    
    feature_files = list(feature_dir.glob("features_*.parquet"))
    if not feature_files:
        raise FileNotFoundError("No existing feature files found")
    
    logger.info(f"Loading {len(feature_files)} existing feature files...")
    
    all_features = []
    for file_path in sorted(feature_files):
        try:
            year_data = pd.read_parquet(file_path)
            all_features.append(year_data)
            logger.info(f"✅ Loaded {len(year_data)} records from {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
    
    if not all_features:
        raise RuntimeError("No valid feature files could be loaded")
    
    combined_data = pd.concat(all_features, ignore_index=True)
    logger.info(f"📊 Combined: {len(combined_data)} records across {len(all_features)} years")
    return combined_data

def generate_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply enhanced time-series feature engineering."""
    logger.info("🔬 Applying enhanced time-series feature engineering...")
    
    enhancer = EnhancedTimeSeriesFeatures()
    enhanced_data = enhancer.engineer_features(data)
    
    original_features = len(data.columns)
    enhanced_features = len(enhanced_data.columns)
    new_features = enhanced_features - original_features
    
    logger.info(f"✅ Enhanced features: {original_features} → {enhanced_features} (+{new_features} new)")
    return enhanced_data

def save_enhanced_features(enhanced_data: pd.DataFrame) -> None:
    """Save enhanced feature files by year."""
    output_dir = Path("data/features_enhanced")
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"💾 Saving enhanced features to {output_dir}/")
    
    years = sorted(enhanced_data['year'].unique())
    for year in years:
        year_data = enhanced_data[enhanced_data['year'] == year].copy()
        
        output_file = output_dir / f"features_enhanced_{year}.parquet"
        year_data.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Saved {len(year_data)} records for {year} ({len(year_data.columns)} features)")

def create_summary_report(original_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> None:
    """Create a summary report of the enhancement process."""
    logger.info("📊 Creating enhancement summary report...")
    
    original_features = len(original_data.columns)
    enhanced_features = len(enhanced_data.columns)
    new_features = enhanced_features - original_features
    
    # Identify new feature categories
    new_feature_cols = [col for col in enhanced_data.columns if col not in original_data.columns]
    
    feature_categories = {
        'Rolling Averages': [col for col in new_feature_cols if '_rolling_' in col],
        'Volatility Measures': [col for col in new_feature_cols if '_volatility_' in col or '_cv_' in col],
        'Trend Analysis': [col for col in new_feature_cols if '_trend_' in col],
        'Momentum Indicators': [col for col in new_feature_cols if '_momentum_' in col],
        'Stability Scores': [col for col in new_feature_cols if '_stability_' in col],
        'Industry Percentiles': [col for col in new_feature_cols if '_percentile_' in col or '_bottom_10' in col],
        'Deviations': [col for col in new_feature_cols if '_dev_from_' in col]
    }
    
    report = [
        "# Enhanced Feature Engineering Report",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- **Original Features**: {original_features}",
        f"- **Enhanced Features**: {enhanced_features}",
        f"- **New Features**: {new_features}",
        f"- **Years Covered**: {enhanced_data['year'].min()} - {enhanced_data['year'].max()}",
        f"- **Total Records**: {len(enhanced_data):,}",
        "",
        "## New Feature Categories",
    ]
    
    for category, features in feature_categories.items():
        if features:
            report.append(f"### {category} ({len(features)} features)")
            # Show data coverage for sample features
            for feat in features[:3]:
                non_null = enhanced_data[feat].notna().sum()
                coverage = non_null / len(enhanced_data) * 100
                report.append(f"- `{feat}`: {coverage:.1f}% coverage")
            if len(features) > 3:
                report.append(f"- ... and {len(features) - 3} more")
            report.append("")
    
    # Top features by data coverage
    coverage_analysis = []
    for col in new_feature_cols:
        non_null = enhanced_data[col].notna().sum()
        coverage = non_null / len(enhanced_data) * 100
        coverage_analysis.append((col, coverage))
    
    coverage_analysis.sort(key=lambda x: x[1], reverse=True)
    
    report.extend([
        "## Data Coverage Analysis",
        "Top 10 new features by data coverage:",
        ""
    ])
    
    for feat, coverage in coverage_analysis[:10]:
        report.append(f"- `{feat}`: {coverage:.1f}%")
    
    # Save report
    report_path = Path("reports/enhanced_feature_engineering_report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"📄 Report saved to {report_path}")

def main():
    """Main execution function."""
    logger.info("🚀 Starting Enhanced Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load existing features
        logger.info("📂 Step 1: Loading existing feature data...")
        original_data = load_existing_features()
        
        # Step 2: Generate enhanced features
        logger.info("🔬 Step 2: Generating enhanced time-series features...")
        enhanced_data = generate_enhanced_features(original_data)
        
        # Step 3: Save enhanced features
        logger.info("💾 Step 3: Saving enhanced feature files...")
        save_enhanced_features(enhanced_data)
        
        # Step 4: Create summary report
        logger.info("📊 Step 4: Creating summary report...")
        create_summary_report(original_data, enhanced_data)
        
        logger.info("✅ Enhanced Feature Engineering Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        # Print next steps
        print("\n🎯 NEXT STEPS:")
        print("1. Run enhanced modeling: uv run python run_enhanced_modeling.py")
        print("2. Compare model performance with enhanced features")
        print("3. Update evaluation dashboards")
        print("4. Validate enhanced feature importance with SHAP")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 