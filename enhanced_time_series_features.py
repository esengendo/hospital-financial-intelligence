#!/usr/bin/env uv run python
"""
Enhanced Time-Series Feature Engineering
========================================

Based on research findings, this script improves our time-series features to increase
their predictive power while working with the existing successful column mappings.

Key Improvements:
1. Multi-year trends (2-year, 3-year rolling averages)
2. Volatility and stability measures
3. Trend classification (improving/declining/stable)
4. Momentum indicators
5. Industry percentile rankings

Research Sources:
- Healthcare financial distress literature shows trend stability is more predictive than YoY changes
- Financial volatility is a key distress indicator
- Multi-year patterns capture long-term deterioration better than single-year changes

Usage:
    uv run python enhanced_time_series_features.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesConfig:
    """Configuration for time-series feature engineering."""
    windows: List[int] = None  # Rolling window sizes
    volatility_windows: List[int] = None  # Windows for volatility calculation
    trend_threshold: float = 0.05  # 5% threshold for trend classification
    min_years_for_trend: int = 3  # Minimum years needed for trend analysis
    
    def __post_init__(self):
        if self.windows is None:
            self.windows = [2, 3]  # 2-year and 3-year rolling averages
        if self.volatility_windows is None:
            self.volatility_windows = [3, 5]  # 3-year and 5-year volatility

class EnhancedTimeSeriesFeatures:
    """
    Enhanced time-series feature engineering focused on financial stability and trends.
    
    This class works with our existing successful features (operating_margin, 
    times_interest_earned, etc.) to create more predictive time-series indicators.
    """
    
    def __init__(self, config: TimeSeriesConfig = None):
        self.config = config or TimeSeriesConfig()
        self.base_features = [
            'operating_margin', 'total_margin', 'times_interest_earned',
            'current_ratio', 'days_cash_on_hand', 'return_on_assets',
            'debt_to_assets', 'asset_turnover'
        ]
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer enhanced time-series features.
        
        Args:
            data: DataFrame with columns [oshpd_id, year, base_features...]
            
        Returns:
            DataFrame with enhanced time-series features
        """
        logger.info(f"Engineering enhanced time-series features for {len(data)} records...")
        
        # Ensure data is sorted
        data = data.sort_values(['oshpd_id', 'year']).copy()
        
        # Create enhanced features
        enhanced_data = data.copy()
        
        # 1. Multi-year rolling averages
        enhanced_data = self._add_rolling_averages(enhanced_data)
        
        # 2. Volatility measures
        enhanced_data = self._add_volatility_features(enhanced_data)
        
        # 3. Trend classification
        enhanced_data = self._add_trend_features(enhanced_data)
        
        # 4. Momentum indicators
        enhanced_data = self._add_momentum_features(enhanced_data)
        
        # 5. Stability scores
        enhanced_data = self._add_stability_features(enhanced_data)
        
        # 6. Industry percentile rankings
        enhanced_data = self._add_percentile_features(enhanced_data)
        
        logger.info(f"Enhanced features created: {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    def _add_rolling_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling averages for key financial metrics."""
        logger.info("Adding rolling average features...")
        
        for feature in self.base_features:
            if feature not in data.columns:
                continue
                
            for window in self.config.windows:
                col_name = f"{feature}_rolling_{window}y"
                data[col_name] = data.groupby('oshpd_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).mean()
                )
                
                # Add deviation from rolling average
                dev_col = f"{feature}_dev_from_{window}y_avg"
                data[dev_col] = data[feature] - data[col_name]
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures for financial stability assessment."""
        logger.info("Adding volatility features...")
        
        for feature in self.base_features:
            if feature not in data.columns:
                continue
                
            for window in self.config.volatility_windows:
                # Standard deviation (volatility)
                vol_col = f"{feature}_volatility_{window}y"
                data[vol_col] = data.groupby('oshpd_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                )
                
                # Coefficient of variation (normalized volatility)
                cv_col = f"{feature}_cv_{window}y"
                rolling_mean = data.groupby('oshpd_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).mean()
                )
                data[cv_col] = data[vol_col] / np.abs(rolling_mean)
                data[cv_col] = data[cv_col].replace([np.inf, -np.inf], np.nan)
        
        return data
    
    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend classification features."""
        logger.info("Adding trend classification features...")
        
        for feature in self.base_features:
            if feature not in data.columns:
                continue
            
            # Calculate 3-year trend slope
            trend_col = f"{feature}_trend_3y"
            data[trend_col] = data.groupby('oshpd_id').apply(
                lambda group: self._calculate_trend_slope(group, feature, 3)
            ).reset_index(level=0, drop=True)
            
            # Classify trend direction
            trend_dir_col = f"{feature}_trend_direction"
            data[trend_dir_col] = np.where(
                data[trend_col] > self.config.trend_threshold, 1,  # Improving
                np.where(data[trend_col] < -self.config.trend_threshold, -1, 0)  # Declining, Stable
            )
            
            # Trend consistency (how consistently is the trend moving in one direction)
            consistency_col = f"{feature}_trend_consistency"
            data[consistency_col] = data.groupby('oshpd_id')[feature].transform(
                lambda x: self._calculate_trend_consistency(x, 3)
            )
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (acceleration/deceleration)."""
        logger.info("Adding momentum indicators...")
        
        for feature in self.base_features:
            if feature not in data.columns:
                continue
            
            # Rate of change acceleration (2nd derivative)
            momentum_col = f"{feature}_momentum"
            data[momentum_col] = data.groupby('oshpd_id')[feature].transform(
                lambda x: x.diff().diff()  # Second difference approximates acceleration
            )
            
            # Momentum direction (positive = accelerating improvement)
            momentum_dir_col = f"{feature}_momentum_direction"
            data[momentum_dir_col] = np.sign(data[momentum_col])
        
        return data
    
    def _add_stability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add financial stability composite scores."""
        logger.info("Adding stability composite features...")
        
        # Key stability indicators
        stability_features = ['operating_margin', 'times_interest_earned', 'days_cash_on_hand']
        
        for feature in stability_features:
            if feature not in data.columns:
                continue
            
            # Stability score (inverse of coefficient of variation)
            cv_col = f"{feature}_cv_3y"
            if cv_col in data.columns:
                stability_col = f"{feature}_stability_score"
                data[stability_col] = 1 / (1 + data[cv_col].fillna(1))
        
        # Composite financial stability score
        stability_cols = [col for col in data.columns if '_stability_score' in col]
        if stability_cols:
            data['composite_stability_score'] = data[stability_cols].mean(axis=1, skipna=True)
        
        return data
    
    def _add_percentile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add industry percentile rankings by year."""
        logger.info("Adding industry percentile features...")
        
        key_features = ['operating_margin', 'times_interest_earned', 'days_cash_on_hand']
        
        for feature in key_features:
            if feature not in data.columns:
                continue
            
            percentile_col = f"{feature}_industry_percentile"
            data[percentile_col] = data.groupby('year')[feature].transform(
                lambda x: x.rank(pct=True)
            )
            
            # Flag hospitals in bottom 10% (high risk)
            bottom_10_col = f"{feature}_bottom_10pct"
            data[bottom_10_col] = (data[percentile_col] <= 0.1).astype(int)
        
        return data
    
    def _calculate_trend_slope(self, group: pd.DataFrame, feature: str, window: int) -> pd.Series:
        """Calculate trend slope using linear regression."""
        def slope_calc(series):
            if len(series) < window:
                return np.nan
            
            # Use last 'window' years
            y = series.tail(window).values
            x = np.arange(len(y))
            
            if len(y) < 2:
                return np.nan
            
            # Simple linear regression slope
            x_mean, y_mean = np.mean(x), np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            return slope
        
        return group[feature].rolling(window=window, min_periods=2).apply(slope_calc)
    
    def _calculate_trend_consistency(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate how consistently a trend moves in one direction."""
        def consistency_calc(subseries):
            if len(subseries) < window:
                return np.nan
            
            # Calculate period-to-period changes
            changes = subseries.diff().dropna()
            if len(changes) == 0:
                return np.nan
            
            # Consistency = proportion of changes in same direction as majority
            positive_changes = (changes > 0).sum()
            negative_changes = (changes < 0).sum()
            
            if positive_changes + negative_changes == 0:
                return 0
            
            majority_direction = max(positive_changes, negative_changes)
            return majority_direction / (positive_changes + negative_changes)
        
        return series.rolling(window=window, min_periods=2).apply(consistency_calc)

def test_enhanced_features():
    """Test enhanced time-series features on our existing data."""
    logger.info("🧪 Testing enhanced time-series features...")
    
    # Load existing feature data
    feature_files = list(Path("data/features").glob("features_*.parquet"))
    if not feature_files:
        logger.error("No feature files found in data/features/")
        return
    
    # Load and combine recent years
    recent_data = []
    for file_path in sorted(feature_files)[-5:]:  # Last 5 years
        year_data = pd.read_parquet(file_path)
        recent_data.append(year_data)
    
    combined_data = pd.concat(recent_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_data)} records from {len(recent_data)} years")
    
    # Test enhancement
    enhancer = EnhancedTimeSeriesFeatures()
    enhanced_data = enhancer.engineer_features(combined_data)
    
    # Report results
    original_features = len(combined_data.columns)
    enhanced_features = len(enhanced_data.columns)
    new_features = enhanced_features - original_features
    
    print(f"\n📊 ENHANCED TIME-SERIES RESULTS:")
    print(f"Original features: {original_features}")
    print(f"Enhanced features: {enhanced_features}")
    print(f"New features added: {new_features}")
    
    # Show sample of new features
    new_feature_cols = [col for col in enhanced_data.columns if col not in combined_data.columns]
    print(f"\n🔍 Sample of new features ({len(new_feature_cols)} total):")
    for feature in new_feature_cols[:15]:
        non_null = enhanced_data[feature].notna().sum()
        print(f"  {feature:40} ({non_null:4d} non-null)")
    
    # Check data quality
    print(f"\n📈 Data quality check:")
    for feature in ['operating_margin_volatility_3y', 'operating_margin_trend_direction', 'composite_stability_score']:
        if feature in enhanced_data.columns:
            non_null = enhanced_data[feature].notna().sum()
            print(f"  {feature:40}: {non_null:4d}/{len(enhanced_data)} ({non_null/len(enhanced_data)*100:.1f}%)")

if __name__ == "__main__":
    test_enhanced_features() 