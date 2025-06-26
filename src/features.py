"""
Feature Engineering for Hospital Financial Distress Prediction

This module creates a rich feature set from processed hospital financial data,
combining standard financial ratios, advanced predictive model components,
and enhanced time-series features for machine learning.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

from .financial_metrics import FinancialMetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesConfig:
    """Configuration for enhanced time-series feature engineering."""
    windows: List[int] = None  # Rolling window sizes
    volatility_windows: List[int] = None  # Windows for volatility calculation
    trend_threshold: float = 0.05  # 5% threshold for trend classification
    min_years_for_trend: int = 3  # Minimum years needed for trend analysis
    
    def __post_init__(self):
        if self.windows is None:
            self.windows = [2, 3]  # 2-year and 3-year rolling averages
        if self.volatility_windows is None:
            self.volatility_windows = [3, 5]  # 3-year and 5-year volatility

class FeatureEngineering:
    """
    Generates a comprehensive feature set for predicting hospital financial distress.
    
    The process involves:
    1. Calculating standard, explainable financial ratios.
    2. Engineering advanced predictive features (e.g., Altman Z-Score components).
    3. Creating enhanced time-series features (volatility, trends, momentum).
    """
    
    def __init__(self, current_year_data: pd.DataFrame, all_data: pd.DataFrame, 
                 ts_config: TimeSeriesConfig = None):
        """
        Initialize the feature engineering process.
        
        Args:
            current_year_data (pd.DataFrame): The filtered data for the specific year to process.
            all_data (pd.DataFrame): The combined, processed financial data for all years for historical context.
            ts_config (TimeSeriesConfig): Configuration for enhanced time-series features.
        """
        self.current_year_data = current_year_data
        self.all_data = all_data
        self.year = self.current_year_data['year'].iloc[0]
        self.ts_config = ts_config or TimeSeriesConfig()
        
        # Create a single calculator instance to reuse across all feature calculations
        logger.info(f"Initializing FinancialMetricsCalculator for year {self.year}...")
        self.calculator = FinancialMetricsCalculator(self.current_year_data)

    def generate_features(self) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.
        
        Returns:
            pd.DataFrame: A DataFrame with the complete feature set for the given year.
        """
        logger.info(f"🚀 Starting feature engineering for year {self.year}...")
        
        # Step 1: Calculate core financial ratios (reuse existing calculator)
        features = self._calculate_core_ratios()
        
        # Step 2: Engineer advanced predictive features (reuse existing calculator)
        advanced_features = self._calculate_advanced_features()
        features = pd.concat([features, advanced_features], axis=1)

        # Step 3: Engineer basic time-series momentum features (reuse existing calculator)
        ts_features = self._calculate_timeseries_features()
        features = pd.concat([features, ts_features], axis=1)

        # Step 4: Engineer enhanced time-series features (NEW)
        enhanced_ts_features = self._calculate_enhanced_timeseries_features()
        if not enhanced_ts_features.empty:
            features = pd.concat([features, enhanced_ts_features], axis=1)

        # Step 5: Preserve identifier columns and combine results
        try:
            # Use the actual column names from our processed data
            identifier_cols = []
            if 'FAC_NO' in self.current_year_data.columns:
                identifier_cols.append('FAC_NO')
            if 'year' in self.current_year_data.columns:
                identifier_cols.append('year')
            
            if identifier_cols:
                identifiers = self.current_year_data[identifier_cols].copy()
                # Rename FAC_NO to oshpd_id for consistency with modeling pipeline
                if 'FAC_NO' in identifiers.columns:
                    identifiers = identifiers.rename(columns={'FAC_NO': 'oshpd_id'})
                
                final_features = pd.concat([identifiers, features], axis=1)
            else:
                logger.warning("No identifier columns found. Proceeding without them.")
                final_features = features
        except Exception as e:
            logger.error(f"Error preserving identifiers: {e}")
            final_features = features

        logger.info(f"✅ Completed feature engineering for {self.year}. Generated {len(features.columns)} features.")
        return final_features

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Perform safe division to avoid zero-division errors."""
        return numerator.div(denominator.replace(0, np.nan)).fillna(0)

    def _calculate_core_ratios(self) -> pd.DataFrame:
        """Calculate standard, explainable financial ratios using the metrics calculator."""
        logger.info("Calculating core financial ratios via FinancialMetricsCalculator...")
        
        # Use the existing, validated calculator
        metrics_dict = self.calculator.calculate_all_metrics()
        
        if not metrics_dict:
            logger.warning(f"No metrics could be calculated for year {self.year}. Returning empty DataFrame.")
            return pd.DataFrame()
            
        # Convert the dictionary of Series to a single DataFrame
        ratios_df = pd.DataFrame(metrics_dict)
        
        logger.info(f"Successfully calculated {len(ratios_df.columns)} core financial ratios.")
        return ratios_df

    def _calculate_advanced_features(self) -> pd.DataFrame:
        """Calculate components of predictive models like Altman Z-Score."""
        logger.info("Engineering advanced predictive features (Altman Z-Score components)...")
        df = pd.DataFrame(index=self.current_year_data.index)
        
        total_assets = self.calculator._get_col('total_assets')
        current_assets = self.calculator._get_col('current_assets')
        current_liabilities = self.calculator._get_col('current_liabilities')
        retained_earnings = self.calculator._get_col('retained_earnings')
        operating_income = self.calculator._get_col('operating_income')
        total_equity = self.calculator._get_col('total_equity')
        total_liabilities = self.calculator._get_col('total_liabilities')
        total_revenue = self.calculator._get_col('total_revenue')

        working_capital = current_assets - current_liabilities
        df['z_working_capital_ratio'] = self._safe_divide(working_capital, total_assets)
        df['z_retained_earnings_ratio'] = self._safe_divide(retained_earnings, total_assets)
        df['z_ebit_ratio'] = self._safe_divide(operating_income, total_assets)
        df['z_equity_to_liability_ratio'] = self._safe_divide(total_equity, total_liabilities)
        df['z_sales_to_assets_ratio'] = self._safe_divide(total_revenue, total_assets)
        
        logger.info(f"Successfully calculated {len(df.columns)} advanced features.")
        return df

    def _calculate_timeseries_features(self) -> pd.DataFrame:
        """Create momentum and trend-based features using historical data."""
        logger.info("Creating time-series momentum features (Year-over-Year)...")
        df = pd.DataFrame(index=self.current_year_data.index)

        prev_year_data = self.all_data[self.all_data['year'] == self.year - 1]
        if prev_year_data.empty:
            logger.warning(f"No data for previous year ({self.year - 1}). Skipping time-series features.")
            return df

        # Calculate previous year ratios using a separate calculator
        prev_calc = FinancialMetricsCalculator(prev_year_data)
        prev_ratios_dict = prev_calc.calculate_all_metrics()
        
        if not prev_ratios_dict:
            return df
        
        # Get current year ratios for comparison
        current_ratios_dict = self.calculator.calculate_all_metrics()
        
        if not current_ratios_dict:
            return df
            
        # Calculate YoY changes for each ratio
        for ratio_name in current_ratios_dict.keys():
            if ratio_name in prev_ratios_dict:
                current_values = current_ratios_dict[ratio_name]
                prev_values = prev_ratios_dict[ratio_name]
                
                yoy_change = self._safe_divide(
                    current_values - prev_values,
                    prev_values.abs()
                ) * 100
                
                df[f'{ratio_name}_yoy_change'] = yoy_change

        logger.info(f"Successfully calculated {len(df.columns)} time-series features.")
        return df

    def _calculate_enhanced_timeseries_features(self) -> pd.DataFrame:
        """
        Create enhanced time-series features using multi-year data.
        
        This method creates sophisticated time-series features including:
        - Rolling averages and deviations
        - Volatility measures (standard deviation, coefficient of variation)
        - Trend classification and consistency
        - Momentum indicators
        - Industry percentile rankings
        - Composite stability scores
        """
        logger.info("Creating enhanced time-series features...")
        
        # Get hospital ID for this year's data
        if 'FAC_NO' not in self.current_year_data.columns:
            logger.warning("No FAC_NO column found. Cannot create enhanced time-series features.")
            return pd.DataFrame()
        
        current_hospital_ids = self.current_year_data['FAC_NO'].unique()
        
        # Filter all_data to hospitals in current year and sort by hospital and year
        hospital_data = self.all_data[self.all_data['FAC_NO'].isin(current_hospital_ids)].copy()
        hospital_data = hospital_data.sort_values(['FAC_NO', 'year'])
        
        if len(hospital_data) < 2:
            logger.warning("Insufficient historical data for enhanced time-series features.")
            return pd.DataFrame()
        
        # Create a temporary DataFrame with key financial metrics for time-series analysis
        metrics_data = self._create_metrics_timeseries(hospital_data)
        
        if metrics_data.empty:
            logger.warning("Could not create metrics time-series. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Apply enhanced time-series engineering
        enhanced_data = metrics_data.copy()
        
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
        
        # Filter to current year only
        current_year_enhanced = enhanced_data[enhanced_data['year'] == self.year].copy()
        
        # Remove identifier columns and return only new features
        feature_cols = [col for col in current_year_enhanced.columns 
                       if col not in ['FAC_NO', 'year'] and col not in metrics_data.columns]
        
        if not feature_cols:
            logger.warning("No enhanced time-series features created.")
            return pd.DataFrame()
        
        # Return features with proper indexing
        result_df = current_year_enhanced[feature_cols].copy()
        result_df.index = self.current_year_data.index
        
        logger.info(f"Successfully calculated {len(result_df.columns)} enhanced time-series features.")
        return result_df

    def _create_metrics_timeseries(self, hospital_data: pd.DataFrame) -> pd.DataFrame:
        """Create a time-series DataFrame with key financial metrics."""
        metrics_list = []
        
        for year in sorted(hospital_data['year'].unique()):
            year_data = hospital_data[hospital_data['year'] == year]
            if year_data.empty:
                continue
                
            # Calculate metrics for this year
            calc = FinancialMetricsCalculator(year_data)
            metrics_dict = calc.calculate_all_metrics()
            
            if not metrics_dict:
                continue
            
            # Convert to DataFrame and add identifiers
            year_metrics = pd.DataFrame(metrics_dict)
            year_metrics['FAC_NO'] = year_data['FAC_NO'].values
            year_metrics['year'] = year
            
            metrics_list.append(year_metrics)
        
        if not metrics_list:
            return pd.DataFrame()
        
        return pd.concat(metrics_list, ignore_index=True)

    def _add_rolling_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling averages for key financial metrics."""
        base_features = ['operating_margin', 'total_margin', 'times_interest_earned', 'days_cash_on_hand']
        
        for feature in base_features:
            if feature not in data.columns:
                continue
                
            for window in self.ts_config.windows:
                col_name = f"{feature}_rolling_{window}y"
                data[col_name] = data.groupby('FAC_NO')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).mean()
                )
                
                # Add deviation from rolling average
                dev_col = f"{feature}_dev_from_{window}y_avg"
                data[dev_col] = data[feature] - data[col_name]
        
        return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures for financial stability assessment."""
        base_features = ['operating_margin', 'total_margin', 'times_interest_earned']
        
        for feature in base_features:
            if feature not in data.columns:
                continue
                
            for window in self.ts_config.volatility_windows:
                # Standard deviation (volatility)
                vol_col = f"{feature}_volatility_{window}y"
                data[vol_col] = data.groupby('FAC_NO')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                )
                
                # Coefficient of variation (normalized volatility)
                cv_col = f"{feature}_cv_{window}y"
                rolling_mean = data.groupby('FAC_NO')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=2).mean()
                )
                data[cv_col] = data[vol_col] / np.abs(rolling_mean)
                data[cv_col] = data[cv_col].replace([np.inf, -np.inf], np.nan)
        
        return data

    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend classification features."""
        base_features = ['operating_margin', 'times_interest_earned']
        
        for feature in base_features:
            if feature not in data.columns:
                continue
            
            # Calculate 3-year trend slope
            trend_col = f"{feature}_trend_3y"
            data[trend_col] = data.groupby('FAC_NO').apply(
                lambda group: self._calculate_trend_slope(group, feature, 3)
            ).reset_index(level=0, drop=True)
            
            # Classify trend direction
            trend_dir_col = f"{feature}_trend_direction"
            data[trend_dir_col] = np.where(
                data[trend_col] > self.ts_config.trend_threshold, 1,  # Improving
                np.where(data[trend_col] < -self.ts_config.trend_threshold, -1, 0)  # Declining, Stable
            )
        
        return data

    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (acceleration/deceleration)."""
        base_features = ['operating_margin', 'times_interest_earned']
        
        for feature in base_features:
            if feature not in data.columns:
                continue
            
            # Rate of change acceleration (2nd derivative)
            momentum_col = f"{feature}_momentum"
            data[momentum_col] = data.groupby('FAC_NO')[feature].transform(
                lambda x: x.diff().diff()  # Second difference approximates acceleration
            )
            
            # Momentum direction (positive = accelerating improvement)
            momentum_dir_col = f"{feature}_momentum_direction"
            data[momentum_dir_col] = np.sign(data[momentum_col])
        
        return data

    def _add_stability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add financial stability composite scores."""
        stability_features = ['operating_margin', 'times_interest_earned']
        
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
        key_features = ['operating_margin', 'times_interest_earned']
        
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