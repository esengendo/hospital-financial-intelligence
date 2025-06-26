"""
Feature Engineering for Hospital Financial Distress Prediction

This module creates a rich feature set from processed hospital financial data,
combining standard financial ratios, advanced predictive model components,
and time-series momentum indicators for machine learning.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

from .financial_metrics import FinancialMetricsCalculator

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Generates a comprehensive feature set for predicting hospital financial distress.
    
    The process involves:
    1. Calculating standard, explainable financial ratios.
    2. Engineering advanced predictive features (e.g., Altman Z-Score components).
    3. Creating time-series momentum features to capture trends.
    """
    
    def __init__(self, current_year_data: pd.DataFrame, all_data: pd.DataFrame):
        """
        Initialize the feature engineering process.
        
        Args:
            current_year_data (pd.DataFrame): The filtered data for the specific year to process.
            all_data (pd.DataFrame): The combined, processed financial data for all years for historical context.
        """
        self.current_year_data = current_year_data
        self.all_data = all_data
        self.year = self.current_year_data['year'].iloc[0]

    def generate_features(self) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.
        
        Returns:
            pd.DataFrame: A DataFrame with the complete feature set for the given year.
        """
        logger.info(f"🚀 Starting feature engineering for year {self.year}...")
        
        # Step 1: Calculate core financial ratios
        features = self._calculate_core_ratios()
        
        # Step 2: Engineer advanced predictive features
        advanced_features = self._calculate_advanced_features(features)
        features = pd.concat([features, advanced_features], axis=1)

        # Step 3: Engineer time-series momentum features
        ts_features = self._calculate_timeseries_features(features)
        features = pd.concat([features, ts_features], axis=1)

        logger.info(f"✅ Completed feature engineering for {self.year}. Generated {len(features.columns)} features.")
        return features.add_prefix('feature_')

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Perform safe division to avoid zero-division errors."""
        return numerator.div(denominator.replace(0, np.nan)).fillna(0)

    def _calculate_core_ratios(self) -> pd.DataFrame:
        """Calculate standard, explainable financial ratios using the metrics calculator."""
        logger.info("Calculating core financial ratios via FinancialMetricsCalculator...")
        
        # Use the existing, validated calculator
        metrics_calculator = FinancialMetricsCalculator(self.current_year_data)
        metrics_dict = metrics_calculator.calculate_all_metrics()
        
        if not metrics_dict:
            logger.warning(f"No metrics could be calculated for year {self.year}. Returning empty DataFrame.")
            return pd.DataFrame()
            
        # Convert the dictionary of Series to a single DataFrame
        ratios_df = pd.DataFrame(metrics_dict)
        
        logger.info(f"Successfully calculated {len(ratios_df.columns)} core financial ratios.")
        return ratios_df

    def _calculate_advanced_features(self, current_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate components of predictive models like Altman Z-Score."""
        logger.info("Engineering advanced predictive features (Altman Z-Score components)...")
        df = pd.DataFrame(index=self.current_year_data.index)
        
        calc = FinancialMetricsCalculator(self.current_year_data)

        total_assets = calc._get_col('total_assets')
        current_assets = calc._get_col('current_assets')
        current_liabilities = calc._get_col('current_liabilities')
        retained_earnings = calc._get_col('retained_earnings')
        operating_income = calc._get_col('operating_income')
        total_equity = calc._get_col('total_equity')
        total_liabilities = calc._get_col('total_liabilities')
        total_revenue = calc._get_col('total_revenue')

        working_capital = current_assets - current_liabilities
        df['z_working_capital_ratio'] = self._safe_divide(working_capital, total_assets)
        df['z_retained_earnings_ratio'] = self._safe_divide(retained_earnings, total_assets)
        df['z_ebit_ratio'] = self._safe_divide(operating_income, total_assets)
        df['z_equity_to_liability_ratio'] = self._safe_divide(total_equity, total_liabilities)
        df['z_sales_to_assets_ratio'] = self._safe_divide(total_revenue, total_assets)
        
        logger.info(f"Successfully calculated {len(df.columns)} advanced features.")
        return df

    def _calculate_timeseries_features(self, current_features: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend-based features using historical data."""
        logger.info("Creating time-series momentum features (Year-over-Year)...")
        df = pd.DataFrame(index=current_features.index)

        prev_year_data = self.all_data[self.all_data['year'] == self.year - 1]

        if prev_year_data.empty:
            logger.warning(f"No data for previous year ({self.year - 1}), skipping time-series features.")
            return df

        prev_calc = FinancialMetricsCalculator(prev_year_data)
        prev_ratios_dict = prev_calc.calculate_all_metrics()
        
        if not prev_ratios_dict:
            return df
        
        prev_ratios_df = pd.DataFrame(prev_ratios_dict).add_suffix('_py')
        merged_df = current_features.merge(prev_ratios_df, left_index=True, right_index=True, how='left')

        for col in current_features.columns:
            prev_col = f"{col}_py"
            if prev_col in merged_df.columns:
                yoy_change = self._safe_divide(merged_df[col] - merged_df[prev_col], merged_df[prev_col].abs())
                df[f'{col}_yoy_change'] = yoy_change.fillna(0)

        logger.info(f"Successfully calculated {len(df.columns)} time-series features.")
        return df 