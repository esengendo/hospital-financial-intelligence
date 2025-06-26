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

        # Step 3: Engineer time-series momentum features (reuse existing calculator)
        ts_features = self._calculate_timeseries_features()
        features = pd.concat([features, ts_features], axis=1)

        # Step 4: Preserve identifier columns and combine results
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