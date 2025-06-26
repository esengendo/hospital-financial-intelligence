"""
Hospital Financial Data Preprocessing Module

This module handles data cleaning, validation, and preprocessing for the
hospital financial analysis pipeline.

Key Features:
- Data cleaning and outlier detection
- Missing value imputation strategies
- Data type standardization
- Quality checks and validation
- Multi-year data integration
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')


class HospitalDataPreprocessor:
    """
    Comprehensive data preprocessing for hospital financial data.
    
    Handles cleaning, validation, and preparation of hospital financial
    data for machine learning and analysis.
    """
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            processed_data_dir: Directory to store processed data
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Data quality thresholds
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        self.missing_threshold = 0.5  # Maximum missing data percentage
        
        # Initialize scalers
        self.scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        
    def load_raw_data(self, raw_data_dir: str = "data/raw") -> pd.DataFrame:
        """
        Load and combine all raw hospital financial data files.
        
        Args:
            raw_data_dir: Directory containing raw data files
            
        Returns:
            Combined DataFrame with all years of data
        """
        raw_path = Path(raw_data_dir)
        data_files = list(raw_path.glob("hospital_financial_*.xlsx"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {raw_path}")
        
        all_data = []
        
        for file_path in data_files:
            try:
                self.logger.info(f"Loading data from {file_path}")
                df = pd.read_excel(file_path)
                
                # Add metadata
                year = file_path.stem.split('_')[-1]
                df['DATA_SOURCE_FILE'] = file_path.name
                df['LOAD_TIMESTAMP'] = pd.Timestamp.now()
                
                all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {str(e)}")
        
        if not all_data:
            raise ValueError("No data files could be loaded successfully")
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} total hospital records")
        
        return combined_df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        df = df.copy()
        
        # Standardize column names
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.replace('-', '_')
        
        # Define standard column mappings
        column_mappings = {
            'FACILITY_NAME': 'HOSPITAL_NAME',
            'BED_CAPACITY': 'LICENSED_BEDS',
            'TOTAL_REVENUE': 'TOTAL_OPERATING_REVENUE',
            'TOTAL_EXPENSES': 'TOTAL_OPERATING_EXPENSES',
        }
        
        df = df.rename(columns=column_mappings)
        
        # Convert data types
        numeric_columns = [
            'TOTAL_OPERATING_REVENUE', 'TOTAL_OPERATING_EXPENSES', 
            'LABOR_EXPENSES', 'CONTRACT_LABOR', 'DEPRECIATION',
            'INTEREST_EXPENSE', 'CASH_ON_HAND', 'TOTAL_ASSETS',
            'TOTAL_LIABILITIES', 'NET_WORTH', 'PATIENT_DAYS',
            'DISCHARGES', 'LICENSED_BEDS'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert categorical columns
        categorical_columns = ['COUNTY', 'HOSPITAL_NAME', 'OSHPD_ID']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect outliers using statistical methods.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (defaults to numeric columns)
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Z-score method
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_flags[f'{col}_outlier_zscore'] = z_scores > self.outlier_threshold
            
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_flags[f'{col}_outlier_iqr'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Combine outlier detection results
        df = pd.concat([df, outlier_flags], axis=1)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Analyze missing patterns
        missing_summary = df.isnull().sum() / len(df) * 100
        high_missing = missing_summary[missing_summary > self.missing_threshold * 100]
        
        if len(high_missing) > 0:
            self.logger.warning(f"Columns with >50% missing data: {high_missing.to_dict()}")
        
        # Define imputation strategies by column type
        
        # Financial ratios - use median imputation
        financial_columns = [
            'OPERATING_MARGIN', 'LABOR_COST_RATIO', 'CONTRACT_LABOR_RATIO',
            'DEBT_TO_ASSETS', 'DAYS_CASH_ON_HAND'
        ]
        
        for col in financial_columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.logger.info(f"Imputed {col} with median value: {median_value:.3f}")
        
        # Absolute financial values - use KNN imputation for better accuracy
        absolute_financial_columns = [
            'TOTAL_OPERATING_REVENUE', 'TOTAL_OPERATING_EXPENSES', 
            'LABOR_EXPENSES', 'CONTRACT_LABOR', 'CASH_ON_HAND'
        ]
        
        knn_columns = [col for col in absolute_financial_columns if col in df.columns]
        if knn_columns:
            knn_imputer = KNNImputer(n_neighbors=5)
            df[knn_columns] = knn_imputer.fit_transform(df[knn_columns])
            self.logger.info(f"Applied KNN imputation to: {knn_columns}")
        
        # Categorical variables - use mode
        categorical_columns = df.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                self.logger.info(f"Imputed {col} with mode: {mode_value}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'validation_timestamp': pd.Timestamp.now(),
            'issues': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        # Check for duplicates
        duplicate_count = df.duplicated(subset=['OSHPD_ID', 'FISCAL_YEAR']).sum()
        if duplicate_count > 0:
            validation_results['issues'].append(f"Found {duplicate_count} duplicate hospital-year records")
        
        # Check for negative financial values where inappropriate
        financial_checks = {
            'TOTAL_OPERATING_REVENUE': 'negative_revenue',
            'TOTAL_OPERATING_EXPENSES': 'negative_expenses', 
            'CASH_ON_HAND': 'negative_cash',
            'TOTAL_ASSETS': 'negative_assets'
        }
        
        for col, check_name in financial_checks.items():
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validation_results['warnings'].append(f"{negative_count} records with negative {col}")
        
        # Check for unrealistic ratios
        if 'LABOR_COST_RATIO' in df.columns:
            unrealistic_labor = ((df['LABOR_COST_RATIO'] < 0) | (df['LABOR_COST_RATIO'] > 1)).sum()
            if unrealistic_labor > 0:
                validation_results['warnings'].append(f"{unrealistic_labor} records with unrealistic labor cost ratios")
        
        # Calculate overall quality score
        issues_score = max(0, 1 - len(validation_results['issues']) * 0.1)
        warnings_score = max(0, 1 - len(validation_results['warnings']) * 0.05)
        completeness_score = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        validation_results['quality_score'] = (issues_score + warnings_score + completeness_score) / 3
        
        return validation_results
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived financial health indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Financial performance ratios
        if 'TOTAL_OPERATING_REVENUE' in df.columns and 'TOTAL_OPERATING_EXPENSES' in df.columns:
            df['NET_INCOME'] = df['TOTAL_OPERATING_REVENUE'] - df['TOTAL_OPERATING_EXPENSES']
            df['OPERATING_MARGIN'] = df['NET_INCOME'] / df['TOTAL_OPERATING_REVENUE']
        
        # Efficiency ratios
        if 'LABOR_EXPENSES' in df.columns and 'TOTAL_OPERATING_EXPENSES' in df.columns:
            df['LABOR_COST_RATIO'] = df['LABOR_EXPENSES'] / df['TOTAL_OPERATING_EXPENSES']
        
        if 'CONTRACT_LABOR' in df.columns and 'LABOR_EXPENSES' in df.columns:
            df['CONTRACT_LABOR_RATIO'] = df['CONTRACT_LABOR'] / df['LABOR_EXPENSES']
        
        # Liquidity ratios  
        if 'CASH_ON_HAND' in df.columns and 'TOTAL_OPERATING_EXPENSES' in df.columns:
            df['DAYS_CASH_ON_HAND'] = (df['CASH_ON_HAND'] / df['TOTAL_OPERATING_EXPENSES']) * 365
        
        # Leverage ratios
        if 'TOTAL_LIABILITIES' in df.columns and 'TOTAL_ASSETS' in df.columns:
            df['DEBT_TO_ASSETS'] = df['TOTAL_LIABILITIES'] / df['TOTAL_ASSETS']
        
        # Utilization metrics
        if 'PATIENT_DAYS' in df.columns and 'LICENSED_BEDS' in df.columns:
            df['OCCUPANCY_RATE'] = df['PATIENT_DAYS'] / (df['LICENSED_BEDS'] * 365)
        
        if 'TOTAL_OPERATING_REVENUE' in df.columns and 'PATIENT_DAYS' in df.columns:
            df['REVENUE_PER_PATIENT_DAY'] = df['TOTAL_OPERATING_REVENUE'] / df['PATIENT_DAYS']
        
        # Year-over-year growth rates (if multiple years available)
        if 'FISCAL_YEAR' in df.columns:
            df = df.sort_values(['OSHPD_ID', 'FISCAL_YEAR'])
            
            growth_columns = ['TOTAL_OPERATING_REVENUE', 'TOTAL_OPERATING_EXPENSES', 'NET_INCOME']
            for col in growth_columns:
                if col in df.columns:
                    df[f'{col}_GROWTH'] = df.groupby('OSHPD_ID')[col].pct_change()
        
        return df
    
    def preprocess_pipeline(self, raw_data_dir: str = "data/raw") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw_data_dir: Directory containing raw data files
            
        Returns:
            Tuple of (processed DataFrame, processing summary)
        """
        self.logger.info("Starting hospital financial data preprocessing...")
        
        # Load raw data
        df = self.load_raw_data(raw_data_dir)
        initial_count = len(df)
        
        # Standardize columns
        df = self.standardize_columns(df)
        
        # Detect outliers
        df = self.detect_outliers(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Final validation
        validation_results = self.validate_data_quality(df)
        
        # Save processed data
        output_path = self.processed_data_dir / "hospital_financial_processed.parquet"
        df.to_parquet(output_path, index=False)
        
        # Generate processing summary
        processing_summary = {
            'input_records': initial_count,
            'output_records': len(df),
            'processing_timestamp': pd.Timestamp.now(),
            'output_file': str(output_path),
            'validation_results': validation_results,
            'columns_created': len(df.columns),
            'data_quality_score': validation_results['quality_score']
        }
        
        self.logger.info(f"Preprocessing completed. Quality score: {validation_results['quality_score']:.3f}")
        self.logger.info(f"Processed data saved to: {output_path}")
        
        return df, processing_summary


if __name__ == "__main__":
    # Example usage
    preprocessor = HospitalDataPreprocessor()
    processed_df, summary = preprocessor.preprocess_pipeline()
    
    print("Preprocessing Summary:")
    print(f"Records processed: {summary['input_records']} -> {summary['output_records']}")
    print(f"Data quality score: {summary['data_quality_score']:.3f}")
    print(f"Output file: {summary['output_file']}")
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")