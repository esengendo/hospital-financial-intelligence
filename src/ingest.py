"""Hospital Financial Data Loader - Processed CHHS Data

This module provides functionality to load and work with the processed California 
hospital financial disclosure data that has been cleaned and standardized.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HospitalDataLoader:
    """Loads and processes cleaned California hospital financial data."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.processed_dir}")
        
        logger.info(f"Initialized data loader with directory: {self.data_dir}")

    def get_available_years(self) -> List[str]:
        """Get list of available fiscal years from processed data.
        
        Returns:
            List of available year strings (e.g., ['2022_2023', '2021_2022'])
        """
        available_files = list(self.processed_dir.glob("processed_financials_*.parquet"))
        years = []
        
        for file in available_files:
            # Extract year from filename like "processed_financials_2022_2023.parquet"
            year_part = file.stem.replace("processed_financials_", "")
            years.append(year_part)
        
        return sorted(years, reverse=True)  # Most recent first

    def load_year_data(self, fiscal_year: str) -> Optional[pd.DataFrame]:
        """Load hospital financial data for a specific fiscal year.
        
        Args:
            fiscal_year: Fiscal year in format "2022_2023" or "2022-2023"
            
        Returns:
            DataFrame with hospital financial data or None if not found
        """
        # Standardize year format
        fiscal_year = fiscal_year.replace("-", "_")
        
        file_path = self.processed_dir / f"processed_financials_{fiscal_year}.parquet"
        
        if not file_path.exists():
            logger.error(f"Data file not found for year {fiscal_year}: {file_path}")
            available = self.get_available_years()
            logger.info(f"Available years: {available}")
            return None
        
        try:
            logger.info(f"Loading data for fiscal year {fiscal_year}...")
            df = pd.read_parquet(file_path)
            logger.info(f"✅ Loaded {len(df)} hospitals with {len(df.columns)} financial metrics")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data for {fiscal_year}: {e}")
            return None

    def load_multiple_years(self, fiscal_years: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple fiscal years.
        
        Args:
            fiscal_years: List of years to load, or None for all available
            
        Returns:
            Dictionary mapping year strings to DataFrames
        """
        if fiscal_years is None:
            fiscal_years = self.get_available_years()
        
        data_dict = {}
        for year in fiscal_years:
            df = self.load_year_data(year)
            if df is not None:
                data_dict[year] = df
        
        logger.info(f"✅ Loaded data for {len(data_dict)} fiscal years")
        return data_dict

    def load_combined_data(self, fiscal_years: List[str] = None, 
                          add_year_column: bool = True) -> pd.DataFrame:
        """Load and combine data from multiple fiscal years.
        
        Args:
            fiscal_years: List of years to combine, or None for all available
            add_year_column: Whether to add a 'fiscal_year' column
            
        Returns:
            Combined DataFrame with all hospitals across years
        """
        data_dict = self.load_multiple_years(fiscal_years)
        
        if not data_dict:
            logger.error("No data loaded for combination")
            return pd.DataFrame()
        
        combined_dfs = []
        for year, df in data_dict.items():
            if add_year_column:
                df = df.copy()
                df['fiscal_year'] = year
            combined_dfs.append(df)
        
        combined_df = pd.concat(combined_dfs, ignore_index=False, sort=False)
        logger.info(f"✅ Combined data: {len(combined_df)} total hospital records")
        
        return combined_df

    def get_data_summary(self) -> Dict:
        """Get summary information about available data.
        
        Returns:
            Dictionary with data summary statistics
        """
        available_years = self.get_available_years()
        summary = {
            "total_years": len(available_years),
            "year_range": f"{available_years[-1]} to {available_years[0]}" if available_years else "None",
            "available_years": available_years,
            "year_details": {}
        }
        
        # Get details for each year
        for year in available_years[:5]:  # Limit to first 5 for performance
            df = self.load_year_data(year)
            if df is not None:
                summary["year_details"][year] = {
                    "hospitals": len(df),
                    "metrics": len(df.columns),
                    "sample_hospitals": df.index[:3].tolist()
                }
        
        return summary

    def analyze_data_quality(self, fiscal_year: str) -> Dict:
        """Analyze data quality for a specific fiscal year.
        
        Args:
            fiscal_year: Fiscal year to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        df = self.load_year_data(fiscal_year)
        if df is None:
            return {}
        
        # Convert non-numeric columns to numeric where possible
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        
        quality_metrics = {
            "fiscal_year": fiscal_year,
            "total_hospitals": len(df),
            "total_metrics": len(df.columns),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "columns_with_all_missing": df.columns[df.isnull().all()].tolist(),
            "columns_with_no_missing": df.columns[df.notnull().all()].tolist(),
            "numeric_columns": len(numeric_df.select_dtypes(include=[np.number]).columns),
            "text_columns": len(df.select_dtypes(include=['object']).columns)
        }
        
        return quality_metrics

    def get_financial_metrics_info(self, fiscal_year: str) -> Dict:
        """Get information about available financial metrics.
        
        Args:
            fiscal_year: Fiscal year to analyze
            
        Returns:
            Dictionary with metrics information
        """
        df = self.load_year_data(fiscal_year)
        if df is None:
            return {}
        
        # Look for key financial metrics based on common naming patterns
        key_metrics = {
            "revenue_metrics": [col for col in df.columns if any(term in col.lower() 
                               for term in ['revenue', 'income', 'receipts'])],
            "expense_metrics": [col for col in df.columns if any(term in col.lower() 
                               for term in ['expense', 'cost', 'expenditure'])],
            "asset_metrics": [col for col in df.columns if any(term in col.lower() 
                             for term in ['asset', 'cash', 'investment'])],
            "liability_metrics": [col for col in df.columns if any(term in col.lower() 
                                 for term in ['liability', 'debt', 'payable'])],
            "equity_metrics": [col for col in df.columns if any(term in col.lower() 
                              for term in ['equity', 'net_worth', 'surplus'])]
        }
        
        metrics_info = {
            "fiscal_year": fiscal_year,
            "total_metrics": len(df.columns),
            "key_metrics": key_metrics,
            "metrics_counts": {category: len(metrics) for category, metrics in key_metrics.items()}
        }
        
        return metrics_info

    def create_hospital_subset(self, fiscal_year: str, hospital_ids: List[str]) -> Optional[pd.DataFrame]:
        """Create a subset of data for specific hospitals.
        
        Args:
            fiscal_year: Fiscal year to load
            hospital_ids: List of hospital IDs to include
            
        Returns:
            DataFrame with subset of hospitals or None if not found
        """
        df = self.load_year_data(fiscal_year)
        if df is None:
            return None
        
        # Filter for requested hospitals
        subset_df = df[df.index.isin(hospital_ids)]
        
        if subset_df.empty:
            logger.warning(f"No hospitals found matching IDs: {hospital_ids}")
            logger.info(f"Available hospital IDs sample: {df.index[:5].tolist()}")
            return None
        
        logger.info(f"✅ Created subset with {len(subset_df)} hospitals")
        return subset_df


def main():
    """Demo function showing how to use the data loader."""
    try:
        # Initialize the loader
        loader = HospitalDataLoader()
        
        # Get summary of available data
        summary = loader.get_data_summary()
        print("📊 Data Summary:")
        print(f"  Years available: {summary['total_years']}")
        print(f"  Range: {summary['year_range']}")
        print(f"  Years: {summary['available_years'][:5]}...")
        
        # Load most recent year
        if summary['available_years']:
            recent_year = summary['available_years'][0]
            df = loader.load_year_data(recent_year)
            print(f"\n📈 Most Recent Year ({recent_year}):")
            print(f"  Hospitals: {len(df)}")
            print(f"  Metrics: {len(df.columns)}")
            
            # Show data quality
            quality = loader.analyze_data_quality(recent_year)
            print(f"  Missing data: {quality['missing_data_percentage']:.1f}%")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()