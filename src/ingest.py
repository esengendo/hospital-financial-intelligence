"""
Hospital Financial Intelligence - Data Ingestion

Robust data loader for California hospital financial data with configurable paths.
Docker-ready with environment variable support.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HospitalDataLoader:
    """Load and manage hospital financial data with configurable paths."""
    
    def __init__(self, processed_data_dir: Union[str, Path]):
        """
        Initialize data loader.
        
        Args:
            processed_data_dir: Directory containing processed parquet files
        """
        self.processed_dir = Path(processed_data_dir)
        
        if not self.processed_dir.exists():
            logger.warning(f"Processed data directory does not exist: {self.processed_dir}")
            
        logger.info(f"📂 Data loader initialized: {self.processed_dir}")
    
    def get_available_years(self) -> List[str]:
        """Get list of available fiscal years from processed files."""
        years = set()
        
        # Look for parquet files in the processed directory
        for file in self.processed_dir.glob("*.parquet"):
            # Extract years from filename patterns like "processed_financials_2019_2020.parquet"
            parts = file.stem.split('_')
            for part in parts:
                if part.isdigit() and len(part) == 4 and part.startswith('20'):
                    years.add(part)
        
        return sorted(list(years))
    
    def load_year_data(self, fiscal_year: Union[str, int]) -> pd.DataFrame:
        """
        Load processed financial data for a specific fiscal year.
        
        Args:
            fiscal_year: The fiscal year to load (e.g., "2023" or 2023)
            
        Returns:
            DataFrame with hospital financial data for the specified year
        """
        # Standardize year format
        fiscal_year = str(fiscal_year).replace("-", "_")
        
        # Find file that contains the fiscal year
        matching_files = list(self.processed_dir.glob(f"*_{fiscal_year}*.parquet"))
        
        if not matching_files:
            # Try searching for files that start with the year
            matching_files = list(self.processed_dir.glob(f"*{fiscal_year}_*.parquet"))

        if not matching_files:
            logger.error(f"Data file not found for year {fiscal_year}")
            available = self.get_available_years()
            logger.info(f"Available years: {available}")
            return pd.DataFrame()
        
        # Use the first matching file
        data_file = matching_files[0]
        logger.info(f"📄 Loading data from: {data_file.name}")
        
        try:
            df = pd.read_parquet(data_file)
            logger.info(f"✅ Loaded {len(df):,} records for year {fiscal_year}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load {data_file}: {e}")
            return pd.DataFrame()
    
    def load_multiple_years(self, years: List[Union[str, int]]) -> List[pd.DataFrame]:
        """
        Load data for multiple years.
        
        Args:
            years: List of fiscal years to load
            
        Returns:
            List of DataFrames, one for each successfully loaded year
        """
        datasets = []
        
        for year in years:
            df = self.load_year_data(year)
            if not df.empty:
                df['fiscal_year'] = str(year)
                datasets.append(df)
        
        logger.info(f"📊 Loaded data for {len(datasets)} out of {len(years)} requested years")
        return datasets
    
    def load_combined_data(self, years: Optional[List[Union[str, int]]] = None, 
                          include_year: bool = True) -> pd.DataFrame:
        """
        Load and combine data from multiple years.
        
        Args:
            years: List of years to load. If None, loads all available years.
            include_year: Whether to include fiscal_year column
            
        Returns:
            Combined DataFrame with data from all specified years
        """
        if years is None:
            years = self.get_available_years()
        
        logger.info(f"🔄 Loading combined data for years: {years}")
        
        datasets = self.load_multiple_years(years)
        
        if not datasets:
            logger.warning("No data loaded for any year")
            return pd.DataFrame()
        
        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True, sort=False)
        
        # Optional: Remove fiscal_year column if not needed
        if not include_year and 'fiscal_year' in combined_df.columns:
            combined_df = combined_df.drop('fiscal_year', axis=1)
        
        logger.info(f"✅ Combined dataset: {len(combined_df):,} total records")
        return combined_df
    
    def get_data_summary(self) -> Dict:
        """
        Get summary information about available data.
        
        Returns:
            Dictionary with data summary statistics
        """
        available_years = self.get_available_years()
        total_records = 0
        year_details = {}
        
        for year in available_years:
            df = self.load_year_data(year)
            if not df.empty:
                year_details[year] = {
                    'records': len(df),
                    'columns': len(df.columns),
                    'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                total_records += len(df)
        
        summary = {
            'available_years': available_years,
            'total_years': len(available_years),
            'total_records': total_records,
            'year_details': year_details,
            'data_directory': str(self.processed_dir)
        }
        
        return summary
    
    def analyze_data_quality(self, sample_years: Optional[List[str]] = None) -> Dict:
        """
        Analyze data quality across years.
        
        Args:
            sample_years: Years to analyze. If None, analyzes all available years.
            
        Returns:
            Dictionary with data quality metrics
        """
        if sample_years is None:
            sample_years = self.get_available_years()[:3]  # Sample first 3 years
        
        quality_report = {}
        
        for year in sample_years:
            df = self.load_year_data(year)
            if df.empty:
                continue
                
            # Calculate quality metrics
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=['number']).columns
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns
            
            quality_report[year] = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_percentage': round(missing_pct, 2),
                'numeric_columns': len(numeric_cols),
                'non_numeric_columns': len(non_numeric_cols),
                'data_completeness': round(100 - missing_pct, 2)
            }
        
        # Calculate overall quality
        if quality_report:
            avg_completeness = sum(report['data_completeness'] for report in quality_report.values()) / len(quality_report)
            quality_report['overall'] = {
                'average_completeness': round(avg_completeness, 2),
                'years_analyzed': len(quality_report) - 1  # Subtract 'overall' key
            }
        
        return quality_report


def main():
    """Demo function showing how to use the data loader."""
    try:
        # Initialize the loader
        loader = HospitalDataLoader()
        
        # Get summary of available data
        summary = loader.get_data_summary()
        print("📊 Data Summary:")
        print(f"  Years available: {summary['total_years']}")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Data directory: {summary['data_directory']}")
        
        # Load most recent year
        if summary['available_years']:
            recent_year = summary['available_years'][0]
            df = loader.load_year_data(recent_year)
            print(f"\n📈 Most Recent Year ({recent_year}):")
            print(f"  Records: {len(df):,}")
            print(f"  Metrics: {len(df.columns)}")
            
            # Show data quality
            quality = loader.analyze_data_quality()
            print(f"  Missing data: {quality['overall']['average_completeness']:.1f}%")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()