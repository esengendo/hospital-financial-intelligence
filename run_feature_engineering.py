#!/usr/bin/env python3
"""
Orchestration script for running the feature engineering pipeline.

This script loads processed hospital financial data, generates a comprehensive
feature set for each year using the FeatureEngineering class, and saves the
output to the 'data/features' directory.
"""

import pandas as pd
import logging
from pathlib import Path
import argparse

from src.config import get_config, Config
from src.ingest import load_multiple_years_data
from src.features import FeatureEngineering

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def run_pipeline(config: Config, start_year: int, end_year: int):
    """
    Execute the feature engineering pipeline for a range of years.

    Args:
        config (Config): The project configuration object.
        start_year (int): The first year to process.
        end_year (int): The last year to process.
    """
    logger.info("🚀 Starting Feature Engineering Pipeline...")
    
    # Load all necessary data at once for efficiency
    years_to_load = list(range(start_year - 1, end_year + 1))
    all_data = load_multiple_years_data(years_to_load, config.processed_data_dir)

    if all_data.empty:
        logger.error("❌ No data loaded. Aborting feature engineering.")
        return

    features_dir = config.base_dir / 'data' / 'features'
    features_dir.mkdir(exist_ok=True)
    logger.info(f"💾 Feature sets will be saved to: {features_dir}")

    for year in range(start_year, end_year + 1):
        logger.info(f"🔧 Processing year: {year}")
        try:
            # Filter the combined dataframe for the specific year being processed
            year_data = all_data[all_data['year'] == year]
            
            if year_data.empty:
                logger.warning(f"No data found for year {year}. Skipping.")
                continue

            # Pass only the relevant year's data and the full history for time-series features
            feature_generator = FeatureEngineering(year_data, all_data)
            
            # Generate the full feature set
            features_df = feature_generator.generate_features()

            if not features_df.empty:
                # Save the feature set to a Parquet file
                output_path = features_dir / f"features_{year}.parquet"
                features_df.to_parquet(output_path, index=True)
                logger.info(f"✅ Successfully saved features for {year} to {output_path}")
            else:
                logger.warning(f"⚠️ No features generated for {year}. Skipping file save.")

        except Exception as e:
            logger.error(f"❌ Failed to process year {year}: {e}", exc_info=True)

    logger.info("🎉 Feature Engineering Pipeline Completed.")

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Run the Hospital Financial Feature Engineering Pipeline.")
    parser.add_argument('--start-year', type=int, default=2003, help="The first year to process for feature engineering.")
    parser.add_argument('--end-year', type=int, default=2023, help="The last year to process for feature engineering.")
    args = parser.parse_args()

    config = get_config()
    run_pipeline(config, args.start_year, args.end_year)

if __name__ == "__main__":
    main() 