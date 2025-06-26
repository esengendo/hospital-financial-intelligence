#!/usr/bin/env python3
"""
Hospital Data Streamliner

This script processes the raw, non-standard CHHS hospital financial data files 
and transforms them into a clean, standardized, and machine-readable format.

It performs the following key steps:
1.  Reads the header rows of each data file to map cryptic column names to 
    (Page, Line) tuples.
2.  Uses the 'Page Column Line Labels' files to map these (Page, Line) tuples
    to standardized, human-readable financial metric names.
3.  Reloads the data, applying the new standardized column names.
4.  Saves the processed, clean data to the 'data/processed' directory.
"""
import pandas as pd
from pathlib import Path
import logging
import re

# --- Configuration ---
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_LEVEL = logging.INFO

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_DIR.mkdir(exist_ok=True)


class DataStreamliner:
    def __init__(self, year_str):
        self.year_str = year_str # e.g., "2022_2023"
        self.data_file_path = RAW_DIR / f"hospital_financial_{self.year_str}.xlsx"
        self.label_file_path = self._find_label_file()
        self.column_mapping = {}

    def _find_label_file(self) -> Path:
        """
        Finds the most appropriate label file for the given year, using highly
        specific logic to ensure the correct file is chosen for each period.
        """
        year = int(self.year_str.split('_')[0])
        
        # This logic now correctly maps each data year to its specific label file.
        if year >= 2015:
            # The 2014-15 labels are the most recent known ones and apply to later years.
            logging.info(f"Using '2014-15' labels for data year {year}.")
            return RAW_DIR / "Page Column Line Labels 2014-15.xlsx"
        elif year == 2014:
            return RAW_DIR / "Page Column Line Labels 2014-15.xlsx"
        elif year == 2013:
            return RAW_DIR / "Page Column Line Labels 2013-2014.xlsx"
        elif 2010 <= year <= 2012:
            # This range uses the comprehensive 2004-2013 label set.
            return RAW_DIR / "Page Column Line Labels 2004 - 2013.xlsx"
        elif 2004 <= year <= 2009:
            # This range uses the specific 2004-2010 documentation.
            return RAW_DIR / "Page Column Line Labels 2004-2010.xlsx"
        elif 2002 <= year <= 2003:
            # The earliest data uses the 2002-2004 documentation.
            return RAW_DIR / "Page Column Line Labels 2002-2004.xls"
        else:
            # As a fallback, use the most recent one, but log a clear warning.
            logging.error(f"Could not find a specific label file for year {year}. Critical fallback to latest.")
            return RAW_DIR / "Page Column Line Labels 2014-15.xlsx"
    
    def create_column_map_from_labels(self):
        """Creates a mapping from (Page, Line) to a standardized metric name."""
        if not self.label_file_path.exists():
            logging.warning(f"Label file not found for {self.year_str} at {self.label_file_path}")
            return
            
        logging.info(f"Loading labels from {self.label_file_path.name}")
        df_labels = pd.read_excel(self.label_file_path)
        
        # Standardize column names in label file
        df_labels.columns = [str(c).replace(' ', '_') for c in df_labels.columns]
        
        # Find the description column
        desc_col = next((c for c in df_labels.columns if 'description' in c.lower()), None)
        if not desc_col:
            logging.error("Could not find a 'Description' column in label file.")
            return

        for _, row in df_labels.iterrows():
            try:
                page = int(row.Page)
                line = int(row.Line)
                # Clean up the metric name to be a valid column name
                metric_name = str(row[desc_col])
                # A more robust way to clean names
                metric_name = re.sub(r'\s+', '_', metric_name) # Replace spaces with underscore
                metric_name = re.sub(r'[^0-9a-zA-Z_]+', '', metric_name) # Remove special chars
                metric_name = metric_name.strip('_')
                self.column_mapping[(page, line)] = metric_name
            except (ValueError, TypeError):
                # Skip rows where Page or Line are not valid numbers
                continue
        
        logging.info(f"Created mapping for {len(self.column_mapping)} metrics.")

    def process_file(self):
        """Processes a single raw data file into a clean DataFrame."""
        if not self.data_file_path.exists():
            logging.error(f"Data file not found: {self.data_file_path}")
            return None

        logging.info(f"--- Processing: {self.data_file_path.name} ---")
        self.create_column_map_from_labels()

        # 1. Read header to map cryptic columns to (Page, Line)
        try:
            df_header = pd.read_excel(self.data_file_path, nrows=3, header=None)
        except Exception as e:
            logging.error(f"Could not read header from {self.data_file_path.name}: {e}")
            return None

        page_row = df_header.iloc[0].tolist()
        line_row = df_header.iloc[2].tolist()  # Use row 2 for Line numbers

        header_map = {}
        for i, (page, line) in enumerate(zip(page_row, line_row)):
            try:
                # Use the standardized metric name from the label file
                header_map[i] = self.column_mapping.get((int(page), int(line)), f"P{page}_L{line}")
            except (ValueError, TypeError):
                # Handle non-numeric page/line values in header
                header_map[i] = f"HEADER_{i}"

        # 2. Read the actual data, skipping header rows and applying new names
        try:
            df_data = pd.read_excel(self.data_file_path, skiprows=3, header=None)
        except Exception as e:
            logging.error(f"Could not read data from {self.data_file_path.name}: {e}")
            return None
        
        # Rename columns using our map
        num_cols_to_rename = min(len(header_map), len(df_data.columns))
        df_data.rename(columns={i: header_map[i] for i in range(num_cols_to_rename)}, inplace=True)

        # 3. Set Hospital ID as index
        df_data.rename(columns={'HEADER_0': 'hospital_id'}, inplace=True)
        df_data.set_index('hospital_id', inplace=True)
        
        # 4. Clean up and De-duplicate Columns
        df_data.dropna(axis=1, how='all', inplace=True)
        df_data = df_data.loc[~df_data.index.isna()]

        # Handle duplicate columns by appending a suffix
        cols = pd.Series(df_data.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df_data.columns = cols
        
        # Convert all object columns to strings to prevent type inference errors
        for col in df_data.select_dtypes(include=['object']).columns:
            df_data[col] = df_data[col].astype(str)

        # Final check: ensure index is also a string before saving
        df_data.index = df_data.index.astype(str)
        
        logging.info(f"Successfully processed. Shape: {df_data.shape}")
        return df_data

def main():
    """Main function to process all detected data files."""
    logging.info("🚀 Starting Hospital Data Streamlining Process")
    
    # --- Process all available years ---
    processed_files = 0
    # Get all standardized financial files
    financial_files = list(RAW_DIR.glob("hospital_financial_*.xlsx"))

    for f in financial_files:
        match = re.search(r'(\d{4}_\d{4})', f.name)
        if match:
            year_to_process = match.group(1)
            streamliner = DataStreamliner(year_to_process)
            clean_df = streamliner.process_file()

            if clean_df is not None and not clean_df.empty:
                output_path = PROCESSED_DIR / f"processed_financials_{year_to_process}.parquet"
                clean_df.to_parquet(output_path, engine='fastparquet')
                logging.info(f"✅ Saved cleaned data to {output_path}")
                processed_files += 1
            else:
                logging.error(f"Failed to process {f.name}")
    
    logging.info(f"🎉 Streamlining complete. Processed {processed_files}/{len(financial_files)} files.")


if __name__ == "__main__":
    main() 