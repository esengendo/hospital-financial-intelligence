#!/usr/bin/env python3
"""
Intelligent Data File Renamer

This script renames the downloaded CHHS hospital financial data files to the 
standardized naming convention required by the project's ingestion scripts.

It handles variations in filenames and ensures consistency.
"""

import os
from pathlib import Path

# --- CONFIGURATION ---

# Directory where the downloaded files are located
RAW_DATA_DIR = Path("data/raw")

# Mapping from year to a unique part of the original filename and the new filename.
# This allows for flexible matching.
FILE_MAPPING = {
    # Tier 1 (2020-2023)
    "2022-2023": ("2022 - 2023", "hospital_financial_2022_2023.xlsx"),
    "2021-2022": ("2021 - 2022", "hospital_financial_2021_2022.xlsx"),
    "2020-2021": ("2020 - 2021", "hospital_financial_2020_2021.xlsx"),
    
    # Tier 2 (2018-2020)
    "2019-2020": ("2019 - 2020", "hospital_financial_2019_2020.xlsx"),
    "2018-2019": ("2018 - 2019", "hospital_financial_2018_2019.xlsx"),
    
    # Tier 3 (2012-2018)
    "2017-2018": ("2017 - 2018", "hospital_financial_2017_2018.xlsx"),
    "2016-2017": ("2016 - 2017", "hospital_financial_2016_2017.xlsx"),
    "2015-2016": ("2015 - 2016", "hospital_financial_2015_2016.xlsx"),
    "2014-2015": ("2014 - 2015", "hospital_financial_2014_2015.xlsx"),
    "2013-2014": ("2013 - 2014", "hospital_financial_2013_2014.xlsx"),
    "2012-2013": ("2012 - 2013", "hospital_financial_2012_2013.xlsx"),

    # Historical Files (Pre-2012) - CORRECTED PATTERNS
    "2011-2012": ("2011-2012", "hospital_financial_2011_2012.xlsx"),
    "2010-2011": ("2010-2011", "hospital_financial_2010_2011.xlsx"),
    "2009-2010": ("2009-2010", "hospital_financial_2009_2010.xlsx"),
    "2008-2009": ("2008-2009", "hospital_financial_2008_2009.xlsx"),
    "2007-2008": ("2007-2008", "hospital_financial_2007_2008.xlsx"),
    "2006-2007": ("2006-2007", "hospital_financial_2006_2007.xlsx"),
    "2005-2006": ("2005-2006", "hospital_financial_2005_2006.xlsx"),
    "2004-2005": ("2004-2005", "hospital_financial_2004_2005.xlsx"),
    "2003-2004": ("2003-2004", "hospital_financial_2003_2004.xlsx"),
    "2002-2003": ("2002-2003", "hospital_financial_2002_2003.xlsx"),
}

def rename_files():
    """
    Renames files in the raw data directory based on the mapping.
    """
    print(f"📁 Checking directory: {RAW_DATA_DIR}")
    if not RAW_DATA_DIR.is_dir():
        print(f"❌ Error: Directory not found. Aborting.")
        return

    files_in_dir = os.listdir(RAW_DATA_DIR)
    renamed_count = 0
    already_correct_count = 0

    print("🔄 Starting file renaming process...")

    for year, (original_pattern, new_name) in FILE_MAPPING.items():
        # Check if the standardized file already exists
        if new_name in files_in_dir:
            already_correct_count += 1
            continue

        # Find the original file that matches the pattern
        found_original = False
        for original_filename in files_in_dir:
            if original_pattern in original_filename and original_filename.endswith(('.xlsx', '.csv')):
                old_path = RAW_DATA_DIR / original_filename
                new_path = RAW_DATA_DIR / new_name
                
                try:
                    os.rename(old_path, new_path)
                    print(f"  ✅ Renamed: '{original_filename}' -> '{new_name}'")
                    renamed_count += 1
                    found_original = True
                    break 
                except OSError as e:
                    print(f"  ❌ Error renaming {original_filename}: {e}")
                    found_original = True # Mark as found to avoid "Not found" message
                    break
        
        # if not found_original:
        #     print(f"  ⚠️  Warning: No file found for year {year} (pattern: '{original_pattern}')")

    print("\n🎉 Renaming process complete!")
    print(f"  Total files renamed: {renamed_count}")
    print(f"  Files already correct: {already_correct_count}")
    
    total_standardized = renamed_count + already_correct_count
    total_expected = len(FILE_MAPPING)
    
    if total_standardized == total_expected:
        print(f"  ✨ Success! All {total_expected} expected files are now standardized.")
    else:
        print(f"  🔍 Status: {total_standardized}/{total_expected} files are standardized.")
        print("     Run 'python download_all_data.py' to see what's missing.")


if __name__ == "__main__":
    rename_files() 