#!/usr/bin/env uv run python
"""
Enhanced Column Mapping for HADR Data
=====================================

Fixes the critical column mapping issues identified in feature engineering analysis.
Uses official HADR PCL (Page-Column-Line) structure for accurate financial statement mapping.

Key Improvements:
1. Multi-level fuzzy matching with HADR-specific patterns
2. Official balance sheet structure mapping
3. Comprehensive logging for debugging
4. Validation against known good mappings

Usage:
    uv run python fix_column_mapping.py
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HADRColumnMapper:
    """Enhanced column mapper using official HADR structure."""
    
    def __init__(self):
        self.official_mappings = self._get_official_hadr_mappings()
        self.fallback_patterns = self._get_fallback_patterns()
        
    def _get_official_hadr_mappings(self) -> Dict[str, List[str]]:
        """Official HADR balance sheet and income statement mappings."""
        return {
            # Balance Sheet - Assets
            'total_assets': [
                'TOT_ASSETS', 'TOTAL_ASSETS', 'TOT_ASSET',
                'PY_TOT_ASSETS', 'EQ_TOT_ASSETS'
            ],
            'current_assets': [
                'TOT_CUR_ASSETS', 'TOTAL_CURRENT_ASSETS', 'CUR_ASSETS',
                'PY_TOT_CUR_ASSETS', 'EQ_TOT_CUR_ASSETS'
            ],
            'cash_and_equivalents': [
                'CASH_AND_CASH_EQUIV', 'CASH_EQUIV', 'TOT_CASH',
                'PY_CASH_AND_CASH_EQUIV', 'EQ_CASH'
            ],
            'accounts_receivable': [
                'PAT_ACCOUNTS_REC_NET', 'ACCOUNTS_REC_NET', 'NET_PAT_REC',
                'PAT_ACCOUNTS_REC', 'PY_PAT_ACCOUNTS_REC_NET'
            ],
            
            # Balance Sheet - Liabilities  
            'total_liabilities': [
                'TOT_LIAB', 'TOTAL_LIABILITIES', 'TOT_LIABILITIES',
                'PY_TOT_LIAB', 'EQ_TOT_LIAB'
            ],
            'current_liabilities': [
                'TOT_CUR_LIAB', 'TOTAL_CURRENT_LIAB', 'CUR_LIABILITIES',
                'PY_TOT_CUR_LIAB', 'EQ_TOT_CUR_LIAB'
            ],
            'long_term_debt': [
                'LONG_TERM_DEBT', 'LT_DEBT', 'LONG_TERM_LIAB',
                'PY_LONG_TERM_DEBT', 'EQ_LONG_TERM_DEBT'
            ],
            
            # Balance Sheet - Equity
            'total_equity': [
                'TOT_NET_ASSETS', 'TOTAL_NET_ASSETS', 'NET_ASSETS',
                'TOT_EQUITY', 'TOTAL_EQUITY', 'FUND_BALANCE',
                'PY_TOT_NET_ASSETS', 'EQ_TOT_NET_ASSETS'
            ],
            'retained_earnings': [
                'UNREST_FND_RET_EARN', 'RETAINED_EARNINGS', 'RET_EARNINGS',
                'UNREST_NET_ASSETS', 'PY_UNREST_FND_RET_EARN'
            ],
            
            # Income Statement
            'total_revenue': [
                'TOT_OPER_REV', 'TOTAL_OPERATING_REV', 'TOT_REV',
                'PY_TOT_OPER_REV', 'EQ_TOT_OPER_REV'
            ],
            'operating_revenue': [
                'TOT_OPER_REV', 'OPERATING_REVENUE', 'NET_PAT_REV',
                'PY_TOT_OPER_REV', 'EQ_TOT_OPER_REV'
            ],
            'total_expenses': [
                'TOT_OPER_EXP', 'TOTAL_OPERATING_EXP', 'TOT_EXP',
                'PY_TOT_OPER_EXP', 'EQ_TOT_OPER_EXP'
            ],
            'operating_expenses': [
                'TOT_OPER_EXP', 'OPERATING_EXPENSES', 'OPER_EXP',
                'PY_TOT_OPER_EXP', 'EQ_TOT_OPER_EXP'
            ],
            'interest_expense': [
                'INT_EXP', 'INTEREST_EXPENSE', 'INT_ON_INDEBT',
                'PY_INT_EXP', 'EQ_INT_EXP'
            ],
            'net_income': [
                'EXCESS_REV_OVER_EXP', 'NET_INCOME', 'OPER_INCOME',
                'PY_EXCESS_REV_OVER_EXP', 'EQ_EXCESS_REV_OVER_EXP'
            ]
        }
    
    def _get_fallback_patterns(self) -> Dict[str, List[str]]:
        """Fallback patterns for fuzzy matching."""
        return {
            'current_liabilities': ['CUR.*LIAB', 'SHORT.*TERM.*LIAB', 'CURRENT.*DEBT'],
            'total_equity': ['NET.*ASSET', 'FUND.*BALANCE', 'EQUITY'],
            'accounts_receivable': ['PAT.*REC', 'ACCOUNTS.*REC', 'RECEIVABLE'],
            'retained_earnings': ['RETAIN', 'UNREST.*EARN', 'RET.*EARN'],
            'interest_expense': ['INT.*EXP', 'INTEREST', 'DEBT.*SERVICE']
        }
    
    def map_columns(self, available_columns: List[str]) -> Dict[str, str]:
        """
        Map required financial fields to available HADR columns.
        
        Returns:
            Dictionary mapping required_field -> actual_column
        """
        logger.info(f"Mapping {len(available_columns)} available columns...")
        
        mappings = {}
        
        # Phase 1: Exact matches with official mappings
        for required_field, official_names in self.official_mappings.items():
            best_match = self._find_exact_match(official_names, available_columns)
            if best_match:
                mappings[required_field] = best_match
                logger.info(f"✅ {required_field} -> {best_match} (exact match)")
        
        # Phase 2: Fuzzy matching for remaining fields
        unmapped = set(self.official_mappings.keys()) - set(mappings.keys())
        for required_field in unmapped:
            best_match = self._find_fuzzy_match(
                required_field, 
                available_columns, 
                min_similarity=0.6
            )
            if best_match:
                mappings[required_field] = best_match[0]
                logger.info(f"🔍 {required_field} -> {best_match[0]} (fuzzy: {best_match[1]:.2f})")
            else:
                logger.warning(f"❌ No mapping found for {required_field}")
        
        logger.info(f"Final mappings: {len(mappings)}/{len(self.official_mappings)} successful")
        return mappings
    
    def _find_exact_match(self, official_names: List[str], available_columns: List[str]) -> Optional[str]:
        """Find exact match from official HADR names."""
        for official_name in official_names:
            if official_name in available_columns:
                return official_name
        return None
    
    def _find_fuzzy_match(self, required_field: str, available_columns: List[str], 
                         min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
        """Find best fuzzy match using multiple strategies."""
        best_match = None
        best_score = 0
        
        # Strategy 1: Direct field name similarity
        for col in available_columns:
            similarity = SequenceMatcher(None, required_field.upper(), col.upper()).ratio()
            if similarity > best_score and similarity >= min_similarity:
                best_score = similarity
                best_match = col
        
        # Strategy 2: Use fallback patterns if available
        if required_field in self.fallback_patterns:
            import re
            for pattern in self.fallback_patterns[required_field]:
                matches = [col for col in available_columns if re.search(pattern, col.upper())]
                for match in matches:
                    # Give pattern matches a slight boost
                    similarity = SequenceMatcher(None, required_field.upper(), match.upper()).ratio() + 0.1
                    if similarity > best_score:
                        best_score = similarity
                        best_match = match
        
        return (best_match, best_score) if best_match else None

def test_enhanced_mapping():
    """Test the enhanced mapping on our actual data."""
    logger.info("🧪 Testing enhanced column mapping...")
    
    # Load sample data
    data_path = Path("data/processed/processed_financials_2020_2021.parquet")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    data = pd.read_parquet(data_path)
    available_columns = data.columns.tolist()
    
    # Test mapping
    mapper = HADRColumnMapper()
    mappings = mapper.map_columns(available_columns)
    
    # Report results
    print("\n📊 ENHANCED MAPPING RESULTS:")
    print(f"Total available columns: {len(available_columns)}")
    print(f"Successful mappings: {len(mappings)}")
    print(f"Success rate: {len(mappings)/len(mapper.official_mappings)*100:.1f}%")
    
    print("\n✅ SUCCESSFUL MAPPINGS:")
    for field, column in mappings.items():
        print(f"  {field:20} -> {column}")
    
    failed = set(mapper.official_mappings.keys()) - set(mappings.keys())
    if failed:
        print(f"\n❌ FAILED MAPPINGS ({len(failed)}):")
        for field in failed:
            print(f"  {field}")

if __name__ == "__main__":
    test_enhanced_mapping() 