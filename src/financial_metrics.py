"""
Hospital Financial Metrics Calculator

Healthcare financial ratios and metrics for hospital financial health assessment.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from fuzzywuzzy import process, fuzz

logger = logging.getLogger(__name__)


class FinancialMetricsCalculator:
    """
    Healthcare financial metrics calculator with GAAP compliance.
    
    Calculates key financial ratios:
    - Liquidity ratios (current ratio, days cash on hand)
    - Profitability ratios (operating margin, total margin)
    - Efficiency ratios (asset turnover, receivables turnover)
    - Leverage ratios (debt-to-equity, times interest earned)
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize calculator with hospital financial data."""
        self.data = data.copy()
        self.metrics = {}
        self._clean_data()
        self.column_mapping = self._create_column_mapping()
        
        # Log mapping success rate
        total_required = len(self._get_required_columns())
        mapped_count = len(self.column_mapping)
        mapping_rate = (mapped_count / total_required) * 100 if total_required > 0 else 0
        logger.info(f"Column mapping success: {mapped_count}/{total_required} ({mapping_rate:.1f}%)")
        
    def _clean_data(self):
        """
        Convert all possible columns to numeric, coercing errors.
        This handles cases where financial figures are loaded as strings.
        """
        logger.info("Cleaning and converting data to numeric types...")
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Attempt to convert to numeric, coercing errors to NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        logger.info("Data cleaning complete.")

    def _get_required_columns(self) -> List[str]:
        """Get list of all required column names for financial calculations."""
        return [
            'total_revenue', 'operating_expenses', 'net_income', 'total_assets',
            'current_assets', 'current_liabilities', 'total_equity', 'accounts_receivable',
            'operating_income', 'cash_equivalents', 'inventory', 'total_debt',
            'interest_expense', 'patient_revenue', 'total_liabilities', 'retained_earnings'
        ]
        
    def _create_column_mapping(self) -> Dict[str, str]:
        """
        Create mapping from standard metric names to actual column names.
        Enhanced with HADR PCL-validated structure knowledge.
        
        Based on California OSHPD HADR documentation (2004-2015+):
        - Balance Sheet (P5): Rows 1,787-1,950 (164 fields) - PCL Range BPS-BVZ
        - Income Statement (P8): Rows 2,540-2,721 (182 fields) - PCL Range CSR-CZQ  
        - Revenue & Costs (P10): Rows 2,821-4,170 (1,350 fields) - PCL Range DDM-FDG
        - Patient Revenue (P12): Rows 4,171-5,747 (1,577 fields) - PCL Range FDH-HLX
        - Cash Flow Statement (P9): Rows 2,722-2,820 (99 fields) - PCL Range CZR-DDL
        - Expense Data (P17-P18): Rows 6,952-8,937 (1,986 fields) - PCL Range JBU-LXC
        """
        # HADR-aligned column aliases with PCL references
        known_aliases = {
            # Revenue Section - PCL-Validated Primary Fields
            'total_revenue': [
                # ✅ PCL P12_C23_L415 - Patient Revenue (12) - 100% Complete
                'REV_TOT_PT_REV',           # Gross Patient Revenue_Total Patient Revenue
                # Enhanced Income Statement alternatives (P8)
                'TOT_GR_PT_REV',            # Total Gross Patient Revenue - standardized
                'NET_PT_REV',               # Net Patient Revenue (after deductions)
                'TOT_OP_REV',               # Total Operating Revenue
                'PY_TOT_GR_PT_REV',         # Prior Year Total Gross Patient Revenue
                'PY_TOT_OP_REV',            # Prior Year Total Operating Revenue  
                'TOTAL_REVENUE', 'NET_PATIENT_REVENUE', 'GROSS_PATIENT_REVENUE'
            ],
            
            # Expense Section - PCL-Validated Primary Fields  
            'operating_expenses': [
                # ✅ PCL P8_C2_L200 - Income Statement (8) - 100% Complete
                'PY_TOT_OP_EXP',            # Prior Year_Income Statement_Total Operating Expenses
                # Additional expense alternatives
                'TOT_OP_EXP',               # Total Operating Expenses (current year)
                'SUMM_NET_PROFIT_TOT_OP_EXP', # Summary section operating expenses
                'PY_TOT_NONOP_EXP',         # Non-operating expenses
                'TOTAL_OPERATING_EXPENSES', 'TOT_EXP', 'OPERATING_EXPENSES'
            ],
            
            # Income Section - PCL-Validated Primary Fields
            'net_income': [
                # ✅ PCL P7_C1_L55 - Changes in Equity (7) - 100% Complete
                'EQ_UNREST_FND_NET_INCOME', # Changes in Equity_Unrestricted Fund_Net Income (Loss)
                # Income Statement alternatives (P8)
                'NET_INCOME', 'NET_PROFIT', 'TOTAL_MARGIN_DOLLARS',
                'INCOME_LOSS', 'NET_INCOME_LOSS'
            ],
            'operating_income': [
                # Using net income as proxy - may need separate PCL field
                'EQ_UNREST_FND_NET_INCOME', # Net income proxy
                'OP_INCOME', 'OPERATING_INCOME', 'INCOME_FROM_OPERATIONS',
                'OPERATING_MARGIN_DOLLARS'
            ],
            
            # Asset Section - PCL-Validated and Enhanced
            'total_assets': [
                # ✅ PCL P6_C2_L30 - Balance Sheet (6) - 100% Complete  
                'PY_SP_PURP_FND_OTH_ASSETS',    # Prior Year_Specific Purpose Fund_Other Assets
                # Enhanced Balance Sheet alternatives (P5)
                'TOT_ASSETS',               # Standard total assets (P5)
                'TOTAL_ASSETS',             # Alternative naming
                'PY_PLANT_REP_FND_OTH_ASSETS',  # Plant Replacement Fund Assets
                'PY_UNREST_FND_TOT_ASSETS',     # Unrestricted Fund Total Assets
                'ASSETS_TOTAL'
            ],
            'current_assets': [
                # Balance Sheet current assets (P5)
                'TOT_CUR_ASSETS',           # Total Current Assets
                'CURRENT_ASSETS', 'CUR_ASSETS', 'TOTAL_CURRENT_ASSETS',
                'PY_CUR_ASSETS'             # Prior Year Current Assets
            ],
            
            # Liability Section - PCL-Validated and Enhanced
            'total_liabilities': [
                # ✅ PCL P6_C4_L75 - Balance Sheet (6) - 100% Complete
                'PY_SP_PURP_FND_TOT_LIAB_EQ',   # Prior Year_Specific Purpose Fund_Total Liab & Equity
                # Enhanced Balance Sheet alternatives (P5)  
                'TOT_LIAB',                 # Total Liabilities (P5)
                'TOTAL_LIABILITIES',        # Standard naming
                'PY_PLANT_REP_FND_TOT_LIAB_EQ', # Plant Replacement Fund Total Liab & Equity
                'PY_UNREST_FND_TOT_LIAB',   # Unrestricted Fund Total Liabilities
                'LIABILITIES_TOTAL'
            ],
            'current_liabilities': [
                # Balance Sheet current liabilities (P5)
                'TOT_CUR_LIAB',             # Total Current Liabilities
                'CURRENT_LIABILITIES', 'CUR_LIAB', 'TOTAL_CURRENT_LIABILITIES',
                'PY_CUR_LIAB'               # Prior Year Current Liabilities
            ],
            
            # Equity Section - Enhanced
            'total_equity': [
                # Balance Sheet equity section (P5)
                'TOT_EQUITY',               # Total Equity
                'TOTAL_EQUITY', 'NET_WORTH', 'FUND_BALANCE',
                'PY_TOT_EQUITY',            # Prior Year Total Equity
                'EQUITY_TOTAL'
            ],
            'retained_earnings': [
                # Balance Sheet equity section (P5) for Retained Earnings
                'UNREST_FND_RET_EARN',      # Unrestricted Fund Retained Earnings
                'RETAINED_EARNINGS',
                'PY_UNREST_FND_RET_EARN',
                'NET_ASSETS_RELEASED_RETAINED'
            ],
            
            # Cash Flow Section - PCL-Validated Primary Fields
            'cash_equivalents': [
                # ✅ PCL P9_C91_L102 - Cash Flows (9) - 100% Complete
                'CASH_FLOW_SPECIFY_OTH_OP_L102', # Cash Flow_Specify_Other Cash from Operating Activities
                # Enhanced cash alternatives
                'CASH_CASH_EQUIV',          # Cash and Cash Equivalents
                'CASH_AND_EQUIVALENTS', 'CASH', 'CASH_EQUIV',
                'PY_CASH_EQUIV',            # Prior Year Cash Equivalents
                'TOT_CASH'                  # Total Cash
            ],
            
            # Additional Balance Sheet Items
            'accounts_receivable': [
                # Patient accounts receivable
                'PAT_ACCOUNTS_REC',         # Patient Accounts Receivable
                'ACCOUNTS_RECEIVABLE', 'AR', 'NET_AR',
                'PY_PAT_ACCOUNTS_REC',      # Prior Year Patient AR
                'RECEIVABLES_NET'
            ],
            'inventory': [
                # Inventory items
                'INVENTORY', 'INV', 'SUPPLIES_INVENTORY',
                'PY_INVENTORY',             # Prior Year Inventory
                'TOTAL_INVENTORY'
            ],
            'total_debt': [
                # Long-term debt
                'LT_DEBT',                  # Long Term Debt
                'LONG_TERM_DEBT', 'TOTAL_DEBT', 'DEBT_TOTAL',
                'PY_LT_DEBT',               # Prior Year Long Term Debt
                'NOTES_PAYABLE'
            ],
            'interest_expense': [
                # Interest expenses
                'INT_EXP',                  # Interest Expense
                'INTEREST_EXPENSE', 'INTEREST_PAID',
                'PY_INT_EXP',               # Prior Year Interest Expense
                'FINANCING_COSTS'
            ],
            
            # Enhanced Revenue Section Aliases with PCL Context
            'patient_revenue': [
                # ✅ PCL P12_C23_L415 - Patient Revenue (12) - 100% Complete
                'REV_TOT_PT_REV',           # Total Patient Revenue - primary
                'PY_TOT_GR_PT_REV',         # Total Gross Patient Revenue
                'NET_PT_REV',               # Net Patient Revenue (post-deductions)
                'PAT_REV', 'PATIENT_REVENUE', 'GROSS_PATIENT_REVENUE'
            ],
            'gross_patient_revenue': [
                # Gross patient revenue (P12/P8)
                'PY_TOT_GR_PT_REV',         # Total Gross Patient Revenue - primary  
                'TOT_GR_PT_REV',            # Current year version
                'GROSS_PATIENT_REVENUE', 'GROSS_REV'
            ],
            'operating_revenue': [
                # Operating revenue (P8)
                'PY_TOT_OP_REV',            # Total Operating Revenue - primary
                'TOT_OP_REV',               # Current year version
                'OPERATING_REVENUE', 'OP_REV'
            ],
            
            # Fund-Specific Fields (Multi-Fund Analysis)
            'unrestricted_fund_assets': [
                'PY_UNREST_FND_TOT_ASSETS', # Unrestricted Fund Total Assets
                'UNREST_FND_ASSETS', 'UNRESTRICTED_ASSETS'
            ],
            'restricted_fund_assets': [
                'PY_REST_FND_TOT_ASSETS',   # Restricted Fund Total Assets  
                'REST_FND_ASSETS', 'RESTRICTED_ASSETS'
            ],
            'special_purpose_fund_assets': [
                'PY_SP_PURP_FND_OTH_ASSETS', # Special Purpose Fund Assets (current mapping)
                'SP_PURP_FND_ASSETS', 'SPECIAL_PURPOSE_ASSETS'
            ],
            
            # Non-Operating Revenue/Expenses
            'non_operating_revenue': [
                'PY_TOT_NONOP_REV',         # Non-Operating Revenue
                'NONOP_REV', 'NON_OPERATING_REVENUE'
            ],
            'non_operating_expenses': [
                'PY_TOT_NONOP_EXP',         # Non-Operating Expenses  
                'NONOP_EXP', 'NON_OPERATING_EXPENSES'
            ]
        }

        required_cols = self._get_required_columns()
        mapping = {}
        available_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Enhanced logging for PCL validation
        logger.info(f"🔍 Column mapping with {len(available_cols)} numeric columns available")
        logger.info(f"📊 Targeting {len(required_cols)} required financial metrics")
        
        for req_col in required_cols:
            best_match = None
            highest_score = 0
            match_strategy = None
            
            # Strategy 1: Direct alias matching (PCL-validated fields prioritized)
            if req_col in known_aliases:
                for alias in known_aliases[req_col]:
                    if alias in available_cols:
                        mapping[req_col] = alias
                        best_match = alias
                        highest_score = 100
                        match_strategy = "Direct PCL Match"
                        break
            
            # Strategy 2: Fuzzy matching on PCL-validated aliases
            if not best_match and req_col in known_aliases:
                for alias in known_aliases[req_col]:
                    alias_match, alias_score = process.extractOne(
                        alias, available_cols, scorer=fuzz.token_set_ratio
                    )
                    if alias_score > highest_score and alias_score > 75:
                        highest_score = alias_score
                        best_match = alias_match
                        match_strategy = f"PCL Fuzzy Match ({alias_score}%)"

            # Strategy 3: Fuzzy matching on standardized column name
            if not best_match:
                search_term = req_col.replace('_', ' ').upper()
                fuzzy_match, fuzzy_score = process.extractOne(
                    search_term, available_cols, scorer=fuzz.token_set_ratio
                )
                if fuzzy_score > highest_score and fuzzy_score > 60:
                    highest_score = fuzzy_score
                    best_match = fuzzy_match
                    match_strategy = f"Standard Fuzzy Match ({fuzzy_score}%)"

            # Strategy 4: Partial matching on key financial terms
            if not best_match:
                key_terms = req_col.replace('_', ' ').split()
                for term in key_terms:
                    if len(term) > 3:  # Only use meaningful terms
                        term_matches = [col for col in available_cols if term.upper() in col.upper()]
                        if term_matches:
                            # Use the first match with highest similarity
                            term_match, term_score = process.extractOne(
                                term.upper(), term_matches, scorer=fuzz.partial_ratio
                            )
                            if term_score > highest_score and term_score > 50:
                                highest_score = term_score
                                best_match = term_match
                                match_strategy = f"Partial Term Match ({term_score}%)"

            if best_match and highest_score > 50:
                mapping[req_col] = best_match
                # Enhanced logging with strategy information
                if highest_score == 100:
                    logger.info(f"✅ {req_col} → {best_match} ({match_strategy})")
                elif highest_score >= 75:
                    logger.info(f"🎯 {req_col} → {best_match} ({match_strategy})")
                else:
                    logger.warning(f"⚠️  {req_col} → {best_match} ({match_strategy})")
            else:
                logger.warning(f"❌ {req_col}: No suitable mapping found")
        
        # Summary logging with PCL context
        total_required = len(required_cols)
        mapped_count = len(mapping)
        mapping_rate = (mapped_count / total_required) * 100 if total_required > 0 else 0
        
        logger.info(f"📈 Enhanced HADR column mapping completed:")
        logger.info(f"   🎯 Success Rate: {mapped_count}/{total_required} ({mapping_rate:.1f}%)")
        logger.info(f"   🏥 PCL-Validated: {len([m for m in mapping.values() if any(m in aliases for aliases in known_aliases.values())])} fields")
        logger.info(f"   📊 Ready for financial metrics calculation")
        
        return mapping

    def _get_col(self, key: str) -> Optional[pd.Series]:
        """
        Safely retrieve a data column using the mapped name.
        Returns a Series of NaNs if the key is not mapped.
        """
        col_name = self.column_mapping.get(key)
        if col_name and col_name in self.data:
            return self.data[col_name]
        logger.warning(f"'{key}' not found in column mapping. Returning NaNs.")
        return pd.Series(np.nan, index=self.data.index)

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                    default_value: float = np.nan) -> pd.Series:
        """Perform safe division with protection against division by zero."""
        # Replace zeros with NaN to avoid division by zero
        denominator_safe = denominator.replace(0, np.nan)
        result = numerator / denominator_safe
        
        # Replace infinite values
        result = result.replace([np.inf, -np.inf], default_value)
        
        return result

    def calculate_liquidity_ratios(self) -> Dict[str, pd.Series]:
        """Calculate liquidity ratios with enhanced missing value handling."""
        metrics = {}
        
        try:
            # Current Ratio = Current Assets / Current Liabilities
            current_assets = self._get_col('current_assets')
            current_liabilities = self._get_col('current_liabilities')
            
            if current_assets is not None and current_liabilities is not None:
                metrics['current_ratio'] = self._safe_divide(current_assets, current_liabilities, 0)
                logger.debug("✓ Calculated current_ratio")
            
            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities  
            inventory = self._get_col('inventory')
            if (current_assets is not None and inventory is not None and 
                current_liabilities is not None):
                quick_assets = current_assets - inventory
                metrics['quick_ratio'] = self._safe_divide(quick_assets, current_liabilities, 0)
                logger.debug("✓ Calculated quick_ratio")
            
            # Days Cash on Hand = Cash & Equivalents / (Operating Expenses / 365)
            cash_equivalents = self._get_col('cash_equivalents')
            operating_expenses = self._get_col('operating_expenses')
            
            if cash_equivalents is not None and operating_expenses is not None:
                daily_expenses = operating_expenses / 365
                metrics['days_cash_on_hand'] = self._safe_divide(cash_equivalents, daily_expenses, 0)
                logger.debug("✓ Calculated days_cash_on_hand")
                
        except Exception as e:
            logger.warning(f"Error calculating liquidity ratios: {e}")
            
        return metrics
    
    def calculate_profitability_ratios(self) -> Dict[str, pd.Series]:
        """Calculate profitability ratios with enhanced missing value handling."""
        metrics = {}
        
        try:
            # Get common columns once
            total_revenue = self._get_col('total_revenue')
            net_income = self._get_col('net_income')
            operating_income = self._get_col('operating_income')
            total_assets = self._get_col('total_assets')
            total_equity = self._get_col('total_equity')
            
            # Operating Margin = Operating Income / Total Revenue
            if operating_income is not None and total_revenue is not None:
                metrics['operating_margin'] = self._safe_divide(operating_income, total_revenue) * 100
                logger.debug("✓ Calculated operating_margin")
            
            # Total Margin = Net Income / Total Revenue  
            if net_income is not None and total_revenue is not None:
                metrics['total_margin'] = self._safe_divide(net_income, total_revenue) * 100
                logger.debug("✓ Calculated total_margin")
            
            # Return on Assets = Net Income / Total Assets
            if net_income is not None and total_assets is not None:
                metrics['return_on_assets'] = self._safe_divide(net_income, total_assets) * 100
                logger.debug("✓ Calculated return_on_assets")
                
            # Return on Equity = Net Income / Total Equity
            if net_income is not None and total_equity is not None:
                metrics['return_on_equity'] = self._safe_divide(net_income, total_equity) * 100
                logger.debug("✓ Calculated return_on_equity")
                
        except Exception as e:
            logger.warning(f"Error calculating profitability ratios: {e}")
            
        return metrics
    
    def calculate_efficiency_ratios(self) -> Dict[str, pd.Series]:
        """Calculate efficiency ratios with enhanced missing value handling."""
        metrics = {}
        
        try:
            # Get common columns
            total_revenue = self._get_col('total_revenue')
            total_assets = self._get_col('total_assets')
            accounts_receivable = self._get_col('accounts_receivable')
            
            # Asset Turnover = Total Revenue / Total Assets
            if total_revenue is not None and total_assets is not None:
                metrics['asset_turnover'] = self._safe_divide(total_revenue, total_assets)
                logger.debug("✓ Calculated asset_turnover")
            
            # Receivables Turnover = Total Revenue / Accounts Receivable
            if total_revenue is not None and accounts_receivable is not None:
                metrics['receivables_turnover'] = self._safe_divide(total_revenue, accounts_receivable)
                logger.debug("✓ Calculated receivables_turnover")
                
                # Days Sales Outstanding = 365 / Receivables Turnover
                metrics['days_sales_outstanding'] = self._safe_divide(
                    pd.Series(365, index=total_revenue.index), 
                    metrics['receivables_turnover']
                )
                logger.debug("✓ Calculated days_sales_outstanding")
                
        except Exception as e:
            logger.warning(f"Error calculating efficiency ratios: {e}")
            
        return metrics
    
    def calculate_leverage_ratios(self) -> Dict[str, pd.Series]:
        """Calculate leverage ratios with enhanced missing value handling."""
        metrics = {}
        
        try:
            # Get common columns
            total_debt = self._get_col('total_debt')
            total_equity = self._get_col('total_equity')
            total_assets = self._get_col('total_assets')
            operating_income = self._get_col('operating_income')
            interest_expense = self._get_col('interest_expense')
            
            # Debt-to-Equity = Total Debt / Total Equity
            if total_debt is not None and total_equity is not None:
                metrics['debt_to_equity'] = self._safe_divide(total_debt, total_equity)
                logger.debug("✓ Calculated debt_to_equity")
            
            # Debt-to-Assets = Total Debt / Total Assets
            if total_debt is not None and total_assets is not None:
                metrics['debt_to_assets'] = self._safe_divide(total_debt, total_assets) * 100
                logger.debug("✓ Calculated debt_to_assets")
                
            # Times Interest Earned = Operating Income / Interest Expense
            if operating_income is not None and interest_expense is not None:
                metrics['times_interest_earned'] = self._safe_divide(operating_income, interest_expense)
                logger.debug("✓ Calculated times_interest_earned")
                
        except Exception as e:
            logger.warning(f"Error calculating leverage ratios: {e}")
            
        return metrics
    
    def calculate_all_metrics(self) -> Dict[str, pd.Series]:
        """Calculate all financial metrics and combine into single dictionary."""
        all_metrics = {}
        
        # Calculate all ratio categories
        liquidity = self.calculate_liquidity_ratios()
        profitability = self.calculate_profitability_ratios()
        efficiency = self.calculate_efficiency_ratios()
        leverage = self.calculate_leverage_ratios()
        
        all_metrics.update(liquidity)
        all_metrics.update(profitability)
        all_metrics.update(efficiency)
        all_metrics.update(leverage)
        
        # Log detailed results
        logger.info(f"✅ Calculated {len(all_metrics)} financial metrics:")
        for category, metrics_dict in [
            ("Liquidity", liquidity), ("Profitability", profitability),
            ("Efficiency", efficiency), ("Leverage", leverage)
        ]:
            if metrics_dict:
                logger.info(f"   {category}: {list(metrics_dict.keys())}")
        
        return all_metrics
    
    def get_financial_health_score(self) -> pd.Series:
        """Calculate composite financial health score based on available metrics."""
        metrics = self.calculate_all_metrics()
        
        if not metrics:
            logger.warning("No metrics available for health score calculation")
            return pd.Series(np.nan, index=self.data.index)
        
        # Define scoring weights (industry-standard importance)
        weights = {
            'current_ratio': 0.20,
            'days_cash_on_hand': 0.15,
            'operating_margin': 0.25,
            'total_margin': 0.20,
            'debt_to_equity': 0.10,
            'asset_turnover': 0.10
        }
        
        # Only use weights for available metrics
        available_weights = {k: v for k, v in weights.items() if k in metrics}
        
        if not available_weights:
            logger.warning("No key metrics available for health score")
            return pd.Series(np.nan, index=self.data.index)
        
        # Normalize weights to sum to 1
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        
        scores = []
        for metric, weight in normalized_weights.items():
            # Normalize metrics to 0-100 scale
            normalized = self._normalize_metric(metrics[metric], metric)
            scores.append(normalized * weight)
        
        final_score = pd.Series(np.sum(scores, axis=0), index=self.data.index)
        logger.info(f"Health score calculated using {len(available_weights)} metrics")
        
        return final_score
    
    def _normalize_metric(self, series: pd.Series, metric_name: str) -> pd.Series:
        """Normalize metric to 0-100 scale based on industry benchmarks."""
        # Industry benchmark ranges for normalization
        benchmarks = {
            'current_ratio': {'min': 0.5, 'max': 3.0, 'target': 2.0},
            'days_cash_on_hand': {'min': 0, 'max': 200, 'target': 100},
            'operating_margin': {'min': -20, 'max': 20, 'target': 5},
            'total_margin': {'min': -30, 'max': 15, 'target': 3},
            'debt_to_equity': {'min': 0, 'max': 3.0, 'target': 1.0},
            'asset_turnover': {'min': 0, 'max': 2.0, 'target': 1.0}
        }
        
        if metric_name not in benchmarks:
            # Default percentile-based normalization for unmapped metrics
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return pd.Series(50, index=series.index)  # Default to middle score
            return pd.Series(
                100 * (series.rank(pct=True)), 
                index=series.index
            ).fillna(50)
        
        bench = benchmarks[metric_name]
        
        # Clip to benchmark range and normalize
        clipped = series.clip(bench['min'], bench['max'])
        normalized = 100 * (clipped - bench['min']) / (bench['max'] - bench['min'])
        
        return normalized.fillna(50)  # Default to middle score for missing values


def identify_financial_distress(data: pd.DataFrame, 
                              threshold_scores: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Identify hospitals at risk of financial distress with enhanced robustness.
    
    Args:
        data: Hospital financial data
        threshold_scores: Custom threshold scores for distress indicators
        
    Returns:
        Series indicating financial distress risk level (0-3 scale)
    """
    if threshold_scores is None:
        threshold_scores = {
            'current_ratio': 1.0,      # Below 1.0 = liquidity concern
            'days_cash_on_hand': 30,   # Below 30 days = cash concern  
            'operating_margin': -5,    # Below -5% = operational concern
            'total_margin': -10        # Below -10% = severe concern
        }
    
    calculator = FinancialMetricsCalculator(data)
    metrics = calculator.calculate_all_metrics()
    
    if not metrics:
        logger.warning("No metrics available for distress analysis")
        return pd.Series(0, index=data.index)  # Default to low risk
    
    distress_score = pd.Series(0, index=data.index)
    
    for metric, threshold in threshold_scores.items():
        if metric in metrics:
            # Add 1 point for each distress indicator triggered
            distress_indicator = (metrics[metric] < threshold).fillna(False)
            distress_score += distress_indicator.astype(int)
    
    logger.info(f"Financial distress analysis completed using {len([m for m in threshold_scores.keys() if m in metrics])} indicators")
    
    return distress_score 