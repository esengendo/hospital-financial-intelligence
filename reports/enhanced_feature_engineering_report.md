# Enhanced Feature Engineering Report
Generated: 2025-06-27 15:02:38

## Summary
- **Original Features**: 33
- **Enhanced Features**: 147
- **New Features**: 114
- **Years Covered**: 2003 - 2023
- **Total Records**: 9,475

## Phase 4 Feature Engineering Validation ✅
- **Altman Z-Score Components**: 5/5 (✅ Complete)
- **Healthcare-Specific Ratios**: 0 implemented
- **Time-Series Features**: 61 unique features
- **Volatility Measures**: 36 unique features
- **Growth Rate Indicators**: 13 unique features

### Altman Z-Score Components Found:
- ✅ `z_working_capital_ratio`
- ✅ `z_retained_earnings_ratio`
- ✅ `z_ebit_ratio`
- ✅ `z_equity_to_liability_ratio`
- ✅ `z_sales_to_assets_ratio`

## New Feature Categories
### Rolling Averages (16 features)
- `operating_margin_rolling_2y`: 19.1% coverage
- `operating_margin_rolling_3y`: 19.1% coverage
- `total_margin_rolling_2y`: 19.1% coverage
- ... and 13 more

### Volatility Measures (32 features)
- `operating_margin_volatility_3y`: 19.1% coverage
- `operating_margin_cv_3y`: 19.1% coverage
- `operating_margin_volatility_5y`: 19.2% coverage
- ... and 29 more

### Trend Analysis (24 features)
- `operating_margin_trend_3y`: 15.2% coverage
- `operating_margin_trend_direction`: 100.0% coverage
- `operating_margin_trend_consistency`: 15.2% coverage
- ... and 21 more

### Momentum Indicators (8 features)
- `operating_margin_momentum_direction`: 15.2% coverage
- `total_margin_momentum_direction`: 15.2% coverage
- `times_interest_earned_momentum_direction`: 11.6% coverage
- ... and 5 more

### Stability Scores (4 features)
- `operating_margin_stability_score`: 100.0% coverage
- `times_interest_earned_stability_score`: 100.0% coverage
- `days_cash_on_hand_stability_score`: 100.0% coverage
- ... and 1 more

### Industry Percentiles (3 features)
- `operating_margin_bottom_10pct`: 100.0% coverage
- `times_interest_earned_bottom_10pct`: 100.0% coverage
- `days_cash_on_hand_bottom_10pct`: 100.0% coverage

### Deviations (16 features)
- `operating_margin_dev_from_2y_avg`: 19.1% coverage
- `operating_margin_dev_from_3y_avg`: 19.1% coverage
- `total_margin_dev_from_2y_avg`: 19.1% coverage
- ... and 13 more

## Data Coverage Analysis
Top 10 new features by data coverage:

- `operating_margin_trend_direction`: 100.0%
- `total_margin_trend_direction`: 100.0%
- `times_interest_earned_trend_direction`: 100.0%
- `current_ratio_trend_direction`: 100.0%
- `days_cash_on_hand_trend_direction`: 100.0%
- `return_on_assets_trend_direction`: 100.0%
- `debt_to_assets_trend_direction`: 100.0%
- `asset_turnover_trend_direction`: 100.0%
- `operating_margin_stability_score`: 100.0%
- `times_interest_earned_stability_score`: 100.0%