#!/usr/bin/env python3
"""
Hospital Financial Intelligence - EDA Execution Script

Professional business execution script for hospital financial analysis.
Docker-ready with configurable paths and environment variable support.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List
import json
import pandas as pd
from datetime import datetime

sys.path.append('src')
from src.eda import HospitalFinancialEDA
from src.config import get_config


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger('plotly').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="🏥 Hospital Financial Intelligence - Professional EDA Platform"
    )
    
    # Analysis parameters (defaults to full analysis)
    parser.add_argument('--years', help='Years to analyze (e.g., "2015-2023" or "2020,2021,2022") - overrides default full analysis')
    parser.add_argument('--single-year-only', action='store_true', help='Analyze only most recent year instead of all years')
    parser.add_argument('--dashboard-only', action='store_true', help='Generate dashboard only')
    parser.add_argument('--sample-size', type=int, help='Random sample size for large datasets')
    
    # Phase 3 Healthcare-Specific Analysis (enabled by default)
    parser.add_argument('--skip-phase3', action='store_true', 
                       help='Skip Phase 3 healthcare-specific analysis (legacy mode for faster execution)')
    
    # Path configuration (Docker-friendly)
    parser.add_argument('--base-dir', 
                       default=os.getenv('PROJECT_BASE_DIR', '.'),
                       help='Base project directory (default: current directory or PROJECT_BASE_DIR env var)')
    parser.add_argument('--data-dir', 
                       help='Data directory (overrides PROCESSED_DATA_DIR env var)')
    parser.add_argument('--output-dir', 
                       help='Output directory (overrides REPORTS_DIR env var)')
    
    # Execution options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    return parser.parse_args()


def parse_years(years_str: str) -> List[int]:
    """Parse years string into list."""
    if '-' in years_str:
        start, end = years_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in years_str:
        return [int(year.strip()) for year in years_str.split(',')]
    else:
        return [int(years_str)]


def validate_environment(config) -> bool:
    """Validate environment setup using config."""
    logger = logging.getLogger(__name__)
    
    is_valid, issues = config.validate_environment()
    
    if not is_valid:
        logger.error(f"❌ Environment validation failed:")
        for issue in issues:
            logger.error(f"   • {issue}")
        return False
    
    # Count data files
    data_files = list(config.processed_data_dir.glob(config.get_data_file_pattern()))
    logger.info(f"✅ Environment validated: {len(data_files)} data files found")
    return True


def print_header():
    """Print business header."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                🏥 HOSPITAL FINANCIAL INTELLIGENCE PLATFORM                   ║
║                    Professional Healthcare Analytics Suite                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_summary(results: dict):
    """Print analysis summary with Phase 3 enhancements."""
    phase3_summary = results.get('phase3_summary', {})
    phase3_enabled = results.get('phase3_enabled', False)
    
    base_summary = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ANALYSIS COMPLETED                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  📊 Records Analyzed: {results.get('records_analyzed', 0):,}                               │
│  📅 Years Covered:    {results.get('years_covered', 'N/A')}                                  │
│  🎯 Data Quality:     {results.get('data_quality_score', 0):.1f}%                                │
│  🏥 HADR Alignment:   {results.get('hadr_alignment_score', 0):.1f}%                                │"""
    
    if phase3_enabled and phase3_summary:
        phase3_section = f"""│  💰 Payer Fields:    {phase3_summary.get('payer_fields_found', 0)} found                                │
│  🌎 Counties:        {phase3_summary.get('market_counties_analyzed', 0)} analyzed                              │
│  ⭐ Quality Metrics: {phase3_summary.get('quality_indicators_found', 0)} indicators                           │"""
    else:
        phase3_section = "│  🚀 Phase 3:         Healthcare analysis included                       │"
    
    footer = f"""│  📁 Dashboard:        {results.get('outputs', {}).get('dashboard_file', 'N/A')} │
│  🚀 Status:           Analysis completed successfully                        │
└─────────────────────────────────────────────────────────────────────────────┘"""
    
    print(base_summary + "\n" + phase3_section + "\n" + footer)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup
    log_level = 'WARNING' if args.quiet else args.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        print_header()
    
    try:
        # Initialize configuration with base directory
        base_dir = Path(args.base_dir).resolve()
        config = get_config(base_dir)
        
        # Override config paths if provided via command line
        if args.data_dir:
            config.processed_data_dir = Path(args.data_dir).resolve()
        if args.output_dir:
            config.reports_dir = Path(args.output_dir).resolve()
            config._create_directories()  # Ensure new directories exist
        
        logger.info(f"📁 Configuration:")
        logger.info(f"   Base Directory: {config.base_dir}")
        logger.info(f"   Data Directory: {config.processed_data_dir}")
        logger.info(f"   Output Directory: {config.reports_dir}")
        
        # Validate environment
        if not validate_environment(config):
            sys.exit(1)
        
        # Initialize platform
        logger.info("🚀 Initializing Hospital Financial Intelligence Platform...")
        eda_platform = HospitalFinancialEDA(config=config)
        
        # Determine years for analysis (default: all available years)
        if args.years:
            # User specified specific years
            years = [str(y) for y in parse_years(args.years)]
        elif args.single_year_only:
            # User wants only most recent year
            years = ['2023']
        else:
            # Default: Find all available years from data files
            data_files = list(config.processed_data_dir.glob(config.get_data_file_pattern()))
            years = []
            for file in data_files:
                # Extract years from filename patterns like "processed_financials_2019_2020.parquet"
                parts = file.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 4 and part.startswith('20'):
                        years.append(part)
            years = sorted(list(set(years)))
            
            # Fallback if no data files found
            if not years:
                years = ['2020', '2021', '2022', '2023']
        
        logger.info(f"📅 Analysis scope: {len(years)} years ({', '.join(years)})")
        
        # Determine Phase 3 execution (enabled by default)
        phase3_enabled = not args.skip_phase3
        if phase3_enabled:
            logger.info("🏥 Phase 3 Healthcare-Specific Analysis: ENABLED (default)")
        else:
            logger.info("⚡ Phase 3 Healthcare-Specific Analysis: SKIPPED (legacy mode)")

        # Run analysis
        all_results = []
        phase3_aggregate = {'payer_fields_found': 0, 'market_counties_analyzed': 0, 'quality_indicators_found': 0}
        
        for year in years:
            logger.info(f"🔍 Analyzing year: {year}")
            try:
                result = eda_platform.run_single_year_analysis(year)
                if result:
                    all_results.append(result)
                    
                    # Aggregate Phase 3 results if available
                    if phase3_enabled and 'phase3_healthcare_analysis' in result:
                        phase3_data = result['phase3_healthcare_analysis'].get('phase3_summary', {})
                        phase3_aggregate['payer_fields_found'] = max(
                            phase3_aggregate['payer_fields_found'], 
                            phase3_data.get('payer_fields_found', 0)
                        )
                        phase3_aggregate['market_counties_analyzed'] = max(
                            phase3_aggregate['market_counties_analyzed'], 
                            phase3_data.get('market_counties_analyzed', 0)
                        )
                        phase3_aggregate['quality_indicators_found'] = max(
                            phase3_aggregate['quality_indicators_found'], 
                            phase3_data.get('quality_indicators_found', 0)
                        )
                    
            except Exception as e:
                logger.error(f"❌ Failed to analyze year {year}: {e}")
                continue

        if all_results:
            # Calculate combined results
            total_records = sum(r.get('records_analyzed', 0) for r in all_results)
            avg_quality = sum(r.get('data_quality_score', 0) for r in all_results) / len(all_results)
            avg_hadr_alignment = sum(r.get('hadr_alignment_score', 0) for r in all_results) / len(all_results)
            
            summary_results = {
                'records_analyzed': total_records,
                'years_covered': f"{years[0]}-{years[-1]}" if len(years) > 1 else years[0],
                'data_quality_score': avg_quality,
                'hadr_alignment_score': avg_hadr_alignment,
                'phase3_enabled': phase3_enabled,
                'phase3_summary': phase3_aggregate if phase3_enabled else {},
                'outputs': {
                    'dashboard_file': f"Multiple dashboards generated in {config.reports_dir}"
                }
            }
            
            if not args.quiet:
                print_summary(summary_results)
            
            logger.info("✅ All years analyzed successfully.")
        else:
            logger.error("❌ No years were successfully analyzed.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Analysis interrupted")
        return 130
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 