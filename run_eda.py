#!/usr/bin/env python3
"""
Hospital Financial Intelligence - EDA Execution Script

Professional EDA execution for hospital financial analysis.
Docker-ready with configurable paths and environment support.
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
    
    # Analysis parameters
    parser.add_argument('--years', help='Years to analyze (e.g., "2015-2023" or "2020,2021,2022")')
    parser.add_argument('--single-year-only', action='store_true', help='Analyze only most recent year')
    parser.add_argument('--dashboard-only', action='store_true', help='Generate dashboard only')
    parser.add_argument('--sample-size', type=int, help='Random sample size for large datasets')
    
    # Healthcare analysis
    parser.add_argument('--skip-phase3', action='store_true', 
                       help='Skip Phase 3 healthcare-specific analysis')
    
    # Path configuration
    parser.add_argument('--base-dir', 
                       default=os.getenv('PROJECT_BASE_DIR', '.'),
                       help='Base project directory')
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
    """Validate environment setup."""
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
    """Print analysis summary."""
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
        # Initialize configuration
        base_dir = Path(args.base_dir).resolve()
        config = get_config(base_dir)
        
        # Override config paths if provided
        if args.data_dir:
            config.processed_data_dir = Path(args.data_dir).resolve()
        if args.output_dir:
            config.reports_dir = Path(args.output_dir).resolve()
            config._create_directories()
        
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
        
        # Determine years for analysis
        if args.years:
            years = [str(y) for y in parse_years(args.years)]
        elif args.single_year_only:
            years = ['2023']
        else:
            # Find all available years from data files
            data_files = list(config.processed_data_dir.glob(config.get_data_file_pattern()))
            years = []
            for file in data_files:
                # Extract years from filename patterns
                parts = file.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 4 and part.startswith('20'):
                        years.append(part)
            years = sorted(list(set(years)))
            
            # Fallback if no data files found
            if not years:
                years = ['2020', '2021', '2022', '2023']
        
        logger.info(f"📅 Analysis scope: {len(years)} years ({', '.join(years)})")
        
        # Determine Phase 3 execution
        phase3_enabled = not args.skip_phase3
        if phase3_enabled:
            logger.info("🏥 Phase 3 Healthcare-Specific Analysis: ENABLED")
        else:
            logger.info("⚡ Phase 3 Healthcare-Specific Analysis: SKIPPED")
        
        # Execute analysis
        if args.dashboard_only:
            logger.info("📊 Dashboard-only mode: Generating financial dashboard...")
            results = eda_platform.generate_dashboard_only(years=years)
        else:
            logger.info("🔍 Full Analysis Mode: Complete EDA pipeline...")
            results = eda_platform.run_comprehensive_analysis(
                years=years,
                sample_size=args.sample_size,
                phase3_enabled=phase3_enabled
            )
        
        # Print summary
        if not args.quiet:
            print_summary(results)
        
        logger.info("✅ EDA execution completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ EDA execution failed: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 