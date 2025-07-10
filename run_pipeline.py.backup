#!/usr/bin/env python3
"""
🏥 Hospital Financial Intelligence - Simple Pipeline Runner
==========================================================
Easy-to-use commands for running the healthcare analytics pipeline.

Quick Commands:
    python run_pipeline.py                    # Default: EDA → Features → Modeling → Dashboard
    python run_pipeline.py --full             # Complete pipeline from data processing
    python run_pipeline.py --dashboard        # Launch dashboard only
    python run_pipeline.py --quick            # Quick analysis with sample data
    python run_pipeline.py --modeling-only    # Retrain models only
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd_args, description):
    """Run a command with description."""
    print(f"\n🚀 {description}")
    print(f"Command: python main.py {' '.join(cmd_args)}")
    print("="*60)
    
    result = subprocess.run([sys.executable, "main.py"] + cmd_args)
    return result.returncode == 0

def main():
    """Simple pipeline runner with predefined workflows."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🏥 Simple Hospital Financial Intelligence Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                    # Default analysis pipeline
    python run_pipeline.py --full             # Complete pipeline
    python run_pipeline.py --dashboard        # Dashboard only
    python run_pipeline.py --quick            # Quick analysis (sample data)
    python run_pipeline.py --modeling-only    # Retrain models only
    python run_pipeline.py --features-only    # Feature engineering only
        """
    )
    
    # Predefined workflows
    parser.add_argument('--full', action='store_true',
                       help='Complete pipeline: data processing → modeling → dashboard')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch dashboard only')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis with sample data (faster EDA)')
    parser.add_argument('--modeling-only', action='store_true',
                       help='Retrain ML models only')
    parser.add_argument('--features-only', action='store_true',
                       help='Run feature engineering only')
    parser.add_argument('--eda-only', action='store_true',
                       help='Run EDA analysis only')
    
    # Configuration
    parser.add_argument('--port', type=int, default=8502,
                       help='Dashboard port (default: 8502)')
    
    args = parser.parse_args()
    
    # Determine workflow
    if args.full:
        success = run_command(['--full-pipeline'], 
                            "Complete Healthcare Analytics Pipeline")
    
    elif args.dashboard:
        success = run_command(['--dashboard-only', '--port', str(args.port)], 
                            f"Launch Dashboard on Port {args.port}")
    
    elif args.quick:
        success = run_command(['--sample-size', '1000'], 
                            "Quick Analysis with Sample Data")
    
    elif args.modeling_only:
        success = run_command(['--phase', 'modeling'], 
                            "ML Model Training Only")
    
    elif args.features_only:
        success = run_command(['--phase', 'feature_engineering'], 
                            "Enhanced Feature Engineering Only")
    
    elif args.eda_only:
        success = run_command(['--phase', 'eda'], 
                            "Exploratory Data Analysis Only")
    
    else:
        # Default: comprehensive analysis pipeline
        success = run_command(['--skip-data-processing'], 
                            "Default Analysis Pipeline (EDA → Features → Modeling → Dashboard)")
    
    if success:
        print(f"\n🎉 Workflow completed successfully!")
        if not args.modeling_only and not args.features_only and not args.eda_only:
            print(f"🌐 Dashboard available at: http://localhost:{args.port}")
    else:
        print(f"\n❌ Workflow failed. Check output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 