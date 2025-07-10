#!/usr/bin/env python3
"""
Hospital Financial Intelligence - Streamlined Pipeline
Production-ready healthcare analytics orchestrator.

Usage:
    python pipeline.py                    # Default workflow
    python pipeline.py --full             # Complete pipeline
    python pipeline.py --dashboard        # Dashboard only
    python pipeline.py --phase eda        # Single phase
"""

import argparse
import logging
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for pipeline execution."""
    log_format = "%(asctime)s | %(name)12s | %(levelname)8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    # Suppress verbose third-party loggers
    logging.getLogger('plotly').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


class HospitalPipelineOrchestrator:
    """Master orchestrator for healthcare analytics pipeline."""
    
    def __init__(self, base_dir: Path, log_level: str = "INFO"):
        self.base_dir = Path(base_dir).resolve()
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pipeline phase configuration
        self.phases = {
            'data_processing': {
                'name': '📊 Data Processing & Validation',
                'script': 'streamline_data.py',
                'required_dirs': ['data/raw'],
                'output_dirs': ['data/processed'],
                'description': 'Process raw CHHS data into clean, validated format'
            },
            'eda': {
                'name': '🔍 Exploratory Data Analysis',
                'script': 'run_eda.py',
                'required_dirs': ['data/processed'],
                'output_dirs': ['reports', 'visuals'],
                'description': 'Comprehensive financial analysis and visualization'
            },
            'feature_engineering': {
                'name': '⚙️ Enhanced Feature Engineering',
                'script': 'run_enhanced_feature_engineering.py',
                'required_dirs': ['data/features'],
                'output_dirs': ['data/features_enhanced'],
                'description': '147 advanced features with Altman Z-Score components'
            },
            'modeling': {
                'name': '🤖 Machine Learning Pipeline',
                'script': 'run_enhanced_modeling.py',
                'required_dirs': ['data/features_enhanced'],
                'output_dirs': ['models', 'visuals'],
                'description': 'XGBoost training with explainability and validation'
            },
            'llm_analysis': {
                'name': '🧠 LLM Integration & Analysis',
                'script': 'groq_hospital_analysis.py',
                'required_dirs': ['models'],
                'output_dirs': ['reports'],
                'description': 'AI-powered financial insights and portfolio analysis'
            },
            'dashboard': {
                'name': '📈 Interactive Dashboard',
                'script': 'streamlit_dashboard_modern.py',
                'required_dirs': ['data/features_enhanced', 'models', 'reports'],
                'output_dirs': [],
                'description': 'Professional healthcare analytics dashboard'
            }
        }
        
        self.pipeline_status = {phase: 'pending' for phase in self.phases.keys()}
        self.execution_times = {}
        self.start_time = datetime.now()
        
    def print_header(self):
        """Print pipeline header."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                🏥 HOSPITAL FINANCIAL INTELLIGENCE PLATFORM                   ║
║                        Streamlined Master Pipeline                          ║
║                     Production Healthcare Analytics Suite                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
    def validate_environment(self) -> bool:
        """Validate environment and dependencies."""
        self.logger.info("🔧 Validating environment setup...")
        
        # Check project root
        required_files = ['pyproject.toml', 'src', 'README.md']
        missing_files = [f for f in required_files if not (self.base_dir / f).exists()]
        
        if missing_files:
            self.logger.error(f"❌ Missing required files/directories: {missing_files}")
            self.logger.error(f"   Please run from project root: {self.base_dir}")
            return False
        
        # Check virtual environment
        if not os.environ.get('VIRTUAL_ENV') and not os.environ.get('CONDA_DEFAULT_ENV'):
            self.logger.warning("⚠️  Virtual environment not detected. Consider activating .venv")
        
        # Create output directories
        output_dirs = ['data', 'reports', 'visuals', 'models', 'logs']
        for dir_name in output_dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
        
        self.logger.info("✅ Environment validation completed")
        return True
        
    def check_phase_requirements(self, phase: str) -> bool:
        """Check phase requirements."""
        phase_config = self.phases[phase]
        
        for req_dir in phase_config['required_dirs']:
            dir_path = self.base_dir / req_dir
            if not dir_path.exists():
                self.logger.error(f"❌ Required directory missing for {phase}: {req_dir}")
                return False
            
            # Check directory content (except raw data)
            if req_dir != 'data/raw' and not any(dir_path.iterdir()):
                self.logger.warning(f"⚠️  Required directory empty for {phase}: {req_dir}")
        
        return True
        
    def execute_phase(self, phase: str, **kwargs) -> bool:
        """Execute pipeline phase."""
        phase_config = self.phases[phase]
        script_path = self.base_dir / phase_config['script']
        
        if not script_path.exists():
            self.logger.error(f"❌ Script not found: {script_path}")
            return False
        
        self.logger.info(f"🚀 Starting {phase_config['name']}")
        self.logger.info(f"   {phase_config['description']}")
        
        start_time = time.time()
        
        try:
            # Special handling for dashboard
            if phase == 'dashboard':
                return self._launch_dashboard(**kwargs)
            
            # Build command
            cmd = [sys.executable, str(script_path)]
            
            # Add phase-specific arguments
            if phase == 'eda' and kwargs.get('sample_size'):
                cmd.extend(['--sample-size', str(kwargs['sample_size'])])
            
            # Execute with real-time output
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.logger.info(f"   {phase}: {line}")
            
            process.wait()
            
            if process.returncode != 0:
                self.logger.error(f"❌ {phase_config['name']} failed with exit code {process.returncode}")
                return False
            
            execution_time = time.time() - start_time
            self.execution_times[phase] = execution_time
            
            self.logger.info(f"✅ {phase_config['name']} completed in {execution_time:.1f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {phase_config['name']} failed: {str(e)}")
            return False
    
    def _launch_dashboard(self, port: int = 8502, **kwargs) -> bool:
        """Launch Streamlit dashboard."""
        try:
            # Try different streamlit execution methods
            streamlit_paths = [
                'streamlit',
                sys.executable + ' -m streamlit'
            ]
            
            streamlit_cmd = None
            for path in streamlit_paths:
                if path.startswith('/') and os.path.exists(path):
                    streamlit_cmd = [path]
                    break
                elif not path.startswith('/'):
                    try:
                        subprocess.run(['which', path], check=True, capture_output=True)
                        streamlit_cmd = [path] if ' -m ' not in path else path.split()
                        break
                    except subprocess.CalledProcessError:
                        continue
            
            if not streamlit_cmd:
                self.logger.error("❌ No working streamlit installation found")
                return False
            
            cmd = streamlit_cmd + [
                'run', 'streamlit_dashboard_modern.py',
                '--server.port', str(port),
                '--server.headless', 'true'
            ]
            
            self.logger.info(f"🌐 Launching dashboard on port {port}")
            self.logger.info(f"   URL: http://localhost:{port}")
            
            if kwargs.get('background', True):
                subprocess.Popen(cmd, cwd=self.base_dir)
                self.logger.info("✅ Dashboard launched in background")
                return True
            else:
                subprocess.run(cmd, cwd=self.base_dir)
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Dashboard launch failed: {str(e)}")
            return False
    
    def run_pipeline(self, phases: List[str], **kwargs) -> bool:
        """Execute pipeline phases."""
        self.logger.info(f"🎯 Starting pipeline execution: {', '.join(phases)}")
        
        success_count = 0
        total_phases = len(phases)
        
        for i, phase in enumerate(phases):
            phase_num = i + 1
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"📋 Phase {phase_num}/{total_phases}: {self.phases[phase]['name']}")
            self.logger.info(f"{'='*80}")
            
            # Check requirements
            if not self.check_phase_requirements(phase):
                self.logger.error(f"❌ Phase {phase} requirements not met")
                self.pipeline_status[phase] = 'failed'
                break
            
            # Execute phase
            if self.execute_phase(phase, **kwargs):
                self.pipeline_status[phase] = 'completed'
                success_count += 1
            else:
                self.pipeline_status[phase] = 'failed'
                break
        
        return success_count == total_phases
    
    def print_summary(self):
        """Print execution summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"📊 PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.1f}s")
        
        for phase, status in self.pipeline_status.items():
            if status == 'completed':
                exec_time = self.execution_times.get(phase, 0)
                print(f"✅ {self.phases[phase]['name']:40} ({exec_time:.1f}s)")
            elif status == 'failed':
                print(f"❌ {self.phases[phase]['name']:40} (FAILED)")
            else:
                print(f"⏸️  {self.phases[phase]['name']:40} (SKIPPED)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hospital Financial Intelligence - Streamlined Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple Commands:
    python pipeline.py                      # Default analysis pipeline
    python pipeline.py --full               # Complete pipeline from data processing
    python pipeline.py --dashboard          # Launch dashboard only
    python pipeline.py --quick              # Quick analysis with sample data
    python pipeline.py --modeling-only      # Retrain models only
    python pipeline.py --features-only      # Feature engineering only
    python pipeline.py --eda-only           # EDA analysis only

Advanced Options:
    python pipeline.py --phase eda          # Run specific phase
    python pipeline.py --log-level DEBUG    # Detailed logging
    python pipeline.py --sample-size 1000   # Custom sample size
    python pipeline.py --port 8503          # Custom dashboard port
        """
    )
    
    # Simple workflow options
    workflow_group = parser.add_mutually_exclusive_group()
    workflow_group.add_argument('--full', action='store_true',
                               help='Complete pipeline: data processing → modeling → dashboard')
    workflow_group.add_argument('--dashboard', action='store_true',
                               help='Launch dashboard only')
    workflow_group.add_argument('--quick', action='store_true',
                               help='Quick analysis with sample data (faster EDA)')
    workflow_group.add_argument('--modeling-only', action='store_true',
                               help='Retrain ML models only')
    workflow_group.add_argument('--features-only', action='store_true',
                               help='Run feature engineering only')
    workflow_group.add_argument('--eda-only', action='store_true',
                               help='Run EDA analysis only')
    workflow_group.add_argument('--phase', choices=['data_processing', 'eda', 'feature_engineering', 'modeling', 'llm_analysis', 'dashboard'],
                               help='Execute single phase')
    
    # Configuration options
    parser.add_argument('--skip-data-processing', action='store_true',
                       help='Skip data processing phase')
    parser.add_argument('--port', type=int, default=8502,
                       help='Dashboard port (default: 8502)')
    parser.add_argument('--sample-size', type=int,
                       help='Sample size for EDA analysis')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO',
                       help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Initialize orchestrator
    base_dir = Path.cwd()
    orchestrator = HospitalPipelineOrchestrator(base_dir, args.log_level)
    
    # Print header
    orchestrator.print_header()
    
    # Validate environment
    if not orchestrator.validate_environment():
        sys.exit(1)
    
    # Determine phases to run
    if args.full:
        phases = ['data_processing', 'eda', 'feature_engineering', 'modeling', 'llm_analysis', 'dashboard']
        print("🚀 Complete Pipeline: Data processing → AI insights → Dashboard")
    elif args.dashboard:
        phases = ['dashboard']
        print(f"🚀 Launch Dashboard on Port {args.port}")
    elif args.quick:
        phases = ['eda', 'feature_engineering']
        print("🚀 Quick Analysis: EDA → Features")
    elif args.modeling_only:
        phases = ['modeling']
        print("🚀 Machine Learning Pipeline Only")
    elif args.features_only:
        phases = ['feature_engineering']
        print("🚀 Enhanced Feature Engineering Only")
    elif args.eda_only:
        phases = ['eda']
        print("🚀 Exploratory Data Analysis Only")
    elif args.phase:
        phases = [args.phase]
        print(f"🚀 Single Phase: {orchestrator.phases[args.phase]['name']}")
    else:
        # Default: Skip data processing, run core analysis
        phases = ['eda', 'feature_engineering', 'modeling', 'dashboard']
        if not args.skip_data_processing:
            print("🚀 Default Analysis Pipeline: EDA → Features → Modeling → Dashboard")
        else:
            print("🚀 Analysis Pipeline (Skip Data Processing)")
    
    # Show phases
    phase_names = [orchestrator.phases[p]['name'] for p in phases]
    print(f"📋 Phases: {' → '.join(phase_names)}")
    print("=" * 80)
    
    # Execute pipeline
    kwargs = {
        'port': args.port,
        'sample_size': args.sample_size,
        'background': True
    }
    
    success = orchestrator.run_pipeline(phases, **kwargs)
    
    # Print summary
    orchestrator.print_summary()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 