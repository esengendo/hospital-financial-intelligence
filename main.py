#!/usr/bin/env python3
"""
🏥 Hospital Financial Intelligence - Master Pipeline
===================================================
Production-ready master workflow orchestrating the complete healthcare analytics pipeline.

Phases:
1. Data Processing & Validation
2. Exploratory Data Analysis  
3. Enhanced Feature Engineering
4. Machine Learning Modeling
5. LLM Integration & Analysis
6. Dashboard Deployment

Usage:
    python main.py --full-pipeline          # Run complete workflow
    python main.py --phase eda              # Run specific phase
    python main.py --skip-data-processing   # Skip data processing if already done
    python main.py --dashboard-only         # Launch dashboard only
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

# Configure professional logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure comprehensive logging for the master pipeline."""
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
    """Master orchestrator for the complete healthcare analytics pipeline."""
    
    def __init__(self, base_dir: Path, log_level: str = "INFO"):
        self.base_dir = Path(base_dir).resolve()
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pipeline configuration
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
        """Print professional pipeline header."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                🏥 HOSPITAL FINANCIAL INTELLIGENCE PLATFORM                   ║
║                          Master Pipeline Orchestrator                       ║
║                     Production Healthcare Analytics Suite                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
    def validate_environment(self) -> bool:
        """Validate the environment and dependencies."""
        self.logger.info("🔧 Validating environment setup...")
        
        # Check if in project root
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
        """Check if phase requirements are met."""
        phase_config = self.phases[phase]
        
        for req_dir in phase_config['required_dirs']:
            dir_path = self.base_dir / req_dir
            if not dir_path.exists():
                self.logger.error(f"❌ Required directory missing for {phase}: {req_dir}")
                return False
            
            # Check if directory has content (except for raw data which might be downloaded separately)
            if req_dir != 'data/raw' and not any(dir_path.iterdir()):
                self.logger.error(f"❌ Required directory empty for {phase}: {req_dir}")
                return False
        
        return True
        
    def execute_phase(self, phase: str, **kwargs) -> bool:
        """Execute a specific pipeline phase."""
        phase_config = self.phases[phase]
        script_path = self.base_dir / phase_config['script']
        
        if not script_path.exists():
            self.logger.error(f"❌ Script not found: {script_path}")
            return False
        
        self.logger.info(f"🚀 Starting {phase_config['name']}")
        self.logger.info(f"   {phase_config['description']}")
        
        start_time = time.time()
        
        try:
            # Special handling for dashboard (non-blocking)
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
            
            # Stream output in real-time
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
        """Launch the Streamlit dashboard."""
        try:
            # Try different streamlit execution methods (Docker-compatible)
            streamlit_paths = [
                'streamlit',  # System PATH (Docker environment)
                sys.executable + ' -m streamlit'  # Module execution
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
                # Launch in background
                subprocess.Popen(cmd, cwd=self.base_dir)
                self.logger.info("✅ Dashboard launched in background")
                return True
            else:
                # Launch in foreground (blocking)
                subprocess.run(cmd, cwd=self.base_dir)
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Dashboard launch failed: {str(e)}")
            return False
    
    def run_pipeline(self, phases: List[str], **kwargs) -> bool:
        """Execute the complete or partial pipeline."""
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
        """Print pipeline execution summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("📊 PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Execution Time: {total_time:.1f}s")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for phase, status in self.pipeline_status.items():
            phase_name = self.phases[phase]['name']
            exec_time = self.execution_times.get(phase, 0)
            status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏸️"
            print(f"{status_emoji} {phase_name:35} | {status:10} | {exec_time:6.1f}s")
        
        completed = sum(1 for s in self.pipeline_status.values() if s == 'completed')
        total = len(self.pipeline_status)
        print(f"\nSUCCESS RATE: {completed}/{total} phases completed ({completed/total*100:.1f}%)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="🏥 Hospital Financial Intelligence - Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Execution modes
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Execute complete pipeline (data → modeling → dashboard)')
    parser.add_argument('--phase', choices=['data_processing', 'eda', 'feature_engineering', 'modeling', 'llm_analysis', 'dashboard'],
                       help='Execute single phase')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Launch dashboard only (skip all processing)')
    parser.add_argument('--skip-data-processing', action='store_true',
                       help='Skip data processing phase')
    
    # Configuration
    parser.add_argument('--port', type=int, default=8502,
                       help='Dashboard port (default: 8502)')
    parser.add_argument('--sample-size', type=int,
                       help='Sample size for EDA analysis')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Initialize orchestrator
    orchestrator = HospitalPipelineOrchestrator(Path.cwd(), args.log_level)
    orchestrator.print_header()
    
    # Validate environment
    if not orchestrator.validate_environment():
        sys.exit(1)
    
    try:
        # Determine execution plan
        if args.dashboard_only:
            phases = ['dashboard']
        elif args.phase:
            phases = [args.phase]
        elif args.full_pipeline:
            phases = list(orchestrator.phases.keys())
            if args.skip_data_processing:
                phases = phases[1:]  # Skip data_processing
        else:
            # Default: run analysis pipeline (assumes data already processed)
            phases = ['eda', 'feature_engineering', 'modeling', 'dashboard']
        
        # Execute pipeline
        kwargs = {
            'port': args.port,
            'sample_size': args.sample_size,
            'background': not args.dashboard_only  # Dashboard foreground only if dashboard-only
        }
        
        success = orchestrator.run_pipeline(phases, **kwargs)
        
        # Print summary
        orchestrator.print_summary()
        
        if success:
            print(f"\n🎉 Pipeline completed successfully!")
            if 'dashboard' in phases:
                print(f"🌐 Dashboard available at: http://localhost:{args.port}")
        else:
            print(f"\n❌ Pipeline execution failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"💥 Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
