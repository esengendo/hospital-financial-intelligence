# 🏥 Hospital Financial Intelligence - Pipeline Guide
*Production-Ready Workflow Orchestration*

## 📋 Quick Start

### ⚡ Simple Commands (Recommended)
```bash
# Activate environment
source .venv/bin/activate

# Most common workflows
python run_pipeline.py                 # Default: EDA → Features → Modeling → Dashboard
python run_pipeline.py --dashboard     # Launch dashboard only
python run_pipeline.py --full          # Complete pipeline from scratch
python run_pipeline.py --quick         # Quick analysis with sample data
```

### 🎯 Advanced Commands
```bash
# Individual phases
python run_pipeline.py --eda-only         # EDA only
python run_pipeline.py --features-only    # Feature engineering only  
python run_pipeline.py --modeling-only    # Model training only

# Custom configuration
python run_pipeline.py --dashboard --port 8503  # Custom port
```

---

## 🏗️ Pipeline Architecture

### 🔄 Complete Workflow
```
📊 Data Processing    →  🔍 EDA Analysis     →  ⚙️ Feature Engineering
       ↓                      ↓                      ↓
🤖 ML Modeling       →  🧠 LLM Integration  →  📈 Dashboard
```

### 📋 Pipeline Phases

| Phase | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| **1. Data Processing** | `streamline_data.py` | `data/raw/` | `data/processed/` | Clean & validate CHHS data |
| **2. EDA Analysis** | `run_eda.py` | `data/processed/` | `reports/`, `visuals/` | Financial analysis & dashboards |
| **3. Feature Engineering** | `run_enhanced_feature_engineering.py` | `data/features/` | `data/features_enhanced/` | 147 advanced features |
| **4. ML Modeling** | `run_enhanced_modeling.py` | `data/features_enhanced/` | `models/`, `visuals/` | XGBoost training & evaluation |
| **5. LLM Integration** | `groq_hospital_analysis.py` | `models/` | `reports/` | AI-powered insights |
| **6. Dashboard** | `streamlit_dashboard_modern.py` | All outputs | Web interface | Interactive analytics |

---

## 🚀 Usage Examples

### 📊 New Project Setup
```bash
# Complete pipeline from raw data
python run_pipeline.py --full
```

### 🔄 Regular Analysis Updates
```bash
# Standard analysis workflow (assumes data already processed)
python run_pipeline.py
```

### ⚡ Quick Development/Testing
```bash
# Fast analysis with sample data
python run_pipeline.py --quick
```

### 🎯 Specific Tasks
```bash
# Retrain models only
python run_pipeline.py --modeling-only

# Update features only
python run_pipeline.py --features-only

# Launch dashboard for demo
python run_pipeline.py --dashboard
```

### 🛠️ Development & Debugging
```bash
# Advanced pipeline control
python main.py --phase eda --sample-size 500 --log-level DEBUG
python main.py --phase modeling --log-file pipeline.log
python main.py --dashboard-only --port 8503
```

---

## 🔧 Advanced Configuration

### 📁 Master Pipeline (`main.py`)
```bash
# Complete control over pipeline execution
python main.py [OPTIONS]

Options:
  --full-pipeline              # Execute all phases
  --phase PHASE_NAME           # Execute single phase
  --dashboard-only             # Launch dashboard only
  --skip-data-processing       # Skip data processing
  --port PORT                  # Dashboard port (default: 8502)
  --sample-size SIZE           # EDA sample size
  --log-level LEVEL            # DEBUG, INFO, WARNING
  --log-file FILE              # Log to file
```

### 📋 Phase Dependencies

**Sequential Dependencies:**
- Data Processing → EDA Analysis
- EDA Analysis → Feature Engineering
- Feature Engineering → ML Modeling
- ML Modeling → LLM Integration
- All phases → Dashboard

**Parallel Execution:**
- EDA Analysis and Feature Engineering can run in parallel if data exists
- LLM Integration can run independently once models exist

---

## 📊 Pipeline Outputs

### 📁 Directory Structure After Full Pipeline
```
Project_Hospital_Financial_Analysis/
├── data/
│   ├── raw/                     # Original CHHS data
│   ├── processed/               # Clean, validated data
│   ├── features/                # Basic financial features
│   └── features_enhanced/       # 147 advanced features
├── models/
│   └── enhanced_xgboost_model/  # Trained ML models
├── reports/
│   ├── executive_summary_*.json    # Business reports
│   ├── model_evaluation_*.json    # ML performance
│   └── *_groq_analysis_*.json     # AI insights
├── visuals/
│   ├── eda_charts/             # EDA visualizations
│   ├── model_evaluation/       # ML evaluation charts
│   └── shap_outputs/           # Explainability plots
└── logs/                       # Pipeline execution logs
```

### 📈 Key Outputs

**EDA Phase:**
- 88 interactive financial dashboards (HTML)
- 44 executive summary reports (JSON)
- Comprehensive data quality assessments

**Feature Engineering:**
- 147 features per hospital per year
- Complete Altman Z-Score components
- 61 time-series features, 36 volatility measures

**ML Modeling:**
- XGBoost model with 99.5% ROC-AUC
- SHAP explainability analysis
- Feature importance rankings

**LLM Integration:**
- AI-powered financial assessments
- Portfolio risk analysis
- Executive recommendations

**Dashboard:**
- Real-time analytics interface
- Hospital selector with 464 real names
- 5 comprehensive analysis sections

---

## 🔍 Monitoring & Logging

### 📊 Pipeline Status Tracking
The pipeline provides comprehensive execution monitoring:
- **Real-time progress**: Live output from each phase
- **Execution times**: Performance tracking per phase  
- **Success/failure status**: Clear phase completion indicators
- **Summary reports**: Final execution summary with statistics

### 📝 Logging Levels
```bash
# Standard logging
python main.py --log-level INFO

# Detailed debugging
python main.py --log-level DEBUG --log-file debug.log

# Minimal output
python main.py --log-level WARNING
```

### 🔧 Error Handling
- **Dependency validation**: Checks required directories and files
- **Graceful failures**: Pipeline stops at first failure with clear error messages
- **Recovery guidance**: Specific instructions for resolving issues
- **Partial execution**: Ability to resume from any completed phase

---

## 🚀 Performance Optimization

### ⚡ Speed Optimizations
```bash
# Quick analysis (sample data)
python run_pipeline.py --quick

# Skip phases if data exists
python main.py --skip-data-processing

# Parallel execution (advanced)
python main.py --phase eda &
python main.py --phase feature_engineering &
wait
```

### 💾 Resource Management
- **Memory-efficient**: Processes data in chunks using Parquet format
- **Disk space**: Intermediate files cleaned up automatically
- **CPU usage**: Configurable parallelization in ML training

---

## 🛠️ Troubleshooting

### Common Issues

**Pipeline fails at data processing:**
```bash
# Ensure raw data exists
ls data/raw/
# If missing, download CHHS data or use existing processed data
python main.py --skip-data-processing
```

**Dashboard won't launch:**
```bash
# Check port availability
python main.py --dashboard-only --port 8503
# Or use simple runner
python run_pipeline.py --dashboard --port 8503
```

**Feature engineering fails:**
```bash
# Ensure feature files exist
ls data/features/
# If missing, run complete pipeline
python run_pipeline.py --full
```

**Modeling phase fails:**
```bash
# Check enhanced features exist  
ls data/features_enhanced/
# Regenerate if needed
python run_pipeline.py --features-only
```

### 🔧 Environment Issues
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Verify installation
python -c "import src.config; print('✅ Package installed')"
```

---

## 🎯 Best Practices

### 🔄 Development Workflow
1. **Start with dashboard**: `python run_pipeline.py --dashboard`
2. **Quick iterations**: `python run_pipeline.py --quick` 
3. **Full validation**: `python run_pipeline.py --full`
4. **Production deployment**: Use containerized deployment

### 📊 Data Updates
1. **New data available**: `python run_pipeline.py --full`
2. **Model retraining**: `python run_pipeline.py --modeling-only`
3. **Feature updates**: `python run_pipeline.py --features-only`

### 🎯 Performance Monitoring
- Monitor execution times for performance regression
- Track memory usage during large dataset processing  
- Validate output quality at each phase
- Use logging for debugging and optimization

---

## 📋 Integration with CI/CD

### 🔄 Automated Pipeline
```yaml
# Example GitHub Actions workflow
name: Hospital Analytics Pipeline
on: [push, schedule]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e .
      - name: Run pipeline
        run: python main.py --full-pipeline --log-level INFO
      - name: Deploy dashboard
        run: python main.py --dashboard-only --port 8502
```

### 🚀 Production Deployment
```bash
# Docker deployment
docker build -t hospital-analytics .
docker run -p 8502:8502 hospital-analytics python main.py --dashboard-only

# Cloud deployment
python main.py --full-pipeline --log-file production.log
```

---

## 📈 Next Steps

### 🔄 Pipeline Enhancements
- **Distributed processing**: Implement Dask for large-scale data processing
- **Real-time updates**: Add streaming data ingestion capabilities
- **Advanced monitoring**: Integrate with monitoring tools (Prometheus, Grafana)
- **Auto-scaling**: Cloud-native deployment with auto-scaling

### 🤖 ML Pipeline Extensions  
- **Model versioning**: MLflow integration for model lifecycle management
- **A/B testing**: Framework for model comparison and gradual rollout
- **Drift detection**: Automated data and model drift monitoring
- **Federated learning**: Multi-hospital collaborative model training

### 📊 Dashboard Enhancements
- **Real-time streaming**: Live data updates and alerts
- **Mobile optimization**: Responsive design for mobile devices
- **Advanced analytics**: Predictive forecasting and scenario modeling
- **Integration APIs**: RESTful APIs for third-party integrations

This pipeline system provides a solid foundation for production healthcare analytics with room for advanced enterprise features. 