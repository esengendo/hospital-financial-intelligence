# 🏥 Hospital Financial AI: Predictive Analytics & Executive Insights

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready, portfolio-grade data science project** that predicts hospital financial distress and generates LLM-based executive summaries using the [California Health and Human Services Hospital Financial Disclosure dataset](https://data.chhs.ca.gov/dataset/hospital-annual-financial-disclosure-report-complete-data-set).

## 📊 Business Objective

This project delivers actionable insights for healthcare administrators by:

1. **🎯 Predicting Financial Risk**: Machine learning models identify hospitals at risk of financial distress
2. **💰 Cost Optimization**: Analyzes labor expense ratios and contract labor inefficiencies  
3. **📈 Executive Summaries**: AI-generated insights using open-source LLMs (Falcon, Mistral, Flan-T5)
4. **📱 Interactive Dashboard**: Browser-based Streamlit interface for real-time analysis
5. **🐳 Production Ready**: Fully containerized with Docker for enterprise deployment

## 🏗️ Architecture

```
Project_Hospital_Financial_Analysis/
├── 📁 data/
│   ├── raw/                    # Original CHHS data files
│   └── processed/              # Cleaned, feature-engineered datasets
├── 📁 models/                  # Trained ML models & artifacts (Phase 4)
├── 📁 reports/                 # ✅ Generated analysis reports & dashboards
├── 📁 visuals/                 # ✅ Charts, plots, and SHAP outputs
│   ├── eda_charts/
│   └── shap_outputs/
├── 📁 notebooks/               # ✅ Jupyter EDA analysis
├── 📁 src/                     # ✅ Core Python modules
│   ├── __init__.py            # ✅ Package initialization
│   ├── config.py              # ✅ Docker-ready configuration system
│   ├── ingest.py              # ✅ Data loading with flexible paths
│   ├── preprocess.py          # ✅ Data cleaning & validation
│   ├── eda.py                 # ✅ Enhanced EDA with HADR PCL validation
│   ├── financial_metrics.py   # ✅ HADR-compliant financial calculations
│   └── visualizations.py      # ✅ Professional charting and dashboards
├── 📊 run_eda.py              # ✅ Docker-compatible EDA execution
├── 🏥 HADR_DATA_STRUCTURE.md  # ✅ Official OSHPD documentation analysis
├── 📦 pyproject.toml          # ✅ UV package management
├── 🔒 uv.lock                 # ✅ Dependency lock file
└── 📚 Documentation files
```

## 🚀 Project Progress

### ✅ Phase 1: Data Ingestion & Setup (Complete)
- Data source identification and access
- UV package manager setup
- Project structure initialization

### ✅ Phase 2: Data Preprocessing (Complete)  
- CHHS data cleaning and standardization
- Data quality validation
- Parquet optimization for performance

### ✅ Phase 3: Enhanced EDA with HADR Integration (Complete)
- **Docker-ready configuration system** with environment variables
- **HADR PCL validation** with official OSHPD compliance
- **4-strategy column mapping** achieving 80% success rate
- **Financial metrics calculator** with 7 core healthcare ratios
- **Comprehensive reporting** with automated dashboards
- **22-year analysis** covering 9,956 hospital records

### 🔄 Phase 4: Feature Engineering (Completed)
- **Comprehensive Feature Library:** Generated over 30 features per hospital per year, including:
    * **Core Financial Ratios:** Covering liquidity, profitability, efficiency, and leverage.
    * **Advanced Predictive Features:** Components of the Altman Z-Score for bankruptcy prediction.
    * **Time-Series Momentum:** Year-over-Year changes to capture trends.
- **Robust Data Pipeline:** The feature engineering pipeline is fully automated, handling data loading, validation, calculation, and saving of feature sets.
- **Scalable Architecture:** The system processes 21 years of data, demonstrating its ability to handle large-scale financial analysis.

### 📋 Phase 5: Predictive Modeling (Current)

*Work in progress...*

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- `uv` package manager
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/hospital-financial-ai.git
cd hospital-financial-ai

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### 2. Environment Configuration
```bash
cp .env.template .env
# Edit .env with your Hugging Face token (optional)
```

### 3. Run Complete Pipeline
```bash
# Option A: Full automated pipeline
./start.sh

# Option B: Interactive Jupyter analysis
jupyter lab notebooks/hospital_financial_analysis.ipynb

# Option C: Launch Streamlit dashboard
streamlit run src/dashboard.py
```

### 4. Docker Deployment
```bash
# Build and run with Docker
docker build -t hospital-financial-ai .
docker run -p 8501:8501 hospital-financial-ai
```
Access dashboard at: http://localhost:8501

## 📊 Data Source

This project uses the **Hospital Annual Financial Disclosure Report** from the California Health and Human Services Open Data Portal:

- **Dataset**: [CHHS Hospital Financial Data](https://data.chhs.ca.gov/dataset/hospital-annual-financial-disclosure-report-complete-data-set)
- **Coverage**: All licensed California hospitals (2015-2023)
- **Features**: Revenue, expenses, balance sheets, utilization metrics
- **Update Frequency**: Annual
- **Format**: XLSX/CSV with 5.5M+ records

## 🔬 Technical Features

### Docker-Ready Configuration System
- **Environment Variables**: Configurable data paths for containerization
- **Command Line Interface**: `--base-dir`, `--data-dir`, `--output-dir` options
- **Auto-Discovery**: Flexible year detection and batch processing
- **Production Ready**: No hardcoded paths, full container compatibility

### Enhanced HADR PCL Validation
- **Official Compliance**: Direct integration with OSHPD HADR documentation
- **PCL References**: Page-Column-Line validation for all financial fields
- **4-Strategy Mapping**: Direct PCL → Fuzzy PCL → Standard → Partial matching
- **Data Quality**: 100% HADR alignment for 2018-2023 data with 80% mapping success

### Financial Metrics Calculator
- **7 Core Metrics**: Liquidity, profitability, efficiency, and leverage indicators
- **Missing Value Handling**: Robust 0-fill strategy following financial standards
- **HADR-Aligned Fields**: Official PCL references for regulatory compliance
- **Business Context**: Healthcare-specific financial ratio analysis

### Machine Learning Pipeline (Phase 4 - Planned)
- **Models**: XGBoost, Random Forest, Logistic Regression ensemble
- **Features**: 45+ engineered financial ratios and trend indicators
- **Validation**: Time-series cross-validation with forward chaining
- **Metrics**: ROC-AUC, Precision-Recall, Financial Impact Analysis

### Explainable AI (Phase 5 - Planned)
- **SHAP Values**: Feature importance and model interpretability
- **Visualizations**: Waterfall plots, force plots, summary plots
- **Business Context**: Financial ratio analysis with domain expertise

## 📈 Key Insights & Results

### Phase 3 (EDA) - Current Status
- **Dataset Coverage**: 22 years (2002-2023) of California hospital data
- **Records Analyzed**: 9,956 hospital financial reports
- **HADR Compliance**: 100% PCL validation for enhanced years (2018-2023)
- **Financial Metrics**: 7 core metrics with 80% successful mapping
- **Data Quality**: 36.9% overall (expected due to specialty field structure)

### Technical Achievements
- **Configuration System**: Full Docker deployment readiness
- **Column Mapping**: 4-strategy approach with 80% success rate
- **HADR Integration**: Official OSHPD PCL references and validation
- **Financial Calculations**: Robust missing value handling and 0-fill strategy
- **Reporting**: Automated dashboards and executive summaries

### HADR PCL Validated Fields (2018-2023)
1. **Patient Revenue** (`REV_TOT_PT_REV` - P12_C23_L415)
2. **Operating Expenses** (`PY_TOT_OP_EXP` - P8_C2_L200)
3. **Net Income** (`EQ_UNREST_FND_NET_INCOME` - P7_C1_L55)
4. **Assets** (`PY_SP_PURP_FND_OTH_ASSETS` - P6_C2_L30)
5. **Cash Flow** (`CASH_FLOW_SPECIFY_OTH_OP_L102` - P9_C91_L102)
6. **Liabilities** (`PY_SP_PURP_FND_TOT_LIAB_EQ` - P6_C4_L75)

### Future Model Performance (Phase 4 - Planned)
- **ROC-AUC**: Target 0.87+ (XGBoost ensemble)
- **Precision**: Target 0.82+ at 20% recall threshold
- **Early Warning**: 12-month prediction horizon

## 🖥️ Dashboard Features

The Streamlit dashboard provides:

- **📊 Executive Overview**: Key metrics and risk indicators
- **🎯 Hospital Search**: Individual facility analysis
- **📈 Trend Analysis**: Multi-year financial trajectories  
- **🤖 AI Insights**: Real-time LLM-generated summaries
- **📋 Risk Reports**: Downloadable PDF assessments
- **🔍 Comparative Analysis**: Peer benchmarking

## 🧪 Development & Testing

### Code Quality
```bash
# Format code
black src/ notebooks/

# Type checking
mypy src/

# Linting
flake8 src/

# Run tests
pytest tests/ --cov=src
```

### Continuous Integration
GitHub Actions workflow automatically:
- ✅ Tests data pipeline integrity
- ✅ Validates model performance
- ✅ Checks code quality standards
- ✅ Builds Docker images

## 📚 Documentation

- **[CONTEXT.md](CONTEXT.md)**: Problem background and healthcare industry context
- **[TASK.md](TASK.md)**: Technical implementation details and module breakdown
- **[Notebook](notebooks/hospital_financial_analysis.ipynb)**: Complete analysis walkthrough
- **[API Docs](docs/)**: Module documentation and API reference

## 🤝 Contributing

This project follows software engineering best practices:

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **California Health and Human Services** for providing comprehensive hospital financial data
- **Hugging Face** for democratizing access to state-of-the-art language models
- **Healthcare Financial Management Association** for domain expertise and validation

## 📞 Contact

**Portfolio Project** - Demonstrating expertise in:
- Healthcare Analytics & Domain Knowledge
- Production MLOps & Model Deployment  
- LLM Integration & Prompt Engineering
- Data Visualization & Executive Communication

---

*This project showcases 20 years of healthcare analytics experience in a modern, production-ready implementation suitable for enterprise healthcare organizations.*
