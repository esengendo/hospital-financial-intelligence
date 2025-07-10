# 🏥 Hospital Financial Intelligence Platform

> **Predicting healthcare financial distress with AI-powered analytics**

A production-ready healthcare analytics system that combines 20+ years of California hospital data, advanced machine learning, and explainable AI to predict financial distress and provide actionable insights for healthcare executives.

## ✨ What Makes This Special

🎯 **Real-World Impact**: Analyzes 441 California hospitals using official state data (2003-2023)  
🧠 **AI-Powered**: Groq LLM integration for intelligent financial analysis  
📊 **Production-Ready**: Complete MLOps pipeline with Docker deployment  
🔍 **Explainable**: SHAP-based model interpretability for regulatory compliance  
⚡ **High Performance**: 99.5% ROC-AUC with 147 engineered features  

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Pull and run from Docker Hub
docker pull esengendo730/hospital-financial-ai:latest
docker run -p 8502:8502 esengendo730/hospital-financial-ai:latest

# Access dashboard at http://localhost:8502
```

### Option 2: Local Development
```bash
# Clone and setup
git clone <repository-url>
cd Project_Hospital_Financial_Analysis
source .venv/bin/activate

# Install dependencies  
uv pip install -r requirements.txt

# Run complete analysis pipeline
python pipeline.py --full

# Launch interactive dashboard
python pipeline.py --dashboard
```

**That's it!** 🎉 Your healthcare analytics platform is running.

## 🎯 Key Features

### 💡 Intelligent Analytics
- **Financial Health Prediction**: XGBoost models with 99.5% accuracy
- **AI-Powered Insights**: Natural language analysis via Groq LLM
- **Real-Time Dashboards**: Interactive Streamlit interface
- **Regulatory Compliance**: SHAP explainability for audits

### 📈 Advanced Engineering
- **147 Features**: From 33 base metrics to comprehensive financial indicators
- **Altman Z-Score**: Complete bankruptcy prediction framework
- **Time-Series Analysis**: Rolling averages, volatility, momentum indicators
- **Data Quality**: HADR PCL compliance validation

### 🏗️ Production Architecture
- **Streamlined Pipeline**: Single-command orchestration
- **Docker Ready**: Containerized deployment
- **Scalable Design**: Modular, testable components
- **MLOps Best Practices**: Experiment tracking, model versioning

## 📊 Performance Highlights

| Metric | Value | Description |
|--------|-------|-------------|
| **ROC-AUC** | 99.5% | Model accuracy on test set |
| **PR-AUC** | 90.9% | Precision-recall performance |
| **Data Coverage** | 20+ years | Historical analysis depth |
| **Hospitals** | 441 | California healthcare facilities |
| **Features** | 147 | Engineered financial indicators |

## 🛠️ Technical Stack

**Core Technologies**
- Python 3.9+ • Pandas • XGBoost • SHAP • Streamlit

**Data & ML**
- Parquet • NumPy • Scikit-learn • Plotly • Groq LLM

**DevOps**
- Docker • Docker Hub • UV Package Manager • Virtual Environments

## 📁 Project Structure

```
├── pipeline.py                    # 🎯 Master orchestrator
├── run_eda.py                     # 📊 Exploratory analysis  
├── run_enhanced_modeling.py       # 🤖 ML training
├── groq_hospital_analysis.py      # 🧠 AI insights
├── streamlit_dashboard_modern.py  # 📈 Interactive dashboard
├── src/                           # 📚 Core modules
├── data/                          # 💾 Data storage
├── models/                        # 🎯 Trained models
└── reports/                       # 📄 Analysis outputs
```

## 🎮 Usage Examples

### Docker Commands
```bash
# Run complete pipeline in Docker
docker run esengendo730/hospital-financial-ai:latest python pipeline.py --full

# Launch dashboard on custom port
docker run -p 8503:8502 esengendo730/hospital-financial-ai:latest

# Interactive shell access
docker run -it esengendo730/hospital-financial-ai:latest /bin/bash
```

### Local Development
```bash
# Complete pipeline from raw data
python pipeline.py --full

# Quick analysis for development
python pipeline.py --quick --sample-size 1000

# Specific analysis phases
python pipeline.py --eda-only
python pipeline.py --modeling-only

# Custom dashboard port
python pipeline.py --dashboard --port 8503
```

## 🏆 Skills Demonstrated

### **Data Science & ML**
- Feature engineering (33 → 147 features)
- Time-series analysis and forecasting
- Imbalanced data handling techniques
- Model interpretability (SHAP)
- Hyperparameter optimization

### **Software Engineering**
- Clean, modular architecture
- Comprehensive testing and validation
- Docker containerization
- CI/CD pipeline design
- Production deployment patterns

### **Healthcare Domain**
- GAAP accounting standards
- HADR PCL compliance
- Healthcare financial ratios
- Regulatory requirements
- Executive dashboard design

### **AI/LLM Integration**
- Groq API integration
- Prompt engineering
- Natural language generation
- Portfolio analysis automation
- Business intelligence workflows

## 📈 Business Impact

**For Healthcare Executives:**
- Early warning system for financial distress
- Portfolio risk assessment and monitoring
- Regulatory compliance reporting
- Strategic planning insights

**For Data Teams:**
- Production-ready ML pipeline
- Explainable AI framework
- Automated report generation
- Scalable analytics platform

## 🔧 Advanced Features

### **AI-Powered Analysis**
```python
# Generate hospital financial insights
analyzer = GroqHospitalAnalyzer()
insights = analyzer.analyze_portfolio(max_hospitals=10)
```

### **Interactive Dashboard**
- Real-time financial health monitoring
- Executive-ready visualizations
- Multi-hospital portfolio views
- Drill-down analysis capabilities

### **Model Explainability**
- SHAP feature importance analysis
- Individual prediction explanations
- Regulatory audit trails
- Business-friendly interpretations

## 🚀 Getting Started

### Quick Demo (Docker)
1. **Pull from Docker Hub**: `docker pull esengendo730/hospital-financial-ai:latest`
2. **Run the dashboard**: `docker run -p 8502:8502 esengendo730/hospital-financial-ai:latest`
3. **Open browser**: Navigate to `http://localhost:8502`

### Local Development
1. **Clone the repository**
2. **Activate virtual environment**: `source .venv/bin/activate`
3. **Install dependencies**: `uv pip install -r requirements.txt`
4. **Run the pipeline**: `python pipeline.py --full`
5. **Explore the dashboard**: `python pipeline.py --dashboard`

## 🎯 What's Next?

This platform demonstrates production-ready healthcare analytics with:
- **Scalable ML pipelines** for enterprise deployment
- **Explainable AI** for regulatory compliance
- **Modern data engineering** practices
- **Healthcare domain expertise** application

Ready to discuss how these skills can drive impact at your organization? Let's connect! 🤝

---

**Built with ❤️ for healthcare analytics and data-driven decision making**

*Showcasing expertise in ML, healthcare analytics, and production system design*
