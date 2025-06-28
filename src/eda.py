"""
Hospital Financial Intelligence - Exploratory Data Analysis

Professional EDA platform with HADR PCL-validated column identification.
"""

import warnings
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from .ingest import HospitalDataLoader
from .visualizations import HospitalVisualizationEngine
from .financial_metrics import FinancialMetricsCalculator, identify_financial_distress
from .config import Config, get_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HospitalFinancialEDA:
    """Hospital Financial Intelligence EDA Platform with HADR PCL validation."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize EDA platform with configuration."""
        self.config = config or get_config()
        self.data_loader = HospitalDataLoader(self.config.processed_data_dir)
        
        # Business color palette
        self.colors = {
            'primary': '#2E4F6B', 'secondary': '#8B5A3C', 'accent': '#F4A261',
            'success': '#2A9D8F', 'warning': '#E76F51', 'neutral': '#495867'
        }
        self.palette = ['#2E4F6B', '#8B5A3C', '#F4A261', '#2A9D8F', '#E76F51', '#495867']
        
        self._configure_theme()
        
        self.data = None
        self.metrics_calculator = None
        self.column_analysis = None
        
        logger.info("🏥 Hospital Financial Intelligence EDA Platform initialized")
        logger.info(f"   📁 Data Directory: {self.config.processed_data_dir}")
        logger.info(f"   📁 Output Directory: {self.config.reports_dir}")
    
    def _configure_theme(self):
        """Configure professional theme."""
        template = go.layout.Template(pio.templates["plotly_white"])
        
        template.layout.update({
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#1A1A1A'},
            'title': {'font': {'size': 18}, 'x': 0.5, 'xanchor': 'center'},
            'colorway': self.palette,
            'margin': {'t': 80, 'r': 40, 'b': 60, 'l': 80}
        })
        
        axis_style = {
            'linecolor': self.colors['neutral'], 'gridcolor': '#E8E8E8',
            'tickcolor': self.colors['neutral']
        }
        template.layout.xaxis = template.layout.yaxis = axis_style
        
        pio.templates['business'] = template
        pio.templates.default = 'business'
    
    def analyze_hadr_columns(self) -> Dict:
        """Comprehensive HADR column identification and validation analysis."""
        logger.info("🔍 Performing comprehensive HADR column analysis...")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Official HADR PCL structure
        hadr_structure = {
            'P5': {'name': 'Balance Sheet - Unrestricted Fund', 'fields': 164, 'pcl_range': 'BPS-BVZ'},
            'P6': {'name': 'Balance Sheet - Restricted Fund', 'fields': 90, 'pcl_range': 'CJY-CNJ'},
            'P7': {'name': 'Changes in Equity', 'fields': 54, 'pcl_range': 'CNK-CPS'},
            'P8': {'name': 'Income Statement', 'fields': 182, 'pcl_range': 'CSR-CZQ'},
            'P9': {'name': 'Cash Flow Statement', 'fields': 99, 'pcl_range': 'CZR-DDL'},
            'P10': {'name': 'Revenue & Costs Summary', 'fields': 1350, 'pcl_range': 'DDM-FDG'},
            'P12': {'name': 'Patient Revenue by Payer', 'fields': 1577, 'pcl_range': 'FDH-HLX'},
            'P17': {'name': 'Expense Trial Balance - Revenue Centers', 'fields': 1102, 'pcl_range': 'JBU-KRP'},
            'P18': {'name': 'Expense Trial Balance - Non-Revenue Centers', 'fields': 884, 'pcl_range': 'KRQ-LXC'}
        }
        
        # PCL-validated HADR field mappings
        pcl_validated_fields = {
            'REV_TOT_PT_REV': {
                'pcl': 'P12_C23_L415',
                'section': 'Patient Revenue (12)',
                'description': 'Gross Patient Revenue_Total Patient Revenue',
                'metric_type': 'Revenue'
            },
            'PY_TOT_OP_EXP': {
                'pcl': 'P8_C2_L200', 
                'section': 'Income Statement (8)',
                'description': 'Prior Year_Income Statement_Total Operating Expenses',
                'metric_type': 'Expense'
            },
            'EQ_UNREST_FND_NET_INCOME': {
                'pcl': 'P7_C1_L55',
                'section': 'Changes in Equity (7)', 
                'description': 'Changes in Equity_Unrestricted Fund_Net Income (Loss)',
                'metric_type': 'Income'
            },
            'PY_SP_PURP_FND_OTH_ASSETS': {
                'pcl': 'P6_C2_L30',
                'section': 'Balance Sheet (6)',
                'description': 'Prior Year_Specific Purpose Fund_Other Assets',
                'metric_type': 'Asset'
            },
            'CASH_FLOW_SPECIFY_OTH_OP_L102': {
                'pcl': 'P9_C91_L102',
                'section': 'Cash Flows (9)',
                'description': 'Cash Flow_Specify_Other Cash from Operating Activities - Line 102', 
                'metric_type': 'Cash Flow'
            },
            'PY_SP_PURP_FND_TOT_LIAB_EQ': {
                'pcl': 'P6_C4_L75',
                'section': 'Balance Sheet (6)',
                'description': 'Prior Year_Specific Purpose Fund_Total Specific Purpose Fund Liabilities and Fund Balance',
                'metric_type': 'Liability'
            }
        }
        
        analysis = {
            'dataset_overview': {
                'total_columns': len(self.data.columns),
                'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
                'total_records': len(self.data),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'hadr_structure_validation': hadr_structure,
            'pcl_validated_mappings': {},
            'column_completeness': {},
            'section_analysis': {},
            'mapping_recommendations': {}
        }
        
        # Analyze PCL-validated field presence and completeness
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for field, info in pcl_validated_fields.items():
            if field in self.data.columns:
                field_data = self.data[field]
                completeness = ((len(field_data) - field_data.isnull().sum()) / len(field_data)) * 100
                
                analysis['pcl_validated_mappings'][field] = {
                    'found': True,
                    'pcl_reference': info['pcl'],
                    'hadr_section': info['section'],
                    'description': info['description'],
                    'metric_type': info['metric_type'],
                    'completeness_pct': round(completeness, 1),
                    'total_records': len(field_data),
                    'non_null_records': len(field_data) - field_data.isnull().sum(),
                    'data_type': str(field_data.dtype)
                }
                
                if completeness == 100.0:
                    logger.info(f"✅ {field}: 100% complete ({info['pcl']})")
                else:
                    logger.warning(f"⚠️  {field}: {completeness:.1f}% complete ({info['pcl']})")
            else:
                analysis['pcl_validated_mappings'][field] = {
                    'found': False,
                    'pcl_reference': info['pcl'],
                    'hadr_section': info['section'],
                    'description': info['description'],
                    'metric_type': info['metric_type']
                }
                logger.warning(f"❌ {field}: Not found in dataset ({info['pcl']})")
        
        # Analyze columns by HADR section patterns
        section_patterns = {
            'Revenue Fields': ['REV_', 'REVENUE', '_REV'],
            'Expense Fields': ['EXP_', 'EXPENSE', '_EXP'],
            'Asset Fields': ['ASSETS', '_ASSETS', 'ASSET_'],
            'Liability Fields': ['LIAB', 'LIABILITY', '_LIAB'],
            'Equity Fields': ['EQ_', 'EQUITY', '_EQUITY'],
            'Cash Flow Fields': ['CASH_', '_CASH', 'FLOW_'],
            'Prior Year Fields': ['PY_'],
            'Fund Fields': ['_FND_', 'FUND_'],
            'Patient Fields': ['PAT_', 'PATIENT', '_PT_'],
            'Operating Fields': ['OP_', 'OPERATING', '_OP']
        }
        
        for section, patterns in section_patterns.items():
            matching_cols = []
            for col in self.data.columns:
                if any(pattern in col.upper() for pattern in patterns):
                    matching_cols.append(col)
            
            analysis['section_analysis'][section] = {
                'column_count': len(matching_cols),
                'columns': matching_cols[:10],  # First 10 for brevity
                'total_found': len(matching_cols)
            }
        
        # Generate column completeness analysis
        for col in numeric_cols:
            if col in self.data.columns:
                col_data = self.data[col]
                completeness = ((len(col_data) - col_data.isnull().sum()) / len(col_data)) * 100
                
                analysis['column_completeness'][col] = {
                    'completeness_pct': round(completeness, 1),
                    'total_records': len(col_data),
                    'non_null_records': len(col_data) - col_data.isnull().sum(),
                    'data_type': str(col_data.dtype),
                    'mean_value': float(col_data.mean()) if completeness > 0 else None,
                    'std_value': float(col_data.std()) if completeness > 0 else None
                }
        
        # Enhanced field mapping recommendations
        recommendations = {
            'immediate_enhancements': [
                {
                    'field': 'TOT_GR_PT_REV',
                    'pcl': 'Income Statement (P8)',
                    'benefit': 'More standardized total gross patient revenue',
                    'priority': 'High'
                },
                {
                    'field': 'NET_PT_REV', 
                    'pcl': 'Income Statement (P8)',
                    'benefit': 'Net patient revenue after deductions',
                    'priority': 'High'
                },
                {
                    'field': 'TOT_ASSETS',
                    'pcl': 'Balance Sheet (P5)',
                    'benefit': 'Standard total assets field',
                    'priority': 'Medium'
                },
                {
                    'field': 'TOT_LIAB',
                    'pcl': 'Balance Sheet (P5)', 
                    'benefit': 'Standard total liabilities field',
                    'priority': 'Medium'
                }
            ],
            'future_opportunities': [
                {
                    'section': 'Cash Flow Statement (P9)',
                    'benefit': 'Complete cash flow analysis capabilities',
                    'fields_available': 99
                },
                {
                    'section': 'Revenue by Payer (P12)',
                    'benefit': 'Detailed payer mix analysis',
                    'fields_available': 1577
                },
                {
                    'section': 'Multi-Fund Balance Sheets (P5-P6)',
                    'benefit': 'Fund-specific financial analysis',
                    'fields_available': 254
                }
            ]
        }
        
        analysis['mapping_recommendations'] = recommendations
        
        # Calculate overall HADR alignment score
        pcl_fields_found = sum(1 for field in analysis['pcl_validated_mappings'].values() if field.get('found', False))
        total_pcl_fields = len(pcl_validated_fields)
        hadr_alignment_score = (pcl_fields_found / total_pcl_fields) * 100
        
        analysis['hadr_alignment'] = {
            'pcl_fields_mapped': pcl_fields_found,
            'total_pcl_fields': total_pcl_fields,
            'alignment_score_pct': round(hadr_alignment_score, 1),
            'status': 'Excellent' if hadr_alignment_score >= 80 else 'Good' if hadr_alignment_score >= 60 else 'Needs Improvement'
        }
        
        self.column_analysis = analysis
        
        logger.info(f"✅ HADR analysis completed:")
        logger.info(f"   📊 {analysis['dataset_overview']['total_columns']} total columns analyzed")
        logger.info(f"   🎯 {pcl_fields_found}/{total_pcl_fields} PCL-validated fields found ({hadr_alignment_score:.1f}%)")
        logger.info(f"   📈 {len([f for f in analysis['pcl_validated_mappings'].values() if f.get('completeness_pct') == 100])} fields with 100% completeness")
        
        return analysis

    def load_data(self, years: Optional[List[int]] = None, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare hospital financial data."""
        logger.info("📊 Loading hospital financial data...")
        
        if years:
            datasets = []
            for year in years:
                try:
                    year_data = self.data_loader.load_year_data(str(year))
                    year_data['fiscal_year'] = year
                    datasets.append(year_data)
                    logger.info(f"   ✓ {year}: {len(year_data):,} records")
                except Exception as e:
                    logger.warning(f"   ✗ {year}: {e}")
            
            self.data = pd.concat(datasets, ignore_index=True) if datasets else None
        else:
            self.data = self.data_loader.load_combined_data(include_year=True)
        
        # Use full dataset - no sampling for production analysis
        if sample_size:
            logger.warning(f"⚠️  Sampling disabled for production analysis. Using full dataset ({len(self.data):,} records)")
        
        self._prepare_data()
        logger.info(f"✅ Data ready: {len(self.data):,} records")
        
        return self.data
    
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Convert types
        if 'fiscal_year' in self.data.columns:
            self.data['fiscal_year'] = pd.to_numeric(self.data['fiscal_year'], errors='coerce')
        
        # Convert object columns to numeric where possible
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                converted = pd.to_numeric(self.data[col], errors='coerce')
                if not converted.isna().all():
                    self.data[col] = converted
        
        # Create derived features
        if 'total_revenue' in self.data.columns:
            self.data['hospital_size'] = pd.cut(
                self.data['total_revenue'], bins=5,
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
                include_lowest=True
            )
        
        if 'fiscal_year' in self.data.columns:
            conditions = [
                (self.data['fiscal_year'] <= 2007),
                (self.data['fiscal_year'].between(2008, 2009)),
                (self.data['fiscal_year'].between(2010, 2019)),
                (self.data['fiscal_year'] >= 2020)
            ]
            choices = ['Pre-Recession', 'Great Recession', 'Recovery Period', 'COVID Era']
            self.data['economic_period'] = np.select(conditions, choices, default='Unknown')
        
        self.metrics_calculator = FinancialMetricsCalculator(self.data)

    def generate_summary(self, year: Optional[str] = None) -> Dict:
        """Generate executive summary with HADR column analysis."""
        logger.info(f"📋 Generating executive summary{f' for {year}' if year else ''}...")
        
        # Ensure column analysis is available
        if self.column_analysis is None:
            self.analyze_hadr_columns()
        
        summary = {
            'overview': {
                'total_records': len(self.data),
                'year_range': (
                    int(self.data['fiscal_year'].min()),
                    int(self.data['fiscal_year'].max())
                ) if 'fiscal_year' in self.data.columns else None,
                'data_quality': (1 - self.data.isnull().sum().sum() / 
                               (len(self.data) * len(self.data.columns))) * 100
            },
            'hadr_analysis': self.column_analysis,
            'financial_health': self._assess_health(),
            'risk_assessment': self._assess_risk()
        }
        
        # Save summary with year-specific filename
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"executive_summary_{year}_{timestamp}.json" if year else f"executive_summary_{timestamp}.json"
        summary_file = self.config.reports_dir / filename
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Summary saved: {summary_file}")
        return summary

    def _assess_health(self) -> Dict:
        """Assess financial health."""
        if not self.metrics_calculator:
            return {'status': 'No metrics available'}
        
        metrics = self.metrics_calculator.calculate_all_metrics()
        health = {}
        
        for name, data in metrics.items():
            clean_data = data.dropna()
            if len(clean_data) > 0:
                health[name] = {
                    'median': float(clean_data.median()),
                    'mean': float(clean_data.mean()),
                    'q25': float(clean_data.quantile(0.25)),
                    'q75': float(clean_data.quantile(0.75))
                }
        
        return health
    
    def _assess_risk(self) -> Dict:
        """Assess financial risk."""
        distress_scores = identify_financial_distress(self.data)
        
        risk_levels = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk', 3: 'Critical Risk'}
        risk_dist = distress_scores.value_counts().to_dict()
        risk_dist = {risk_levels.get(k, f'Level {k}'): v for k, v in risk_dist.items()}
        
        total = len(distress_scores)
        risk_pct = {k: (v/total)*100 for k, v in risk_dist.items()}
        
        return {
            'distribution': risk_dist,
            'percentages': risk_pct,
            'high_risk_count': risk_dist.get('High Risk', 0) + risk_dist.get('Critical Risk', 0),
            'avg_risk_score': float(distress_scores.mean())
        }

    def create_dashboard(self, year: Optional[str] = None) -> go.Figure:
        """Create executive financial dashboard."""
        logger.info(f"📊 Creating executive dashboard{f' for {year}' if year else ''}...")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Financial Trends", "Risk Distribution", "Regional Performance",
                "Size vs Performance", "Economic Impact", "Key Metrics"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.12
        )
        
        # Save dashboard with year-specific filename
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"financial_dashboard_{year}_{timestamp}.html" if year else f"financial_dashboard_{timestamp}.html"
        dashboard_file = self.config.reports_dir / filename
        
        # Add basic visualization placeholder
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Sample Data"),
            row=1, col=1
        )
        
        fig.update_layout(
            title=f"Hospital Financial Intelligence Dashboard{f' - {year}' if year else ''}",
            height=800
        )
        
        fig.write_html(dashboard_file)
        logger.info(f"✅ Dashboard saved: {dashboard_file}")
        
        return fig

    def _create_eda_visualizations(self, year: str):
        """Create comprehensive EDA visualizations and save to files."""
        logger.info(f"📊 Creating comprehensive EDA visualizations for {year}...")
        
        # Initialize visualization engine
        from .visualizations import HospitalVisualizationEngine
        viz_engine = HospitalVisualizationEngine(self.config)
        
        try:
            # 1. Hospital Size Distribution
            if 'LICENSED_BED_SIZE' in self.data.columns:
                logger.info("📈 Creating hospital size distribution...")
                fig1 = viz_engine.create_distribution_plot(
                    self.data, 
                    'LICENSED_BED_SIZE',
                    title=f"Hospital Size Distribution ({year})",
                    save_file=f"hospital_size_distribution_{year}.html"
                )
            
            # 2. Geographic Distribution
            if 'COUNTY_NAME' in self.data.columns:
                logger.info("🗺️ Creating geographic distribution...")
                county_counts = self.data['COUNTY_NAME'].value_counts().head(15)
                
                fig2 = go.Figure(data=[
                    go.Bar(x=county_counts.values, y=county_counts.index, 
                           orientation='h', marker_color=viz_engine.colors['primary'])
                ])
                fig2.update_layout(
                    title=f"Top 15 Counties by Hospital Count ({year})",
                    xaxis_title="Number of Hospitals",
                    yaxis_title="County",
                    height=500
                )
                viz_engine._save_chart(fig2, f"geographic_distribution_{year}.html")
            
            # 3. Financial Metrics Analysis (if available)
            if self.metrics_calculator:
                logger.info("💰 Creating financial metrics visualizations...")
                metrics_result = self.metrics_calculator.calculate_all_metrics()
                
                # Convert dict of Series to DataFrame properly
                if isinstance(metrics_result, dict) and metrics_result:
                    # Convert dict of Series to DataFrame (proper way)
                    metrics_df = pd.DataFrame(metrics_result)
                    logger.info(f"✅ Converted metrics to DataFrame: {metrics_df.shape}")
                else:
                    metrics_df = metrics_result
                
                if not metrics_df.empty and len(metrics_df) > 0:
                    # Operating Margin Distribution
                    if 'operating_margin' in metrics_df.columns:
                        fig3 = viz_engine.create_distribution_plot(
                            metrics_df,
                            'operating_margin',
                            title=f"Operating Margin Distribution ({year})",
                            save_file=f"operating_margin_distribution_{year}.html"
                        )
                    
                    # Financial Health Dashboard
                    key_metrics = ['operating_margin', 'days_cash_on_hand', 'asset_turnover', 'debt_to_assets']
                    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
                    
                    if available_metrics:
                        logger.info("📊 Creating financial health dashboard...")
                        fig4 = self._create_financial_dashboard(metrics_df, available_metrics, year)
                        viz_engine._save_chart(fig4, f"financial_health_dashboard_{year}.html")
                else:
                    logger.warning("⚠️ No financial metrics data available for visualization")
            
            # 4. Data Quality Overview
            logger.info("🔍 Creating data quality overview...")
            self._create_data_quality_chart(year, viz_engine)
            
            # 5. HADR PCL Validation Chart
            if self.column_analysis:
                logger.info("🏥 Creating HADR PCL validation chart...")
                self._create_hadr_validation_chart(year, viz_engine)
            
            logger.info(f"✅ EDA visualizations completed for {year}")
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations for {year}: {e}")

    def _create_financial_dashboard(self, metrics_df: pd.DataFrame, metrics: List[str], year: str) -> go.Figure:
        """Create financial metrics dashboard subplot."""
        rows = 2
        cols = 2
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        
        for i, metric in enumerate(metrics[:4]):
            if metric in metrics_df.columns:
                data = metrics_df[metric].dropna()
                if len(data) > 0:
                    row, col = positions[i]
                    fig.add_trace(
                        go.Histogram(
                            x=data, 
                            name=metric, 
                            marker_color=colors[i], 
                            showlegend=False,
                            nbinsx=20
                        ),
                        row=row, col=col
                    )
                    logger.debug(f"Added histogram for {metric}: {len(data)} data points")
                else:
                    logger.warning(f"No data available for metric: {metric}")
        
        fig.update_layout(
            title_text=f"Financial Metrics Dashboard ({year})",
            height=600,
            showlegend=False
        )
        
        return fig

    def _create_data_quality_chart(self, year: str, viz_engine):
        """Create data quality overview chart."""
        # Calculate completion rates for key columns
        key_columns = [
            'REV_TOT_PT_REV', 'PY_TOT_OP_EXP', 'EQ_UNREST_FND_NET_INCOME',
            'PY_SP_PURP_FND_OTH_ASSETS', 'CASH_FLOW_SPECIFY_OTH_OP_L102',
            'PY_SP_PURP_FND_TOT_LIAB_EQ'
        ]
        
        completion_data = []
        for col in key_columns:
            if col in self.data.columns:
                completion_rate = (self.data[col].notna().sum() / len(self.data)) * 100
                completion_data.append({
                    'Field': col.replace('_', ' ').title(),
                    'Completion Rate (%)': completion_rate
                })
        
        if completion_data:
            completion_df = pd.DataFrame(completion_data)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=completion_df['Completion Rate (%)'],
                    y=completion_df['Field'],
                    orientation='h',
                    marker_color=viz_engine.colors['success']
                )
            ])
            
            fig.update_layout(
                title=f"HADR Field Completion Rates ({year})",
                xaxis_title="Data Completeness (%)",
                yaxis_title="HADR Field",
                height=400
            )
            
            viz_engine._save_chart(fig, f"data_quality_overview_{year}.html")

    def _create_hadr_validation_chart(self, year: str, viz_engine):
        """Create HADR PCL validation visualization."""
        if not self.column_analysis or 'pcl_validated_mappings' not in self.column_analysis:
            return
        
        mappings = self.column_analysis['pcl_validated_mappings']
        
        # Prepare data for visualization
        chart_data = []
        for field, info in mappings.items():
            chart_data.append({
                'Field': field.replace('_', ' ').title(),
                'Status': 'Found' if info['found'] else 'Missing',
                'Completeness': info.get('completeness_pct', 0) if info['found'] else 0,
                'PCL_Reference': info['pcl_reference']
            })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            # Create color mapping
            colors = ['green' if status == 'Found' else 'red' for status in chart_df['Status']]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=chart_df['Completeness'],
                    y=chart_df['Field'],
                    orientation='h',
                    marker_color=colors,
                    text=chart_df['PCL_Reference'],
                    textposition='inside'
                )
            ])
            
            fig.update_layout(
                title=f"HADR PCL Field Validation ({year})",
                xaxis_title="Completeness (%)",
                yaxis_title="PCL Field",
                height=400
            )
            
            viz_engine._save_chart(fig, f"hadr_pcl_validation_{year}.html")

    def analyze_payer_mix(self) -> Dict:
        """
        Phase 3: Payer Mix Analysis
        Analyze hospital revenue composition by payer type.
        """
        logger.info("💰 Analyzing payer mix composition...")
        
        payer_fields = [
            'MEDICARE_REV', 'MEDICAID_REV', 'PRIVATE_INS_REV', 
            'SELF_PAY_REV', 'OTHER_PAYER_REV'
        ]
        
        # Look for payer-related columns in the dataset
        available_payer_fields = []
        for field in payer_fields:
            possible_names = [
                field, f'REV_{field}', f'PT_{field}', f'{field}_PATIENT',
                f'NET_{field}', f'GROSS_{field}'
            ]
            for name in possible_names:
                if name in self.data.columns:
                    available_payer_fields.append((field, name))
                    break
        
        # Calculate total revenue for normalization
        total_rev_field = None
        for field in ['REV_TOT_PT_REV', 'TOTAL_PATIENT_REV', 'NET_PT_REV']:
            if field in self.data.columns:
                total_rev_field = field
                break
        
        payer_analysis = {
            'available_payer_fields': len(available_payer_fields),
            'total_revenue_field': total_rev_field,
            'payer_metrics': {},
            'diversity_scores': {},
            'government_exposure': {}
        }
        
        if available_payer_fields and total_rev_field:
            total_revenue = self.data[total_rev_field].fillna(0)
            
            for standard_name, actual_field in available_payer_fields:
                payer_revenue = self.data[actual_field].fillna(0)
                
                # Calculate payer dependency ratio
                payer_dependency = np.where(
                    total_revenue > 0, 
                    payer_revenue / total_revenue, 
                    0
                )
                
                payer_analysis['payer_metrics'][standard_name] = {
                    'mean_dependency': float(np.mean(payer_dependency)),
                    'median_dependency': float(np.median(payer_dependency)),
                    'std_dependency': float(np.std(payer_dependency)),
                    'hospitals_with_data': int(np.sum(payer_revenue > 0))
                }
            
            # Calculate payer diversity index (1 - Herfindahl Index)
            if len(available_payer_fields) >= 2:
                payer_revenues = np.array([
                    self.data[field].fillna(0) for _, field in available_payer_fields
                ]).T
                
                # Calculate diversity index for each hospital
                diversity_scores = []
                for hospital_revenues in payer_revenues:
                    total = np.sum(hospital_revenues)
                    if total > 0:
                        proportions = hospital_revenues / total
                        herfindahl = np.sum(proportions ** 2)
                        diversity = 1 - herfindahl
                        diversity_scores.append(diversity)
                
                if diversity_scores:
                    payer_analysis['diversity_scores'] = {
                        'mean_diversity': float(np.mean(diversity_scores)),
                        'median_diversity': float(np.median(diversity_scores)),
                        'hospitals_analyzed': len(diversity_scores)
                    }
            
            # Government exposure (Medicare + Medicaid)
            gov_fields = [(name, field) for name, field in available_payer_fields 
                         if 'MEDICARE' in name or 'MEDICAID' in name]
            
            if gov_fields:
                gov_revenue = sum(self.data[field].fillna(0) for _, field in gov_fields)
                gov_exposure = np.where(total_revenue > 0, gov_revenue / total_revenue, 0)
                
                payer_analysis['government_exposure'] = {
                    'mean_exposure': float(np.mean(gov_exposure)),
                    'median_exposure': float(np.median(gov_exposure)),
                    'high_exposure_hospitals': int(np.sum(gov_exposure > 0.6))
                }
        
        logger.info(f"✅ Payer mix analysis completed: {len(available_payer_fields)} payer types found")
        return payer_analysis

    def analyze_market_competition(self) -> Dict:
        """
        Phase 3: Market Competition Indicators
        Analyze hospital market position and competitive landscape.
        """
        logger.info("🏆 Analyzing market competition indicators...")
        
        market_analysis = {
            'geographic_concentration': {},
            'size_distribution': {},
            'service_concentration': {},
            'competitive_indicators': {}
        }
        
        # Geographic concentration analysis
        if 'COUNTY_NAME' in self.data.columns:
            county_counts = self.data['COUNTY_NAME'].value_counts()
            
            # Calculate market concentration (HHI) by county
            total_hospitals = len(self.data)
            county_shares = county_counts / total_hospitals
            geographic_hhi = np.sum(county_shares ** 2)
            
            market_analysis['geographic_concentration'] = {
                'total_counties': len(county_counts),
                'herfindahl_index': float(geographic_hhi),
                'top_county_share': float(county_shares.iloc[0]),
                'top_5_counties_share': float(county_shares.head(5).sum()),
                'counties_with_single_hospital': int(np.sum(county_counts == 1))
            }
        
        # Hospital size distribution analysis
        size_fields = ['LICENSED_BED_SIZE', 'STAFFED_BEDS', 'TOTAL_BEDS']
        size_field = None
        for field in size_fields:
            if field in self.data.columns:
                size_field = field
                break
        
        if size_field:
            sizes = self.data[size_field].dropna()
            if len(sizes) > 0:
                # Define size categories
                size_categories = pd.cut(
                    sizes, 
                    bins=[0, 25, 100, 300, 1000, np.inf],
                    labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
                )
                size_dist = size_categories.value_counts(normalize=True)
                
                market_analysis['size_distribution'] = {
                    'size_categories': size_dist.to_dict(),
                    'mean_size': float(sizes.mean()),
                    'median_size': float(sizes.median()),
                    'size_concentration_hhi': float(np.sum((size_dist) ** 2))
                }
        
        # Service line concentration (if service data available)
        service_fields = [col for col in self.data.columns 
                         if any(service in col.upper() for service in 
                               ['SURGERY', 'ICU', 'EMERGENCY', 'MATERNITY', 'CARDIAC'])]
        
        if service_fields:
            market_analysis['service_concentration'] = {
                'service_fields_available': len(service_fields),
                'hospitals_with_service_data': int(
                    self.data[service_fields].notna().any(axis=1).sum()
                )
            }
        
        # Competitive position indicators
        if 'REV_TOT_PT_REV' in self.data.columns and 'COUNTY_NAME' in self.data.columns:
            revenue = self.data['REV_TOT_PT_REV'].fillna(0)
            
            # Calculate market share within county for each hospital
            county_market_shares = []
            for county in self.data['COUNTY_NAME'].unique():
                if pd.notna(county):
                    county_data = self.data[self.data['COUNTY_NAME'] == county]
                    county_revenue = county_data['REV_TOT_PT_REV'].fillna(0)
                    total_county_revenue = county_revenue.sum()
                    
                    if total_county_revenue > 0:
                        county_shares = county_revenue / total_county_revenue
                        county_market_shares.extend(county_shares.tolist())
            
            if county_market_shares:
                market_analysis['competitive_indicators'] = {
                    'mean_market_share': float(np.mean(county_market_shares)),
                    'median_market_share': float(np.median(county_market_shares)),
                    'market_leaders': int(np.sum(np.array(county_market_shares) > 0.3)),
                    'competitive_markets': int(np.sum(np.array(county_market_shares) < 0.2))
                }
        
        logger.info("✅ Market competition analysis completed")
        return market_analysis

    def analyze_financial_health_comprehensive(self) -> Dict:
        """
        Phase 3: Comprehensive Financial Health Analysis
        Advanced financial health assessment using multiple dimensions.
        """
        logger.info("💊 Analyzing comprehensive financial health...")
        
        health_analysis = {
            'financial_stability_scores': {},
            'liquidity_health': {},
            'profitability_health': {},
            'efficiency_health': {},
            'leverage_health': {},
            'composite_health_score': {},
            'health_benchmarks': {},
            'distress_indicators': {}
        }
        
        if not self.metrics_calculator:
            health_analysis['status'] = 'No metrics calculator available'
            return health_analysis
        
        try:
            # Calculate all financial metrics
            metrics = self.metrics_calculator.calculate_all_metrics()
            if isinstance(metrics, dict) and metrics:
                metrics_df = pd.DataFrame(metrics)
            else:
                metrics_df = metrics
            
            if metrics_df.empty:
                health_analysis['status'] = 'No financial metrics available'
                return health_analysis
            
            # 1. Liquidity Health Assessment
            liquidity_metrics = ['current_ratio', 'quick_ratio', 'days_cash_on_hand']
            available_liquidity = [m for m in liquidity_metrics if m in metrics_df.columns]
            
            if available_liquidity:
                liquidity_scores = []
                for metric in available_liquidity:
                    data = metrics_df[metric].dropna()
                    if len(data) > 0:
                        # Define healthy thresholds
                        if metric == 'current_ratio':
                            healthy_threshold = 1.5  # Current ratio > 1.5 is healthy
                            score = (data >= healthy_threshold).mean()
                        elif metric == 'quick_ratio':
                            healthy_threshold = 1.0  # Quick ratio > 1.0 is healthy
                            score = (data >= healthy_threshold).mean()
                        elif metric == 'days_cash_on_hand':
                            healthy_threshold = 90  # 90+ days cash is healthy
                            score = (data >= healthy_threshold).mean()
                        else:
                            score = 0.5  # neutral score for unknown metrics
                        
                        liquidity_scores.append(score)
                        health_analysis['liquidity_health'][metric] = {
                            'healthy_percentage': float(score * 100),
                            'mean_value': float(data.mean()),
                            'median_value': float(data.median()),
                            'threshold': healthy_threshold
                        }
                
                health_analysis['liquidity_health']['overall_score'] = float(np.mean(liquidity_scores)) if liquidity_scores else 0.0
            
            # 2. Profitability Health Assessment
            profitability_metrics = ['operating_margin', 'total_margin', 'return_on_assets', 'return_on_equity']
            available_profitability = [m for m in profitability_metrics if m in metrics_df.columns]
            
            if available_profitability:
                profitability_scores = []
                for metric in available_profitability:
                    data = metrics_df[metric].dropna()
                    if len(data) > 0:
                        # Define healthy thresholds
                        if 'margin' in metric:
                            healthy_threshold = 0.02  # 2% margin is healthy
                        elif 'return' in metric:
                            healthy_threshold = 0.05  # 5% return is healthy
                        else:
                            healthy_threshold = 0.0
                        
                        score = (data >= healthy_threshold).mean()
                        profitability_scores.append(score)
                        health_analysis['profitability_health'][metric] = {
                            'healthy_percentage': float(score * 100),
                            'mean_value': float(data.mean()),
                            'median_value': float(data.median()),
                            'threshold': healthy_threshold
                        }
                
                health_analysis['profitability_health']['overall_score'] = float(np.mean(profitability_scores)) if profitability_scores else 0.0
            
            # 3. Efficiency Health Assessment
            efficiency_metrics = ['asset_turnover', 'receivables_turnover']
            available_efficiency = [m for m in efficiency_metrics if m in metrics_df.columns]
            
            if available_efficiency:
                efficiency_scores = []
                for metric in available_efficiency:
                    data = metrics_df[metric].dropna()
                    if len(data) > 0:
                        # Higher turnover is generally better
                        healthy_threshold = data.quantile(0.6)  # Above 60th percentile is healthy
                        score = (data >= healthy_threshold).mean()
                        efficiency_scores.append(score)
                        health_analysis['efficiency_health'][metric] = {
                            'healthy_percentage': float(score * 100),
                            'mean_value': float(data.mean()),
                            'median_value': float(data.median()),
                            'threshold': float(healthy_threshold)
                        }
                
                health_analysis['efficiency_health']['overall_score'] = float(np.mean(efficiency_scores)) if efficiency_scores else 0.0
            
            # 4. Leverage Health Assessment
            leverage_metrics = ['debt_to_equity', 'debt_to_assets', 'times_interest_earned']
            available_leverage = [m for m in leverage_metrics if m in metrics_df.columns]
            
            if available_leverage:
                leverage_scores = []
                for metric in available_leverage:
                    data = metrics_df[metric].dropna()
                    if len(data) > 0:
                        if 'debt_to' in metric:
                            # Lower debt ratios are healthier
                            healthy_threshold = 0.5  # Debt ratio < 50% is healthy
                            score = (data <= healthy_threshold).mean()
                        elif 'times_interest' in metric:
                            # Higher interest coverage is healthier
                            healthy_threshold = 2.5  # Coverage > 2.5x is healthy
                            score = (data >= healthy_threshold).mean()
                        else:
                            score = 0.5
                        
                        leverage_scores.append(score)
                        health_analysis['leverage_health'][metric] = {
                            'healthy_percentage': float(score * 100),
                            'mean_value': float(data.mean()),
                            'median_value': float(data.median()),
                            'threshold': healthy_threshold
                        }
                
                health_analysis['leverage_health']['overall_score'] = float(np.mean(leverage_scores)) if leverage_scores else 0.0
            
            # 5. Composite Health Score
            category_scores = []
            for category in ['liquidity_health', 'profitability_health', 'efficiency_health', 'leverage_health']:
                if category in health_analysis and 'overall_score' in health_analysis[category]:
                    category_scores.append(health_analysis[category]['overall_score'])
            
            if category_scores:
                composite_score = np.mean(category_scores)
                health_analysis['composite_health_score'] = {
                    'overall_score': float(composite_score),
                    'grade': self._get_health_grade(composite_score),
                    'categories_included': len(category_scores),
                    'interpretation': self._interpret_health_score(composite_score)
                }
            
            # 6. Health Benchmarks
            health_analysis['health_benchmarks'] = {
                'excellent': 'Score ≥ 0.8 (80%+ hospitals healthy in all categories)',
                'good': 'Score ≥ 0.6 (60%+ hospitals healthy in most categories)',
                'fair': 'Score ≥ 0.4 (40%+ hospitals healthy in some categories)',
                'poor': 'Score < 0.4 (Less than 40% hospitals healthy)'
            }
            
            # 7. Financial Distress Indicators
            try:
                from .financial_metrics import identify_financial_distress
                distress_scores = identify_financial_distress(self.data)
                if len(distress_scores) > 0:
                    health_analysis['distress_indicators'] = {
                        'hospitals_analyzed': len(distress_scores),
                        'low_risk_count': int(np.sum(distress_scores == 0)),
                        'moderate_risk_count': int(np.sum(distress_scores == 1)),
                        'high_risk_count': int(np.sum(distress_scores == 2)),
                        'critical_risk_count': int(np.sum(distress_scores == 3)),
                        'average_risk_score': float(distress_scores.mean())
                    }
            except Exception as e:
                logger.warning(f"Could not calculate distress indicators: {e}")
        
        except Exception as e:
            logger.warning(f"Error in comprehensive financial health analysis: {e}")
            health_analysis['error'] = str(e)
        
        logger.info("✅ Comprehensive financial health analysis completed")
        return health_analysis
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade."""
        if score >= 0.8:
            return 'A (Excellent)'
        elif score >= 0.6:
            return 'B (Good)'
        elif score >= 0.4:
            return 'C (Fair)'
        else:
            return 'D (Poor)'
    
    def _interpret_health_score(self, score: float) -> str:
        """Provide interpretation of health score."""
        if score >= 0.8:
            return 'Strong financial health across all categories'
        elif score >= 0.6:
            return 'Generally healthy with some areas for improvement'
        elif score >= 0.4:
            return 'Mixed financial health requiring attention'
        else:
            return 'Financial health concerns requiring immediate action'

    def analyze_quality_financial_integration(self) -> Dict:
        """
        Phase 3: Quality-Financial Integration
        Analyze relationships between quality indicators and financial performance.
        """
        logger.info("⭐ Analyzing quality-financial integration...")
        
        quality_analysis = {
            'quality_indicators_available': [],
            'financial_quality_correlations': {},
            'efficiency_metrics': {},
            'risk_indicators': {}
        }
        
        # Look for quality-related fields
        quality_fields = []
        quality_patterns = [
            'QUALITY', 'RATING', 'SCORE', 'VIOLATION', 'COMPLIANCE',
            'SAFETY', 'SATISFACTION', 'READMISSION', 'MORTALITY'
        ]
        
        for col in self.data.columns:
            if any(pattern in col.upper() for pattern in quality_patterns):
                quality_fields.append(col)
        
        quality_analysis['quality_indicators_available'] = quality_fields
        
        # Financial efficiency metrics
        if self.metrics_calculator:
            try:
                metrics = self.metrics_calculator.calculate_all_metrics()
                if isinstance(metrics, dict) and metrics:
                    metrics_df = pd.DataFrame(metrics)
                else:
                    metrics_df = metrics
                
                if not metrics_df.empty:
                    # Cost efficiency indicators
                    if 'asset_turnover' in metrics_df.columns:
                        asset_turnover = metrics_df['asset_turnover'].dropna()
                        if len(asset_turnover) > 0:
                            quality_analysis['efficiency_metrics']['asset_turnover'] = {
                                'mean': float(asset_turnover.mean()),
                                'median': float(asset_turnover.median()),
                                'high_efficiency_hospitals': int(np.sum(asset_turnover > asset_turnover.quantile(0.75)))
                            }
                    
                    # Operating efficiency
                    if 'operating_margin' in metrics_df.columns:
                        operating_margin = metrics_df['operating_margin'].dropna()
                        if len(operating_margin) > 0:
                            quality_analysis['efficiency_metrics']['operating_efficiency'] = {
                                'mean_margin': float(operating_margin.mean()),
                                'profitable_hospitals': int(np.sum(operating_margin > 0)),
                                'high_performers': int(np.sum(operating_margin > 0.05))
                            }
            except Exception as e:
                logger.warning(f"Could not calculate efficiency metrics: {e}")
        
        # Financial risk indicators
        risk_fields = [col for col in self.data.columns 
                      if any(risk in col.upper() for risk in 
                            ['DEBT', 'LIABILITY', 'LOSS', 'DEFICIT'])]
        
        if risk_fields:
            quality_analysis['risk_indicators'] = {
                'risk_fields_available': len(risk_fields),
                'hospitals_with_risk_data': int(
                    self.data[risk_fields].notna().any(axis=1).sum()
                )
            }
        
        # Quality-financial correlations (if quality data available)
        if quality_fields and self.metrics_calculator:
            try:
                # Calculate correlation between quality metrics and financial performance
                correlations = {}
                financial_metrics = ['operating_margin', 'asset_turnover', 'debt_to_assets']
                
                for quality_field in quality_fields[:3]:  # Limit to first 3 for performance
                    quality_data = self.data[quality_field].dropna()
                    if len(quality_data) > 10:  # Need sufficient data
                        for fin_metric in financial_metrics:
                            if hasattr(self.metrics_calculator, f'calculate_{fin_metric}'):
                                try:
                                    fin_data = getattr(self.metrics_calculator, f'calculate_{fin_metric}')()
                                    if len(fin_data) > 0:
                                        # Align data
                                        common_index = quality_data.index.intersection(fin_data.index)
                                        if len(common_index) > 10:
                                            correlation = np.corrcoef(
                                                quality_data.loc[common_index],
                                                fin_data.loc[common_index]
                                            )[0, 1]
                                            
                                            if not np.isnan(correlation):
                                                correlations[f"{quality_field}_{fin_metric}"] = float(correlation)
                                except Exception:
                                    continue
                
                quality_analysis['financial_quality_correlations'] = correlations
            except Exception as e:
                logger.warning(f"Could not calculate quality-financial correlations: {e}")
        
        logger.info(f"✅ Quality-financial integration analysis completed: {len(quality_fields)} quality indicators found")
        return quality_analysis

    def run_phase3_healthcare_analysis(self) -> Dict:
        """
        Execute complete Phase 3 Healthcare-Specific Enhancement Analysis.
        This method runs all Phase 3 components from the project plan.
        """
        logger.info("🏥 Starting Phase 3 Healthcare-Specific Enhancement Analysis...")
        
        phase3_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'phase3_components': {
                'payer_mix_analysis': {},
                'market_competition_indicators': {},
                'quality_financial_integration': {}
            },
            'phase3_summary': {}
        }
        
        try:
            # 1. Comprehensive Financial Health Analysis
            logger.info("💊 Running Comprehensive Financial Health Analysis...")
            financial_health_analysis = self.analyze_financial_health_comprehensive()
            phase3_results['phase3_components']['financial_health_analysis'] = financial_health_analysis
            
            # 2. Payer Mix Analysis
            logger.info("📊 Running Payer Mix Analysis...")
            payer_analysis = self.analyze_payer_mix()
            phase3_results['phase3_components']['payer_mix_analysis'] = payer_analysis
            
            # 3. Market Competition Indicators
            logger.info("🏆 Running Market Competition Analysis...")
            market_analysis = self.analyze_market_competition()
            phase3_results['phase3_components']['market_competition_indicators'] = market_analysis
            
            # 4. Quality-Financial Integration
            logger.info("⭐ Running Quality-Financial Integration Analysis...")
            quality_analysis = self.analyze_quality_financial_integration()
            phase3_results['phase3_components']['quality_financial_integration'] = quality_analysis
            
            # Generate Phase 3 summary
            phase3_results['phase3_summary'] = {
                'financial_health_score': financial_health_analysis.get('composite_health_score', {}).get('overall_score', 0),
                'financial_health_grade': financial_health_analysis.get('composite_health_score', {}).get('grade', 'N/A'),
                'payer_fields_found': payer_analysis.get('available_payer_fields', 0),
                'market_counties_analyzed': market_analysis.get('geographic_concentration', {}).get('total_counties', 0),
                'quality_indicators_found': len(quality_analysis.get('quality_indicators_available', [])),
                'efficiency_metrics_calculated': len(quality_analysis.get('efficiency_metrics', {})),
                'phase3_completion_status': 'completed'
            }
            
            logger.info("✅ Phase 3 Healthcare-Specific Enhancement Analysis completed successfully")
            logger.info(f"   💊 Financial Health: {phase3_results['phase3_summary']['financial_health_grade']} ({phase3_results['phase3_summary']['financial_health_score']:.3f})")
            logger.info(f"   💰 Payer fields: {phase3_results['phase3_summary']['payer_fields_found']}")
            logger.info(f"   🌎 Counties: {phase3_results['phase3_summary']['market_counties_analyzed']}")
            logger.info(f"   ⭐ Quality indicators: {phase3_results['phase3_summary']['quality_indicators_found']}")
            
        except Exception as e:
            logger.error(f"❌ Phase 3 analysis failed: {e}")
            phase3_results['phase3_summary'] = {
                'phase3_completion_status': 'failed',
                'error_message': str(e)
            }
        
        return phase3_results

    def run_single_year_analysis(self, year: str) -> Dict:
        """Run complete EDA for a single fiscal year with HADR column analysis."""
        logger.info(f"🚀 Starting comprehensive analysis for year {year}...")
        
        try:
            # Load data for the year
            self.load_data(years=[int(year)])
            
            if self.data is None or len(self.data) == 0:
                logger.error(f"No data loaded for year {year}")
                return {}
            
            # Perform HADR column analysis
            hadr_analysis = self.analyze_hadr_columns()
            
            # Phase 3: Run Healthcare-Specific Enhancement Analysis
            phase3_analysis = self.run_phase3_healthcare_analysis()
            
            # Create comprehensive visualizations
            self._create_eda_visualizations(year)
            
            # Generate summary and dashboard
            summary = self.generate_summary(year)
            dashboard = self.create_dashboard(year)
            
            # Calculate enhanced analysis results
            data_quality = summary['overview']['data_quality']
            records_analyzed = summary['overview']['total_records']
            pcl_alignment = hadr_analysis['hadr_alignment']['alignment_score_pct']
            
            result = {
                'year': year,
                'records_analyzed': records_analyzed,
                'data_quality_score': data_quality,
                'hadr_alignment_score': pcl_alignment,
                'pcl_fields_mapped': hadr_analysis['hadr_alignment']['pcl_fields_mapped'],
                'total_columns': hadr_analysis['dataset_overview']['total_columns'],
                'numeric_columns': hadr_analysis['dataset_overview']['numeric_columns'],
                'financial_health': summary['financial_health'],
                'risk_assessment': summary['risk_assessment'],
                'column_analysis': hadr_analysis,
                'phase3_healthcare_analysis': phase3_analysis
            }
            
            logger.info(f"✅ Year {year} analysis completed:")
            logger.info(f"   📊 Records: {records_analyzed:,}")
            logger.info(f"   🎯 Data Quality: {data_quality:.1f}%")
            logger.info(f"   🏥 HADR Alignment: {pcl_alignment:.1f}%")
            logger.info(f"   📈 PCL Fields: {hadr_analysis['hadr_alignment']['pcl_fields_mapped']}/6")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze year {year}: {e}")
            return {}

    def run_analysis(self, years: list = None, sample_size: int = None) -> dict:
        """Run complete financial analysis (legacy method for compatibility)."""
        logger.info("🚀 Starting comprehensive financial analysis...")
        
        # Load data
        self.load_data(years, sample_size)
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded for analysis")
        
        # Perform HADR analysis
        hadr_analysis = self.analyze_hadr_columns()
        
        # Phase 3: Run Healthcare-Specific Enhancement Analysis
        phase3_analysis = self.run_phase3_healthcare_analysis()
        
        # Generate components
        summary = self.generate_summary()
        dashboard = self.create_dashboard()
        
        # Calculate metrics
        financial_metrics = self.metrics_calculator.calculate_all_metrics()
        logger.info(f"📊 Calculated {len(financial_metrics)} financial metrics")
        
        results = {
            'summary': summary,
            'hadr_analysis': hadr_analysis,
            'phase3_healthcare_analysis': phase3_analysis,
            'metrics_calculated': len(financial_metrics),
            'records_analyzed': len(self.data),
            'data_quality_score': summary['overview']['data_quality'],
            'hadr_alignment_score': hadr_analysis['hadr_alignment']['alignment_score_pct'],
            'outputs': {
                'dashboard_file': f"Dashboard saved to {self.config.reports_dir}"
            }
        }
        
        logger.info("✅ Analysis completed successfully")
        logger.info(f"   📊 Phase 3 Payer Fields: {phase3_analysis['phase3_summary'].get('payer_fields_found', 0)}")
        logger.info(f"   🌎 Phase 3 Counties: {phase3_analysis['phase3_summary'].get('market_counties_analyzed', 0)}")
        logger.info(f"   ⭐ Phase 3 Quality Indicators: {phase3_analysis['phase3_summary'].get('quality_indicators_found', 0)}")
        return results 