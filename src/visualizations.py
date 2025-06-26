"""
Hospital Financial Intelligence - Visualization Engine

Professional business visualizations with modern design and configurable output paths.
Docker-ready with environment variable support.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from typing import Optional, Dict, List, Union
import logging

from .config import Config, get_config

logger = logging.getLogger(__name__)


class HospitalVisualizationEngine:
    """Professional visualization engine for hospital financial data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize visualization engine with configuration.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.output_dir = self.config.visuals_dir / "eda_charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modern professional color scheme (2024 trends)
        self.colors = {
            'primary': '#2E4F6B',     # Professional blue
            'secondary': '#8B5A3C',   # Warm brown
            'accent': '#F4A261',      # Amber orange
            'success': '#2A9D8F',     # Teal green
            'warning': '#E76F51',     # Coral red
            'neutral': '#495867',     # Charcoal
            'light': '#F8F9FA',       # Light background
            'text': '#1A1A1A'         # Dark text
        }
        
        self.palette = [
            self.colors['primary'], self.colors['secondary'], self.colors['accent'],
            self.colors['success'], self.colors['warning'], self.colors['neutral']
        ]
        
        # Configure professional theme
        self._setup_theme()
        
        logger.info(f"📊 Visualization engine initialized")
        logger.info(f"   📁 Output directory: {self.output_dir}")
    
    def _setup_theme(self):
        """Setup professional business theme for all charts."""
        # Create custom theme
        template = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': self.colors['text']},
                'title': {'font': {'size': 18, 'color': self.colors['text']}, 'x': 0.5},
                'colorway': self.palette,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'t': 80, 'r': 40, 'b': 60, 'l': 80},
                'xaxis': {
                    'linecolor': self.colors['neutral'],
                    'gridcolor': '#E8E8E8',
                    'tickcolor': self.colors['neutral'],
                    'title': {'font': {'size': 14, 'color': self.colors['text']}}
                },
                'yaxis': {
                    'linecolor': self.colors['neutral'],
                    'gridcolor': '#E8E8E8',
                    'tickcolor': self.colors['neutral'],
                    'title': {'font': {'size': 14, 'color': self.colors['text']}}
                }
            }
        }
        
        pio.templates['hospital_professional'] = template
        pio.templates.default = 'hospital_professional'
    
    def create_financial_trends(self, data: pd.DataFrame, metrics: List[str], 
                               title: str = "Financial Trends Over Time",
                               save_file: Optional[str] = None) -> go.Figure:
        """
        Create financial trends visualization.
        
        Args:
            data: DataFrame with financial data
            metrics: List of metric column names to plot
            title: Chart title
            save_file: Optional filename to save chart
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        if 'fiscal_year' not in data.columns:
            logger.warning("No fiscal_year column found for trends analysis")
            return fig
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric in data.columns:
                yearly_data = data.groupby('fiscal_year')[metric].mean()
                
                fig.add_trace(go.Scatter(
                    x=yearly_data.index,
                    y=yearly_data.values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=self.palette[i % len(self.palette)], width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Fiscal Year",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        if save_file:
            self._save_chart(fig, save_file)
        
        return fig
    
    def create_distribution_plot(self, data: pd.DataFrame, metric: str,
                               group_by: Optional[str] = None,
                               title: Optional[str] = None,
                               save_file: Optional[str] = None) -> go.Figure:
        """
        Create distribution visualization (histogram or box plot).
        
        Args:
            data: DataFrame with data
            metric: Column name for the metric to analyze
            group_by: Optional column to group by
            title: Chart title
            save_file: Optional filename to save chart
            
        Returns:
            Plotly figure object
        """
        if metric not in data.columns:
            logger.warning(f"Metric '{metric}' not found in data")
            return go.Figure()
        
        title = title or f"Distribution of {metric.replace('_', ' ').title()}"
        
        if group_by and group_by in data.columns:
            # Box plot by group
            fig = px.box(data, x=group_by, y=metric, 
                        title=title,
                        color=group_by,
                        color_discrete_sequence=self.palette)
        else:
            # Simple histogram
            fig = px.histogram(data, x=metric, 
                             title=title,
                             color_discrete_sequence=[self.colors['primary']])
        
        fig.update_layout(height=500)
        
        if save_file:
            self._save_chart(fig, save_file)
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                  metrics: Optional[List[str]] = None,
                                  title: str = "Financial Metrics Correlation",
                                  save_file: Optional[str] = None) -> go.Figure:
        """
        Create correlation heatmap for financial metrics.
        
        Args:
            data: DataFrame with financial data
            metrics: List of metrics to include. If None, uses all numeric columns.
            title: Chart title
            save_file: Optional filename to save chart
            
        Returns:
            Plotly figure object
        """
        # Select numeric columns
        if metrics:
            numeric_data = data[metrics].select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning("No numeric data found for correlation analysis")
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600
        )
        
        if save_file:
            self._save_chart(fig, save_file)
        
        return fig
    
    def create_performance_comparison(self, data: pd.DataFrame,
                                    x_metric: str, y_metric: str,
                                    size_metric: Optional[str] = None,
                                    color_metric: Optional[str] = None,
                                    title: Optional[str] = None,
                                    save_file: Optional[str] = None) -> go.Figure:
        """
        Create scatter plot for performance comparison.
        
        Args:
            data: DataFrame with financial data
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            size_metric: Optional metric for bubble size
            color_metric: Optional metric for color coding
            title: Chart title
            save_file: Optional filename to save chart
            
        Returns:
            Plotly figure object
        """
        missing_metrics = [m for m in [x_metric, y_metric] if m not in data.columns]
        if missing_metrics:
            logger.warning(f"Metrics not found: {missing_metrics}")
            return go.Figure()
        
        title = title or f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}"
        
        # Create scatter plot
        fig = px.scatter(data, x=x_metric, y=y_metric,
                        size=size_metric if size_metric and size_metric in data.columns else None,
                        color=color_metric if color_metric and color_metric in data.columns else None,
                        title=title,
                        color_continuous_scale='Viridis')
        
        # Update colors if no color metric
        if not color_metric or color_metric not in data.columns:
            fig.update_traces(marker=dict(color=self.colors['primary']))
        
        fig.update_layout(height=500)
        
        if save_file:
            self._save_chart(fig, save_file)
        
        return fig
    
    def create_risk_dashboard(self, data: pd.DataFrame,
                            risk_scores: pd.Series,
                            title: str = "Financial Risk Assessment Dashboard",
                            save_file: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive risk dashboard.
        
        Args:
            data: DataFrame with financial data
            risk_scores: Series with risk scores
            title: Dashboard title
            save_file: Optional filename to save chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Risk Distribution", "Risk by Hospital Size", 
                          "Risk Trends", "High Risk Hospitals"],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Risk distribution pie chart
        risk_levels = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}
        risk_counts = risk_scores.value_counts()
        
        fig.add_trace(go.Pie(
            labels=[risk_levels.get(i, f'Level {i}') for i in risk_counts.index],
            values=risk_counts.values,
            name="Risk Distribution"
        ), row=1, col=1)
        
        # 2. Risk by hospital size (if available)
        if 'hospital_size' in data.columns:
            size_risk = data.groupby('hospital_size')[risk_scores.name if risk_scores.name else 'risk_score'].mean()
            fig.add_trace(go.Bar(
                x=size_risk.index,
                y=size_risk.values,
                name="Risk by Size",
                marker_color=self.colors['warning']
            ), row=1, col=2)
        
        # 3. Risk trends over time (if fiscal_year available)
        if 'fiscal_year' in data.columns:
            yearly_risk = data.groupby('fiscal_year')[risk_scores.name if risk_scores.name else 'risk_score'].mean()
            fig.add_trace(go.Scatter(
                x=yearly_risk.index,
                y=yearly_risk.values,
                mode='lines+markers',
                name="Risk Trends"
            ), row=2, col=1)
        
        # 4. High risk count
        high_risk_count = (risk_scores >= 2).sum()
        fig.add_trace(go.Bar(
            x=['High Risk Hospitals'],
            y=[high_risk_count],
            name="High Risk Count",
            marker_color=self.colors['warning']
        ), row=2, col=2)
        
        fig.update_layout(title=title, height=800)
        
        if save_file:
            self._save_chart(fig, save_file)
        
        return fig
    
    def _save_chart(self, fig: go.Figure, filename: str):
        """
        Save chart to output directory.
        
        Args:
            fig: Plotly figure to save
            filename: Filename (with or without .html extension)
        """
        if not filename.endswith('.html'):
            filename += '.html'
        
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        logger.info(f"📊 Chart saved: {filepath}")
    
    def create_executive_summary_charts(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create a complete set of executive summary charts.
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            Dictionary of chart names to figures
        """
        charts = {}
        
        # Get numeric columns for analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'fiscal_year' in numeric_cols:
            numeric_cols.remove('fiscal_year')
        
        # Key financial metrics (if available)
        revenue_metrics = [col for col in numeric_cols if 'revenue' in col.lower()][:3]
        if revenue_metrics:
            charts['revenue_trends'] = self.create_financial_trends(
                data, revenue_metrics, "Revenue Trends", "revenue_trends.html"
            )
        
        # Performance comparison
        if len(numeric_cols) >= 2:
            charts['performance_comparison'] = self.create_performance_comparison(
                data, numeric_cols[0], numeric_cols[1], 
                title="Financial Performance Comparison",
                save_file="performance_comparison.html"
            )
        
        # Correlation analysis
        if len(numeric_cols) >= 3:
            charts['correlation_heatmap'] = self.create_correlation_heatmap(
                data, numeric_cols[:10],  # Limit to first 10 metrics
                save_file="correlation_heatmap.html"
            )
        
        # Distribution analysis
        if numeric_cols:
            charts['distribution'] = self.create_distribution_plot(
                data, numeric_cols[0],
                group_by='hospital_size' if 'hospital_size' in data.columns else None,
                save_file="metric_distribution.html"
            )
        
        logger.info(f"✅ Created {len(charts)} executive summary charts")
        return charts 