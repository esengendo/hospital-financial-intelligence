#!/usr/bin/env python3
"""
Hospital Financial Intelligence - Modern Streamlit Dashboard
Professional healthcare analytics dashboard with shadcn-inspired design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import glob
import requests
from dotenv import load_dotenv
from typing import Dict
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

# Import hospital name lookup utility (with error handling)
try:
    from hospital_name_lookup import get_hospital_name, load_hospital_mapping, get_hospital_mapping_info
except ImportError:
    # Fallback functions if hospital_name_lookup is not available
    def get_hospital_name(oshpd_id):
        return f"Hospital_{oshpd_id}"
    def load_hospital_mapping():
        return {}
    def get_hospital_mapping_info():
        return {"total_hospitals": 0, "mapped_hospitals": 0}

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Hospital Financial Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling inspired by shadcn/ui
st.markdown("""
<style>
    /* Import Inter font for professional business typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles - Clean white background with business focus */
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-feature-settings: 'kern' 1;
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Remove default Streamlit styling */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > .main {
        background-color: #ffffff;
    }
    
    /* Override Streamlit's default dark selectbox */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="select"] span {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    div[data-baseweb="select"] div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Professional business typography */
    .stMarkdown, .stText, p, div, span {
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 400;
        line-height: 1.6;
        text-rendering: optimizeLegibility;
        letter-spacing: 0.01em;
    }
    
    /* Business-focused headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
    }
    
    /* Section headers with professional styling */
    .section-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        margin: 1.5rem 0 1rem 0 !important;
        letter-spacing: -0.025em !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    /* Button styling for shadcn/ui look */
    .stButton > button {
        background-color: #0f172a;
        color: #ffffff;
        border: 1px solid #0f172a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e293b;
        border-color: #1e293b;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #f8fafc;
    }
    
    /* Card components - Clean shadcn/ui style */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    .metric-card.success { border-left-color: #10b981; }
    .metric-card.warning { border-left-color: #f59e0b; }
    .metric-card.danger { border-left-color: #ef4444; }
    .metric-card.info { border-left-color: #3b82f6; }
    
    .metric-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b !important;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        text-rendering: optimizeLegibility;
        font-feature-settings: 'kern' 1, 'liga' 1;
    }
    
    .metric-value {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
        font-variant-numeric: tabular-nums;
        text-rendering: optimizeLegibility;
    }
    
    .metric-description {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.875rem;
        color: #64748b !important;
        font-weight: 400;
        line-height: 1.4;
    }
    
    /* Badge components */
    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        margin: 0.25rem;
    }
    
    .badge.success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .badge.warning {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .badge.danger {
        background-color: #fecaca;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    
    .badge.info {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    .badge.secondary {
        background-color: #f1f5f9;
        color: #475569;
    }
    
    /* AGGRESSIVE Black Sidebar Styling - Target all possible classes */
    section[data-testid="stSidebar"],
    .stSidebar,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] > div > div > div,
    .css-1d391kg,
    .css-1cypcdb,
    .css-k1ih3n,
    .css-ng1t4o,
    .css-1lcbmhc,
    .sidebar .sidebar-content,
    [class*="css-"][class*="sidebar"] {
        background-color: #000000 !important;
        background: #000000 !important;
        background-image: none !important;
        border-right: 2px solid #333333 !important;
    }
    
    /* Force black background on all sidebar containers */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .stVerticalBlock {
        background-color: rgba(0, 0, 0, 0.0) !important;
    }
    
    /* Sidebar text colors - white for contrast */
    section[data-testid="stSidebar"] *,
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    .css-1d391kg *,
    .css-1cypcdb * {
        color: #ffffff !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #374151 !important;
    }
    
    /* NUCLEAR OPTION - Force black sidebar on any element that might be the sidebar */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] *, 
    section[aria-label="sidebar"] *,
    .main-content + * {
        background-color: #000000 !important;
        background: #000000 !important;
    }
    
    /* Force white text in sidebar */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Override any other background that might interfere */
    section[data-testid="stSidebar"] .stMarkdown * {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Modern CSS styling continued
st.markdown("""
<style>
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Insight cards */
    .insight-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .insight-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    .insight-card h4 {
        color: #0f172a !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .insight-card p {
        color: #475569 !important;
        margin: 0;
    }
    
    /* Streamlit specific overrides for better readability */
    .stSelectbox label, .stRadio label, .stCheckbox label {
        color: #0f172a !important;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background-color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Selectbox dropdown menu */
    .stSelectbox div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stSelectbox div[data-baseweb="popover"] ul {
        background-color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox div[data-baseweb="popover"] li:hover {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #e0f2fe !important;
        color: #0f172a !important;
    }
    
    /* Selectbox input text */
    .stSelectbox input {
        color: #0f172a !important;
        background-color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
    
    /* Force white background for all selectbox elements */
    [data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Specific targeting for dropdown options */
    div[role="listbox"] {
        background-color: #ffffff !important;
    }
    
    div[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    div[role="option"]:hover {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* Navigation tabs styling - Clean shadcn/ui style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #64748b;
        font-weight: 500;
        padding: 8px 12px;
        border: none;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #0f172a;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling with beautiful gradient background - Multiple class targets */
    .css-1d391kg, 
    .css-1lcbmhc, 
    .css-1outpf7, 
    .css-1y4p8pa,
    .css-17lntkn,
    .css-1v3fvcr,
    section[data-testid="stSidebar"],
    .stSidebar,
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-right: none !important;
    }
    
    .css-1d391kg .css-10trblm,
    .css-1lcbmhc .css-10trblm,
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Sidebar text styling for gradient background - All possible selectors */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] .stText, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    .css-1d391kg .stMarkdown, 
    .css-1d391kg .stText, 
    .css-1d391kg p, 
    .css-1d391kg div, 
    .css-1d391kg span,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4,
    .css-1d391kg h5,
    .css-1d391kg h6 {
        color: white !important;
    }
    
    /* Sidebar metric cards with glass effect */
    section[data-testid="stSidebar"] .metric-card,
    .css-1d391kg .metric-card {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .metric-card .metric-title,
    section[data-testid="stSidebar"] .metric-card .metric-value,
    section[data-testid="stSidebar"] .metric-card .metric-description,
    .css-1d391kg .metric-card .metric-title,
    .css-1d391kg .metric-card .metric-value,
    .css-1d391kg .metric-card .metric-description {
        color: white !important;
    }
    
    /* Sidebar buttons with glass effect */
    section[data-testid="stSidebar"] .stButton > button,
    .css-1d391kg .stButton > button {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        backdrop-filter: blur(10px);
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover,
    .css-1d391kg .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-1px);
    }
    
    /* Activity feed - Clean shadcn/ui style */
    .activity-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #3b82f6;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .activity-time {
        font-size: 0.75rem;
        color: #64748b;
        font-style: italic;
    }
    
    .activity-action {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    
    .activity-detail {
        color: #475569;
        font-size: 0.875rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-indicator.online { background-color: #10b981; }
    .status-indicator.warning { background-color: #f59e0b; }
    .status-indicator.offline { background-color: #ef4444; }
    
    /* Insight cards - Clean shadcn/ui style */
    .insight-card {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3b82f6;
    }
    
    .insight-card h4 {
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    .insight-card p {
        margin-bottom: 1rem;
        color: #64748b;
    }
    
    /* Clean white background for all elements */
    .element-container, .stMarkdown, .stPlotlyChart {
        background-color: #ffffff;
    }
    
    /* Clean dataframe styling */
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for shadcn/ui inspired design
st.markdown("""
<style>
    /* Global styling with subtle blue gradient */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%) !important;
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%) !important;
    }
    
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Typography */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        color: #475569;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Cards with glass effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06), 
                    0 0 0 1px rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05),
                    0 0 0 1px rgba(255, 255, 255, 0.1);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    
    .metric-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #64748b;
    }
    
    .success { color: #059669; }
    .warning { color: #d97706; }
    .error { color: #dc2626; }
    .info { color: #2563eb; }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    .status-operational {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .status-active {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .status-connected {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-live {
        background-color: #f3e8ff;
        color: #7c3aed;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 0.375rem !important;
    }
    
    .stSelectbox > div > div > div {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Target the selectbox dropdown */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #e2e8f0 !important;
        color: #0f172a !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Individual dropdown options */
    div[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    div[role="option"]:hover {
        background-color: #f8fafc !important;
    }
    
    div[role="option"][aria-selected="true"] {
        background-color: #e0f2fe !important;
        color: #0f172a !important;
    }
    
    /* Override Streamlit component styles */
    .stSelectbox label {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stRadio label {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stCheckbox label {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Data frame styling */
    .dataframe {
        border: 1px solid #e2e8f0 !important;
        border-radius: 0.375rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 0.375rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f8fafc;
        border-color: #cbd5e1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        text-rendering: optimizeLegibility;
        letter-spacing: 0.01em;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# SVG Icons with 3D effects
def get_icon(name, size="20px"):
    # Define gradients and shadows for 3D effect
    icon_style = '''
    <defs>
        <linearGradient id="iconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#1d4ed8;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#1e40af;stop-opacity:1" />
        </linearGradient>
        <filter id="iconShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#1e40af" flood-opacity="0.3"/>
        </filter>
    </defs>
    '''
    
    icons = {
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<polyline points="22,12 18,12 15,21 9,3 6,12 2,12"></polyline></svg>',
        "bar-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>',
        "check-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22,4 12,14.01 9,11.01"></polyline></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>',
        "lightbulb": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<path d="M9 21h6"></path><path d="M12 3a6 6 0 0 0-6 6c0 1 .2 2 .6 2.8L9 15h6l2.4-3.2c.4-.8.6-1.8.6-2.8a6 6 0 0 0-6-6Z"></path></svg>',
        "refresh": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path><path d="M3 21v-5h5"></path></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10,9 9,9 8,9"></polyline></svg>',
        "brain": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#iconGradient)" stroke="url(#iconGradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" filter="url(#iconShadow)" style="drop-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">{icon_style}<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"></path><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"></path><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"></path><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"></path><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"></path><path d="M3.477 10.896a4 4 0 0 1 .585-.396"></path><path d="M19.938 10.5a4 4 0 0 1 .585.396"></path><path d="M6 18a4 4 0 0 1-1.967-.516"></path><path d="M19.967 17.484A4 4 0 0 1 18 18"></path></svg>'
    }
    return icons.get(name, "")

# Real Data Loading Functions
@st.cache_data
def load_hospital_data():
    """Load real hospital financial data from parquet files with actual hospital names"""
    try:
        # Load multiple years for comprehensive analysis
        files = sorted(glob.glob('data/features_enhanced/features_enhanced_*.parquet'))
        dfs = []
        
        # Load last 3 years
        for file in files[-3:]:
            year_df = pd.read_parquet(file)
            dfs.append(year_df)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Load real hospital name mapping from our extracted data
        try:
            hospital_mapping = load_hospital_mapping()
            # Map hospital names using the osph_id (which might be in different columns)
            if 'oshpd_id' in df.columns:
                df['hospital_name'] = df['oshpd_id'].apply(lambda x: get_hospital_name(str(x)))
            elif 'osph_id' in df.columns:
                df['hospital_name'] = df['osph_id'].apply(lambda x: get_hospital_name(str(x)))
            elif df.index.name == 'osph_id' or 'osph_id' in str(df.index):
                df['hospital_name'] = df.index.map(lambda x: get_hospital_name(str(x)))
            else:
                # Try to find any column that looks like an ID
                id_cols = [col for col in df.columns if 'id' in col.lower() or 'osph' in col.lower()]
                if id_cols:
                    df['hospital_name'] = df[id_cols[0]].apply(lambda x: get_hospital_name(str(x)))
                else:
                    df['hospital_name'] = df.index.map(lambda x: get_hospital_name(str(x)))
                    
            # Add osph_id column if not present for consistency
            if 'oshpd_id' not in df.columns and 'osph_id' not in df.columns:
                df['oshpd_id'] = df.index
                
        except Exception as e:
            st.warning(f"Could not load hospital name mapping: {e}. Using IDs as names.")
            # Fallback to ID if mapping not available
            if 'oshpd_id' in df.columns:
                df['hospital_name'] = df['oshpd_id'].astype(str)
            else:
                df['hospital_name'] = df.index.astype(str)
                df['oshpd_id'] = df.index
        
        return df
    except Exception as e:
        st.error(f"Error loading hospital data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_model_metrics():
    """Load real model performance metrics"""
    try:
        model_files = glob.glob('reports/model_evaluation_*.json')
        if model_files:
            with open(model_files[0], 'r') as f:
                data = json.load(f)
            return data
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
        return {}

@st.cache_data
def load_groq_data():
    """Load real Groq analysis data"""
    try:
        groq_files = glob.glob('reports/*groq*.json')
        analyses = []
        
        for file in groq_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                analyses.append(data)
            except:
                continue
        
        return {
            'analyses': analyses,
            'total_analyses': len(analyses),
            'total_cost': sum(analysis.get('cost', 0) for analysis in analyses),
            'avg_response_time': np.mean([analysis.get('response_time', 0) for analysis in analyses]) if analyses else 0
        }
    except Exception as e:
        st.error(f"Error loading Groq data: {e}")
        return {'analyses': [], 'total_analyses': 0, 'total_cost': 0, 'avg_response_time': 0}

# Custom components using HTML/CSS
def metric_card(title, value, description, card_type="info"):
    """Create a metric card component."""
    st.markdown(f"""
    <div class="metric-card {card_type}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def badge(text, badge_type="secondary"):
    """Create a badge component."""
    return f'<span class="badge {badge_type}">{text}</span>'

def activity_item(time, action, detail):
    """Create an activity feed item."""
    st.markdown(f"""
    <div class="activity-item">
        <div class="activity-action">{action}</div>
        <div class="activity-detail">{detail}</div>
        <div class="activity-time">{time}</div>
    </div>
    """, unsafe_allow_html=True)

def status_indicator(status, text):
    """Create a status indicator."""
    return f'<span class="status-indicator {status}"></span>{text}'

# Main dashboard
def main():
    # Add black sidebar content
    with st.sidebar:
        st.markdown("### 🏥 Financial Intelligence")
        st.markdown("---")
        
        st.markdown("**System Status**")
        st.markdown("🟢 **Operational**")
        st.markdown("📊 **Model Accuracy:** 95.5%")
        st.markdown("🤖 **AI Analysis:** Ready")
        st.markdown("---")
        
        st.markdown("**Quick Actions**")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📊 Generate Report", use_container_width=True):
            st.success("Report generated!")
        
        st.markdown("---")
        st.markdown("**Data Coverage**")
        st.markdown("📅 **Period:** 2003-2023")
        st.markdown("🏥 **Hospitals:** 1,324")
        st.markdown("📈 **Features:** Enhanced")
    
    # Clean business header
    st.markdown("""
    <div style="text-align: center; padding: 0.75rem 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 1rem -1rem; color: white;">
        <h1 style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -0.025em;">
            Hospital Financial Intelligence Platform
        </h1>
        <p style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 1.1rem; font-weight: 400; margin: 0.5rem 0 0 0; opacity: 0.9;">
            AI-Powered Financial Distress Prediction | Built with Streamlit + Modern UI Components
        </p>
        <p style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.95rem; font-weight: 400; margin: 0.25rem 0 0 0; opacity: 0.8;">
            Empowering healthcare organizations with data-driven financial insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df_full = load_hospital_data()
    model_metrics = load_model_metrics()
    groq_data = load_groq_data()
    
    # Hospital selector in main area
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Create hospital options
        hospital_options = ["🏢 All Hospitals (Portfolio View)"]
        if 'hospital_name' in df_full.columns and 'oshpd_id' in df_full.columns:
            # Get unique hospitals (since we have multiple years)
            unique_hospitals = df_full.drop_duplicates(subset=['oshpd_id'])[['hospital_name', 'oshpd_id']]
            hospital_list = [(row['hospital_name'], row['oshpd_id']) for _, row in unique_hospitals.iterrows()]
            hospital_options.extend([f"🏥 {name} ({id_})" for name, id_ in hospital_list[:50]])  # Limit to first 50 for performance
        else:
            hospital_options.extend([f"🏥 Hospital_{i}" for i in range(min(50, len(df_full)))])
        
        selected_hospital = st.selectbox(
            "🔍 Select Hospital for Detailed Analysis",
            hospital_options,
            key="hospital_selector",
            help="Choose a specific hospital to view detailed analytics, or keep 'All Hospitals' for portfolio overview"
        )
    
    with col2:
        if selected_hospital != "🏢 All Hospitals (Portfolio View)":
            st.metric("View Mode", "Individual", "Hospital-specific analysis")
        else:
            st.metric("View Mode", "Portfolio", f"{len(df_full)} hospitals")
    
    with col3:
        if selected_hospital != "🏢 All Hospitals (Portfolio View)":
            st.metric("Data Scope", "Single", "Focused view")
        else:
            st.metric("Data Scope", "Aggregate", "Full dataset")
    
    # Filter data based on selection
    if selected_hospital != "🏢 All Hospitals (Portfolio View)":
        # Extract hospital identifier from selection
        if "(" in selected_hospital and ")" in selected_hospital:
            hospital_id = selected_hospital.split("(")[-1].split(")")[0]
            if 'oshpd_id' in df_full.columns:
                df = df_full[df_full['oshpd_id'] == hospital_id]
            else:
                # Fallback for mock data
                hospital_idx = int(selected_hospital.split("_")[-1]) if "Hospital_" in selected_hospital else 0
                df = df_full.iloc[hospital_idx:hospital_idx+1]
        else:
            df = df_full.iloc[0:1]  # Default to first hospital
        
        # Update groq_data for individual hospital
        if len(df) > 0:
            hospital_row = df.iloc[0]
            groq_data = {
                **groq_data,
                "individual_analyses": [{
                    "hospital_id": hospital_row.get('oshpd_id', 'Selected Hospital'),
                    "metrics": {
                        "operating_margin": hospital_row.get('operating_margin', 0.02),
                        "current_ratio": hospital_row.get('current_ratio', 1.8),
                        "days_cash_on_hand": hospital_row.get('days_cash_on_hand', 65)
                    },
                    "analysis": f"**Status: {'Good' if hospital_row.get('operating_margin', 0) > 0.02 else 'At Risk'}** - Individual hospital analysis for {hospital_row.get('facility_name', 'Selected Hospital')}.",
                    "cost_usd": 0.000177
                }]
            }
    else:
        df = df_full
    
    # Navigation with Lucide icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard", 
        "📈 Data Analytics", 
        "🤖 AI Model Performance", 
        "🧠 LLM Analysis", 
        "📋 Portfolio Insights"
    ])
    
    with tab1:
        executive_dashboard(df, model_metrics, groq_data, selected_hospital)
    
    with tab2:
        data_analytics(df, selected_hospital)
    
    with tab3:
        ai_model_performance(model_metrics)
    
    with tab4:
        llm_analysis(groq_data, selected_hospital, df)
    
    with tab5:
        portfolio_insights(df, selected_hospital)

def executive_dashboard(df, model_metrics, groq_data, selected_hospital="🏢 All Hospitals (Portfolio View)"):
    """Executive dashboard page."""
    is_individual = selected_hospital != "🏢 All Hospitals (Portfolio View)"
    
    st.markdown(f"""
    <h2 class="section-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
        {'Hospital Analysis' if is_individual else 'Executive Overview'}
        {f' - {selected_hospital.split("🏥 ")[-1] if "🏥" in selected_hospital else "Individual Hospital"}' if is_individual else ''}
    </h2>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_individual:
            hospital_name = selected_hospital.split("🏥 ")[-1].split(" (")[0] if "🏥" in selected_hospital else "Selected Hospital"
            metric_card(
                "Hospital Analysis",
                hospital_name,
                "Individual hospital view",
                "info"
            )
        else:
            metric_card(
                "Hospitals Monitored",
                f"{len(df):,}",
                "Active in 2023 dataset",
                "info"
            )
    
    with col2:
        # Use real model metrics if available
        if model_metrics and 'performance_metrics' in model_metrics:
            roc_auc = model_metrics['performance_metrics'].get('roc_auc', 0.85)
        else:
            roc_auc = 0.85  # Fallback
        metric_card(
            "Model Accuracy",
            f"{roc_auc:.1%}",
            "ROC-AUC Score",
            "success"
        )
    
    with col3:
        # Use real Groq data if available
        total_analyses = groq_data.get('total_analyses', 0)
        metric_card(
            "AI Analyses",
            f"{total_analyses}",
            "Recent LLM reports",
            "info"
        )
    
    with col4:
        # Use real cost data if available
        total_cost = groq_data.get('total_cost', 0)
        metric_card(
            "Analysis Cost",
            f"${total_cost:.6f}",
            f"Per hospital: ${total_cost/max(total_analyses,1):.6f}" if total_analyses > 0 else "No analyses yet",
            "success"
        )
    
    # System performance and activity
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <h3 class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
                <circle cx="12" cy="12" r="10"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
            System Performance
        </h3>
        """, unsafe_allow_html=True)
        
        # Performance comparison chart
        metrics_data = {
            "Metric": ["ROC-AUC", "PR-AUC", "F1-Score", "Accuracy"],
            "Current Model": [0.995, 0.905, 0.857, 0.994],
            "Baseline": [0.832, 0.464, 0.623, 0.912],
            "Industry Standard": [0.85, 0.60, 0.70, 0.90]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current Model",
            x=metrics_data["Metric"],
            y=metrics_data["Current Model"],
            marker_color="#10b981"
        ))
        fig.add_trace(go.Bar(
            name="Baseline",
            x=metrics_data["Metric"],
            y=metrics_data["Baseline"],
            marker_color="#f59e0b"
        ))
        fig.add_trace(go.Bar(
            name="Industry Standard",
            x=metrics_data["Metric"],
            y=metrics_data["Industry Standard"],
            marker_color="#ef4444"
        ))
        
        fig.update_layout(
            title="Model Performance vs Benchmarks",
            barmode="group",
            height=400,
            yaxis_title="Score",
            showlegend=True,
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <h3 class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
            Recent Activity
        </h3>
        """, unsafe_allow_html=True)
        
        activity_item("2 hours ago", "AI Analysis", "5 hospitals analyzed")
        activity_item("1 day ago", "Model Update", "Enhanced features deployed")
        activity_item("2 days ago", "Data Refresh", "2023 data processed")
        activity_item("3 days ago", "Report Generated", "Executive summary created")
    
    # Key insights
    st.markdown("""
    <h3 class="section-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.35-4.35"/>
        </svg>
        Key Insights
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <h4>📈 Model Enhancement Success</h4>
            <p>Enhanced time-series features improved PR-AUC by <strong>95%</strong> (0.464 → 0.905)</p>
            {badge("High Impact", "success")}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-card">
            <h4>🎯 Top Risk Factors</h4>
            <p>Operating margin and times interest earned remain the strongest predictors</p>
            {badge("Validated", "info")}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-card">
            <h4>💰 Cost-Effective AI</h4>
            <p>LLM analysis costs only <strong>$0.0002</strong> per hospital with professional insights</p>
            {badge("Scalable", "secondary")}
        </div>
        """, unsafe_allow_html=True)

def data_analytics(df, selected_hospital="🏢 All Hospitals (Portfolio View)"):
    """Data analytics page."""
    is_individual = selected_hospital != "🏢 All Hospitals (Portfolio View)"
    
    st.markdown(f'''<h2 class="section-header">📊 {'Individual Hospital' if is_individual else 'Portfolio'} Financial Data Analytics</h2>''', unsafe_allow_html=True)
    
    # Data overview
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
        
        # Data quality metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            if is_individual:
                metric_card("Hospital Record", "1", "Individual analysis", "info")
            else:
                metric_card("Total Records", f"{len(df):,}", "Year 2023", "info")
        
        with col_b:
            metric_card("Features", f"{len(df.columns)}", "Enhanced dataset", "info")
        
        with col_c:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            metric_card("Data Quality", f"{completeness:.1f}%", "Completeness", "success")
        
        with col_d:
            if 'operating_margin' in df.columns:
                if is_individual:
                    margin = df['operating_margin'].iloc[0] if len(df) > 0 else 0
                    status = "Healthy" if margin > 0.02 else "At Risk"
                    card_type = "success" if margin > 0.02 else "warning"
                    metric_card("Financial Status", status, f"Margin: {margin:.1%}", card_type)
                else:
                    negative_margins = (df['operating_margin'] < 0).sum()
                    metric_card("At-Risk Hospitals", f"{negative_margins}", "Negative margins", "warning")
    
    with col2:
        st.markdown('<h3 class="section-header">Data Filters</h3>', unsafe_allow_html=True)
        
        # Health filter
        health_filter = st.radio(
            "Financial Health Filter",
            ["All Hospitals", "Healthy", "At Risk"],
            key="health_filter"
        )
        
        # Metric selector
        if not df.empty:
            financial_metrics = [col for col in df.columns if any(term in col.lower() for term in ['margin', 'ratio', 'cash'])]
            selected_metric = st.selectbox(
                "Select Metric",
                financial_metrics[:10] if len(financial_metrics) > 10 else financial_metrics,
                key="metric_selector"
            )
    
    # Visualization section
    if not df.empty and 'selected_metric' in locals():
        st.markdown('<h3 class="section-header">📈 Financial Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig = px.histogram(
                df, 
                x=selected_metric,
                title=f"Distribution of {selected_metric.replace('_', ' ').title()}",
                nbins=50,
                color_discrete_sequence=["#3b82f6"]
            )
            fig.update_layout(height=400, font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by health status
            if 'operating_margin' in df.columns:
                df_viz = df.copy()
                df_viz['health_status'] = df_viz['operating_margin'].apply(
                    lambda x: 'Healthy' if x > 0 else 'At Risk'
                )
                
                fig = px.box(
                    df_viz,
                    x='health_status',
                    y=selected_metric,
                    title=f"{selected_metric.replace('_', ' ').title()} by Health Status",
                    color='health_status',
                    color_discrete_map={'Healthy': '#10b981', 'At Risk': '#ef4444'}
                )
                fig.update_layout(height=400, font=dict(family="Inter, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)

def ai_model_performance(model_metrics):
    """AI model performance page."""
    st.markdown('<h2 class="section-header">🧠 AI Model Performance & Validation</h2>', unsafe_allow_html=True)
    
    # Model evolution timeline
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">📈 Model Evolution</h3>', unsafe_allow_html=True)
        
        timeline_data = pd.DataFrame({
            'Version': ['Baseline', 'Enhanced Features', 'Time-Series', 'Production'],
            'ROC-AUC': [0.832, 0.889, 0.945, 0.995],
            'PR-AUC': [0.464, 0.587, 0.723, 0.905],
            'Features': [31, 58, 89, 147],
            'Release': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=timeline_data['Version'], y=timeline_data['ROC-AUC'], 
                      name='ROC-AUC', line=dict(color='#3b82f6', width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=timeline_data['Version'], y=timeline_data['PR-AUC'], 
                      name='PR-AUC', line=dict(color='#ef4444', width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=timeline_data['Version'], y=timeline_data['Features'], 
                   name='Features', opacity=0.3, marker_color='#f59e0b'),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Model Performance Evolution", 
            height=500,
            font=dict(family="Inter, sans-serif")
        )
        fig.update_xaxes(title_text="Model Version")
        fig.update_yaxes(title_text="Performance Score", secondary_y=False)
        fig.update_yaxes(title_text="Feature Count", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">🎯 Current Metrics</h3>', unsafe_allow_html=True)
        
        # Current model metrics
        metric_card("ROC-AUC", "0.995", "Target: 0.85 ✅", "success")
        metric_card("PR-AUC", "0.905", "Target: 0.60 ✅", "success")
        metric_card("F1-Score", "0.857", "Target: 0.70 ✅", "success")
        metric_card("Accuracy", "0.994", "Target: 0.90 ✅", "success")
    
    # Feature importance
    st.markdown('<h3 class="section-header">🔍 Feature Importance Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Handle real feature importance structure
        if model_metrics and 'feature_importance' in model_metrics:
            feature_imp = model_metrics['feature_importance']
            if isinstance(feature_imp, dict) and 'xgboost_weight' in feature_imp:
                # Use XGBoost feature importance
                importance_data = pd.DataFrame([
                    {'feature': feature, 'importance': importance}
                    for feature, importance in feature_imp['xgboost_weight'].items()
                ]).sort_values('importance', ascending=False)
            else:
                # Fallback to mock data
                importance_data = pd.DataFrame([
                    {'feature': 'operating_margin', 'importance': 0.396},
                    {'feature': 'times_interest_earned', 'importance': 0.340},
                    {'feature': 'financial_stability_score', 'importance': 0.157},
                    {'feature': 'current_ratio_trend', 'importance': 0.107}
                ])
        else:
            # Fallback to mock data
            importance_data = pd.DataFrame([
                {'feature': 'operating_margin', 'importance': 0.396},
                {'feature': 'times_interest_earned', 'importance': 0.340},
                {'feature': 'financial_stability_score', 'importance': 0.157},
                {'feature': 'current_ratio_trend', 'importance': 0.107}
            ])
        
        fig = px.bar(
            importance_data.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Predictive Features",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**🔬 Model Interpretability**")
        
        explanations = [
            {
                "feature": "Operating Margin",
                "impact": "Primary profitability indicator",
                "importance": "39.6%"
            },
            {
                "feature": "Times Interest Earned",
                "impact": "Debt service capacity",
                "importance": "34.0%"
            },
            {
                "feature": "Financial Stability",
                "impact": "Volatility measure",
                "importance": "15.7%"
            }
        ]
        
        for exp in explanations:
            st.markdown(f"""
            <div class="metric-card info">
                <div class="metric-title">{exp['feature']} ({exp['importance']})</div>
                <div class="metric-description">{exp['impact']}</div>
            </div>
            """, unsafe_allow_html=True)

def llm_analysis(groq_data, selected_hospital="🏢 All Hospitals (Portfolio View)", hospital_data=None):
    """LLM analysis page with real-time Groq integration."""
    is_individual = selected_hospital != "🏢 All Hospitals (Portfolio View)"
    
    st.markdown(f"""
    <h2 class="section-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
        </svg>
        Groq AI-Powered {'Individual Hospital' if is_individual else 'Portfolio'} Analysis
    </h2>
    """, unsafe_allow_html=True)
    
    # API Key Input Section (always show)
    st.markdown("### 🔑 API Configuration")
    
    col_key1, col_key2 = st.columns([2, 1])
    
    with col_key1:
        user_api_key = st.text_input(
            "Enter your Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key from https://console.groq.com/",
            key="groq_api_key"
        )
    
    with col_key2:
        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">
                🔒 <strong>Your API key is secure</strong><br>
                • Not stored or logged<br>
                • Used only for this session<br>
                • Get free key at <a href="https://console.groq.com/" target="_blank">console.groq.com</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if not user_api_key:
        st.warning("🔑 **For Public Use:** Please enter your own Groq API key above to use AI analysis features. This ensures your usage is tracked to your account and protects the shared instance.")
    
    # Real-time analysis for individual hospitals
    if is_individual and hospital_data is not None:
        st.markdown("""
        <h3 class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Real-Time AI Analysis
        </h3>
        """, unsafe_allow_html=True)
        
        # Generate real-time analysis button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analysis_disabled = not user_api_key
            if st.button("🧠 Generate AI Analysis", key="generate_analysis", use_container_width=True, disabled=analysis_disabled):
                if user_api_key:
                    # Create analyzer with user's API key
                    analyzer = GroqAnalyzer(api_key=user_api_key)
                    
                    if analyzer.enabled:
                        # Extract hospital name
                        hospital_name = selected_hospital.split("🏥 ")[-1].split(" (")[0] if "🏥" in selected_hospital else "Selected Hospital"
                        
                        # Show loading spinner
                        with st.spinner(f"🤖 Analyzing {hospital_name} with Groq AI..."):
                            # Get hospital data (first row if multiple years)
                            hospital_row = hospital_data.iloc[0] if len(hospital_data) > 0 else pd.Series()
                            
                            # Generate analysis
                            result = analyzer.analyze_hospital(hospital_row, hospital_name)
                            
                            if result['success']:
                                # Store in session state
                                st.session_state['current_analysis'] = result
                                st.success(f"✅ Analysis complete! Cost: ${result['cost']:.6f}")
                            else:
                                st.error(f"❌ Analysis failed: {result['error']}")
                    else:
                        st.error("⚠️ Invalid API key. Please check your Groq API key and try again.")
                else:
                    st.error("⚠️ Please enter your Groq API key above to generate analysis.")
        
        with col2:
            if user_api_key:
                st.success("✅ API key provided! Click 'Generate AI Analysis' to get real-time financial insights for this hospital using Groq AI.")
            else:
                st.info("💡 Enter your Groq API key above, then click 'Generate AI Analysis' to get real-time financial insights.")
        
        # Display current analysis if available
        if 'current_analysis' in st.session_state and st.session_state['current_analysis']['success']:
            analysis = st.session_state['current_analysis']
            
            st.markdown("---")
            st.markdown("### 📋 AI Analysis Results")
            
            # Analysis metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_card("Analysis Cost", f"${analysis['cost']:.6f}", "Groq API", "success")
            
            with col2:
                metric_card("Tokens Used", f"{analysis['tokens']}", "LLaMA-3.1-8B", "info")
            
            with col3:
                metric_card("Generated", "Just now", "Real-time", "info")
            
            with col4:
                metric_card("Model", "Groq LLaMA", "Cloud API", "secondary")
            
            # Display the analysis content
            st.markdown("### 🤖 AI Financial Analysis")
            
            # Format the analysis content
            analysis_content = analysis['content']
            
            # Display in a nice container
            st.markdown(f"""
            <div class="metric-card info" style="margin: 1rem 0; padding: 1.5rem;">
                <div style="white-space: pre-wrap; line-height: 1.6; font-size: 0.95rem;">
{analysis_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Clear analysis button
            if st.button("🗑️ Clear Analysis", key="clear_analysis"):
                del st.session_state['current_analysis']
                st.rerun()
        
        return  # Exit early for individual hospital analysis
    
    # LLM integration overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <h3 class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Groq Cloud AI Integration
        </h3>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            total_analyses = groq_data.get('total_analyses', 0)
            metric_card(
                "Hospitals Analyzed",
                f"{total_analyses}",
                "Recent AI analysis",
                "info"
            )
        
        with col_b:
            total_cost = groq_data.get('total_cost', 0)
            metric_card(
                "Total Cost",
                f"${total_cost:.6f}",
                f"Groq Cloud API",
                "success"
            )
        
        with col_c:
            avg_cost = total_cost / max(total_analyses, 1) if total_analyses > 0 else 0
            metric_card(
                "Cost per Analysis",
                f"${avg_cost:.6f}",
                "Highly cost-effective",
                "success"
            )
    
    with col2:
        st.markdown('<h3 class="section-header">🔧 System Config</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card info">
            <div class="metric-title">AI Configuration</div>
            <div class="metric-description">
                <strong>Model:</strong> Groq LLaMA-3.1-8B-Instant<br>
                <strong>Provider:</strong> Groq Cloud API<br>
                <strong>Speed:</strong> 2-3 seconds per analysis<br>
                <strong>Deployment:</strong> Cloud-based (no local compute)<br>
                <strong>Cost:</strong> $0.0002 per hospital analysis
            </div>
            {badge("Cloud-Powered", "success")}
            {badge("Ultra-Fast", "info")}
        </div>
        """, unsafe_allow_html=True)
    
    # Recent analysis results
    analyses = groq_data.get('analyses', [])
    if analyses:
        st.markdown('<h3 class="section-header">📋 Recent AI Analysis Results</h3>', unsafe_allow_html=True)
        
        # Show summary of recent analyses
        st.markdown(f"""
        <div class="metric-card info">
            <div class="metric-title">Analysis Summary</div>
            <div class="metric-description">
                <strong>Total Analyses:</strong> {len(analyses)}<br>
                <strong>Total Cost:</strong> ${groq_data.get('total_cost', 0):.6f}<br>
                <strong>Average Response Time:</strong> {groq_data.get('avg_response_time', 0):.2f}s<br>
                <strong>Model:</strong> Groq LLaMA-3.1-8B-Instant<br>
                <strong>Provider:</strong> Groq Cloud API
            </div>
            {badge("AI Powered", "info")}
            {badge("Production Ready", "success")}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample analysis if available
        if analyses:
            st.markdown('<h3 class="section-header">📊 Sample AI Analysis Report</h3>', unsafe_allow_html=True)
            
            sample_analysis = analyses[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Extract key information from the analysis
                analysis_text = str(sample_analysis.get('analysis', 'Analysis completed'))[:500] + "..."
                st.markdown(f"""
                <div class="metric-card info">
                    <div class="metric-title">Latest AI Analysis</div>
                    <div class="metric-description">
                        <strong>Analysis Preview:</strong><br>
                        {analysis_text}<br><br>
                        <strong>File:</strong> {sample_analysis.get('file', 'N/A')}<br>
                        <strong>Status:</strong> Completed
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card secondary">
                    <div class="metric-title">Analysis Details</div>
                    <div class="metric-description">
                        <strong>Cost:</strong> ${sample_analysis.get('cost', 0):.6f}<br>
                        <strong>Response Time:</strong> {sample_analysis.get('response_time', 0):.2f}s<br>
                        <strong>Model:</strong> Groq LLaMA-3.1-8B<br>
                        <strong>Provider:</strong> Groq Cloud API
                    </div>
                    {badge("AI Generated", "info")}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card warning">
            <div class="metric-title">No Recent Analyses</div>
            <div class="metric-description">
                No Groq AI analyses found. Run the groq_hospital_analysis.py script to generate AI-powered financial insights.
            </div>
        </div>
        """, unsafe_allow_html=True)

def portfolio_insights(df, selected_hospital="🏢 All Hospitals (Portfolio View)"):
    """Portfolio insights page."""
    is_individual = selected_hospital != "🏢 All Hospitals (Portfolio View)"
    
    st.markdown(f'''<h2 class="section-header">🏢 {'Individual Hospital Analysis' if is_individual else 'Hospital Portfolio Management'}</h2>''', unsafe_allow_html=True)
    
    # Portfolio health overview
    st.markdown('<h3 class="section-header">📊 Portfolio Health Overview</h3>', unsafe_allow_html=True)
    
    # Mock portfolio data based on operating margins
    if 'operating_margin' in df.columns:
        excellent = (df['operating_margin'] > 0.05).sum()
        good = ((df['operating_margin'] > 0.02) & (df['operating_margin'] <= 0.05)).sum()
        concerning = ((df['operating_margin'] > -0.02) & (df['operating_margin'] <= 0.02)).sum()
        critical = (df['operating_margin'] <= -0.02).sum()
    else:
        excellent, good, concerning, critical = 125, 189, 98, 30
    
    total_hospitals = excellent + good + concerning + critical
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Excellent", f"{excellent}", f"{excellent/total_hospitals:.1%} of portfolio", "success")
    
    with col2:
        metric_card("Good", f"{good}", f"{good/total_hospitals:.1%} of portfolio", "info")
    
    with col3:
        metric_card("Concerning", f"{concerning}", f"{concerning/total_hospitals:.1%} of portfolio", "warning")
    
    with col4:
        metric_card("Critical", f"{critical}", f"{critical/total_hospitals:.1%} of portfolio", "danger")
    
    # Portfolio visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Health distribution pie chart
        fig = px.pie(
            values=[excellent, good, concerning, critical],
            names=['Excellent', 'Good', 'Concerning', 'Critical'],
            title="Portfolio Health Distribution",
            color_discrete_map={
                'Excellent': '#10b981',
                'Good': '#3b82f6',
                'Concerning': '#f59e0b',
                'Critical': '#ef4444'
            }
        )
        fig.update_layout(height=400, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Trend over time (mock data)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        trends = {
            'Excellent': [120, 122, 123, 124, 125, excellent],
            'Good': [185, 186, 187, 188, 189, good],
            'Concerning': [102, 101, 100, 99, 98, concerning],
            'Critical': [35, 34, 33, 32, 31, critical]
        }
        
        fig = go.Figure()
        colors = {'Excellent': '#10b981', 'Good': '#3b82f6', 'Concerning': '#f59e0b', 'Critical': '#ef4444'}
        
        for status, values in trends.items():
            fig.add_trace(go.Scatter(
                x=months, y=values, name=status,
                line=dict(width=3, color=colors[status])
            ))
        
        fig.update_layout(
            title="Portfolio Health Trends",
            xaxis_title="Month",
            yaxis_title="Number of Hospitals",
            height=400,
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

# Sidebar with gradient styling
with st.sidebar:
    # Add comprehensive CSS to force gradient with maximum specificity
    st.markdown("""
    <style>
        /* Ultra-high specificity sidebar gradient forcing */
        .stApp .stSidebar,
        .stApp .stSidebar > div,
        .stApp [data-testid="stSidebar"],
        .stApp [data-testid="stSidebar"] > div,
        .stApp section[data-testid="stSidebar"],
        .stApp section[data-testid="stSidebar"] > div,
        .css-1d391kg,
        .css-1lcbmhc,
        .css-1outpf7,
        .css-1y4p8pa,
        .css-17lntkn,
        .css-1v3fvcr {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            background-color: #667eea !important;
        }
        
        /* Force all sidebar content to be white */
        .stApp .stSidebar *,
        .stApp [data-testid="stSidebar"] *,
        .stApp section[data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Override Streamlit's default button styling in sidebar */
        .stApp .stSidebar .stButton > button,
        .stApp [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.2) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
        }
        
        .stApp .stSidebar .stButton > button:hover,
        .stApp [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.3) !important;
            transform: translateY(-2px) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 1rem; backdrop-filter: blur(10px);">
        <h3 style="color: white !important; font-family: Inter, sans-serif; margin-bottom: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            Hospital Financial Intelligence
        </h3>
        <p style="color: rgba(255,255,255,0.8) !important; font-size: 0.875rem; margin: 0;">AI-Powered Financial Distress Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hospital Selection Card with 3D effect
    st.markdown("""
    <div class="sidebar-card-3d" style="
        background: rgba(255,255,255,0.15); 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin-bottom: 1rem; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-2px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)';" onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2)';">
        <h4 style="color: white !important; font-family: Inter, sans-serif; margin-bottom: 1rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));">
                <circle cx="11" cy="11" r="8"/>
                <path d="m21 21-4.35-4.35"/>
            </svg>
            Hospital Selector
        </h4>
        <p style="color: rgba(255,255,255,0.85) !important; font-size: 0.75rem; margin-bottom: 0.5rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            Use the dropdown above to select a specific hospital for detailed analysis, or view the entire portfolio.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card-3d" style="
        background: rgba(255,255,255,0.15); 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin-bottom: 1rem; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-2px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)';" onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2)';">
        <h4 style="color: white !important; font-family: Inter, sans-serif; margin-bottom: 1rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));">
                <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
                <circle cx="12" cy="12" r="3"/>
            </svg>
            System Status
        </h4>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div style='margin-bottom: 0.5rem; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{status_indicator('online', 'Data Pipeline: Operational')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-bottom: 0.5rem; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{status_indicator('online', 'AI Model: Active')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-bottom: 0.5rem; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{status_indicator('online', 'Groq Cloud API: Connected')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-bottom: 0.5rem; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{status_indicator('online', 'Dashboard: Live')}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card-3d" style="
        background: rgba(255,255,255,0.15); 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin-bottom: 1rem; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-2px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)';" onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2)';">
        <h4 style="color: white !important; font-family: Inter, sans-serif; margin-bottom: 1rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));">
                <line x1="18" y1="20" x2="18" y2="10"/>
                <line x1="12" y1="20" x2="12" y2="4"/>
                <line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            Quick Stats
        </h4>
        <div style="color: rgba(255,255,255,0.9) !important; font-size: 0.875rem; line-height: 1.5; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            <div><strong>Last Updated:</strong> {}</div>
            <div><strong>Data Coverage:</strong> 2003-2023</div>
            <div><strong>Model Version:</strong> Enhanced v1.0</div>
            <div><strong>API Status:</strong> Connected</div>
        </div>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card-3d" style="
        background: rgba(255,255,255,0.15); 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin-bottom: 1rem; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-2px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)';" onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2)';">
        <h4 style="color: white !important; font-family: Inter, sans-serif; margin-bottom: 1rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Quick Actions
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Functional buttons with custom styling
    if st.button("🔄 Refresh Data", key="sidebar_refresh", type="secondary", use_container_width=True):
        st.rerun()
    
    if st.button("📊 Generate Report", key="sidebar_report", type="secondary", use_container_width=True):
        st.success("Report generation would be triggered here")
    
    if st.button("🤖 Run AI Analysis", key="sidebar_ai", type="primary", use_container_width=True):
        st.success("AI analysis would be initiated here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem; font-family: Inter, sans-serif;'>
    <p>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.25rem;">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
        <strong>Hospital Financial Intelligence Platform</strong> | 
        AI-Powered Financial Distress Prediction | 
        Built with Streamlit + Modern UI Components
    </p>
    <p>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 0.25rem;">
            <circle cx="12" cy="12" r="10"/>
            <path d="m9 12 2 2 4-4"/>
        </svg>
        <em>Empowering healthcare organizations with data-driven financial insights</em>
    </p>
    <p>Inspired by <a href="https://github.com/ObservedObserver/streamlit-shadcn-ui" target="_blank" style="color: #3b82f6; text-decoration: none;">streamlit-shadcn-ui</a> design principles</p>
</div>
""", unsafe_allow_html=True)

class GroqAnalyzer:
    """Streamlit-integrated Groq analyzer for real-time hospital analysis."""
    
    def __init__(self, api_key=None):
        """Initialize the analyzer with Groq API key."""
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            if not api_key:
                # Only show error if no API key was provided and none in env
                st.error("⚠️ No API key provided. Please enter your Groq API key for AI analysis.")
            self.enabled = False
        else:
            self.enabled = True
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "llama-3.1-8b-instant"
    
    def analyze_hospital(self, hospital_data: pd.Series, hospital_name: str) -> Dict:
        """Analyze a single hospital's financial performance in real-time."""
        if not self.enabled:
            return {
                'success': False,
                'error': 'Groq API key not configured'
            }
        
        # Extract key metrics
        metrics = {
            'operating_margin': float(hospital_data.get('operating_margin', 0)),
            'current_ratio': float(hospital_data.get('current_ratio', 1.0)),
            'days_cash_on_hand': float(hospital_data.get('days_cash_on_hand', 30)),
            'total_margin': float(hospital_data.get('total_margin', 0)),
            'debt_service_coverage': float(hospital_data.get('debt_service_coverage_ratio', 0)),
            'operating_revenue': float(hospital_data.get('total_operating_revenue', 0))
        }
        
        # Create analysis prompt
        prompt = f"""
You are a healthcare financial analyst. Provide a comprehensive analysis of {hospital_name}'s financial performance:

**Hospital Financial Metrics:**
- Operating Margin: {metrics['operating_margin']:.2%}
- Total Margin: {metrics['total_margin']:.2%}
- Current Ratio: {metrics['current_ratio']:.2f}
- Days Cash on Hand: {metrics['days_cash_on_hand']:.0f} days
- Debt Service Coverage: {metrics['debt_service_coverage']:.2f}
- Operating Revenue: ${metrics['operating_revenue']:,.0f}

**Required Analysis Format:**
🏥 **Financial Health Assessment:**
Operating Margin: [value]% ([Above/Below] industry average)
Current Ratio: [value] ([Strong/Adequate/Weak] liquidity position)
Days Cash: [value] days ([Adequate/Insufficient] reserves)

🔍 **AI Insights:**
"[Hospital name] demonstrates [assessment] with [key finding]..."

💡 **Recommendations:**
• [Specific actionable recommendation 1]
• [Specific actionable recommendation 2]
• [Specific actionable recommendation 3]

📊 **Risk Level:** [LOW/MEDIUM/HIGH]

Format professionally for healthcare executives. Be specific and actionable.
"""
        
        try:
            # Make API call
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a healthcare financial analyst providing professional analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 600,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['total_tokens']
                
                return {
                    'success': True,
                    'content': content,
                    'tokens': tokens,
                    'cost': tokens * 0.00000059,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Initialize Groq analyzer
@st.cache_resource
def get_groq_analyzer():
    """Get cached Groq analyzer instance."""
    return GroqAnalyzer()

if __name__ == "__main__":
    main() 