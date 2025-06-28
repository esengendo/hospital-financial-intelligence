"""
Hospital Financial Analysis - LLM Integration Module
Production-ready AI analysis using Groq API
"""

from .external_llm_client import ExternalLLMClient, LLMProvider, LLMResponse
from .input_parsers import (
    HospitalFinancialData,
    parse_hospital_data,
    JSONParser,
    HTMLParser,
    PDFParser,
    CSVParser,
    ExcelParser,
    MultiFormatParser
)
from .prompt_templates import PromptTemplates
from .streamlined_generators import StreamlinedReportGenerator

# Main exports for production use
__all__ = [
    'ExternalLLMClient',
    'LLMProvider', 
    'LLMResponse',
    'HospitalFinancialData',
    'parse_hospital_data',
    'JSONParser',
    'HTMLParser', 
    'PDFParser',
    'CSVParser',
    'ExcelParser',
    'MultiFormatParser',
    'PromptTemplates',
    'StreamlinedReportGenerator'
]

__version__ = "1.0.0"
__author__ = "Hospital Financial Intelligence"
__description__ = "Production-ready LLM integration for hospital financial analysis" 