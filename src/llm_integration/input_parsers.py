"""
Multi-format Input Parsers for Hospital Financial Analysis

Supports parsing hospital financial data from various formats:
- JSON (executive summaries, model outputs)
- HTML (dashboard reports)
- PDF (financial statements)
- CSV/Excel (raw financial data)
- XML (healthcare data standards)
"""

import json
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
import re

# Optional imports for PDF and HTML parsing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HospitalFinancialData:
    """Standardized hospital financial data structure."""
    hospital_id: str
    hospital_name: str
    location: Optional[str] = None
    year: Optional[int] = None
    bed_count: Optional[int] = None
    financial_metrics: Dict[str, float] = None
    risk_prediction: Dict[str, Any] = None
    shap_explanations: Dict[str, float] = None
    trends: Dict[str, str] = None
    peer_comparison: Dict[str, float] = None
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.financial_metrics is None:
            self.financial_metrics = {}
        if self.risk_prediction is None:
            self.risk_prediction = {}
        if self.shap_explanations is None:
            self.shap_explanations = {}
        if self.trends is None:
            self.trends = {}
        if self.peer_comparison is None:
            self.peer_comparison = {}
        if self.raw_data is None:
            self.raw_data = {}


class InputParser:
    """Base class for input parsers."""
    
    def __init__(self):
        self.supported_formats = []
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def parse(self, input_data: Union[str, Path, Dict]) -> List[HospitalFinancialData]:
        """Parse input data and return standardized hospital data."""
        raise NotImplementedError


class JSONParser(InputParser):
    """Parser for JSON files containing hospital financial data."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.json']
    
    def parse(self, input_data: Union[str, Path, Dict]) -> List[HospitalFinancialData]:
        """Parse JSON data."""
        try:
            if isinstance(input_data, (str, Path)):
                with open(input_data, 'r') as f:
                    data = json.load(f)
            elif isinstance(input_data, dict):
                data = input_data
            else:
                raise ValueError("Input must be file path or dictionary")
            
            return self._parse_json_data(data)
            
        except Exception as e:
            logger.error(f"Error parsing JSON data: {e}")
            return []
    
    def _parse_json_data(self, data: Dict[str, Any]) -> List[HospitalFinancialData]:
        """Parse JSON data structure."""
        hospitals = []
        
        # Handle different JSON structures
        if 'hospitals' in data:
            # Multiple hospitals in array
            for hospital_data in data['hospitals']:
                hospitals.append(self._create_hospital_data(hospital_data))
        elif 'hospital_data' in data:
            # Single hospital
            hospitals.append(self._create_hospital_data(data['hospital_data']))
        elif 'provider_id' in data or 'hospital_id' in data:
            # Direct hospital data
            hospitals.append(self._create_hospital_data(data))
        else:
            # Try to extract from executive summary format
            hospitals.extend(self._parse_executive_summary(data))
        
        return hospitals
    
    def _create_hospital_data(self, data: Dict[str, Any]) -> HospitalFinancialData:
        """Create standardized hospital data from JSON."""
        return HospitalFinancialData(
            hospital_id=data.get('provider_id', data.get('hospital_id', 'unknown')),
            hospital_name=data.get('hospital_name', data.get('facility_name', 'Unknown Hospital')),
            location=data.get('location', data.get('city_state')),
            year=data.get('year', data.get('reporting_year')),
            bed_count=data.get('bed_count', data.get('total_beds')),
            financial_metrics=data.get('financial_metrics', {}),
            risk_prediction=data.get('risk_prediction', {}),
            shap_explanations=data.get('shap_explanations', {}),
            trends=data.get('trends', {}),
            peer_comparison=data.get('peer_comparison', {}),
            raw_data=data
        )
    
    def _parse_executive_summary(self, data: Dict[str, Any]) -> List[HospitalFinancialData]:
        """Parse executive summary format."""
        hospitals = []
        
        # Look for hospital data in various summary formats
        if 'financial_health' in data and isinstance(data['financial_health'], dict):
            for hospital_id, hospital_data in data['financial_health'].items():
                hospitals.append(HospitalFinancialData(
                    hospital_id=hospital_id,
                    hospital_name=hospital_data.get('name', f'Hospital {hospital_id}'),
                    financial_metrics=hospital_data.get('metrics', {}),
                    risk_prediction=hospital_data.get('risk', {}),
                    raw_data=hospital_data
                ))
        
        return hospitals


class HTMLParser(InputParser):
    """Parser for HTML dashboard files."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.html', '.htm']
    
    def parse(self, input_data: Union[str, Path]) -> List[HospitalFinancialData]:
        """Parse HTML dashboard data."""
        if not HTML_AVAILABLE:
            logger.error("BeautifulSoup not available for HTML parsing")
            return []
        
        try:
            with open(input_data, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            return self._extract_hospital_data_from_html(soup)
            
        except Exception as e:
            logger.error(f"Error parsing HTML data: {e}")
            return []
    
    def _extract_hospital_data_from_html(self, soup: BeautifulSoup) -> List[HospitalFinancialData]:
        """Extract hospital data from HTML dashboard."""
        hospitals = []
        
        # Look for JSON data embedded in script tags
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and ('hospital' in script.string.lower() or 'financial' in script.string.lower()):
                try:
                    # Try to extract JSON data
                    json_match = re.search(r'(\{.*\})', script.string, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group(1))
                        json_parser = JSONParser()
                        hospitals.extend(json_parser._parse_json_data(json_data))
                except:
                    continue
        
        # Look for table data
        tables = soup.find_all('table')
        for table in tables:
            hospitals.extend(self._parse_html_table(table))
        
        return hospitals
    
    def _parse_html_table(self, table) -> List[HospitalFinancialData]:
        """Parse hospital data from HTML table."""
        hospitals = []
        
        try:
            # Convert table to DataFrame for easier processing
            df = pd.read_html(str(table))[0]
            
            # Look for hospital identifier columns
            hospital_cols = [col for col in df.columns if 'hospital' in str(col).lower() or 'provider' in str(col).lower()]
            
            if hospital_cols:
                for _, row in df.iterrows():
                    hospital_data = HospitalFinancialData(
                        hospital_id=str(row.get(hospital_cols[0], 'unknown')),
                        hospital_name=str(row.get('Hospital Name', row.get('Facility Name', 'Unknown'))),
                        financial_metrics=self._extract_financial_metrics_from_row(row),
                        raw_data=row.to_dict()
                    )
                    hospitals.append(hospital_data)
        
        except Exception as e:
            logger.debug(f"Could not parse HTML table: {e}")
        
        return hospitals
    
    def _extract_financial_metrics_from_row(self, row: pd.Series) -> Dict[str, float]:
        """Extract financial metrics from table row."""
        metrics = {}
        
        # Common financial metric patterns
        metric_patterns = {
            'operating_margin': ['operating margin', 'op margin', 'operating_margin'],
            'total_margin': ['total margin', 'total_margin', 'net margin'],
            'current_ratio': ['current ratio', 'current_ratio'],
            'debt_to_equity': ['debt to equity', 'debt_to_equity', 'debt/equity'],
            'return_on_assets': ['return on assets', 'roa', 'return_on_assets'],
            'days_cash_on_hand': ['days cash', 'cash days', 'days_cash_on_hand']
        }
        
        for metric_name, patterns in metric_patterns.items():
            for pattern in patterns:
                for col in row.index:
                    if pattern in str(col).lower():
                        try:
                            value = float(str(row[col]).replace('%', '').replace('$', '').replace(',', ''))
                            metrics[metric_name] = value
                            break
                        except:
                            continue
        
        return metrics


class PDFParser(InputParser):
    """Parser for PDF financial statements."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.pdf']
    
    def parse(self, input_data: Union[str, Path]) -> List[HospitalFinancialData]:
        """Parse PDF financial statements."""
        if not PDF_AVAILABLE:
            logger.error("pdfplumber not available for PDF parsing")
            return []
        
        try:
            with pdfplumber.open(input_data) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                
                return self._extract_hospital_data_from_text(text)
                
        except Exception as e:
            logger.error(f"Error parsing PDF data: {e}")
            return []
    
    def _extract_hospital_data_from_text(self, text: str) -> List[HospitalFinancialData]:
        """Extract hospital data from PDF text."""
        hospitals = []
        
        # Look for hospital name patterns
        hospital_patterns = [
            r'(\w+\s+(?:Hospital|Medical Center|Health System|Healthcare))',
            r'Hospital Name:\s*([^\n]+)',
            r'Facility:\s*([^\n]+)'
        ]
        
        hospital_name = "Unknown Hospital"
        for pattern in hospital_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hospital_name = match.group(1).strip()
                break
        
        # Extract financial metrics using regex patterns
        financial_metrics = self._extract_financial_metrics_from_text(text)
        
        if financial_metrics:
            hospitals.append(HospitalFinancialData(
                hospital_id=f"PDF_{hash(hospital_name) % 10000}",
                hospital_name=hospital_name,
                financial_metrics=financial_metrics,
                raw_data={"source": "PDF", "text_length": len(text)}
            ))
        
        return hospitals
    
    def _extract_financial_metrics_from_text(self, text: str) -> Dict[str, float]:
        """Extract financial metrics from PDF text."""
        metrics = {}
        
        # Financial metric patterns
        patterns = {
            'operating_margin': [
                r'Operating Margin[:\s]+([+-]?\d+\.?\d*%?)',
                r'Op\.?\s+Margin[:\s]+([+-]?\d+\.?\d*%?)'
            ],
            'total_margin': [
                r'Total Margin[:\s]+([+-]?\d+\.?\d*%?)',
                r'Net Margin[:\s]+([+-]?\d+\.?\d*%?)'
            ],
            'current_ratio': [
                r'Current Ratio[:\s]+(\d+\.?\d*)',
                r'Liquidity Ratio[:\s]+(\d+\.?\d*)'
            ],
            'days_cash_on_hand': [
                r'Days Cash[:\s]+(\d+\.?\d*)',
                r'Cash on Hand[:\s]+(\d+\.?\d*)\s+days'
            ]
        }
        
        for metric_name, metric_patterns in patterns.items():
            for pattern in metric_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value_str = match.group(1).replace('%', '')
                        value = float(value_str)
                        # Convert percentage to decimal if needed
                        if '%' in match.group(1) and abs(value) > 1:
                            value = value / 100
                        metrics[metric_name] = value
                        break
                    except:
                        continue
        
        return metrics


class CSVParser(InputParser):
    """Parser for CSV financial data files."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.csv']
    
    def parse(self, input_data: Union[str, Path]) -> List[HospitalFinancialData]:
        """Parse CSV financial data."""
        try:
            df = pd.read_csv(input_data)
            return self._parse_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error parsing CSV data: {e}")
            return []
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[HospitalFinancialData]:
        """Parse DataFrame into hospital data."""
        hospitals = []
        
        # Identify key columns
        id_cols = [col for col in df.columns if any(x in col.lower() for x in ['provider', 'hospital', 'id', 'ccn'])]
        name_cols = [col for col in df.columns if any(x in col.lower() for x in ['name', 'facility'])]
        
        if not id_cols:
            logger.warning("No hospital ID column found in CSV")
            return []
        
        id_col = id_cols[0]
        name_col = name_cols[0] if name_cols else None
        
        for _, row in df.iterrows():
            financial_metrics = {}
            
            # Extract financial metrics from numeric columns
            for col in df.columns:
                if col not in [id_col, name_col] and pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        value = float(row[col])
                        if not pd.isna(value):
                            financial_metrics[col.lower().replace(' ', '_')] = value
                    except:
                        continue
            
            hospitals.append(HospitalFinancialData(
                hospital_id=str(row[id_col]),
                hospital_name=str(row[name_col]) if name_col else f"Hospital {row[id_col]}",
                financial_metrics=financial_metrics,
                raw_data=row.to_dict()
            ))
        
        return hospitals


class ExcelParser(InputParser):
    """Parser for Excel financial data files."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.xlsx', '.xls']
    
    def parse(self, input_data: Union[str, Path]) -> List[HospitalFinancialData]:
        """Parse Excel financial data."""
        try:
            # Try to read all sheets
            excel_file = pd.ExcelFile(input_data)
            all_hospitals = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(input_data, sheet_name=sheet_name)
                csv_parser = CSVParser()
                hospitals = csv_parser._parse_dataframe(df)
                all_hospitals.extend(hospitals)
            
            return all_hospitals
            
        except Exception as e:
            logger.error(f"Error parsing Excel data: {e}")
            return []


class MultiFormatParser:
    """Main parser that handles multiple input formats."""
    
    def __init__(self):
        self.parsers = [
            JSONParser(),
            HTMLParser(),
            PDFParser(),
            CSVParser(),
            ExcelParser()
        ]
    
    def parse(self, input_data: Union[str, Path, Dict]) -> List[HospitalFinancialData]:
        """Parse input data using appropriate parser."""
        if isinstance(input_data, dict):
            # Direct dictionary input
            json_parser = JSONParser()
            return json_parser.parse(input_data)
        
        file_path = Path(input_data)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Find appropriate parser
        for parser in self.parsers:
            if parser.can_parse(file_path):
                logger.info(f"Using {parser.__class__.__name__} for {file_path}")
                return parser.parse(file_path)
        
        logger.error(f"No parser available for file type: {file_path.suffix}")
        return []
    
    def get_supported_formats(self) -> List[str]:
        """Get list of all supported file formats."""
        formats = []
        for parser in self.parsers:
            formats.extend(parser.supported_formats)
        return sorted(set(formats))


# Convenience function
def parse_hospital_data(input_data: Union[str, Path, Dict]) -> List[HospitalFinancialData]:
    """Parse hospital financial data from various input formats."""
    parser = MultiFormatParser()
    return parser.parse(input_data) 