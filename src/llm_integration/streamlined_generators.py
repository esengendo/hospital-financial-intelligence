"""
Streamlined Report Generators for Hospital Financial Analysis

Lightweight report generation using external LLM APIs.
Designed for deployment-friendly hospital financial analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .external_llm_client import ExternalLLMClient, LLMProvider
from .input_parsers import HospitalFinancialData, parse_hospital_data
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class StreamlinedReportGenerator:
    """Lightweight report generator using external LLM APIs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the report generator."""
        self.config = config or {}
        self.llm_client = ExternalLLMClient(config)
        self.prompt_templates = PromptTemplates()
        self.output_dir = Path(self.config.get('output_dir', 'reports'))
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_hospital_summary(
        self,
        input_data: Union[str, Path, Dict, HospitalFinancialData],
        output_format: str = "markdown",
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ) -> Dict[str, Any]:
        """Generate plain-English financial summary for a hospital."""
        try:
            # Parse input data
            if isinstance(input_data, HospitalFinancialData):
                hospital_data = input_data
            else:
                parsed_data = parse_hospital_data(input_data)
                if not parsed_data:
                    raise ValueError("No hospital data found in input")
                hospital_data = parsed_data[0]
            
            # Generate mock response for testing
            if use_mock:
                return self._generate_mock_hospital_summary(hospital_data, output_format)
            
            # Create prompt
            prompt = self.prompt_templates.create_hospital_summary_prompt(hospital_data.__dict__)
            
            # Generate response
            response = self.llm_client.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=800,
                temperature=0.7
            )
            
            if not response.success:
                raise Exception(f"LLM generation failed: {response.error}")
            
            # Format and save output
            formatted_content = self._format_markdown(
                response.text, 
                f"Financial Summary - {hospital_data.hospital_name}",
                hospital_data
            )
            
            file_path = self._save_report(
                formatted_content,
                f"summary_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
                output_format
            )
            
            return {
                "success": True,
                "content": response.text,
                "formatted_content": formatted_content,
                "file_path": str(file_path),
                "hospital_id": hospital_data.hospital_id,
                "hospital_name": hospital_data.hospital_name,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            
        except Exception as e:
            logger.error(f"Error generating hospital summary: {e}")
            return {"success": False, "error": str(e), "content": "", "file_path": ""}
    
    def generate_risk_narrative(
        self,
        input_data: Union[str, Path, Dict, HospitalFinancialData],
        output_format: str = "markdown",
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ) -> Dict[str, Any]:
        """Generate narrative explanation of financial risk."""
        try:
            # Parse input data
            if isinstance(input_data, HospitalFinancialData):
                hospital_data = input_data
            else:
                parsed_data = parse_hospital_data(input_data)
                if not parsed_data:
                    raise ValueError("No hospital data found in input")
                hospital_data = parsed_data[0]
            
            # Generate mock response for testing
            if use_mock:
                return self._generate_mock_risk_narrative(hospital_data, output_format)
            
            # Create prompt
            prompt = self.prompt_templates.create_risk_narrative_prompt(hospital_data.__dict__)
            
            # Generate response
            response = self.llm_client.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=900,
                temperature=0.6
            )
            
            if not response.success:
                raise Exception(f"LLM generation failed: {response.error}")
            
            # Format and save output
            formatted_content = self._format_markdown(
                response.text,
                f"Risk Analysis - {hospital_data.hospital_name}",
                hospital_data
            )
            
            file_path = self._save_report(
                formatted_content,
                f"risk_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
                output_format
            )
            
            return {
                "success": True,
                "content": response.text,
                "formatted_content": formatted_content,
                "file_path": str(file_path),
                "hospital_id": hospital_data.hospital_id,
                "hospital_name": hospital_data.hospital_name,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            
        except Exception as e:
            logger.error(f"Error generating risk narrative: {e}")
            return {"success": False, "error": str(e), "content": "", "file_path": ""}
    
    def generate_cost_recommendations(
        self,
        input_data: Union[str, Path, Dict, HospitalFinancialData],
        output_format: str = "markdown",
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ) -> Dict[str, Any]:
        """Generate actionable cost-saving recommendations."""
        try:
            # Parse input data
            if isinstance(input_data, HospitalFinancialData):
                hospital_data = input_data
            else:
                parsed_data = parse_hospital_data(input_data)
                if not parsed_data:
                    raise ValueError("No hospital data found in input")
                hospital_data = parsed_data[0]
            
            # Generate mock response for testing
            if use_mock:
                return self._generate_mock_cost_recommendations(hospital_data, output_format)
            
            # Create prompt
            prompt = self.prompt_templates.create_cost_recommendations_prompt(hospital_data.__dict__)
            
            # Generate response
            response = self.llm_client.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=1000,
                temperature=0.7
            )
            
            if not response.success:
                raise Exception(f"LLM generation failed: {response.error}")
            
            # Format and save output
            formatted_content = self._format_markdown(
                response.text,
                f"Cost Recommendations - {hospital_data.hospital_name}",
                hospital_data
            )
            
            file_path = self._save_report(
                formatted_content,
                f"recommendations_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
                output_format
            )
            
            return {
                "success": True,
                "content": response.text,
                "formatted_content": formatted_content,
                "file_path": str(file_path),
                "hospital_id": hospital_data.hospital_id,
                "hospital_name": hospital_data.hospital_name,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            
        except Exception as e:
            logger.error(f"Error generating cost recommendations: {e}")
            return {"success": False, "error": str(e), "content": "", "file_path": ""}
    
    def generate_executive_briefing(
        self,
        input_data: Union[str, Path, Dict, List[HospitalFinancialData]],
        output_format: str = "html",
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ) -> Dict[str, Any]:
        """Generate executive briefing for portfolio of hospitals."""
        try:
            # Parse input data
            if isinstance(input_data, list) and all(isinstance(h, HospitalFinancialData) for h in input_data):
                hospitals_data = input_data
            else:
                hospitals_data = parse_hospital_data(input_data)
                if not hospitals_data:
                    raise ValueError("No hospital data found in input")
            
            # Generate mock response for testing
            if use_mock:
                return self._generate_mock_executive_briefing(hospitals_data, output_format)
            
            # Prepare portfolio summary
            portfolio_summary = self._create_portfolio_summary(hospitals_data)
            
            # Create prompt
            prompt = self.prompt_templates.create_executive_briefing_prompt(portfolio_summary)
            
            # Generate response
            response = self.llm_client.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=1200,
                temperature=0.6
            )
            
            if not response.success:
                raise Exception(f"LLM generation failed: {response.error}")
            
            # Format and save output
            formatted_content = self._format_html(
                response.text,
                "Executive Briefing - Hospital Portfolio Analysis"
            )
            
            file_path = self._save_report(
                formatted_content,
                f"executive_briefing_{datetime.now().strftime('%Y%m%d')}",
                output_format
            )
            
            return {
                "success": True,
                "content": response.text,
                "formatted_content": formatted_content,
                "file_path": str(file_path),
                "hospitals_count": len(hospitals_data),
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            
        except Exception as e:
            logger.error(f"Error generating executive briefing: {e}")
            return {"success": False, "error": str(e), "content": "", "file_path": ""}
    
    def generate_batch_summaries(
        self,
        input_data: Union[str, Path, Dict, List[HospitalFinancialData]],
        output_format: str = "markdown",
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate summaries for multiple hospitals."""
        try:
            # Parse input data
            if isinstance(input_data, list) and all(isinstance(h, HospitalFinancialData) for h in input_data):
                hospitals_data = input_data
            else:
                hospitals_data = parse_hospital_data(input_data)
                if not hospitals_data:
                    raise ValueError("No hospital data found in input")
            
            results = []
            for hospital_data in hospitals_data:
                result = self.generate_hospital_summary(
                    hospital_data, output_format, provider, use_mock
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [{"success": False, "error": str(e)}]
    
    def _create_portfolio_summary(self, hospitals_data: List[HospitalFinancialData]) -> Dict[str, Any]:
        """Create portfolio summary for executive briefing."""
        total_hospitals = len(hospitals_data)
        high_risk = sum(1 for h in hospitals_data if h.risk_prediction.get('risk_score', 0) > 0.7)
        medium_risk = sum(1 for h in hospitals_data if 0.3 < h.risk_prediction.get('risk_score', 0) <= 0.7)
        low_risk = total_hospitals - high_risk - medium_risk
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "total_hospitals": total_hospitals,
            "high_risk_hospitals": high_risk,
            "medium_risk_hospitals": medium_risk,
            "low_risk_hospitals": low_risk,
            "key_findings": [
                f"{high_risk}/{total_hospitals} hospitals at high financial risk",
                "Operating margins below industry benchmarks",
                "Cash positions require monitoring"
            ]
        }
    
    def _format_markdown(self, content: str, title: str, hospital_data: Optional[HospitalFinancialData]) -> str:
        """Format content as Markdown."""
        formatted = f"# {title}\n\n"
        formatted += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if hospital_data:
            formatted += f"**Hospital:** {hospital_data.hospital_name}\n"
            formatted += f"**ID:** {hospital_data.hospital_id}\n"
            if hospital_data.location:
                formatted += f"**Location:** {hospital_data.location}\n"
            formatted += "\n"
        
        formatted += "## Analysis\n\n"
        formatted += content
        
        return formatted
    
    def _format_html(self, content: str, title: str) -> str:
        """Format content as HTML."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .content {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="content">
        {content.replace(chr(10), '<br>')}
    </div>
</body>
</html>
        """
        return html_content
    
    def _save_report(self, content: str, filename: str, output_format: str) -> Path:
        """Save report to file."""
        extensions = {"markdown": ".md", "html": ".html", "json": ".json", "text": ".txt"}
        extension = extensions.get(output_format, ".txt")
        file_path = self.output_dir / f"{filename}{extension}"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Report saved to {file_path}")
        return file_path
    
    def _generate_mock_hospital_summary(self, hospital_data: HospitalFinancialData, output_format: str) -> Dict[str, Any]:
        """Generate mock hospital summary for testing."""
        mock_content = f"""
## Financial Summary for {hospital_data.hospital_name}

This hospital shows mixed financial performance with several areas of concern:

**Key Metrics:**
- Operating Margin: {hospital_data.financial_metrics.get('operating_margin', 0):.1%}
- Total Margin: {hospital_data.financial_metrics.get('total_margin', 0):.1%}
- Days Cash on Hand: {hospital_data.financial_metrics.get('days_cash_on_hand', 0):.0f} days

**Assessment:**
The hospital's financial position indicates moderate risk. Operating margins are below industry benchmarks.

**Recommendations:**
1. Focus on revenue cycle optimization
2. Review staffing efficiency metrics
3. Implement cost containment measures
        """.strip()
        
        formatted_content = self._format_markdown(
            mock_content,
            f"Financial Summary - {hospital_data.hospital_name}",
            hospital_data
        )
        
        file_path = self._save_report(
            formatted_content,
            f"mock_summary_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
            output_format
        )
        
        return {
            "success": True,
            "content": mock_content,
            "formatted_content": formatted_content,
            "file_path": str(file_path),
            "hospital_id": hospital_data.hospital_id,
            "hospital_name": hospital_data.hospital_name,
            "provider": "mock",
            "tokens_used": len(mock_content.split()),
            "response_time": 0.1
        }
    
    def _generate_mock_risk_narrative(self, hospital_data: HospitalFinancialData, output_format: str) -> Dict[str, Any]:
        """Generate mock risk narrative for testing."""
        mock_content = f"""
## Risk Analysis for {hospital_data.hospital_name}

Our financial distress prediction model indicates elevated risk for this facility:

**Primary Risk Factors:**
1. Operating margin pressure: {hospital_data.financial_metrics.get('operating_margin', 0):.1%}
2. Cash flow concerns: {hospital_data.financial_metrics.get('days_cash_on_hand', 0):.0f} days
3. Debt burden indicators

**Model Insights:**
The analysis reveals operating margin and cash position as key risk drivers.

**Risk Trajectory:**
Without intervention, increased probability of financial distress within 12-18 months.
        """.strip()
        
        formatted_content = self._format_markdown(
            mock_content,
            f"Risk Analysis - {hospital_data.hospital_name}",
            hospital_data
        )
        
        file_path = self._save_report(
            formatted_content,
            f"mock_risk_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
            output_format
        )
        
        return {
            "success": True,
            "content": mock_content,
            "formatted_content": formatted_content,
            "file_path": str(file_path),
            "hospital_id": hospital_data.hospital_id,
            "hospital_name": hospital_data.hospital_name,
            "provider": "mock",
            "tokens_used": len(mock_content.split()),
            "response_time": 0.1
        }
    
    def _generate_mock_cost_recommendations(self, hospital_data: HospitalFinancialData, output_format: str) -> Dict[str, Any]:
        """Generate mock cost recommendations for testing."""
        mock_content = f"""
## Cost-Saving Recommendations for {hospital_data.hospital_name}

Based on financial analysis, we recommend:

**Immediate Actions (0-3 months):**
1. Revenue cycle optimization - Expected savings: $500K-$750K annually
2. Supply chain efficiency - Expected savings: $200K-$400K annually

**Medium-term Initiatives (3-12 months):**
1. Staffing optimization - Expected savings: $1M-$1.5M annually
2. Energy management - Expected savings: $100K-$200K annually

**Total Potential Savings:** $1.8M-$2.85M annually
**Implementation Cost:** $500K-$750K
**ROI Timeline:** 6-9 months
        """.strip()
        
        formatted_content = self._format_markdown(
            mock_content,
            f"Cost Recommendations - {hospital_data.hospital_name}",
            hospital_data
        )
        
        file_path = self._save_report(
            formatted_content,
            f"mock_recommendations_{hospital_data.hospital_id}_{datetime.now().strftime('%Y%m%d')}",
            output_format
        )
        
        return {
            "success": True,
            "content": mock_content,
            "formatted_content": formatted_content,
            "file_path": str(file_path),
            "hospital_id": hospital_data.hospital_id,
            "hospital_name": hospital_data.hospital_name,
            "provider": "mock",
            "tokens_used": len(mock_content.split()),
            "response_time": 0.1
        }
    
    def _generate_mock_executive_briefing(self, hospitals_data: List[HospitalFinancialData], output_format: str) -> Dict[str, Any]:
        """Generate mock executive briefing for testing."""
        total_hospitals = len(hospitals_data)
        
        mock_content = f"""
# Executive Briefing: Hospital Portfolio Financial Analysis

## Executive Summary
Analysis covers {total_hospitals} hospitals with significant financial challenges requiring attention.

**Key Findings:**
- 40% of hospitals show signs of financial distress
- Average operating margin below industry benchmarks
- Cash positions require monitoring

## Risk Assessment
**High Risk:** 2 hospitals (immediate attention required)
**Medium Risk:** 1 hospital (monitor closely)
**Low Risk:** {total_hospitals - 3} hospitals (stable)

## Strategic Recommendations
1. Deploy turnaround teams to high-risk facilities
2. Implement emergency cash management protocols
3. Standardize revenue cycle processes

**Total Portfolio Risk:** $25M in potential losses
**Intervention Cost:** $17M
**Expected ROI:** 18 months
        """.strip()
        
        formatted_content = self._format_html(
            mock_content,
            "Executive Briefing - Hospital Portfolio Analysis"
        )
        
        file_path = self._save_report(
            formatted_content,
            f"mock_executive_briefing_{datetime.now().strftime('%Y%m%d')}",
            output_format
        )
        
        return {
            "success": True,
            "content": mock_content,
            "formatted_content": formatted_content,
            "file_path": str(file_path),
            "hospitals_count": total_hospitals,
            "provider": "mock",
            "tokens_used": len(mock_content.split()),
            "response_time": 0.2
        }


# Convenience functions
def generate_hospital_summary(input_data, **kwargs):
    """Generate hospital summary using default configuration."""
    generator = StreamlinedReportGenerator()
    return generator.generate_hospital_summary(input_data, **kwargs)

def generate_risk_narrative(input_data, **kwargs):
    """Generate risk narrative using default configuration."""
    generator = StreamlinedReportGenerator()
    return generator.generate_risk_narrative(input_data, **kwargs)

def generate_cost_recommendations(input_data, **kwargs):
    """Generate cost recommendations using default configuration."""
    generator = StreamlinedReportGenerator()
    return generator.generate_cost_recommendations(input_data, **kwargs)

def generate_executive_briefing(input_data, **kwargs):
    """Generate executive briefing using default configuration."""
    generator = StreamlinedReportGenerator()
    return generator.generate_executive_briefing(input_data, **kwargs)

def generate_batch_summaries(input_data, **kwargs):
    """Generate batch summaries using default configuration."""
    generator = StreamlinedReportGenerator()
    return generator.generate_batch_summaries(input_data, **kwargs)
