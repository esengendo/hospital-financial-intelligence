"""
Prompt Templates for Hospital Financial Analysis LLM Integration

Provides structured, context-aware prompts for generating:
- Financial summaries
- Risk narratives  
- Cost-saving recommendations
- Executive briefings
"""

from typing import Dict, Any, List, Optional
import json


class PromptTemplates:
    """Collection of structured prompt templates for financial analysis."""
    
    @staticmethod
    def financial_summary_prompt(
        hospital_data: Dict[str, Any],
        risk_score: float,
        risk_level: str,
        key_metrics: Dict[str, float]
    ) -> str:
        """
        Generate prompt for hospital financial summary.
        
        Args:
            hospital_data: Hospital information and financial data
            risk_score: Model-predicted risk score
            risk_level: Risk classification (Low/Medium/High)
            key_metrics: Important financial ratios and metrics
            
        Returns:
            Structured prompt for financial summary generation
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior healthcare financial analyst with 15+ years of experience analyzing hospital financial performance. Your expertise includes GAAP accounting, healthcare regulations, and financial distress prediction. Provide clear, professional analysis for hospital executives and board members.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate a comprehensive financial summary for the following hospital:

**HOSPITAL INFORMATION:**
- Provider ID: {hospital_data.get('provider_id', 'N/A')}
- Hospital Name: {hospital_data.get('hospital_name', 'Unknown')}
- Year: {hospital_data.get('year', 'N/A')}
- State: {hospital_data.get('state', 'N/A')}
- Ownership Type: {hospital_data.get('ownership_type', 'N/A')}

**FINANCIAL DISTRESS ASSESSMENT:**
- Risk Score: {risk_score:.3f}
- Risk Classification: {risk_level}
- Assessment Date: {hospital_data.get('year', 'Current')}

**KEY FINANCIAL METRICS:**
{_format_metrics_table(key_metrics)}

**REQUIREMENTS:**
1. Executive Summary (2-3 sentences)
2. Financial Health Overview
3. Key Performance Indicators Analysis
4. Areas of Concern (if any)
5. Strengths and Opportunities

**OUTPUT FORMAT:**
- Use professional, accessible language
- Include specific numbers and percentages
- Highlight critical insights
- Keep total length under 400 words
- Use bullet points for clarity

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    @staticmethod
    def risk_narrative_prompt(
        hospital_data: Dict[str, Any],
        prediction_details: Dict[str, Any],
        shap_explanations: Dict[str, float],
        historical_trends: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """
        Generate prompt for explaining why a hospital is at financial risk.
        
        Args:
            hospital_data: Hospital information
            prediction_details: Model prediction results and confidence
            shap_explanations: SHAP feature importance values
            historical_trends: Historical financial trends
            
        Returns:
            Structured prompt for risk explanation
        """
        shap_formatted = _format_shap_explanations(shap_explanations)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert healthcare financial analyst specializing in risk assessment and regulatory compliance. Your role is to explain complex machine learning predictions in terms that hospital administrators and board members can understand and act upon.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Explain why this hospital has been flagged for financial risk:

**HOSPITAL PROFILE:**
- Provider: {hospital_data.get('provider_id', 'N/A')}
- Year: {hospital_data.get('year', 'N/A')}
- Risk Score: {prediction_details.get('risk_score', 0):.3f}
- Confidence: {prediction_details.get('confidence', 0):.1%}

**MODEL EXPLANATION (SHAP Analysis):**
The following factors contributed most to this risk assessment:

{shap_formatted}

**HISTORICAL CONTEXT:**
{_format_historical_trends(historical_trends) if historical_trends else "Historical data not available for this analysis."}

**TASK:**
Provide a clear, narrative explanation that:

1. **Root Cause Analysis**: Identify the primary drivers of financial risk
2. **Business Impact**: Explain how these factors affect hospital operations
3. **Regulatory Considerations**: Note any compliance or reporting implications
4. **Interconnected Risks**: Describe how different factors compound each other
5. **Urgency Assessment**: Indicate timeline for potential financial distress

**REQUIREMENTS:**
- Use plain English, avoid technical jargon
- Reference specific metrics and their thresholds
- Include industry context and benchmarks
- Maintain professional, objective tone
- Structure with clear headers and bullet points
- Keep under 500 words

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    @staticmethod
    def cost_recommendations_prompt(
        hospital_data: Dict[str, Any],
        financial_metrics: Dict[str, float],
        benchmark_data: Dict[str, float],
        risk_factors: List[str]
    ) -> str:
        """
        Generate prompt for cost-saving recommendations.
        
        Args:
            hospital_data: Hospital information
            financial_metrics: Current financial performance
            benchmark_data: Industry benchmarks for comparison
            risk_factors: Identified areas of financial concern
            
        Returns:
            Structured prompt for recommendation generation
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a healthcare operations consultant with expertise in financial turnaround strategies, operational efficiency, and regulatory compliance. You specialize in providing actionable cost-saving recommendations for hospitals facing financial challenges.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Develop cost-saving recommendations for this hospital:

**HOSPITAL OVERVIEW:**
- Provider ID: {hospital_data.get('provider_id', 'N/A')}
- Year: {hospital_data.get('year', 'N/A')}
- Ownership: {hospital_data.get('ownership_type', 'N/A')}
- Location: {hospital_data.get('state', 'N/A')}

**CURRENT FINANCIAL PERFORMANCE:**
{_format_metrics_comparison(financial_metrics, benchmark_data)}

**IDENTIFIED RISK FACTORS:**
{_format_risk_factors(risk_factors)}

**ANALYSIS REQUIREMENTS:**
Generate specific, actionable recommendations in these categories:

1. **Immediate Actions (0-90 days)**
   - Quick wins with minimal investment
   - Emergency cost controls
   - Revenue optimization opportunities

2. **Short-term Initiatives (3-12 months)**
   - Operational efficiency improvements
   - Staffing optimization
   - Technology upgrades with ROI

3. **Strategic Long-term Plans (1-3 years)**
   - Service line optimization
   - Capital investment priorities
   - Partnership opportunities

**FOR EACH RECOMMENDATION:**
- Estimated cost savings ($ amount or %)
- Implementation timeline
- Required resources/investment
- Potential risks or challenges
- Success metrics

**CONSTRAINTS:**
- Maintain quality of patient care
- Ensure regulatory compliance
- Consider local market conditions
- Respect union agreements (if applicable)
- Maintain community service obligations

**OUTPUT FORMAT:**
- Executive summary of total potential savings
- Prioritized action plan
- Implementation roadmap
- Risk mitigation strategies
- Success measurement framework

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    @staticmethod
    def executive_briefing_prompt(
        portfolio_summary: Dict[str, Any],
        key_findings: List[str],
        urgent_actions: List[str],
        market_context: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for executive briefing report.
        
        Args:
            portfolio_summary: High-level portfolio statistics
            key_findings: Critical insights from analysis
            urgent_actions: Immediate actions required
            market_context: Healthcare market conditions
            
        Returns:
            Structured prompt for executive briefing
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Chief Financial Officer (CFO) preparing a board presentation on hospital financial analysis. Your audience includes board members, investors, and senior executives who need high-level insights with strategic implications and clear action items.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Create an executive briefing based on comprehensive hospital financial analysis:

**PORTFOLIO OVERVIEW:**
- Analysis Period: {portfolio_summary.get('analysis_period', 'N/A')}
- Hospitals Analyzed: {portfolio_summary.get('total_hospitals', 0):,}
- Geographic Coverage: {portfolio_summary.get('states_covered', 'N/A')} states
- Risk Distribution: {portfolio_summary.get('risk_distribution', {})}

**CRITICAL FINDINGS:**
{_format_key_findings(key_findings)}

**IMMEDIATE ACTIONS REQUIRED:**
{_format_urgent_actions(urgent_actions)}

**MARKET CONTEXT:**
{_format_market_context(market_context)}

**BRIEFING STRUCTURE:**

1. **Executive Summary**
   - Key financial health indicators
   - Overall risk assessment
   - Strategic implications

2. **Performance Highlights**
   - Top performing hospitals
   - Concerning trends
   - Benchmark comparisons

3. **Risk Assessment**
   - High-risk facilities identification
   - Financial distress probability
   - Geographic risk patterns

4. **Strategic Recommendations**
   - Portfolio optimization
   - Investment priorities
   - Divestiture considerations

5. **Action Plan**
   - 30/60/90-day priorities
   - Resource allocation
   - Success metrics

**REQUIREMENTS:**
- Board-ready presentation format
- Data-driven insights with specific numbers
- Clear financial impact quantification
- Regulatory and compliance considerations
- Competitive landscape implications
- Maximum 800 words

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


# Helper functions for formatting
def _format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable table."""
    formatted_lines = []
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in metric.lower() or 'margin' in metric.lower():
                formatted_lines.append(f"- {metric}: {value:.2%}")
            else:
                formatted_lines.append(f"- {metric}: {value:,.2f}")
        else:
            formatted_lines.append(f"- {metric}: {value}")
    return "\n".join(formatted_lines)


def _format_shap_explanations(shap_values: Dict[str, float]) -> str:
    """Format SHAP values for readable explanation."""
    if not shap_values:
        return "SHAP explanations not available."
    
    # Sort by absolute importance
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    formatted_lines = []
    for feature, importance in sorted_features[:5]:  # Top 5 features
        direction = "INCREASES" if importance > 0 else "DECREASES"
        formatted_lines.append(f"- {feature}: {direction} risk by {abs(importance):.3f}")
    
    return "\n".join(formatted_lines)


def _format_historical_trends(trends: Dict[str, List[float]]) -> str:
    """Format historical trends data."""
    if not trends:
        return "No historical trend data available."
    
    formatted_lines = []
    for metric, values in trends.items():
        if len(values) >= 2:
            trend_direction = "improving" if values[-1] > values[0] else "declining"
            formatted_lines.append(f"- {metric}: {trend_direction} trend over analysis period")
    
    return "\n".join(formatted_lines)


def _format_metrics_comparison(current: Dict[str, float], benchmark: Dict[str, float]) -> str:
    """Format current metrics vs benchmarks."""
    formatted_lines = []
    for metric in current.keys():
        current_val = current[metric]
        benchmark_val = benchmark.get(metric, 0)
        
        if benchmark_val > 0:
            vs_benchmark = ((current_val - benchmark_val) / benchmark_val) * 100
            comparison = f"vs benchmark: {vs_benchmark:+.1f}%"
        else:
            comparison = "benchmark not available"
            
        formatted_lines.append(f"- {metric}: {current_val:.2f} ({comparison})")
    
    return "\n".join(formatted_lines)


def _format_risk_factors(factors: List[str]) -> str:
    """Format risk factors list."""
    return "\n".join(f"• {factor}" for factor in factors)


def _format_key_findings(findings: List[str]) -> str:
    """Format key findings list."""
    return "\n".join(f"{i+1}. {finding}" for i, finding in enumerate(findings))


def _format_urgent_actions(actions: List[str]) -> str:
    """Format urgent actions list."""
    return "\n".join(f"→ {action}" for action in actions)


def _format_market_context(context: Dict[str, Any]) -> str:
    """Format market context information."""
    formatted_lines = []
    for key, value in context.items():
        formatted_lines.append(f"- {key}: {value}")
    return "\n".join(formatted_lines) 