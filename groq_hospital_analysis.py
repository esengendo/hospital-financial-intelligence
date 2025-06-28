#!/usr/bin/env python3
"""
Production-Ready Hospital Financial Analysis using Groq API
Hospital Financial Intelligence - Phase 6 LLM Integration
"""

import os
import json
import pandas as pd
import numpy as np
import requests
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union

# Load environment variables
load_dotenv()

class GroqHospitalAnalyzer:
    """Production-ready hospital financial analyzer using Groq API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer with Groq API key."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file.")
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "llama-3.1-8b-instant"
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_hospital_data(self, year: int = 2023) -> pd.DataFrame:
        """Load hospital financial data for analysis."""
        try:
            data_file = f"data/features_enhanced/features_enhanced_{year}.parquet"
            df = pd.read_parquet(data_file)
            
            # Fill missing values with reasonable defaults
            df['operating_margin'] = df['operating_margin'].fillna(0)
            df['current_ratio'] = df['current_ratio'].fillna(1.0)
            df['days_cash_on_hand'] = df['days_cash_on_hand'].fillna(30)
            df['total_margin'] = df['total_margin'].fillna(0)
            
            print(f"✅ Loaded {len(df)} hospitals from {year}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_single_hospital(
        self, 
        hospital_data: pd.Series, 
        hospital_id: str = None
    ) -> Dict:
        """Analyze a single hospital's financial performance."""
        
        hospital_id = hospital_id or str(hospital_data.get('provider_id', 'Unknown'))
        
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
You are a healthcare financial analyst. Provide a comprehensive analysis of this hospital's 2023 financial performance:

**Hospital Financial Metrics:**
- Operating Margin: {metrics['operating_margin']:.2%}
- Total Margin: {metrics['total_margin']:.2%}
- Current Ratio: {metrics['current_ratio']:.2f}
- Days Cash on Hand: {metrics['days_cash_on_hand']:.0f} days
- Debt Service Coverage: {metrics['debt_service_coverage']:.2f}
- Operating Revenue: ${metrics['operating_revenue']:,.0f}

**Required Analysis:**
1. **Financial Health Status**: (Excellent/Good/Concerning/Critical)
2. **Key Strengths**: (2-3 bullet points)
3. **Primary Risk Factors**: (2-3 bullet points)
4. **Strategic Recommendations**: (3-4 actionable items)
5. **Executive Summary**: (2-3 sentences for board presentation)

Format professionally for healthcare executives. Be specific and actionable.
"""
        
        # Make API call
        try:
            response = self._call_groq_api(prompt, max_tokens=600)
            
            if response['success']:
                return {
                    'hospital_id': hospital_id,
                    'metrics': metrics,
                    'analysis': response['content'],
                    'tokens_used': response['tokens'],
                    'cost_usd': response['tokens'] * 0.00000059,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            else:
                return {
                    'hospital_id': hospital_id,
                    'error': response['error'],
                    'success': False
                }
                
        except Exception as e:
            return {
                'hospital_id': hospital_id,
                'error': str(e),
                'success': False
            }
    
    def analyze_portfolio(
        self, 
        max_hospitals: int = 10,
        year: int = 2023
    ) -> Dict:
        """Analyze multiple hospitals in a portfolio."""
        
        print(f"🏥 Analyzing Hospital Portfolio ({max_hospitals} hospitals)")
        print("=" * 60)
        
        # Load data
        df = self.load_hospital_data(year)
        df_sample = df.head(max_hospitals)
        
        results = []
        total_tokens = 0
        successful_analyses = 0
        
        for idx, (_, hospital) in enumerate(df_sample.iterrows(), 1):
            hospital_id = str(hospital.get('provider_id', f'Hospital_{idx}'))
            print(f"\n🔍 Analyzing {idx}/{len(df_sample)}: {hospital_id}")
            
            # Analyze hospital
            result = self.analyze_single_hospital(hospital, hospital_id)
            
            if result['success']:
                results.append(result)
                total_tokens += result['tokens_used']
                successful_analyses += 1
                print(f"   ✅ Complete ({result['tokens_used']} tokens, ${result['cost_usd']:.6f})")
                
                # Rate limiting
                time.sleep(0.5)
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        # Generate portfolio summary
        if results:
            portfolio_summary = self._generate_portfolio_summary(results, year)
            
            # Save comprehensive report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'year': year,
                    'hospitals_analyzed': successful_analyses,
                    'total_tokens': total_tokens,
                    'total_cost_usd': total_tokens * 0.00000059,
                    'model': self.model,
                    'provider': 'groq'
                },
                'portfolio_summary': portfolio_summary,
                'individual_analyses': results
            }
            
            # Save JSON report
            json_file = self.reports_dir / f"portfolio_analysis_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Save Markdown report
            md_file = self.reports_dir / f"portfolio_analysis_{timestamp}.md"
            self._save_markdown_report(report_data, md_file)
            
            print(f"\n🎉 Portfolio Analysis Complete!")
            print(f"📊 Successful analyses: {successful_analyses}/{max_hospitals}")
            print(f"🔢 Total tokens: {total_tokens}")
            print(f"💰 Total cost: ${total_tokens * 0.00000059:.6f}")
            print(f"📄 Reports saved:")
            print(f"   📋 {md_file}")
            print(f"   📊 {json_file}")
            
            return report_data
        
        else:
            print("❌ No successful analyses completed")
            return {'success': False, 'error': 'No analyses completed'}
    
    def _call_groq_api(
        self, 
        prompt: str, 
        max_tokens: int = 500, 
        temperature: float = 0.7
    ) -> Dict:
        """Make API call to Groq."""
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result["choices"][0]["message"]["content"],
                    'tokens': result.get("usage", {}).get("total_tokens", 0)
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Request failed: {str(e)}"
            }
    
    def _generate_portfolio_summary(self, results: List[Dict], year: int) -> Dict:
        """Generate portfolio-level summary statistics."""
        
        # Calculate portfolio metrics
        operating_margins = [r['metrics']['operating_margin'] for r in results]
        current_ratios = [r['metrics']['current_ratio'] for r in results]
        days_cash = [r['metrics']['days_cash_on_hand'] for r in results]
        
        return {
            'total_hospitals': len(results),
            'year': year,
            'portfolio_metrics': {
                'avg_operating_margin': np.mean(operating_margins),
                'median_operating_margin': np.median(operating_margins),
                'avg_current_ratio': np.mean(current_ratios),
                'avg_days_cash': np.mean(days_cash),
                'hospitals_with_negative_margins': sum(1 for m in operating_margins if m < 0),
                'hospitals_with_low_liquidity': sum(1 for d in days_cash if d < 30)
            },
            'cost_analysis': {
                'total_tokens': sum(r['tokens_used'] for r in results),
                'total_cost_usd': sum(r['cost_usd'] for r in results),
                'avg_cost_per_hospital': np.mean([r['cost_usd'] for r in results])
            }
        }
    
    def _save_markdown_report(self, report_data: Dict, file_path: Path):
        """Save comprehensive markdown report."""
        
        metadata = report_data['metadata']
        summary = report_data['portfolio_summary']
        
        content = f"""# Hospital Portfolio Financial Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Year:** {metadata['year']}  
**Hospitals Analyzed:** {metadata['hospitals_analyzed']}  
**AI Model:** Groq {self.model}

## Executive Summary

AI-powered financial analysis of {summary['total_hospitals']} hospitals reveals portfolio performance and strategic recommendations.

## Portfolio Performance Metrics

### Financial Health Overview
- **Average Operating Margin:** {summary['portfolio_metrics']['avg_operating_margin']:.2%}
- **Median Operating Margin:** {summary['portfolio_metrics']['median_operating_margin']:.2%}
- **Average Current Ratio:** {summary['portfolio_metrics']['avg_current_ratio']:.2f}
- **Average Days Cash:** {summary['portfolio_metrics']['avg_days_cash']:.0f} days

### Risk Assessment
- **Hospitals with Negative Margins:** {summary['portfolio_metrics']['hospitals_with_negative_margins']}/{summary['total_hospitals']} ({summary['portfolio_metrics']['hospitals_with_negative_margins']/summary['total_hospitals']*100:.1f}%)
- **Hospitals with Low Liquidity:** {summary['portfolio_metrics']['hospitals_with_low_liquidity']}/{summary['total_hospitals']} ({summary['portfolio_metrics']['hospitals_with_low_liquidity']/summary['total_hospitals']*100:.1f}%)

## Individual Hospital Analyses

"""
        
        for i, result in enumerate(report_data['individual_analyses'], 1):
            content += f"""### {i}. Hospital {result['hospital_id']}

**Financial Metrics:**
- Operating Margin: {result['metrics']['operating_margin']:.2%}
- Current Ratio: {result['metrics']['current_ratio']:.2f}
- Days Cash: {result['metrics']['days_cash_on_hand']:.0f} days
- Total Margin: {result['metrics']['total_margin']:.2%}

**AI Analysis:**
{result['analysis']}

**Analysis Cost:** {result['tokens_used']} tokens (${result['cost_usd']:.6f})

---

"""
        
        content += f"""## Cost Analysis

- **Total Tokens Used:** {metadata['total_tokens']}
- **Total Analysis Cost:** ${metadata['total_cost_usd']:.6f}
- **Average Cost per Hospital:** ${summary['cost_analysis']['avg_cost_per_hospital']:.6f}

## Recommendations

### Immediate Actions
1. **Priority Review:** Focus on hospitals with negative operating margins
2. **Liquidity Monitoring:** Implement weekly cash flow monitoring for low-liquidity hospitals
3. **Best Practice Sharing:** Identify and replicate strategies from top performers

### Strategic Initiatives
1. **Quarterly AI Analysis:** Implement regular AI-powered financial assessments
2. **Predictive Monitoring:** Use trends to identify at-risk hospitals early
3. **Benchmarking:** Compare performance against industry standards

---
*Report generated using Groq AI - {metadata['timestamp']}*
"""
        
        with open(file_path, 'w') as f:
            f.write(content)


def main():
    """Main CLI interface for hospital analysis."""
    
    parser = argparse.ArgumentParser(description='Hospital Financial Analysis using Groq AI')
    parser.add_argument('--mode', choices=['single', 'portfolio'], default='portfolio',
                       help='Analysis mode: single hospital or portfolio')
    parser.add_argument('--hospitals', type=int, default=5,
                       help='Number of hospitals to analyze in portfolio mode')
    parser.add_argument('--year', type=int, default=2023,
                       help='Year of data to analyze')
    parser.add_argument('--hospital-id', type=str,
                       help='Specific hospital ID for single mode')
    
    args = parser.parse_args()
    
    try:
        analyzer = GroqHospitalAnalyzer()
        
        if args.mode == 'portfolio':
            result = analyzer.analyze_portfolio(
                max_hospitals=args.hospitals,
                year=args.year
            )
            
        elif args.mode == 'single':
            df = analyzer.load_hospital_data(args.year)
            
            if args.hospital_id:
                # Find specific hospital
                hospital_row = df[df['provider_id'] == args.hospital_id]
                if hospital_row.empty:
                    print(f"❌ Hospital {args.hospital_id} not found")
                    return
                hospital_data = hospital_row.iloc[0]
            else:
                # Use first hospital
                hospital_data = df.iloc[0]
            
            result = analyzer.analyze_single_hospital(hospital_data)
            
            if result['success']:
                print(f"\n🏥 Analysis for Hospital {result['hospital_id']}")
                print("=" * 50)
                print(result['analysis'])
                print(f"\n💰 Cost: {result['tokens_used']} tokens (${result['cost_usd']:.6f})")
            else:
                print(f"❌ Analysis failed: {result['error']}")
        
        print(f"\n🎉 Analysis complete! Check the reports/ directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 