from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class MetricsCalculator:
    """
    Calculate agricultural marketing metrics and KPIs
    Works with CSV data and campaign performance
    """
    
    @staticmethod
    def calculate_delivery_rate(delivered: int, sent: int) -> float:
        """Calculate message delivery rate (%)"""
        return (delivered / sent * 100) if sent > 0 else 0
    
    @staticmethod
    def calculate_engagement_rate(engaged: int, delivered: int) -> float:
        """Calculate farmer engagement rate (%)"""
        return (engaged / delivered * 100) if delivered > 0 else 0
    
    @staticmethod
    def calculate_conversion_rate(converted: int, engaged: int) -> float:
        """Calculate conversion rate from engagement (%)"""
        return (converted / engaged * 100) if engaged > 0 else 0
    
    @staticmethod
    def calculate_ctr(clicks: int, impressions: int) -> float:
        """Calculate Click-Through Rate"""
        return (clicks / impressions * 100) if impressions > 0 else 0
    
    @staticmethod
    def calculate_roas(revenue: float, cost: float) -> float:
        """Calculate Return on Ad Spend"""
        return (revenue / cost) if cost > 0 else 0
    
    @staticmethod
    def calculate_cac(cost: float, conversions: int) -> float:
        """Calculate Customer Acquisition Cost"""
        return (cost / conversions) if conversions > 0 else 0
    
    @staticmethod
    def calculate_visit_conversion(visits: int, submissions: int) -> float:
        """Calculate landing page to form submission conversion"""
        return (submissions / visits * 100) if visits > 0 else 0
    
    @staticmethod
    def calculate_impression_to_visit(impressions: int, visits: int) -> float:
        """Calculate impression to landing page visit conversion"""
        return (visits / impressions * 100) if impressions > 0 else 0
    
    @staticmethod
    def calculate_trend(values: List[float], period: int = 7) -> str:
        """
        Determine trend direction (UP, DOWN, STABLE)
        """
        if len(values) < period:
            return "INSUFFICIENT_DATA"
        
        recent = np.mean(values[-period:])
        previous = np.mean(values[-2*period:-period])
        
        change_percent = ((recent - previous) / previous * 100) if previous != 0 else 0
        
        if change_percent > 5:
            return "UP"
        elif change_percent < -5:
            return "DOWN"
        else:
            return "STABLE"
    
    @staticmethod
    def calculate_variance(values: List[float]) -> float:
        """Calculate statistical variance in metrics"""
        return float(np.var(values)) if values else 0
    
    @staticmethod
    def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for a metric"""
        if len(values) < 2:
            return (0, 0)
        
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        margin = 1.96 * std_err  # 95% CI
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def analyze_regional_performance(data: pd.DataFrame, metric: str = 'submissions') -> Dict:
        """
        Analyze performance metrics by region
        """
        if data.empty:
            return {}
        
        groupby_col = 'state' if 'state' in data.columns else 'region'
        
        if metric == 'submissions':
            regional = data.groupby(groupby_col)['lead_form_submission'].sum().to_dict()
        elif metric == 'visits':
            regional = data.groupby(groupby_col)['landing_page_visits'].sum().to_dict()
        elif metric == 'impressions':
            regional = data.groupby(groupby_col)['social_post_impression'].sum().to_dict()
        else:
            regional = {}
        
        return regional
    
    @staticmethod
    def analyze_product_performance(data: pd.DataFrame) -> Dict:
        """
        Analyze campaign performance by product
        """
        if data.empty:
            return {}
        
        product_perf = data.groupby('campaign_product').agg({
            'social_post_impression': 'sum',
            'landing_page_visits': 'sum',
            'lead_form_submission': 'sum'
        }).to_dict('index')
        
        return product_perf
    
    @staticmethod
    def identify_top_performers(data: pd.DataFrame, metric: str = 'submissions', top_n: int = 5) -> List[Dict]:
        """
        Identify top performing campaigns/products
        """
        if data.empty:
            return []
        
        if metric == 'submissions':
            grouped = data.groupby('campaign_id')['lead_form_submission'].sum().nlargest(top_n)
        elif metric == 'conversions':
            data['conversion_rate'] = data['lead_form_submission'] / (data['landing_page_visits'] + 1)
            grouped = data.groupby('campaign_id')['conversion_rate'].mean().nlargest(top_n)
        else:
            grouped = data.groupby('campaign_id')['social_post_impression'].sum().nlargest(top_n)
        
        return [{'id': idx, 'value': val} for idx, val in grouped.items()]
