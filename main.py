"""
Lumina Board - AI-Powered Agricultural Marketing & Outreach Platform
Main application orchestrator - works with CSV datasets
"""

import os
import json
import logging
from typing import Dict
import pandas as pd

from utils.data_processors import DataProcessor
from utils.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agrimarketing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgriculturalMarketingPlatform:
    """
    Main platform orchestrator
    Works with CSV datasets for growers, campaigns, products, and retailers
    """
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize the platform with CSV data"""
        logger.info("Initializing Agricultural Marketing Platform...")
        
        self.data_processor = DataProcessor(data_dir)
        self.metrics_calc = MetricsCalculator()
        
        # Load datasets
        logger.info("Loading datasets from CSV...")
        self.datasets = self.data_processor.load_all_datasets()
        logger.info(f"Loaded datasets: {list(self.datasets.keys())}")
        
        logger.info("Platform initialization complete!")
    
    def analyze_campaign_performance(self) -> Dict:
        """
        Analyze all campaign performance from CSV data
        """
        logger.info("Analyzing campaign performance...")
        
        if 'campaigns' not in self.datasets:
            logger.warning("No campaign data loaded")
            return {}
        
        campaigns_df = self.datasets['campaigns']
        
        analysis = {
            'total_campaigns': campaigns_df['campaign_id'].nunique(),
            'total_impressions': campaigns_df['social_post_impression'].sum(),
            'total_visits': campaigns_df['landing_page_visits'].sum(),
            'total_submissions': campaigns_df['lead_form_submission'].sum(),
            'avg_ctr': self.metrics_calc.calculate_impression_to_visit(
                campaigns_df['social_post_impression'].sum(),
                campaigns_df['landing_page_visits'].sum()
            ),
            'avg_conversion': self.metrics_calc.calculate_visit_conversion(
                campaigns_df['landing_page_visits'].sum(),
                campaigns_df['lead_form_submission'].sum()
            ),
            'by_product': self.metrics_calc.analyze_product_performance(campaigns_df),
            'by_crop': self.metrics_calc.analyze_regional_performance(campaigns_df, 'crop')
        }
        
        return analysis
    
    def get_grower_segmentation(self) -> Dict:
        """
        Get grower segmentation from CSV data
        """
        logger.info("Analyzing grower segmentation...")
        
        if 'growers' not in self.datasets:
            logger.warning("No grower data loaded")
            return {}
        
        growers_df = self.datasets['growers']
        
        segmentation = {
            'total_growers': len(growers_df),
            'by_state': growers_df['state'].value_counts().to_dict(),
            'by_device': growers_df['device_type'].value_counts().to_dict(),
            'by_language': growers_df['language'].value_counts().to_dict(),
            'avg_age': growers_df['grower_age'].mean(),
            'avg_farm_size': growers_df['grower_farm_size'].mean(),
            'gender_distribution': growers_df['gender'].value_counts().to_dict()
        }
        
        return segmentation
    
    def get_regional_trends(self, state: str = None) -> Dict:
        """
        Get regional trends from grower data
        """
        logger.info(f"Analyzing regional trends for state: {state}")
        
        if 'growers' not in self.datasets:
            return {}
        
        growers_df = self.datasets['growers']
        
        if state:
            growers_df = growers_df[growers_df['state'] == state]
        
        trends = {
            'total_growers': len(growers_df),
            'avg_farm_size': growers_df['grower_farm_size'].mean(),
            'device_adoption': {
                'smartphone': len(growers_df[growers_df['device_type'] == 'smartphone']),
                'keypad': len(growers_df[growers_df['device_type'] == 'keypad']),
                'unknown': len(growers_df[growers_df['device_type'] == 'unknown'])
            },
            'languages': growers_df['language'].unique().tolist()
        }
        
        return trends
    
    def generate_marketing_insights(self) -> Dict:
        """
        Generate comprehensive marketing insights
        """
        logger.info("Generating marketing insights...")
        
        insights = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'campaign_analysis': self.analyze_campaign_performance(),
            'grower_segmentation': self.get_grower_segmentation(),
            'dataset_summary': {
                'total_growers': len(self.datasets.get('growers', pd.DataFrame())),
                'total_campaigns': len(self.datasets.get('campaigns', pd.DataFrame()).groupby('campaign_id')) if 'campaigns' in self.datasets else 0,
                'total_retailers': len(self.datasets.get('retailers', pd.DataFrame())),
                'total_reps': len(self.datasets.get('reps', pd.DataFrame()))
            }
        }
        
        return insights

# Example usage
if __name__ == "__main__":
    # Initialize platform
    platform = AgriculturalMarketingPlatform()
    
    # Generate insights
    insights = platform.generate_marketing_insights()
    
    # Print results
    print("\n" + "="*60)
    print("AGRICULTURAL MARKETING PLATFORM - INSIGHTS")
    print("="*60)
    print(json.dumps(insights, indent=2, default=str))
    
    # Analyze specific region
    print("\n" + "="*60)
    print("REGIONAL ANALYSIS - PUNJAB")
    print("="*60)
    punjab_trends = platform.get_regional_trends('Punjab')
    print(json.dumps(punjab_trends, indent=2))
