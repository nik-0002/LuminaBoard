import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json

class DataProcessor:
    """
    Utility functions for data transformation and validation
    Works directly with CSV datasets
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.grower_data = None
        self.campaign_data = None
        self.rep_data = None
        self.retailer_data = None
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV datasets from data directory
        """
        datasets = {}
        
        # Load growers data
        growers_path = os.path.join(self.data_dir, 'growers.csv')
        if os.path.exists(growers_path):
            self.grower_data = pd.read_csv(growers_path)
            datasets['growers'] = self.grower_data
        
        # Load campaign data
        campaign_path = os.path.join(self.data_dir, 'digital_funnel_weekly.csv')
        if os.path.exists(campaign_path):
            self.campaign_data = pd.read_csv(campaign_path)
            datasets['campaigns'] = self.campaign_data
        
        # Load reps/territory data
        reps_path = os.path.join(self.data_dir, 'reps_territory.csv')
        if os.path.exists(reps_path):
            self.rep_data = pd.read_csv(reps_path)
            datasets['reps'] = self.rep_data
        
        # Load retailer data
        retailer_path = os.path.join(self.data_dir, 'retailers.csv')
        if os.path.exists(retailer_path):
            self.retailer_data = pd.read_csv(retailer_path)
            datasets['retailers'] = self.retailer_data
        
        return datasets
    
    def clean_grower_data(self, grower_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Validate and clean grower profile data from CSV
        """
        if grower_data is None:
            grower_data = self.grower_data
        
        if grower_data is None:
            return pd.DataFrame()
        
        df = grower_data.copy()
        
        # Required fields
        required_fields = ['grower_id', 'state', 'district', 'language', 'device_type']
        df = df.dropna(subset=required_fields)
        
        # Fill defaults
        df['device_type'].fillna('smartphone', inplace=True)
        df['language'].fillna('Hindi', inplace=True)
        df['grower_age'].fillna(df['grower_age'].median(), inplace=True)
        
        return df
    
    def extract_temporal_features(self, timestamp: str) -> Dict:
        """
        Extract temporal features from ISO timestamp
        """
        try:
            dt = datetime.fromisoformat(timestamp)
        except:
            dt = datetime.strptime(timestamp, '%Y-%m-%d')
        
        return {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'day_of_month': dt.day,
            'month': dt.month,
            'quarter': (dt.month - 1) // 3 + 1,
            'is_weekend': dt.weekday() >= 5,
            'season': self._get_agricultural_season(dt.month)
        }
    
    def _get_agricultural_season(self, month: int) -> str:
        """
        Get agricultural season for India
        Rabi: Oct-Mar (winter crops)
        Kharif: Jun-Sep (monsoon crops)
        Summer: Mar-May
        """
        if month in [3, 4, 5]:
            return 'summer'
        elif month in [6, 7, 8, 9]:
            return 'kharif'
        elif month in [10, 11]:
            return 'rabi'
        else:
            return 'rabi'
    
    def parse_crop_calendar(self, crop_json: str) -> Dict:
        """
        Parse crop calendar from JSON string in grower data
        """
        try:
            if isinstance(crop_json, str):
                return json.loads(crop_json)
            return crop_json
        except:
            return {}
    
    def aggregate_campaign_data(self, campaign_data: Optional[pd.DataFrame] = None, 
                               group_by: str = 'state') -> pd.DataFrame:
        """
        Aggregate campaign data by specified dimension from CSV
        """
        if campaign_data is None:
            campaign_data = self.campaign_data
        
        if campaign_data is None:
            return pd.DataFrame()
        
        df = campaign_data.copy()
        
        if group_by == 'state':
            return df.groupby('state').agg({
                'social_post_impression': 'sum',
                'landing_page_visits': 'sum',
                'lead_form_submission': 'sum'
            }).reset_index()
        
        elif group_by == 'crop':
            return df.groupby('campaign_crop').agg({
                'social_post_impression': 'sum',
                'landing_page_visits': 'sum',
                'lead_form_submission': 'sum'
            }).reset_index()
        
        elif group_by == 'product':
            return df.groupby('campaign_product').agg({
                'social_post_impression': 'sum',
                'landing_page_visits': 'sum',
                'lead_form_submission': 'sum'
            }).reset_index()
        
        return df
    
    def get_growers_by_region(self, state: str, crop: str = None) -> pd.DataFrame:
        """
        Get growers filtered by state and optionally crop
        """
        if self.grower_data is None:
            return pd.DataFrame()
        
        df = self.grower_data[self.grower_data['state'] == state].copy()
        
        if crop:
            # Parse crop calendar and filter
            df = df[df['grower_crop_calendar'].str.contains(crop, case=False, na=False)]
        
        return df
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict:
        """
        Get aggregated metrics for a campaign from CSV
        """
        if self.campaign_data is None:
            return {}
        
        campaign_rows = self.campaign_data[self.campaign_data['campaign_id'] == campaign_id]
        
        if campaign_rows.empty:
            return {}
        
        return {
            'campaign_id': campaign_id,
            'total_impressions': campaign_rows['social_post_impression'].sum(),
            'total_visits': campaign_rows['landing_page_visits'].sum(),
            'total_submissions': campaign_rows['lead_form_submission'].sum(),
            'avg_conversion': campaign_rows['lead_form_submission'].sum() / max(campaign_rows['landing_page_visits'].sum(), 1),
            'product': campaign_rows['campaign_product'].iloc[0] if not campaign_rows.empty else None,
            'crop': campaign_rows['campaign_crop'].iloc[0] if not campaign_rows.empty else None
        }
