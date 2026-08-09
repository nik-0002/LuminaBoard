import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

class CropSegmentationEngine:
    """
    Segments farmers into crop-based clusters for targeted marketing
    Uses KMeans clustering on agricultural and behavioral features
    """
    
    def __init__(self, n_clusters=15, random_state=42):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_names = None
        self.crop_profiles = {}
        self.is_fitted = False
    
    def extract_farmer_features(self, farmer_data):
        """
        Extract features for clustering
        """
        features = {
            'farm_size': farmer_data.get('farm_size_acres', 2.5),
            'crop_diversity': len(farmer_data.get('crops', [])),
            'primary_crop_revenue': farmer_data.get('primary_crop_revenue', 50000),
            'irrigation_capacity': farmer_data.get('irrigation_capacity', 0),  # 0-100
            'soil_fertility': farmer_data.get('soil_fertility', 0),  # 0-100
            'market_access_km': farmer_data.get('market_access_km', 20),
            'smartphone_adoption': farmer_data.get('smartphone_adoption', 0),  # 0-100
            'education_years': farmer_data.get('education_years', 8),
            'prior_product_usage': len(farmer_data.get('product_history', [])),
            'risk_aversion': farmer_data.get('risk_aversion', 50),  # 0-100
        }
        return features
    
    def fit(self, farmer_data_list):
        """
        Fit the clustering model on farmer data
        """
        feature_list = []
        for farmer in farmer_data_list:
            features = self.extract_farmer_features(farmer)
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        self.feature_names = feature_df.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(feature_df)
        self.kmeans.fit(X_scaled)
        
        # Generate cluster profiles
        self._generate_cluster_profiles(feature_df)
        self.is_fitted = True
    
    def _generate_cluster_profiles(self, feature_df):
        """Generate descriptive profiles for each cluster"""
        feature_df['cluster'] = self.kmeans.labels_
        
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_data = feature_df[feature_df['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'characteristics': cluster_data.drop('cluster', axis=1).mean().to_dict(),
                'suggested_crops': self._suggest_crops_for_cluster(cluster_data),
                'marketing_channel': self._suggest_channel(cluster_data),
                'content_preference': self._suggest_content(cluster_data),
                'engagement_strategy': self._suggest_strategy(cluster_data)
            }
            self.crop_profiles[cluster_id] = profile
    
    def _suggest_crops_for_cluster(self, cluster_data):
        """Suggest suitable crops for a cluster"""
        avg_farm_size = cluster_data['farm_size'].mean()
        avg_irrigation = cluster_data['irrigation_capacity'].mean()
        
        crop_suitability = {
            'high_value_vegetables': 0,
            'cereals': 0,
            'pulses': 0,
            'cash_crops': 0,
            'dairy_farming': 0
        }
        
        if avg_farm_size < 2:
            crop_suitability['high_value_vegetables'] = 0.9
            crop_suitability['pulses'] = 0.7
        elif avg_farm_size < 5:
            crop_suitability['cereals'] = 0.8
            crop_suitability['pulses'] = 0.8
        else:
            crop_suitability['cash_crops'] = 0.8
            crop_suitability['cereals'] = 0.7
        
        if avg_irrigation > 70:
            crop_suitability['high_value_vegetables'] = 0.95
            crop_suitability['dairy_farming'] = 0.85
        
        return sorted(crop_suitability.items(), key=lambda x: x[1], reverse=True)
    
    def _suggest_channel(self, cluster_data):
        """Suggest communication channel based on cluster characteristics"""
        avg_smartphone = cluster_data['smartphone_adoption'].mean()
        avg_education = cluster_data['education_years'].mean()
        
        if avg_smartphone > 70:
            return 'whatsapp_video'
        elif avg_smartphone > 50:
            return 'whatsapp_text'
        elif avg_education > 10:
            return 'sms'
        else:
            return 'voice_call'
    
    def _suggest_content(self, cluster_data):
        """Suggest content preference"""
        avg_education = cluster_data['education_years'].mean()
        
        if avg_education > 12:
            return 'detailed_technical'
        elif avg_education > 8:
            return 'mixed_text_visual'
        else:
            return 'visual_heavy_simple'
    
    def _suggest_strategy(self, cluster_data):
        """Suggest engagement strategy"""
        avg_risk_aversion = cluster_data['risk_aversion'].mean()
        avg_prior_usage = cluster_data['prior_product_usage'].mean()
        
        if avg_prior_usage > 5:
            return 'loyalty_and_upsell'
        elif avg_risk_aversion > 70:
            return 'trust_building_gradual'
        else:
            return 'innovative_early_adoption'
    
    def predict_cluster(self, farmer_data):
        """Predict cluster for a new farmer"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        features = self.extract_farmer_features(farmer_data)
        feature_array = np.array([features[fn] for fn in self.feature_names]).reshape(1, -1)
        X_scaled = self.scaler.transform(feature_array)
        
        cluster_id = self.kmeans.predict(X_scaled)[0]
        profile = self.crop_profiles.get(cluster_id, {})
        
        return {
            'cluster_id': cluster_id,
            'profile': profile,
            'confidence': 1 - (self.kmeans.transform(X_scaled).min() / 
                             self.kmeans.transform(X_scaled).max())
        }
    
    def get_cluster_profiles(self):
        """Get all cluster profiles"""
        return self.crop_profiles