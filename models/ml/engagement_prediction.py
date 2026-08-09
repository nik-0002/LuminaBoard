import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class EngagementPredictor:
    """
    Predicts farmer engagement likelihood with marketing campaigns
    Factors: crop type, region, language, device, campaign content type, time of day, season
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.is_fitted = False
    
    def preprocess_features(self, campaign_data):
        """
        Convert categorical and numerical features for model input
        """
        features_list = []
        
        for record in campaign_data:
            feature_vector = {
                # Farmer demographics
                'farm_size_acres': record.get('farm_size', 2.5),
                'primary_crop_encoded': self._encode_feature('crop', record.get('primary_crop', 'rice')),
                'region_encoded': self._encode_feature('region', record.get('region', 'north')),
                'language_encoded': self._encode_feature('language', record.get('language', 'hi')),
                'device_type_encoded': self._encode_feature('device', record.get('device_type', 'smartphone')),
                
                # Campaign characteristics
                'content_type_encoded': self._encode_feature('content_type', record.get('content_type', 'text')),
                'channel_encoded': self._encode_feature('channel', record.get('channel', 'whatsapp')),
                'message_length': len(record.get('message_text', '')),
                'has_visual': 1 if record.get('has_image', False) else 0,
                'has_audio': 1 if record.get('has_audio', False) else 0,
                
                # Temporal features
                'hour_sent': record.get('timestamp', {}).get('hour', 10),
                'day_of_week': record.get('timestamp', {}).get('day_of_week', 3),
                'is_peak_season': 1 if record.get('is_peak_season', False) else 0,
                'days_since_last_message': record.get('days_since_last_message', 7),
                
                # Contextual features
                'pest_pressure_score': record.get('pest_pressure', 0),
                'weather_relevance': record.get('weather_relevance', 0.5),
                'product_relevance': record.get('product_relevance', 0.5),
                'urgency_score': record.get('urgency_score', 0.5),
                
                # Historical features
                'previous_engagement_rate': record.get('previous_engagement_rate', 0),
                'response_time_avg': record.get('response_time_avg', 180),  # seconds
                'num_interactions_history': record.get('num_interactions', 0),
            }
            features_list.append(feature_vector)
        
        return pd.DataFrame(features_list)
    
    def _encode_feature(self, category, value):
        """Encode categorical features"""
        if category not in self.encoders:
            self.encoders[category] = LabelEncoder()
        
        try:
            return self.encoders[category].transform([value])[0]
        except ValueError:
            # Handle unseen categories
            return 0
    
    def fit(self, campaign_data, engagement_labels):
        """
        Train the engagement prediction model
        engagement_labels: array of 0 (no engagement) or 1 (engaged)
        """
        X = self.preprocess_features(campaign_data)
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, engagement_labels)
        self.is_fitted = True
    
    def predict_engagement(self, campaign_data):
        """
        Predict engagement probability for campaigns
        Returns: engagement scores (0-100)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = self.preprocess_features(campaign_data)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)
        engagement_scores = (probabilities[:, 1] * 100).astype(int)
        
        return {
            'engagement_scores': engagement_scores,
            'engagement_levels': ['HIGH' if score > 70 else 'MEDIUM' if score > 40 else 'LOW' 
                                for score in engagement_scores],
            'raw_probabilities': probabilities
        }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.is_fitted:
            return None
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoders = data['encoders']
        self.feature_names = data['feature_names']
        self.is_fitted = True

class CampaignOptimizer:
    """
    Multi-armed bandit approach for campaign optimization
    Uses Thompson Sampling for optimal message variant selection
    """
    
    def __init__(self, n_variants=5, decay_rate=0.95):
        self.n_variants = n_variants
        self.decay_rate = decay_rate
        self.variant_stats = {i: {'success': 0, 'total': 0} for i in range(n_variants)}
        self.daily_stats = {}
    
    def thompson_sample(self):
        """
        Thompson Sampling: probabilistically select variant based on success rate
        """
        samples = []
        for variant_id, stats in self.variant_stats.items():
            success = stats['success'] + 1
            failures = stats['total'] - stats['success'] + 1
            
            # Beta distribution sampling
            sample = np.random.beta(success, failures)
            samples.append(sample)
        
        return np.argmax(samples)
    
    def update_variant_performance(self, variant_id, success):
        """Update performance metrics for a variant"""
        self.variant_stats[variant_id]['total'] += 1
        if success:
            self.variant_stats[variant_id]['success'] += 1
    
    def get_variant_scores(self):
        """Get performance scores for all variants"""
        scores = {}
        for variant_id, stats in self.variant_stats.items():
            if stats['total'] == 0:
                scores[variant_id] = 0
            else:
                scores[variant_id] = stats['success'] / stats['total']
        return scores
    
    def decay_historical_data(self):
        """Apply decay to historical performance for seasonal freshness"""
        for variant_id in self.variant_stats:
            self.variant_stats[variant_id]['success'] *= self.decay_rate
            self.variant_stats[variant_id]['total'] *= self.decay_rate