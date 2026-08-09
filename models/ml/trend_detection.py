import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json

class TrendDetector:
    """
    Detects anomalies and trends in agricultural data:
    - Pest outbreak acceleration
    - Disease spread patterns
    - Weather anomalies
    - Market price swings
    """
    
    def __init__(self, contamination=0.1, n_estimators=100):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_fitted = False
        
    def extract_features(self, time_series_data):
        """
        Extract temporal and statistical features from time series
        """
        features = {}
        
        # Temporal features
        features['trend'] = self._calculate_trend(time_series_data)
        features['volatility'] = np.std(time_series_data)
        features['rate_of_change'] = self._calculate_roc(time_series_data)
        features['acceleration'] = self._calculate_acceleration(time_series_data)
        
        # Statistical features
        features['skewness'] = self._calculate_skewness(time_series_data)
        features['kurtosis'] = self._calculate_kurtosis(time_series_data)
        features['moving_avg_5d'] = self._moving_average(time_series_data, 5)
        features['moving_std_5d'] = self._moving_std(time_series_data, 5)
        
        return features
    
    def _calculate_trend(self, data):
        """Linear regression slope"""
        x = np.arange(len(data))
        if len(data) < 2:
            return 0
        return np.polyfit(x, data, 1)[0]
    
    def _calculate_roc(self, data):
        """Rate of change"""
        if len(data) < 2:
            return 0
        return (data[-1] - data[0]) / (data[0] + 1e-10)
    
    def _calculate_acceleration(self, data):
        """Second derivative"""
        if len(data) < 3:
            return 0
        diffs = np.diff(data)
        return np.mean(np.diff(diffs))
    
    def _calculate_skewness(self, data):
        """Statistical skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Statistical kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _moving_average(self, data, window):
        """Simple moving average"""
        if len(data) < window:
            return np.mean(data)
        return np.mean(data[-window:])
    
    def _moving_std(self, data, window):
        """Moving standard deviation"""
        if len(data) < window:
            return np.std(data)
        return np.std(data[-window:])
    
    def fit(self, X_features):
        """
        Fit the isolation forest model
        X_features: list of feature dictionaries
        """
        feature_df = pd.DataFrame(X_features)
        self.feature_names = feature_df.columns.tolist()
        X_scaled = self.scaler.fit_transform(feature_df)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
    def detect_anomalies(self, X_features):
        """
        Detect anomalies in new data
        Returns: anomaly scores (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        feature_df = pd.DataFrame(X_features)
        X_scaled = self.scaler.transform(feature_df)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        return {
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': predictions == -1
        }
    
    def get_severity_score(self, anomaly_score):
        """
        Convert anomaly score to severity (0-100)
        Lower anomaly scores = higher severity
        """
        # Normalize to 0-100 range
        severity = max(0, min(100, (1 - (anomaly_score + 10) / 10) * 100))
        return severity

class PestOutbreakDetector(TrendDetector):
    """Specialized detector for pest and disease outbreaks"""
    
    def predict_outbreak_risk(self, historical_data, weather_data, region):
        """
        Predict pest outbreak risk based on historical patterns and weather
        """
        features = {
            'pest_incidence_trend': self.extract_features(historical_data['pest_count']),
            'temperature': self.extract_features(weather_data['temp']),
            'humidity': self.extract_features(weather_data['humidity']),
            'rainfall': self.extract_features(weather_data['rainfall']),
        }
        
        # Flatten features
        flat_features = self._flatten_features(features)
        anomalies = self.detect_anomalies([flat_features])
        
        risk_level = "HIGH" if anomalies['is_anomaly'][0] else "MEDIUM"
        severity = self.get_severity_score(anomalies['anomaly_scores'][0])
        
        return {
            'risk_level': risk_level,
            'severity_score': severity,
            'recommended_action': self._get_action_recommendation(risk_level, region)
        }
    
    def _flatten_features(self, nested_dict):
        """Flatten nested feature dictionary"""
        flat = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat[f"{key}_{subkey}"] = subval
            else:
                flat[key] = value
        return flat
    
    def _get_action_recommendation(self, risk_level, region):
        """Generate action recommendations"""
        recommendations = {
            "HIGH": f"Immediate intervention required in {region}. Scout fields daily.",
            "MEDIUM": f"Monitor pest populations closely in {region}. Prepare control measures.",
            "LOW": f"Continue routine monitoring in {region}."
        }
        return recommendations.get(risk_level, "No specific action required")