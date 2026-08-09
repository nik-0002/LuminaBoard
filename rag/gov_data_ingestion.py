import requests
import json
from datetime import datetime
from typing import List, Dict

class IndianGovernmentDataIngestion:
    """
    Ingests data from official Indian government agricultural APIs
    Sources:
    - IMD Weather (India Meteorological Department)
    - AgriNet (Department of Agriculture)
    - PPQS (Plant Protection and Quarantine Services)
    - ICAR (Indian Council of Agricultural Research)
    """
    
    def __init__(self):
        self.imd_api = "https://api.openweathermap.org/data/2.5"
        self.agrisnet_api = "https://agrisnet.dadisnet.gov.in/api"
        self.ppqs_api = "https://ppqs.gov.in/api"
        self.documents = []
    
    def fetch_imd_weather_data(self, lat: float, lon: float, city: str) -> Dict:
        """
        Fetch weather data from IMD/OpenWeather API
        """
        try:
            # Using OpenWeather as proxy for IMD data
            response = requests.get(
                f"{self.imd_api}/weather",
                params={
                    'lat': lat,
                    'lon': lon,
                    'appid': 'YOUR_API_KEY'  # Replace with actual key
                },
                timeout=10
            )
            
            if response.status_code == 200:
                weather_data = response.json()
                
                advisory = self._generate_weather_advisory(weather_data)
                
                return {
                    'success': True,
                    'temperature': weather_data['main']['temp'],
                    'humidity': weather_data['main']['humidity'],
                    'rainfall': weather_data.get('rain', {}).get('1h', 0),
                    'wind_speed': weather_data['wind']['speed'],
                    'conditions': weather_data['weather'][0]['main'],
                    'advisory': advisory,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_weather_advisory(self, weather_data: Dict) -> str:
        """Generate agricultural advisory based on weather"""
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        conditions = weather_data['weather'][0]['main']
        
        advisory = ""
        
        if conditions == 'Rainy' and humidity > 80:
            advisory = "High humidity and rainfall - Favorable for fungal diseases. Monitor crops closely. Consider fungicide application."
        elif temp > 35:
            advisory = "High temperature stress expected. Ensure adequate irrigation. Monitor for heat-sensitive pests."
        elif temp < 10:
            advisory = "Cold weather - Slow pest activity. Reduced metabolic rate in pests. Monitor for frost damage."
        
        return advisory
    
    def fetch_pest_surveillance_data(self, district: str, state: str) -> Dict:
        """
        Fetch pest surveillance reports from PPQS
        """
        try:
            # This would connect to actual PPQS API when available
            pest_data = {
                'district': district,
                'state': state,
                'pests_active': [],
                'diseases_active': [],
                'severity_index': 0
            }
            
            # Placeholder for actual API call
            return {
                'success': True,
                'data': pest_data,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fetch_crop_sowing_data(self, state: str, district: str = None) -> Dict:
        """
        Fetch government crop sowing data and advisories
        """
        try:
            sowing_data = {
                'state': state,
                'district': district,
                'crops_in_season': [],
                'recommended_varieties': {},
                'sowing_window_status': {}
            }
            
            return {
                'success': True,
                'data': sowing_data
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_rag_documents(self, data_dict: Dict) -> List[Dict]:
        """
        Convert government data into RAG-ready documents
        """
        documents = []
        
        # Weather advisory document
        if 'advisory' in data_dict:
            documents.append({
                'text': data_dict['advisory'],
                'metadata': {
                    'type': 'government_advisory',
                    'source': 'IMD/OpenWeather',
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        # Pest surveillance documents
        if 'pests_active' in data_dict:
            for pest in data_dict['pests_active']:
                documents.append({
                    'text': f"Active pest: {pest['name']} in {data_dict.get('region', '')}. Control measures: {pest.get('control', '')}",
                    'metadata': {
                        'type': 'pest_alert',
                        'source': 'PPQS',
                        'pest_id': pest.get('id'),
                        'severity': pest.get('severity', 'moderate')
                    }
                })
        
        return documents