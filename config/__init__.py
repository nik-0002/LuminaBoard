import yaml
import os
from typing import Dict, Any

class Config:
    """
    Configuration management for the platform
    """
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = config_dir
        self.config = self._load_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        config = {}
        
        for file in os.listdir(self.config_dir):
            if file.endswith('.yaml'):
                with open(os.path.join(self.config_dir, file), 'r') as f:
                    config.update(yaml.safe_load(f) or {})
        
        return config
    
    def _setup_paths(self):
        """Setup required directories"""
        required_dirs = [
            './data/raw',
            './data/processed',
            './data/vector_indices',
            './logs',
            './models/checkpoints'
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def ollama_url(self) -> str:
        return self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
    
    @property
    def ollama_model(self) -> str:
        return self.config.get('ollama', {}).get('model', 'qwen2.5')
    
    @property
    def adb_enabled(self) -> bool:
        return self.config.get('messaging', {}).get('adb', {}).get('enabled', True)
    
    @property
    def twilio_enabled(self) -> bool:
        return self.config.get('messaging', {}).get('twilio', {}).get('enabled', False)
    
    @property
    def twilio_account_sid(self) -> str:
        return os.getenv('TWILIO_ACCOUNT_SID', '')
    
    @property
    def twilio_auth_token(self) -> str:
        return os.getenv('TWILIO_AUTH_TOKEN', '')
    
    @property
    def twilio_phone(self) -> str:
        return self.config.get('messaging', {}).get('twilio', {}).get('phone_number', '')