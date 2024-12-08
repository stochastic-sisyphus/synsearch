from typing import Dict, Optional
import json
from pathlib import Path

class UserConfig:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path('config/user_config.json')
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load user configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return self._create_default_config()
        
    def _create_default_config(self) -> Dict:
        """Create default user configuration"""
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'max_seq_length': 128
            },
            'summarization': {
                'style': 'concise',  # or 'detailed', 'technical'
                'max_length': 150
            },
            'clustering': {
                'min_cluster_size': 5,
                'min_samples': 5
            }
        }
        self._save_config(config)
        return config
        
    def _save_config(self, config: Dict) -> None:
        """Save user configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def update_config(self, updates: Dict) -> None:
        """Update user configuration"""
        self.config.update(updates)
        self._save_config(self.config) 