from pathlib import Path
import pandas as pd
from typing import Union, Dict, List
import json
from datasets import load_dataset
import logging

class FlexibleDataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, source: Union[str, Path, Dict]) -> pd.DataFrame:
        """Load data from various sources"""
        if isinstance(source, (str, Path)):
            return self._load_from_path(source)
        elif isinstance(source, Dict):
            return self._load_from_config(source)
            
    def _load_from_path(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from local file"""
        path = Path(path)
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        elif path.suffix == '.jsonl':
            return pd.read_json(path, lines=True)
            
    def _load_from_config(self, config: Dict) -> pd.DataFrame:
        """Load data based on configuration"""
        source_type = config.get('type', '')
        
        if source_type == 'huggingface':
            dataset = load_dataset(config['dataset_name'])
            return pd.DataFrame(dataset[config.get('split', 'train')])
        elif source_type == 'scisummnet':
            return self._load_scisummnet(config['path'])
            
    def _load_scisummnet(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load ScisummNet dataset"""
        # Your existing ScisummNet loading logic
        pass 