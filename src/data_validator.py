import pandas as pd
from typing import Dict, List, Any
import logging
import yaml

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate processed dataset against quality criteria"""
        checks = {
            'missing_values': self._check_missing_values(df),
            'text_length': self._check_text_length(df),
            'language': self._check_language(df),
            'duplicates': self._check_duplicates(df)
        }
        
        self.logger.info(f"Validation results: {checks}")
        return checks
    
    def _check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check if missing values are below threshold (5%)"""
        missing_pct = df.isnull().sum() / len(df) * 100
        return all(missing_pct < 5)
    
    def _check_text_length(self, df: pd.DataFrame) -> bool:
        """Check if text lengths meet minimum requirements"""
        min_length = 100  # Configurable
        text_lengths = df['text'].str.split().str.len()
        return all(text_lengths >= min_length)
    
    def _check_language(self, df: pd.DataFrame) -> bool:
        """Check if texts are in English using spacy's language detector"""
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            
            # Sample a subset of texts for efficiency
            sample_size = min(100, len(df))
            sample_texts = df['text'].sample(n=sample_size)
            
            english_count = sum(
                1 for text in sample_texts 
                if nlp(text[:100]).lang_ == 'en'  # Check first 100 chars
            )
            
            # Require 95% of sampled texts to be English
            return (english_count / sample_size) >= 0.95
            
        except Exception as e:
            self.logger.error(f"Language check failed: {e}")
            return False
    
    def _check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate entries"""
        duplicate_ratio = df.duplicated(subset=['text']).sum() / len(df)
        return duplicate_ratio < 0.05  # Allow up to 5% duplicates
    
    def get_detailed_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate detailed statistics about the dataset"""
        stats = {
            'total_documents': len(df),
            'avg_text_length': df['text'].str.split().str.len().mean(),
            'avg_summary_length': df['summary'].str.split().str.len().mean(),
            'missing_values_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_ratio': df.duplicated(subset=['text']).sum() / len(df),
        }
        
        self.logger.info(f"Dataset statistics: {stats}")
        return stats
    
    def validate_with_thresholds(self, df: pd.DataFrame, config: Dict) -> Dict[str, bool]:
        """Validate dataset against configurable thresholds"""
        thresholds = config.get('preprocessing', {}).get('validation', {})
        
        checks = {
            'missing_values': all(
                pct < thresholds.get('missing_threshold', 5.0)
                for pct in (df.isnull().sum() / len(df) * 100)
            ),
            'dataset_size': len(df) >= thresholds.get('min_dataset_size', 10000),
            'text_length': all(
                thresholds.get('min_text_length', 100) <= length <= thresholds.get('max_text_length', 1000)
                for length in df['text'].str.split().str.len()
            )
        }
        
        self.logger.info(f"Validation results with thresholds: {checks}")
        return checks
    
    # Add more validation methods as needed 

class ConfigValidator:
    """Validates configuration settings for the pipeline."""
    
    REQUIRED_FIELDS = {
        'data': {
            'input_path': str,
            'output_path': str,
            'scisummnet_path': str,
            'processed_path': str,
            'batch_size': int
        },
        'preprocessing': {
            'min_length': int,
            'max_length': int,
            'validation': dict  # For validation thresholds
        },
        'embedding': {
            'model_name': str,
            'dimension': int,  # Added this required field
            'batch_size': int,
            'max_seq_length': int,
            'device': str
        },
        'clustering': {
            'algorithm': str,
            'min_cluster_size': int,
            'min_samples': int,
            'metric': str,
            'params': dict,
            'output_dir': str
        },
        'visualization': {
            'enabled': bool,
            'output_dir': str
        },
        'summarization': {
            'model_name': str,
            'max_length': int,
            'min_length': int,
            'batch_size': int
        },
        'logging': {
            'level': str,
            'format': str
        },
        'checkpoints': {
            'dir': str
        }
    }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validates the configuration dictionary against required fields.
        Returns True if valid, raises ValueError if invalid.
        """
        try:
            self._validate_section(config, self.REQUIRED_FIELDS)
            return True
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def _validate_section(self, config: Dict[str, Any], required: Dict[str, Any], path: str = "") -> None:
        """Recursively validates configuration sections."""
        for key, value_type in required.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config:
                raise ValueError(f"Missing required field: {current_path}")
            
            if isinstance(value_type, dict):
                if not isinstance(config[key], dict):
                    raise ValueError(f"Field {current_path} must be a dictionary")
                self._validate_section(config[key], value_type, current_path)
            else:
                if not isinstance(config[key], value_type):
                    raise ValueError(
                        f"Field {current_path} must be of type {value_type.__name__}, "
                        f"got {type(config[key]).__name__}"
                    )