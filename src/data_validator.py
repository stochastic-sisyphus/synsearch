import pandas as pd
from typing import Dict, List
import logging

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