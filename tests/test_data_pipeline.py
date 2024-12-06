import pytest
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from pathlib import Path

def test_full_pipeline():
    """Test the entire data loading and preprocessing pipeline"""
    # Initialize components
    loader = DataLoader("/path/to/scisummnet")
    preprocessor = TextPreprocessor()
    
    # Test XL-Sum loading
    xlsum = loader.load_xlsum()
    assert xlsum is not None
    assert 'train' in xlsum
    
    # Test ScisummNet loading
    scisummnet = loader.load_scisummnet()
    assert scisummnet is not None
    assert len(scisummnet) > 0
    
    # Test preprocessing
    sample_size = 10
    sample_data = scisummnet.head(sample_size)
    processed = preprocessor.process_dataset(
        sample_data,
        'text',
        'summary'
    )
    assert len(processed) == sample_size
    assert 'processed_text' in processed.columns 