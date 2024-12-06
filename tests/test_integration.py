import pytest
import pandas as pd
from pathlib import Path
import yaml

from src.data_loader import DataLoader
from src.data_preparation import DataPreparator
from src.data_validator import DataValidator
from src.embedding_generator import EmbeddingGenerator
from src.visualization.embedding_visualizer import EmbeddingVisualizer

@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_small_pipeline():
    """Test the entire pipeline with a small dataset"""
    # Create small test dataset
    test_data = pd.DataFrame({
        'text': [
            "This is a test document for pipeline validation.",
            "Another document to ensure end-to-end functionality.",
            "A third document with sufficient length for testing."
        ],
        'summary': [
            "Test document.",
            "End-to-end test.",
            "Third test."
        ],
        'id': [1, 2, 3]
    })
    
    # 1. Validate data
    validator = DataValidator()
    validation_results = validator.validate_dataset(test_data)
    assert isinstance(validation_results, dict)
    
    # 2. Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(test_data['text'].tolist())
    assert embeddings.shape[0] == len(test_data)
    
    # 3. Visualize embeddings
    visualizer = EmbeddingVisualizer()
    viz_result = visualizer.visualize_embeddings(
        embeddings,
        metadata=test_data,
        color_column='id'
    )
    assert 'figure' in viz_result
    assert 'reduced_embeddings' in viz_result

def test_config_compatibility(config):
    """Verify all components work with the config"""
    # Test DataLoader
    loader = DataLoader(config['data']['scisummnet_path'])
    assert loader is not None
    
    # Test DataPreparator
    preparator = DataPreparator()
    assert preparator is not None
    
    # Test EmbeddingGenerator
    generator = EmbeddingGenerator(
        model_name=config['embedding']['model_name']
    )
    assert generator is not None 