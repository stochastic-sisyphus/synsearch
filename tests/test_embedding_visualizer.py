import pytest
import numpy as np
import pandas as pd
from src.visualization.embedding_visualizer import EmbeddingVisualizer

@pytest.fixture
def sample_embeddings():
    return np.random.rand(100, 384)  # Simulated embeddings

@pytest.fixture
def sample_metadata():
    return pd.DataFrame({
        'category': ['A'] * 50 + ['B'] * 50,
        'id': range(100)
    })

def test_dimension_reduction():
    visualizer = EmbeddingVisualizer()
    embeddings = np.random.rand(100, 384)
    reduced = visualizer.reduce_dimensions(embeddings)
    
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (100, 2)

def test_visualization_generation(sample_embeddings, sample_metadata, tmp_path):
    visualizer = EmbeddingVisualizer()
    save_path = tmp_path / "embedding_viz.html"
    
    result = visualizer.visualize_embeddings(
        sample_embeddings,
        metadata=sample_metadata,
        color_column='category',
        save_path=save_path
    )
    
    assert save_path.exists()
    assert 'figure' in result
    assert isinstance(result['reduced_embeddings'], np.ndarray)