import pytest
import numpy as np
from src.embedding_generator import EmbeddingGenerator
import tempfile
from pathlib import Path

@pytest.fixture
def embedding_generator():
    return EmbeddingGenerator(model_name='all-mpnet-base-v2')

@pytest.fixture
def sample_texts():
    return [
        "This is a test document about machine learning.",
        "Another document about natural language processing.",
        "A third document about deep learning."
    ]

def test_embedding_generation(embedding_generator, sample_texts):
    embeddings = embedding_generator.generate_embeddings(sample_texts)
    
    # Check shape and type
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] > 0  # Should have some embedding dimensions
    
def test_save_load_embeddings(embedding_generator, sample_texts):
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(sample_texts)
    metadata = {'texts': sample_texts}
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save embeddings
        embedding_generator.save_embeddings(embeddings, metadata, tmpdir)
        
        # Load embeddings
        loaded_embeddings, loaded_metadata = embedding_generator.load_embeddings(tmpdir)
        
        # Check if loaded data matches original
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        assert metadata == loaded_metadata

def test_batch_processing(embedding_generator):
    # Create a large list of texts
    texts = [f"Document {i} about topic {i%5}" for i in range(100)]
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Check shape
    assert embeddings.shape[0] == len(texts)