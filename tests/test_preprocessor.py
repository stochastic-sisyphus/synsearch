import pytest
import numpy as np
import torch
from src.preprocessing.domain_agnostic_preprocessor import DomainAgnosticPreprocessor

@pytest.fixture
def preprocessor():
    """Create preprocessor instance for testing."""
    return DomainAgnosticPreprocessor()

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "This is a test document.",
        "Another test document for processing.",
        "Technical document with p-value < 0.05."
    ]

def test_preprocessing_basic(preprocessor, sample_texts):
    """Test basic preprocessing functionality."""
    processed = preprocessor.preprocess_texts(sample_texts)
    assert len(processed) == len(sample_texts)
    assert all(isinstance(text, str) for text in processed)

def test_preprocessing_empty_input(preprocessor):
    """Test handling of empty input."""
    result = preprocessor.preprocess_texts([])
    assert isinstance(result, list)
    assert len(result) == 0

def test_large_batch(preprocessor):
    """Test processing of large text batches."""
    large_texts = ["test" for _ in range(10000)]
    results = preprocessor.preprocess_texts(large_texts)
    assert len(results) == len(large_texts)

import pytest
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.enhanced_summarizer import EnhancedSummarizer
from src.summarization.hybrid_summarizer import HybridSummarizer
from src.preprocessing.domain_agnostic_preprocessor import DomainAgnosticPreprocessor
from src.data_loader import DataLoader
import numpy as np
import spacy

@pytest.fixture
def preprocessor():
    return DomainAgnosticPreprocessor()

def test_embedding_refinement():
    generator = EnhancedEmbeddingGenerator(config={})
    test_texts = ["This is a test document", "Another test document"]
    embeddings = generator.generate_embeddings(test_texts)
    
    assert embeddings.shape[1] == 768  # Check embedding dimension
    assert len(embeddings) == len(test_texts)

def test_dynamic_clustering():
    cluster_manager = DynamicClusterManager({})
    # Add test implementation

def test_enhanced_summarization():
    summarizer = EnhancedSummarizer()
    # Add test implementation 

def test_hybrid_summarization():
    summarizer = HybridSummarizer(model_name='t5-small')
    test_texts = [
        {
            'processed_text': "This is a test document about machine learning.",
            'reference_summary': "Test document about ML."
        },
        {
            'processed_text': "Another document discussing AI and ML concepts.",
            'reference_summary': "Document about AI/ML."
        }
    ]
    
    cluster_texts = {'0': test_texts}
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    
    assert '0' in summaries
    assert 'summary' in summaries['0']
    assert len(summaries['0']['summary']) > 0
    assert 'themes' in summaries['0']

def test_scientific_preprocessing(preprocessor):
    text = "The experiment yielded p < 0.05 with 1.23e-4 significance."
    processed = preprocessor.preprocess_text(text, domain='scientific')
    assert '[NUM]' in processed
    assert 'experiment' in processed

def test_legal_preprocessing(preprocessor):
    text = "According to Section 123 of 17 U.S.C. § 107"
    processed = preprocessor.preprocess_text(text, domain='legal')
    assert '[SECTION]' in processed
    assert '[LEGAL_REF]' in processed

def test_batch_processing(preprocessor):
    texts = ["Text one.", "Text two.", "Text three."]
    processed = preprocessor.preprocess_texts(texts)
    assert len(processed) == 3
    assert all(isinstance(text, str) for text in processed)

def test_entity_extraction():
    preprocessor = DomainAgnosticPreprocessor()
    text = "Google and Microsoft are leading AI research in California."
    entities = preprocessor.extract_entities(text)
    
    assert 'ORG' in entities
    assert 'GPE' in entities
    assert 'Google' in entities['ORG']
    assert 'California' in entities['GPE']

def test_embedding_generation():
    generator = EnhancedEmbeddingGenerator(model_name='all-mpnet-base-v2', config={})
    test_texts = ["This is a test document.", "Another test document."]
    embeddings = generator.generate_embeddings(test_texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.shape[1] == generator.embedding_dim

def test_style_aware_summarization():
    summarizer = HybridSummarizer()
    test_texts = ["This is a test document for technical summary.",
                 "Another test document with technical content."]
    
    # Test technical style
    tech_summary = summarizer.summarize_cluster(test_texts, style='technical')
    assert isinstance(tech_summary, dict)
    assert 'summary' in tech_summary
    
    # Test concise style
    concise_summary = summarizer.summarize_cluster(test_texts, style='concise')
    assert len(concise_summary['summary']) < len(tech_summary['summary'])

def test_pipeline_integration():
    # Test full pipeline integration
    preprocessor = DomainAgnosticPreprocessor()
    generator = EnhancedEmbeddingGenerator(config={})
    summarizer = HybridSummarizer()
    
    # Process sample text
    text = "This is a test document for pipeline integration."
    processed_text = preprocessor.preprocess_text(text)
    embeddings = generator.generate_embeddings([processed_text])
    summary = summarizer.summarize_cluster([processed_text])
    
    assert isinstance(summary, dict)
    assert 'summary' in summary
    assert isinstance(embeddings, np.ndarray)

def test_full_pipeline_integration():
    """Test complete pipeline with all components."""
    sample_text = "This is a test document for testing the complete pipeline integration."
    config = {
        'embedding': {'model_name': 'all-mpnet-base-v2'},
        'clustering': {'min_cluster_size': 5},
        'summarization': {'model_name': 'facebook/bart-large-cnn'}
    }
    
    # Initialize components
    preprocessor = DomainAgnosticPreprocessor()
    embedding_gen = EnhancedEmbeddingGenerator(config['embedding'])
    cluster_manager = DynamicClusterManager(config['clustering'])
    summarizer = HybridSummarizer(config['summarization'])
    
    # Run pipeline
    processed_text = preprocessor.preprocess_text(sample_text)
    embeddings = embedding_gen.generate_embeddings([processed_text])
    clusters = cluster_manager.fit_predict(embeddings)
    summaries = summarizer.summarize_all_clusters(clusters)
    
    assert isinstance(summaries, dict)
    assert len(summaries) > 0

def test_empty_input(preprocessor):
    result = preprocessor.preprocess_texts([])
    assert isinstance(result, list)
    assert len(result) == 0

def test_large_batch(preprocessor):
    large_texts = ["test" for _ in range(10000)]
    results = preprocessor.preprocess_texts(large_texts)
    assert len(results) == len(large_texts)