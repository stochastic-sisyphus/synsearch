import pytest
from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer

@pytest.fixture
def pipeline_components():
    api = ArxivAPI()
    embedding_generator = EnhancedEmbeddingGenerator()
    cluster_manager = DynamicClusterManager({
        'min_cluster_size': 5,
        'min_samples': 2
    })
    summarizer = EnhancedHybridSummarizer({
        'model_name': 'facebook/bart-large-cnn',
        'max_length': 150,
        'min_length': 50
    })
    return api, embedding_generator, cluster_manager, summarizer

def test_end_to_end_pipeline(pipeline_components):
    api, embedding_generator, cluster_manager, summarizer = pipeline_components
    
    # Test paper fetching
    papers = api.fetch_papers_batch("machine learning", max_papers=10)
    assert len(papers) > 0
    
    # Test embedding generation
    texts = [p['summary'] for p in papers]
    embeddings = embedding_generator.generate_embeddings(texts)
    assert len(embeddings) == len(papers)
    
    # Test clustering
    labels, metrics = cluster_manager.fit_predict(embeddings)
    assert len(labels) == len(papers)
    
    # Test summarization
    cluster_texts = {}
    for i, label in enumerate(labels):
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append({
            'processed_text': texts[i],
            'embedding': embeddings[i]
        })
    
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    assert len(summaries) > 0

def test_error_handling(pipeline_components):
    api, _, _, _ = pipeline_components
    
    # Test invalid query
    papers = api.fetch_papers_batch("", max_papers=10)
    assert papers == []
    
    # Test invalid max_papers
    papers = api.fetch_papers_batch("test", max_papers=-1)
    assert papers == [] 