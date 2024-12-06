import pytest
import numpy as np
from src.evaluation.metrics import EvaluationMetrics
import tempfile
from pathlib import Path

@pytest.fixture
def evaluator():
    return EvaluationMetrics()

@pytest.fixture
def sample_embeddings():
    # Create synthetic embeddings with clear clusters
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.1, (50, 10))
    cluster2 = np.random.normal(3, 0.1, (50, 10))
    return np.vstack([cluster1, cluster2])

@pytest.fixture
def sample_labels():
    return np.array([0] * 50 + [1] * 50)

@pytest.fixture
def sample_summaries_and_references():
    summaries = [
        "Deep learning has transformed computer vision research.",
        "Natural language processing uses transformer models."
    ]
    references = [
        "Deep learning revolutionized how we process and analyze images.",
        "Modern NLP relies heavily on transformer-based architectures."
    ]
    return summaries, references

def test_clustering_metrics(evaluator, sample_embeddings, sample_labels):
    metrics = evaluator.calculate_clustering_metrics(sample_embeddings, sample_labels)
    
    assert 'silhouette_score' in metrics
    assert 'davies_bouldin_score' in metrics
    assert metrics['silhouette_score'] > 0  # Should be good for well-separated clusters
    assert metrics['davies_bouldin_score'] > 0

def test_rouge_scores(evaluator, sample_summaries_and_references):
    summaries, references = sample_summaries_and_references
    scores = evaluator.calculate_rouge_scores(summaries, references)
    
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores
    
    for metric in scores.values():
        assert 'precision' in metric
        assert 'recall' in metric
        assert 'fmeasure' in metric

def test_save_metrics(evaluator):
    metrics = {
        'clustering': {'silhouette_score': 0.8},
        'rouge': {'rouge1': {'fmeasure': 0.7}}
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator.save_metrics(metrics, tmpdir, prefix='test')
        files = list(Path(tmpdir).glob('*.json'))
        assert len(files) == 1
        assert files[0].name.startswith('test_metrics_') 