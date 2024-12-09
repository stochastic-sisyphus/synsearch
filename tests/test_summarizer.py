import pytest
from src.summarization.summarizer import ClusterSummarizer
from src.summarization.adaptive_summarizer import AdaptiveSummarizer
from src.utils.metrics_utils import calculate_rouge_scores, calculate_bleu_scores
import tempfile
from pathlib import Path

@pytest.fixture
def summarizer():
    return ClusterSummarizer(model_name='t5-small')  # Using smaller model for tests

@pytest.fixture
def adaptive_summarizer():
    config = {
        'model_name': 't5-small',
        'style_params': {
            'concise': {'max_length': 50, 'min_length': 20},
            'detailed': {'max_length': 150, 'min_length': 50},
            'technical': {'max_length': 100, 'min_length': 30}
        },
        'num_beams': 4,
        'length_penalty': 2.0,
        'early_stopping': True
    }
    return AdaptiveSummarizer(config)

@pytest.fixture
def sample_clusters():
    return {
        '0': [
            "Deep learning has revolutionized computer vision.",
            "Convolutional neural networks are effective for image processing.",
            "Image recognition has improved significantly with deep learning."
        ],
        '1': [
            "Natural language processing uses transformer models.",
            "BERT and GPT have changed how we process text.",
            "Language models can understand context better than ever."
        ]
    }

def test_single_cluster_summary(summarizer):
    texts = [
        "Deep learning has revolutionized computer vision.",
        "CNNs are effective for image processing."
    ]
    
    summary = summarizer.summarize_cluster(texts, '0')
    
    assert isinstance(summary, dict)
    assert 'cluster_id' in summary
    assert 'summary' in summary
    assert 'num_docs' in summary
    assert summary['num_docs'] == 2

def test_all_clusters_summary(summarizer, sample_clusters):
    summaries = summarizer.summarize_all_clusters(sample_clusters)
    
    assert len(summaries) == len(sample_clusters)
    assert all('summary' in s for s in summaries)
    assert all('cluster_id' in s for s in summaries)

def test_save_summaries(summarizer, sample_clusters):
    summaries = summarizer.summarize_all_clusters(sample_clusters)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer.save_summaries(summaries, tmpdir)
        assert (Path(tmpdir) / 'summaries.json').exists()

def test_adaptive_summarizer_single_cluster(adaptive_summarizer):
    texts = [
        "Deep learning has revolutionized computer vision.",
        "CNNs are effective for image processing."
    ]
    
    summary = adaptive_summarizer.summarize_cluster(texts, np.random.rand(2, 768), '0')
    
    assert isinstance(summary, dict)
    assert 'cluster_id' in summary
    assert 'summary' in summary
    assert 'style' in summary
    assert 'metrics' in summary
    assert summary['num_docs'] == 2

def test_adaptive_summarizer_all_clusters(adaptive_summarizer, sample_clusters):
    embeddings = {cluster_id: np.random.rand(len(texts), 768) for cluster_id, texts in sample_clusters.items()}
    summaries = adaptive_summarizer.summarize_all_clusters(sample_clusters, embeddings)
    
    assert len(summaries) == len(sample_clusters)
    assert all('summary' in s for s in summaries.values())
    assert all('style' in s for s in summaries.values())
    assert all('metrics' in s for s in summaries.values())

def test_save_and_load_summaries(adaptive_summarizer, sample_clusters):
    embeddings = {cluster_id: np.random.rand(len(texts), 768) for cluster_id, texts in sample_clusters.items()}
    summaries = adaptive_summarizer.summarize_all_clusters(sample_clusters, embeddings)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        adaptive_summarizer.save_summaries(summaries, tmpdir)
        assert (Path(tmpdir) / 'summaries.json').exists()
        
        loaded_summaries = adaptive_summarizer.load_summaries(tmpdir)
        assert loaded_summaries == summaries

def test_calculate_rouge_scores():
    summaries = ["Deep learning has revolutionized computer vision."]
    references = ["Deep learning has transformed computer vision."]
    
    scores = calculate_rouge_scores(summaries, references)
    
    assert isinstance(scores, dict)
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores

def test_calculate_bleu_scores():
    summaries = ["Deep learning has revolutionized computer vision."]
    references = ["Deep learning has transformed computer vision."]
    
    scores = calculate_bleu_scores(summaries, references)
    
    assert isinstance(scores, dict)
    assert 'bleu' in scores
