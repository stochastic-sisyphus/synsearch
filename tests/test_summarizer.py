import pytest
from src.summarization.summarizer import ClusterSummarizer
import tempfile
from pathlib import Path

@pytest.fixture
def summarizer():
    return ClusterSummarizer(model_name='t5-small')  # Using smaller model for tests

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