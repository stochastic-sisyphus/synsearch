import pytest
import numpy as np
import json
import os
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


# =============================================================================
# Configuration Validation Tests
# =============================================================================

class TestConfigurationValidation:
    """Tests for invalid configuration handling."""

    def test_missing_model_name(self):
        """Test handling of missing required model_name parameter."""
        config = {
            'style_params': {
                'concise': {'max_length': 50, 'min_length': 20}
            }
        }
        with pytest.raises((KeyError, TypeError, ValueError)):
            AdaptiveSummarizer(config)

    def test_invalid_style_parameters(self):
        """Test handling of invalid style parameters."""
        config = {
            'model_name': 't5-small',
            'style_params': {
                'concise': {'max_length': 'invalid', 'min_length': 20}  # Invalid type
            }
        }
        with pytest.raises((TypeError, ValueError)):
            summarizer = AdaptiveSummarizer(config)
            # May fail during initialization or during summarization
            summarizer.summarize_cluster(["test"], np.random.rand(1, 768), '0')

    def test_negative_numeric_values(self):
        """Test handling of negative numeric values in config."""
        config = {
            'model_name': 't5-small',
            'style_params': {
                'concise': {'max_length': -50, 'min_length': -20}  # Negative values
            },
            'num_beams': -1  # Negative beams
        }
        with pytest.raises((ValueError, RuntimeError)):
            summarizer = AdaptiveSummarizer(config)
            summarizer.summarize_cluster(["test"], np.random.rand(1, 768), '0')

    def test_min_length_greater_than_max_length(self):
        """Test handling of min_length > max_length."""
        config = {
            'model_name': 't5-small',
            'style_params': {
                'concise': {'max_length': 20, 'min_length': 100}  # Invalid: min > max
            }
        }
        with pytest.raises((ValueError, RuntimeError)):
            summarizer = AdaptiveSummarizer(config)
            summarizer.summarize_cluster(["test text"], np.random.rand(1, 768), '0')


# =============================================================================
# Basic Summarizer Tests
# =============================================================================

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


# =============================================================================
# Adaptive Summarizer Tests with Enhanced Validation
# =============================================================================

def test_adaptive_summarizer_single_cluster(adaptive_summarizer):
    """Test adaptive summarizer with validation of summary content and style."""
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

    # Validate summary content is non-empty
    assert len(summary['summary']) > 0
    assert isinstance(summary['summary'], str)

    # Validate style is one of the expected styles
    expected_styles = ['concise', 'detailed', 'technical']
    assert summary['style'] in expected_styles

    # Validate length constraints match selected style
    style_params = {
        'concise': {'max_length': 50, 'min_length': 20},
        'detailed': {'max_length': 150, 'min_length': 50},
        'technical': {'max_length': 100, 'min_length': 30}
    }
    selected_style = summary['style']
    # Note: Actual token length may vary; this is a soft check
    assert len(summary['summary'].split()) <= style_params[selected_style]['max_length'] * 2


def test_adaptive_summarizer_all_clusters(adaptive_summarizer, sample_clusters):
    embeddings = {cluster_id: np.random.rand(len(texts), 768) for cluster_id, texts in sample_clusters.items()}
    summaries = adaptive_summarizer.summarize_all_clusters(sample_clusters, embeddings)

    assert len(summaries) == len(sample_clusters)
    assert all('summary' in s for s in summaries.values())
    assert all('style' in s for s in summaries.values())
    assert all('metrics' in s for s in summaries.values())

    # Validate all summaries have non-empty content
    for cluster_id, summary in summaries.items():
        assert len(summary['summary']) > 0, f"Summary for cluster {cluster_id} is empty"


def test_adaptive_summarizer_keyword_presence(adaptive_summarizer):
    """Test that summaries contain relevant keywords from input texts."""
    texts = [
        "Machine learning algorithms improve prediction accuracy.",
        "Neural networks are used in machine learning systems.",
        "Deep learning is a subset of machine learning."
    ]

    summary = adaptive_summarizer.summarize_cluster(texts, np.random.rand(3, 768), '0')

    # Check that at least one key term from inputs appears in summary
    key_terms = ['learning', 'machine', 'neural', 'deep']
    summary_lower = summary['summary'].lower()
    assert any(term in summary_lower for term in key_terms), \
        f"Summary should contain relevant keywords. Got: {summary['summary']}"


# =============================================================================
# Save/Load Error Handling Tests
# =============================================================================

class TestSaveLoadErrorHandling:
    """Tests for save/load error handling."""

    def test_save_and_load_summaries(self, adaptive_summarizer, sample_clusters):
        embeddings = {cluster_id: np.random.rand(len(texts), 768) for cluster_id, texts in sample_clusters.items()}
        summaries = adaptive_summarizer.summarize_all_clusters(sample_clusters, embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            adaptive_summarizer.save_summaries(summaries, tmpdir)
            assert (Path(tmpdir) / 'summaries.json').exists()

            loaded_summaries = adaptive_summarizer.load_summaries(tmpdir)
            assert loaded_summaries == summaries

    def test_load_from_invalid_path(self, adaptive_summarizer):
        """Test loading summaries from non-existent path."""
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            adaptive_summarizer.load_summaries('/non/existent/path')

    def test_load_corrupted_json(self, adaptive_summarizer):
        """Test loading corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted JSON file
            corrupted_file = Path(tmpdir) / 'summaries.json'
            with open(corrupted_file, 'w') as f:
                f.write('{invalid json content')

            with pytest.raises((json.JSONDecodeError, ValueError)):
                adaptive_summarizer.load_summaries(tmpdir)

    def test_save_to_readonly_directory(self, adaptive_summarizer, sample_clusters):
        """Test saving to a read-only directory."""
        embeddings = {cluster_id: np.random.rand(len(texts), 768) for cluster_id, texts in sample_clusters.items()}
        summaries = adaptive_summarizer.summarize_all_clusters(sample_clusters, embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only
            os.chmod(tmpdir, 0o444)
            try:
                with pytest.raises((PermissionError, OSError)):
                    adaptive_summarizer.save_summaries(summaries, tmpdir)
            finally:
                # Restore permissions for cleanup
                os.chmod(tmpdir, 0o755)

    @pytest.fixture
    def adaptive_summarizer(self):
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
    def sample_clusters(self):
        return {
            '0': [
                "Deep learning has revolutionized computer vision.",
                "Convolutional neural networks are effective for image processing.",
            ]
        }


# =============================================================================
# ROUGE Score Tests with Comprehensive Validation
# =============================================================================

class TestRougeScores:
    """Comprehensive tests for ROUGE score calculation."""

    def test_calculate_rouge_scores_basic(self):
        """Test basic ROUGE score calculation."""
        summaries = ["Deep learning has revolutionized computer vision."]
        references = ["Deep learning has transformed computer vision."]

        scores = calculate_rouge_scores(summaries, references)

        assert isinstance(scores, dict)
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores

    def test_rouge_scores_in_valid_range(self):
        """Test that ROUGE scores are within valid range [0, 1]."""
        summaries = ["The quick brown fox jumps over the lazy dog."]
        references = ["A fast brown fox leaps over a sleepy dog."]

        scores = calculate_rouge_scores(summaries, references)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            assert 0.0 <= scores[metric]['precision'] <= 1.0, \
                f"{metric} precision out of range: {scores[metric]['precision']}"
            assert 0.0 <= scores[metric]['recall'] <= 1.0, \
                f"{metric} recall out of range: {scores[metric]['recall']}"
            assert 0.0 <= scores[metric]['fmeasure'] <= 1.0, \
                f"{metric} fmeasure out of range: {scores[metric]['fmeasure']}"

    def test_rouge_identical_texts(self):
        """Test ROUGE scores for identical texts (should be 1.0)."""
        text = "This is a test sentence for ROUGE scoring."
        summaries = [text]
        references = [text]

        scores = calculate_rouge_scores(summaries, references)

        # Identical texts should have perfect scores
        assert scores['rouge1']['fmeasure'] == pytest.approx(1.0, rel=0.01)
        assert scores['rougeL']['fmeasure'] == pytest.approx(1.0, rel=0.01)

    def test_rouge_completely_different_texts(self):
        """Test ROUGE scores for completely different texts."""
        summaries = ["apple banana cherry"]
        references = ["xylophone yellow zebra"]

        scores = calculate_rouge_scores(summaries, references)

        # Completely different texts should have low scores
        assert scores['rouge1']['fmeasure'] < 0.2
        assert scores['rouge2']['fmeasure'] < 0.1

    def test_rouge_empty_strings(self):
        """Test ROUGE handling of empty strings."""
        summaries = [""]
        references = ["Some reference text"]

        # Should handle gracefully without crashing
        scores = calculate_rouge_scores(summaries, references)
        assert isinstance(scores, dict)

    def test_rouge_multiple_references(self):
        """Test ROUGE with multiple summary-reference pairs."""
        summaries = [
            "Deep learning transforms AI.",
            "NLP uses transformer models.",
            "Computer vision improved with CNNs."
        ]
        references = [
            "Deep learning has revolutionized artificial intelligence.",
            "Natural language processing utilizes transformer architectures.",
            "Computer vision has been enhanced by convolutional neural networks."
        ]

        scores = calculate_rouge_scores(summaries, references)

        # Verify all metrics are computed
        assert all(metric in scores for metric in ['rouge1', 'rouge2', 'rougeL'])
        # Verify averaging worked (scores should be reasonable averages)
        assert 0.0 < scores['rouge1']['fmeasure'] < 1.0


# =============================================================================
# BLEU Score Tests
# =============================================================================

class TestBleuScores:
    """Tests for BLEU score calculation."""

    def test_calculate_bleu_scores_basic(self):
        """Test basic BLEU score calculation."""
        summaries = ["Deep learning has revolutionized computer vision."]
        references = ["Deep learning has transformed computer vision."]

        scores = calculate_bleu_scores(summaries, references)

        assert isinstance(scores, dict)
        assert 'bleu' in scores

    def test_bleu_score_in_valid_range(self):
        """Test that BLEU score is within valid range [0, 1]."""
        summaries = ["The cat sat on the mat."]
        references = ["A cat is sitting on the mat."]

        scores = calculate_bleu_scores(summaries, references)

        assert 0.0 <= scores['bleu'] <= 1.0

    def test_bleu_identical_texts(self):
        """Test BLEU score for identical texts."""
        text = "This is a test sentence."
        summaries = [text]
        references = [text]

        scores = calculate_bleu_scores(summaries, references)

        # Identical texts should have high score
        assert scores['bleu'] > 0.8

    def test_bleu_with_punctuation(self):
        """Test BLEU handles punctuation correctly with word_tokenize."""
        summaries = ["Hello, world! How are you?"]
        references = ["Hello world. How are you doing?"]

        scores = calculate_bleu_scores(summaries, references)

        # Should handle punctuation without errors
        assert isinstance(scores['bleu'], float)
        assert 0.0 <= scores['bleu'] <= 1.0
