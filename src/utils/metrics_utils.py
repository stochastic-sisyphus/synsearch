from typing import List, Dict, Any
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt', quiet=True)

def calculate_lexical_diversity(texts: List[str]) -> float:
    """Calculate lexical diversity (unique tokens / total tokens)."""
    # Combine all texts and split into tokens
    all_tokens = ' '.join(texts).lower().split()
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    return unique_tokens / total_tokens if total_tokens > 0 else 0.0

def calculate_cluster_variance(embeddings: np.ndarray) -> float:
    """Calculate the variance of embeddings within a cluster."""
    return float(np.var(embeddings).mean())

def calculate_text_complexity(text: str) -> Dict[str, float]:
    """Calculate various text complexity metrics."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    
    if not words or not sentences:
        return {"avg_sentence_length": 0.0, "avg_word_length": 0.0}
        
    return {
        "avg_sentence_length": len(words) / len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words)
    } 

def calculate_cluster_metrics(embeddings: np.ndarray, texts: List[str]) -> Dict[str, float]:
    """Calculate comprehensive metrics for a cluster."""
    return {
        'variance': calculate_cluster_variance(embeddings),
        'lexical_diversity': calculate_lexical_diversity(texts),
        'size': len(texts)
    }

def calculate_rouge_scores(summaries: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate ROUGE scores for summaries.

    Uses running sums for O(1) memory complexity instead of storing intermediate lists.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize running sums for O(1) memory usage
    sums = {
        metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        for metric in ['rouge1', 'rouge2', 'rougeL']
    }
    count = 0

    for summary, reference in zip(summaries, references):
        score = scorer.score(reference, summary)
        count += 1

        for metric, values in score.items():
            sums[metric]['precision'] += values.precision
            sums[metric]['recall'] += values.recall
            sums[metric]['fmeasure'] += values.fmeasure

    # Calculate averages using dictionary comprehension
    return {
        metric: {k: v / count if count > 0 else 0.0 for k, v in values.items()}
        for metric, values in sums.items()
    }

def calculate_bleu_scores(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores for summaries.

    Uses NLTK's word_tokenize for proper tokenization that handles punctuation,
    contractions, and other edge cases better than simple whitespace splitting.
    """
    smoothie = SmoothingFunction().method4
    total_score = 0.0
    count = 0

    for summary, reference in zip(summaries, references):
        # Use word_tokenize for accurate tokenization
        reference_tokens = [word_tokenize(reference.lower())]
        summary_tokens = word_tokenize(summary.lower())
        score = sentence_bleu(reference_tokens, summary_tokens, smoothing_function=smoothie)
        total_score += score
        count += 1

    return {
        'bleu': total_score / count if count > 0 else 0.0
    }
