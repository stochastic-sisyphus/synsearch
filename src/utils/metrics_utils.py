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
    """Calculate ROUGE scores for summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for summary, reference in zip(summaries, references):
        score = scorer.score(reference, summary)
        
        for metric, values in score.items():
            scores[metric]['precision'].append(values.precision)
            scores[metric]['recall'].append(values.recall)
            scores[metric]['fmeasure'].append(values.fmeasure)
    
    # Calculate averages
    averaged_scores = {}
    for metric, values in scores.items():
        averaged_scores[metric] = {
            k: float(np.mean(v)) for k, v in values.items()
        }
        
    return averaged_scores

def calculate_bleu_scores(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores for summaries."""
    smoothie = SmoothingFunction().method4
    scores = []
    
    for summary, reference in zip(summaries, references):
        reference_tokens = [reference.split()]
        summary_tokens = summary.split()
        score = sentence_bleu(reference_tokens, summary_tokens, smoothing_function=smoothie)
        scores.append(score)
    
    return {
        'bleu': float(np.mean(scores))
    }
