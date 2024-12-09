from typing import List, Dict, Any
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt', quiet=True)

def calculate_lexical_diversity(texts: List[str]) -> float:
    """Calculate lexical diversity (TTR) across all texts."""
    all_tokens = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        # Filter out punctuation and numbers
        tokens = [t for t in tokens if t.isalpha()]
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return 0.0
        
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    return unique_tokens / total_tokens

def calculate_cluster_variance(embeddings: np.ndarray) -> float:
    """Calculate the average variance of embeddings within a cluster."""
    if len(embeddings) <= 1:
        return 0.0
    return float(np.mean(np.var(embeddings, axis=0)))

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