from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import spacy

class AdaptiveStyleSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')
        
    def compute_lexical_diversity(self, text: str) -> float:
        """Compute lexical diversity score."""
        tokens = [token.text.lower() for token in self.nlp(text) 
                 if not token.is_punct and not token.is_space]
        return len(set(tokens)) / len(tokens) if tokens else 0
        
    def compute_cluster_variance(self, embeddings: np.ndarray) -> float:
        """Compute variance of embeddings within cluster."""
        return np.var(embeddings, axis=0).mean()
        
    def determine_style(self, 
                       embeddings: np.ndarray, 
                       texts: List[str]) -> str:
        """Determine summarization style based on cluster properties."""
        # Compute metrics
        variance = self.compute_cluster_variance(embeddings)
        avg_lex_div = np.mean([self.compute_lexical_diversity(text) 
                              for text in texts])
        
        # Style selection logic
        if variance > self.config['style_thresholds']['high_variance']:
            return 'detailed'
        elif avg_lex_div > self.config['style_thresholds']['high_lexical_div']:
            return 'technical'
        else:
            return 'balanced'
