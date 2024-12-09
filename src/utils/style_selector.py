from typing import List, Dict, Any
import numpy as np
from .metrics_utils import (
    calculate_lexical_diversity,
    calculate_cluster_variance,
    calculate_text_complexity
)

class AdaptiveStyleSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Default thresholds can be overridden via config
        self.style_thresholds = config.get('style_thresholds', {
            'lexical_diversity': {'low': 0.3, 'high': 0.6},
            'variance': {'low': 0.1, 'high': 0.3},
            'complexity': {'low': 15, 'high': 25}
        })
    
    def determine_cluster_style(
        self, 
        embeddings: np.ndarray, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Determine appropriate summarization style based on cluster characteristics.
        Uses multiple domain-agnostic metrics to make the decision.
        """
        # Calculate core metrics
        metrics = {
            'lexical_diversity': calculate_lexical_diversity(texts),
            'variance': calculate_cluster_variance(embeddings),
            'complexity_scores': [
                calculate_text_complexity(text)["avg_sentence_length"] 
                for text in texts
            ]
        }
        
        metrics['avg_complexity'] = np.mean(metrics['complexity_scores']) if metrics['complexity_scores'] else 0
        
        # Determine style based on combined metrics
        style = self._select_style(
            metrics['lexical_diversity'],
            metrics['variance'],
            metrics['avg_complexity']
        )
        
        return {
            'style': style,
            'metrics': {
                'lexical_diversity': metrics['lexical_diversity'],
                'variance': metrics['variance'],
                'avg_complexity': metrics['avg_complexity']
            }
        }
    
    def _select_style(
        self, 
        lex_div: float, 
        variance: float, 
        complexity: float
    ) -> str:
        """
        Select summarization style based on multiple metrics.
        Returns: 'detailed', 'balanced', or 'concise'
        """
        th = self.style_thresholds
        score = 0
        
        # Calculate style score based on all metrics
        if lex_div > th['lexical_diversity']['high']: score += 1
        elif lex_div < th['lexical_diversity']['low']: score -= 1
        
        if variance > th['variance']['high']: score += 1
        elif variance < th['variance']['low']: score -= 1
        
        if complexity > th['complexity']['high']: score += 1
        elif complexity < th['complexity']['low']: score -= 1
        
        # Convert score to style
        if score >= 1:
            return 'detailed'
        elif score <= -1:
            return 'concise'
        return 'balanced'
 