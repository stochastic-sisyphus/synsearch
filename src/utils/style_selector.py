from typing import Dict, Any, List
import numpy as np

class AdaptiveStyleSelector:
    """Selects appropriate summarization style based on cluster characteristics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.style_params = {
            'concise': {
                'max_length': 100,
                'min_length': 30,
                'length_penalty': 1.0
            },
            'detailed': {
                'max_length': 200,
                'min_length': 100,
                'length_penalty': 2.0
            },
            'balanced': {
                'max_length': 150,
                'min_length': 50,
                'length_penalty': 1.5
            }
        }
    
    def select_style(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Determine appropriate summarization style based on cluster characteristics."""
        # Calculate metrics
        avg_text_length = np.mean([len(text.split()) for text in texts])
        embedding_variance = np.var(embeddings, axis=0).mean()
        
        # Select style based on characteristics
        if embedding_variance > 0.5 or avg_text_length > 300:
            style = 'detailed'
        elif embedding_variance < 0.2 and avg_text_length < 150:
            style = 'concise'
        else:
            style = 'balanced'
            
        return {
            'style': style,
            'params': self.style_params[style],
            'metrics': {
                'avg_text_length': avg_text_length,
                'embedding_variance': float(embedding_variance)
            }
        }
 