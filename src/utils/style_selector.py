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
                'length_penalty': 1.5,
                'num_beams': 3
            },
            'detailed': {
                'max_length': 250,
                'min_length': 100,
                'length_penalty': 1.0,
                'num_beams': 4
            },
            'balanced': {
                'max_length': 150,
                'min_length': 50,
                'length_penalty': 1.2,
                'num_beams': 4
            }
        }
    
    def select_style(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Determine appropriate summarization style based on cluster characteristics."""
        # Calculate metrics
        avg_text_length = np.mean([len(text.split()) for text in texts])
        embedding_variance = np.var(embeddings)
        
        # Determine style based on characteristics
        if avg_text_length < 100 and embedding_variance < 0.5:
            style = 'concise'
        elif avg_text_length > 200 or embedding_variance > 1.0:
            style = 'detailed'
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

def determine_cluster_style(embeddings: np.ndarray, texts: List[str], config: Dict[str, Any] = None) -> str:
    """Determine the appropriate summarization style for a cluster."""
    selector = AdaptiveStyleSelector(config)
    result = selector.select_style(embeddings, texts)
    return result['style']

def get_style_parameters(style: str) -> Dict[str, Any]:
    """Get the parameters for a given summarization style."""
    selector = AdaptiveStyleSelector()
    return selector.style_params.get(style, selector.style_params['balanced'])
 