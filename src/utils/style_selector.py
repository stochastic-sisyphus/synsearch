from typing import Dict, List, Any
import numpy as np
from utils.metrics_utils import calculate_cluster_variance, calculate_lexical_diversity, calculate_cluster_metrics

def determine_cluster_style(embeddings: np.ndarray, texts: List[str], config: Dict[str, Any]) -> str:
    """
    Determine the appropriate summarization style based on cluster characteristics.
    
    Args:
        embeddings: Cluster document embeddings
        texts: List of preprocessed texts in the cluster
        config: Configuration dictionary with thresholds
        
    Returns:
        str: Selected style ('technical', 'narrative', 'concise', or 'detailed')
    """
    # Calculate key metrics
    variance = calculate_cluster_variance(embeddings)
    lexical_diversity = calculate_lexical_diversity(texts)
    cluster_metrics = calculate_cluster_metrics(embeddings, texts)
    
    # Get thresholds from config or use defaults
    thresholds = config.get('style_selection', {}).get('thresholds', {
        'high_variance': 0.7,
        'high_lexical_diversity': 0.6,
        'large_cluster': 10
    })
    
    # Decision logic for style selection
    if variance > thresholds['high_variance']:
        if lexical_diversity > thresholds['high_lexical_diversity']:
            # High variance and diversity suggests complex, varied content
            # Use detailed style to capture nuances
            return 'detailed'
        else:
            # High variance but lower diversity suggests technical content
            # Use technical style for precision
            return 'technical'
    else:
        if len(texts) > thresholds['large_cluster']:
            # Large cluster with low variance suggests related content
            # Use concise style to avoid redundancy
            return 'concise'
        else:
            # Small cluster with low variance suggests focused content
            # Use narrative style for readability
            return 'narrative'

def get_style_parameters(style: str) -> Dict[str, Any]:
    """
    Get the summarization parameters for a given style.
    
    Args:
        style: The selected summarization style
        
    Returns:
        Dict containing style-specific parameters
    """
    style_params = {
        'technical': {
            'max_length': 150,
            'min_length': 50,
            'length_penalty': 2.0,
            'num_beams': 4,
            'preserve_keywords': True
        },
        'narrative': {
            'max_length': 200,
            'min_length': 100,
            'length_penalty': 1.0,
            'num_beams': 4,
            'preserve_keywords': False
        },
        'concise': {
            'max_length': 100,
            'min_length': 30,
            'length_penalty': 0.8,
            'num_beams': 3,
            'preserve_keywords': True
        },
        'detailed': {
            'max_length': 250,
            'min_length': 150,
            'length_penalty': 1.5,
            'num_beams': 5,
            'preserve_keywords': True
        }
    }
    
    return style_params.get(style, style_params['narrative'])
 