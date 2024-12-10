from typing import List, Dict, Any
import numpy as np
from .dynamic_cluster_manager import DynamicClusterManager
from ..utils.metrics_utils import calculate_cluster_metrics
import logging
from multiprocessing import Pool, cpu_count

def process_clusters(
    texts: List[str],
    embeddings: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhanced cluster processing with new features.

    Args:
        texts (List[str]): List of input texts.
        embeddings (np.ndarray): Array of embeddings.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Processed clusters and metrics.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize clustering manager with all features
        cluster_manager = DynamicClusterManager(config['clustering'])
        
        # Perform clustering with enhanced features
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Group texts by cluster with explanations
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {
                    'texts': [],
                    'explanation': metrics.get('explanations', {}).get(str(label), {})
                }
            clusters[label]['texts'].append(texts[idx])
        
        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'metrics': metrics,
            'streaming_enabled': config['clustering']['streaming']['enabled']
        }
        
    except Exception as e:
        logger.error(f"Error in process_clusters: {str(e)}")
        raise
