
from typing import List, Dict, Any
import numpy as np
from .dynamic_cluster_manager import DynamicClusterManager
from ..utils.metrics_utils import calculate_cluster_metrics
import logging

def process_clusters(
    texts: List[str],
    embeddings: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process texts and embeddings through clustering pipeline.
    
    Args:
        texts: List of preprocessed texts
        embeddings: Document embeddings array
        config: Configuration dictionary
        
    Returns:
        Dictionary containing clustering results and metrics
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize clustering manager
        cluster_manager = DynamicClusterManager(config['clustering'])
        
        # Perform clustering
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])
            
        # Calculate additional metrics
        cluster_metrics = calculate_cluster_metrics(embeddings, labels)
        metrics.update(cluster_metrics)
        
        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error in process_clusters: {str(e)}")
        raise