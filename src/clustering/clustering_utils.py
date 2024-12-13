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
    try:
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
            'metrics': metrics
        }
        
    except Exception as e:
        logging.error(f"Error in process_clusters: {str(e)}")
        raise
