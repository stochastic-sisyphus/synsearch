from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Dict, List

class ClusterEvaluator:
    """Comprehensive evaluation of clustering quality"""
    
    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate clustering using multiple metrics"""
        return {
            'silhouette': silhouette_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels),
            'cluster_sizes': self._get_cluster_sizes(labels),
            'cluster_density': self._calculate_cluster_density(embeddings, labels)
        } 