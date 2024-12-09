from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Dict, Any, Tuple

class DynamicClusterSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithms = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'hdbscan': hdbscan.HDBSCAN
        }
    
    def select_best_algorithm(self, embeddings: np.ndarray) -> Tuple[str, float]:
        """Select best clustering algorithm based on data characteristics."""
        scores = {}
        
        for algo_name, algo_class in self.algorithms.items():
            # Initialize with default params from config
            clusterer = algo_class(**self.config['clustering'][algo_name])
            labels = clusterer.fit_predict(embeddings)
            
            # Skip if all points assigned to noise (-1) for DBSCAN/HDBSCAN
            if len(np.unique(labels)) <= 1:
                continue
                
            scores[algo_name] = silhouette_score(embeddings, labels)
        
        best_algo = max(scores.items(), key=lambda x: x[1])
        return best_algo 