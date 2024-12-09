from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

class DynamicClusterManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        self.metrics = {}
        self.hybrid_mode = config.get('clustering', {}).get('hybrid_mode', False)
        
    def select_algorithm(self, embeddings: np.ndarray) -> str:
        """Dynamically select best clustering algorithm based on data characteristics"""
        scores = {}
        
        # Try different algorithms
        algorithms = {
            'kmeans': KMeans(n_clusters=self.config['clustering']['n_clusters']),
            'dbscan': DBSCAN(eps=self.config['clustering']['eps']),
            'hdbscan': hdbscan.HDBSCAN(
                min_cluster_size=self.config['clustering']['min_cluster_size']
            )
        }
        
        for name, algo in algorithms.items():
            try:
                labels = algo.fit_predict(embeddings)
                if len(np.unique(labels[labels != -1])) > 1:  # Check if meaningful clusters found
                    scores[name] = silhouette_score(embeddings, labels)
            except Exception as e:
                self.logger.warning(f"Algorithm {name} failed: {e}")
                
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'hdbscan'
    
    def _analyze_data_characteristics(self, embeddings: np.ndarray) -> Dict:
        """Analyze embedding characteristics to inform algorithm selection"""
        stats = {
            'variance': np.var(embeddings),
            'density': self._estimate_density(embeddings),
            'dimensionality': embeddings.shape[1]
        }
        return stats
        
    def _estimate_density(self, embeddings: np.ndarray) -> float:
        """Estimate data density using average pairwise distances"""
        sample_size = min(1000, embeddings.shape[0])
        sample_idx = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sample = embeddings[sample_idx]
        distances = np.linalg.norm(sample[:, np.newaxis] - sample, axis=2)
        return np.mean(distances)
    
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Enhanced fit_predict with hybrid clustering support"""
        try:
            # Select best algorithm based on data characteristics
            data_stats = self._analyze_data_characteristics(embeddings)
            algo_name = self.select_algorithm(embeddings)
            
            if self.hybrid_mode:
                # Combine results from multiple algorithms
                labels_primary = self._get_primary_clustering(embeddings, algo_name)
                labels_secondary = self._get_secondary_clustering(embeddings)
                labels = self._combine_clustering_results(labels_primary, labels_secondary)
            else:
                labels = super().fit_predict(embeddings)[0]
            
            self.metrics.update({
                'data_characteristics': data_stats,
                'hybrid_mode': self.hybrid_mode
            })
            
            return labels, self.metrics
            
        except Exception as e:
            self.logger.error(f"Enhanced clustering failed: {e}")
            raise
