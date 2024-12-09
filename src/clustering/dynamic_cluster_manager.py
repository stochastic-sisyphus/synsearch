import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple, List, Any
import logging

class DynamicClusterManager:
    """Dynamic clustering manager that adapts to data characteristics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        self.labels_ = None
        
    def _analyze_data_characteristics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Analyze embedding space characteristics to inform clustering strategy."""
        # Calculate data density
        distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
        density = np.mean(np.partition(distances, 5, axis=1)[:, 1:6])
        
        # Calculate data spread
        spread = np.std(embeddings)
        
        return {
            'density': float(density),
            'spread': float(spread),
            'dimensionality': embeddings.shape[1]
        }
        
    def _select_algorithm(self, characteristics: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
        """Select best clustering algorithm based on data characteristics."""
        if characteristics['density'] < self.config['clustering'].get('density_threshold', 0.5):
            # Sparse data: Use HDBSCAN
            return 'hdbscan', {
                'min_cluster_size': self.config['clustering']['params']['min_cluster_size'],
                'min_samples': self.config['clustering']['params'].get('min_samples', 5)
            }
        elif characteristics['spread'] > self.config['clustering'].get('spread_threshold', 1.0):
            # Well-separated data: Use DBSCAN
            return 'dbscan', {
                'eps': self.config['clustering']['params'].get('eps', 0.5),
                'min_samples': self.config['clustering']['params'].get('min_samples', 5)
            }
        else:
            # Dense, well-structured data: Use KMeans
            return 'kmeans', {
                'n_clusters': self.config['clustering']['params']['n_clusters']
            }
            
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform adaptive clustering and return labels with metrics."""
        # Analyze data characteristics
        characteristics = self._analyze_data_characteristics(embeddings)
        self.logger.info(f"Data characteristics: {characteristics}")
        
        # Select appropriate algorithm
        algo_name, params = self._select_algorithm(characteristics)
        self.logger.info(f"Selected algorithm: {algo_name} with params: {params}")
        
        # Initialize and fit clusterer
        if algo_name == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(**params)
        elif algo_name == 'dbscan':
            self.clusterer = DBSCAN(**params)
        else:
            self.clusterer = KMeans(**params)
            
        self.labels_ = self.clusterer.fit_predict(embeddings)
        
        # Calculate clustering metrics
        metrics = self._calculate_metrics(embeddings)
        metrics['algorithm'] = algo_name
        metrics['data_characteristics'] = characteristics
        
        return self.labels_, metrics
        
    def _calculate_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        if len(np.unique(self.labels_)) > 1:  # More than one cluster
            metrics['silhouette_score'] = float(silhouette_score(embeddings, self.labels_))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, self.labels_))
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
            
        return metrics
        
    def get_cluster_documents(self, documents: List[Dict], labels: np.ndarray) -> Dict[int, List[Dict]]:
        """Group documents by cluster label."""
        clusters = {}
        for doc, label in zip(documents, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)
        return clusters
