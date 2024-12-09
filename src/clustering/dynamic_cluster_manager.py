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
        self.clustering_methods = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'hdbscan': hdbscan.HDBSCAN
        }
        
    def analyze_data_characteristics(self, embeddings):
        """Analyze data to determine best clustering approach"""
        characteristics = {
            'density': self._calculate_density(embeddings),
            'variance': np.var(embeddings),
            'dimensionality': embeddings.shape[1]
        }
        return characteristics
        
    def _calculate_density(self, embeddings):
        """Calculate data density using average pairwise distances"""
        sample_size = min(1000, len(embeddings))
        sample = embeddings[np.random.choice(len(embeddings), sample_size)]
        distances = np.linalg.norm(sample[:, np.newaxis] - sample, axis=2)
        return np.mean(distances)
        
    def select_algorithm(self, characteristics):
        """Select best clustering algorithm based on data characteristics"""
        if characteristics['density'] < 0.5:  # Dense data
            return 'kmeans'
        elif characteristics['variance'] > 2.0:  # High variance
            return 'hdbscan'
        else:
            return 'dbscan'
            
    def fit_predict(self, embeddings):
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(embeddings)
        
        # Select best algorithm
        algorithm = self.select_algorithm(characteristics)
        
        # Get clustering parameters
        params = self.config['clustering']['params']
        
        # Initialize and fit clusterer
        clusterer = self.clustering_methods[algorithm](**params)
        labels = clusterer.fit_predict(embeddings)
        
        # Calculate metrics
        metrics = {
            'algorithm': algorithm,
            'silhouette': silhouette_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels),
            'data_characteristics': characteristics
        }
        
        return labels, metrics
