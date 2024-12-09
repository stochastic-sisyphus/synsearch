import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score

class ClusterManager:
    """
    Manages dynamic clustering operations with adaptive algorithm selection.
    """
    def __init__(self, config):
        self.config = config
        self.clusterer = None
        self.method = self.config['clustering']['method']
        
    def perform_clustering(self, embeddings):
        """Perform clustering on embeddings and return labels."""
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config['clustering']['min_cluster_size'],
            min_samples=self.config['clustering']['min_samples']
        )
        labels = self.clusterer.fit_predict(embeddings)
        
        # Calculate clustering quality metrics
        if len(np.unique(labels)) > 1:  # More than one cluster
            silhouette = silhouette_score(embeddings, labels)
        else:
            silhouette = 0
            
        return labels, {'silhouette_score': silhouette} 
