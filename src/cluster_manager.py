import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score
import logging

class ClusterManager:
    def __init__(self, config):
        """
        Initialize the ClusterManager with configuration settings.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.clusterer = None
        self.logger = logging.getLogger(__name__)
        
    def perform_clustering(self, embeddings):
        """
        Perform clustering on embeddings and return labels.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            tuple: Cluster labels and silhouette score.
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error performing clustering: {e}")
            return np.array([]), {'silhouette_score': 0}
