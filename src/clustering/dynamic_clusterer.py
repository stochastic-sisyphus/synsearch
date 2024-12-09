from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

class DynamicClusterer:
    def __init__(self, config):
        self.config = config
        self.metrics = {}
    
    def select_best_algorithm(self, embeddings: np.ndarray) -> tuple:
        """Dynamically select the best clustering algorithm based on data characteristics."""
        algorithms = {
            'hdbscan': (hdbscan.HDBSCAN(
                min_cluster_size=self.config['clustering']['min_size'],
                metric='euclidean'
            ), True),  # (algorithm, handles_noise)
            'kmeans': (KMeans(
                n_clusters=self.config['clustering']['n_clusters'],
                random_state=42
            ), False)
        }
        
        best_score = -1
        best_labels = None
        best_algo = None
        
        for name, (algo, handles_noise) in algorithms.items():
            labels = algo.fit_predict(embeddings)
            if not handles_noise:  # Skip evaluation if algorithm can't handle noise
                labels = labels[labels != -1]
            
            if len(np.unique(labels)) > 1:  # Only evaluate if we have valid clusters
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_algo = name
        
        return best_labels, best_algo, best_score