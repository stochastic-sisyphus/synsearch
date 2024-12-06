from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ClusterManager:
    def __init__(self, config: Dict):
        """Initialize the cluster manager with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.method = config['clustering']['method']
        self.clusterer = self._initialize_clusterer()
        
    def _initialize_clusterer(self):
        """Initialize the clustering algorithm based on config"""
        params = self.config['clustering']['params']
        
        if self.method == 'hdbscan':
            return hdbscan.HDBSCAN(**params)
        elif self.method == 'kmeans':
            return KMeans(**params)
        elif self.method == 'dbscan':
            return DBSCAN(**params)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")
    
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Fit the clustering algorithm and return labels with metrics"""
        self.logger.info(f"Clustering {len(embeddings)} documents using {self.method}")
        
        # Perform clustering
        labels = self.clusterer.fit_predict(embeddings)
        
        # Calculate metrics
        metrics = self._calculate_metrics(embeddings, labels)
        
        return labels, metrics
    
    def _calculate_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics"""
        metrics = {}
        
        # Skip metrics if all points are noise (-1)
        if len(set(labels)) <= 1:
            self.logger.warning("No clusters found, skipping metrics calculation")
            return metrics
        
        try:
            metrics['silhouette_score'] = silhouette_score(embeddings, labels)
        except Exception as e:
            self.logger.warning(f"Failed to calculate silhouette score: {e}")
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
        except Exception as e:
            self.logger.warning(f"Failed to calculate Davies-Bouldin score: {e}")
        
        metrics['num_clusters'] = len(set(labels) - {-1})  # Exclude noise points
        metrics['noise_points'] = sum(labels == -1)
        
        return metrics
    
    def get_cluster_documents(
        self,
        documents: List[Dict],
        labels: np.ndarray
    ) -> Dict[int, List[Dict]]:
        """Group documents by cluster"""
        clusters = {}
        for doc, label in zip(documents, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)
        return clusters
    
    def save_results(
        self,
        clusters: Dict,
        metrics: Dict,
        output_dir: Path
    ) -> None:
        """Save clustering results and metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / f"clustering_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save cluster assignments
        clusters_file = output_dir / f"clusters_{datetime.now():%Y%m%d_%H%M%S}.json"
        cluster_summary = {
            str(label): {
                'size': len(docs),
                'document_ids': [doc.get('id', i) for i, doc in enumerate(docs)]
            }
            for label, docs in clusters.items()
        }
        
        with open(clusters_file, 'w') as f:
            json.dump(cluster_summary, f, indent=2)
            
        self.logger.info(f"Saved clustering results to {output_dir}")