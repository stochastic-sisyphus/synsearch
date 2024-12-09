from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score

class DynamicClusterManager:
    def __init__(self, config=None):
        self.config = config or {
            'thresholds': {
                'density': 0.5,
                'variance': 0.3,
                'min_cluster_size': 5
            }
        }
        self.available_algorithms = {
            'hdbscan': hdbscan.HDBSCAN,
            'kmeans': KMeans,
            'dbscan': DBSCAN
        }
        
    def select_algorithm(self, embeddings):
        """Dynamically select clustering algorithm based on data characteristics"""
        density = self._calculate_density(embeddings)
        variance = np.var(embeddings)
        
        if density > self.config['thresholds']['density']:
            return 'kmeans'
        elif variance > self.config['thresholds']['variance']:
            return 'dbscan'
        else:
            return 'hdbscan'
            
    def _calculate_density(self, embeddings):
        """Calculate data density using average pairwise distances"""
        sample = embeddings if len(embeddings) < 1000 else embeddings[np.random.choice(len(embeddings), 1000)]
        distances = np.linalg.norm(sample[:, np.newaxis] - sample, axis=2)
        return 1 / (np.mean(distances) + 1e-6)
        
    def fit_predict(self, embeddings):
        algo_name = self.select_algorithm(embeddings)
        
        if algo_name == 'kmeans':
            n_clusters = max(2, len(embeddings) // 50)  # Heuristic for number of clusters
            clusterer = self.available_algorithms[algo_name](n_clusters=n_clusters)
        elif algo_name == 'hdbscan':
            clusterer = self.available_algorithms[algo_name](
                min_cluster_size=self.config['thresholds']['min_cluster_size']
            )
        else:  # dbscan
            clusterer = self.available_algorithms[algo_name](
                eps=0.5,
                min_samples=self.config['thresholds']['min_cluster_size']
            )
            
        labels = clusterer.fit_predict(embeddings)
        
        # Calculate clustering metrics
        metrics = {
            'silhouette': silhouette_score(embeddings, labels) if len(np.unique(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(embeddings, labels) if len(np.unique(labels)) > 1 else 0,
            'algorithm': algo_name
        }
        
        return labels, metrics

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