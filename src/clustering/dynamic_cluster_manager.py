from typing import Dict, List, Tuple, Any
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import MiniBatchKMeans
import logging
from pathlib import Path
import torch
import json
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

class DynamicClusterManager:
    """Manages dynamic clustering with adaptive thresholds and online updates."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        
        # Add memory tracking
        self.memory_tracker = PerformanceOptimizer()
        self.batch_size = self.memory_tracker.get_optimal_batch_size()
        
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        try:
            self.logger.info(f"Starting clustering with embeddings shape: {embeddings.shape}")
            
            # Initialize HDBSCAN with optimal parameters
            self.clusterer = HDBSCAN(
                min_cluster_size=self.config.get('min_cluster_size', 5),
                min_samples=self.config.get('min_samples', 3),
                metric='euclidean',
                core_dist_n_jobs=self.memory_tracker.get_optimal_workers()
            )
            
            # Process in batches if needed
            if len(embeddings) > self.batch_size:
                return self._batch_process_clustering(embeddings)
            
            labels = self.clusterer.fit_predict(embeddings)
            metrics = self._calculate_metrics(embeddings, labels)
            
            return labels, metrics
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {e}")
            raise

    def _batch_process_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        batches = [embeddings[i:i + self.batch_size] 
                  for i in range(0, len(embeddings), self.batch_size)]
        
        all_labels = []
        for batch in batches:
            labels = self.clusterer.fit_predict(batch)
            all_labels.extend(labels)
            
        final_labels = np.array(all_labels)
        metrics = self._calculate_metrics(embeddings, final_labels)
        
        return final_labels, metrics

    def _adapt_clustering(self, embeddings: np.ndarray, metrics: Dict) -> np.ndarray:
        """Adapt clustering parameters based on metrics."""
        self.logger.info("Starting _adapt_clustering method")
        self.logger.debug(f"Metrics: {metrics}")
        
        try:
            # Implement adaptive logic here
            # For example, adjust min_cluster_size based on silhouette score
            new_min_cluster_size = max(3, self.min_cluster_size - 1)
            
            adapted_clusterer = HDBSCAN(
                min_cluster_size=new_min_cluster_size,
                min_samples=self.min_samples
            )
            return adapted_clusterer.fit_predict(embeddings)
        except Exception as e:
            self.logger.error(f"Error in adapting clustering: {e}")
            raise

    def get_clusters(self, texts: List[str], labels: np.ndarray) -> Dict[int, List[str]]:
        """Group texts by cluster labels."""
        self.logger.info("Starting get_clusters method")
        self.logger.debug(f"Number of texts: {len(texts)}, Number of labels: {len(labels)}")
        
        try:
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("Texts must be a list of strings")
            if not isinstance(labels, np.ndarray):
                raise ValueError("Labels must be a numpy array")
            if len(texts) != len(labels):
                raise ValueError("Texts and labels must have the same length")
            
            clusters = {}
            for text, label in zip(texts, labels):
                if label != -1:  # Skip noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(text)
            return clusters
        except Exception as e:
            self.logger.error(f"Error in get_clusters: {e}")
            raise

    def _calculate_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering metrics."""
        self.logger.info("Starting _calculate_metrics method")
        self.logger.debug(f"Embeddings shape: {embeddings.shape}, Labels: {labels}")
        
        try:
            # Placeholder for actual metric calculation
            return {
                'silhouette_score': silhouette_score(embeddings, labels)  # Example metric
            }
        except Exception as e:
            self.logger.error(f"Error in _calculate_metrics: {e}")
            raise

    def _save_intermediate_outputs(self, embeddings: np.ndarray, labels: np.ndarray, metrics: Dict) -> None:
        """Save intermediate outputs after clustering."""
        self.logger.info("Starting _save_intermediate_outputs method")
        self.logger.debug(f"Embeddings shape: {embeddings.shape}, Labels: {labels}, Metrics: {metrics}")
        
        try:
            output_dir = Path("outputs/clustering")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings_file = output_dir / "embeddings.npy"
            labels_file = output_dir / "labels.npy"
            metrics_file = output_dir / "metrics.json"
            
            np.save(embeddings_file, embeddings)
            np.save(labels_file, labels)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
            
            self.logger.info(f"Saved intermediate outputs to {output_dir}")
        except Exception as e:
            self.logger.error(f"Error in _save_intermediate_outputs: {e}")
            raise

    def enable_approximate_nearest_neighbors(self):
        """Enable approximate nearest neighbors for HDBSCAN."""
        self.logger.info("Starting enable_approximate_nearest_neighbors method")
        
        try:
            self.hdbscan = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                approx_min_span_tree=True
            )
        except Exception as e:
            self.logger.error(f"Error in enable_approximate_nearest_neighbors: {e}")
            raise

    def process_embeddings_in_parallel_chunks(self, embeddings: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """Process embeddings in parallel chunks."""
        self.logger.info("Starting process_embeddings_in_parallel_chunks method")
        self.logger.debug(f"Embeddings shape: {embeddings.shape}, Chunk size: {chunk_size}")
        
        try:
            n_chunks = len(embeddings) // chunk_size + (1 if len(embeddings) % chunk_size != 0 else 0)
            results = Parallel(n_jobs=-1)(
                delayed(self.clusterer.fit_predict)(embeddings[i * chunk_size:(i + 1) * chunk_size])
                for i in range(n_chunks)
            )
            return np.concatenate(results)
        except Exception as e:
            self.logger.error(f"Error in process_embeddings_in_parallel_chunks: {e}")
            raise

    def save_partial_clustering_results(self, labels: np.ndarray, output_dir: str = "outputs/clustering") -> None:
        """Save partial clustering results."""
        self.logger.info("Starting save_partial_clustering_results method")
        self.logger.debug(f"Labels: {labels}, Output directory: {output_dir}")
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            labels_file = output_dir / "partial_labels.npy"
            np.save(labels_file, labels)
            self.logger.info(f"Saved partial clustering results to {labels_file}")
        except Exception as e:
            self.logger.error(f"Error in save_partial_clustering_results: {e}")
            raise

    def adjust_clustering_parameters_based_on_silhouette(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Automatically adjust clustering parameters based on silhouette scores."""
        self.logger.info("Starting adjust_clustering_parameters_based_on_silhouette method")
        self.logger.debug(f"Embeddings shape: {embeddings.shape}, Labels: {labels}")
        
        try:
            silhouette_avg = silhouette_score(embeddings, labels)
            if silhouette_avg < 0.5:
                self.logger.info("Adjusting clustering parameters based on silhouette score...")
                self.min_cluster_size = max(3, self.min_cluster_size - 1)
                self.clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples
                )
                labels = self.clusterer.fit_predict(embeddings)
            return labels
        except Exception as e:
            self.logger.error(f"Error in adjust_clustering_parameters_based_on_silhouette: {e}")
            raise
