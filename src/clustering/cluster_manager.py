"""
Manages dynamic clustering operations with adaptive algorithm selection
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
from datetime import datetime
from pathlib import Path
import json
import torch
import multiprocessing
from joblib import parallel_backend, Parallel, delayed
from torch.utils.data import DataLoader, Dataset

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class ClusterManager:
    """
    Manages dynamic clustering operations with adaptive algorithm selection.
    """
    
    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize the cluster manager with parallel processing support.

        Args:
            config (Dict): Configuration dictionary.
            device (Optional[str], optional): Device to use for computation. Defaults to None.
            n_jobs (Optional[int], optional): Number of CPU cores to use. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Set number of CPU cores to use
        self.n_jobs = n_jobs if n_jobs is not None else max(1, multiprocessing.cpu_count() - 1)
        self.logger.info(f"Using {self.n_jobs} CPU cores for parallel processing")
        
        # Set device for GPU operations
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize clusterer with parallel processing
        self._initialize_clusterer()
    
    def _initialize_clusterer(self):
        """
        Initialize clustering algorithm with parallel processing support.
        """
        params = self.config.get('clustering_params', {})
        
        if self.method == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(
                **params,
                core_dist_n_jobs=self.n_jobs
            )
        elif self.method == 'kmeans':
            self.clusterer = KMeans(
                **params,
                n_jobs=self.n_jobs
            )
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(
                **params,
                n_jobs=self.n_jobs
            )
            
    def fit_predict(self, embeddings: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, Dict]:
        """
        Fit and predict clusters using parallel processing.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Tuple[np.ndarray, Dict]: Cluster labels and metrics.
        """
        try:
            self.logger.info(f"Starting clustering with {self.method} on {len(embeddings)} documents")
            
            # Move embeddings to GPU if available and algorithm supports it
            if self.device == 'cuda' and self.method == 'kmeans':
                embeddings_tensor = torch.tensor(embeddings, device=self.device)
                self.labels_ = self._gpu_kmeans(embeddings_tensor)
            else:
                # Use parallel CPU processing
                dataset = EmbeddingDataset(embeddings)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                all_labels = []
                for batch in dataloader:
                    with parallel_backend('loky', n_jobs=self.n_jobs):
                        labels = self.clusterer.fit_predict(batch)
                        all_labels.append(labels)
                
                self.labels_ = np.concatenate(all_labels)
            
            metrics = self._calculate_metrics(embeddings)
            return self.labels_, metrics
        except Exception as e:
            self.logger.error(f"Error in fit_predict: {e}")
            return np.array([]), {}

    def _gpu_kmeans(self, embeddings_tensor: torch.Tensor) -> np.ndarray:
        """
        Perform K-means clustering on GPU.

        Args:
            embeddings_tensor (torch.Tensor): Tensor of embeddings.

        Returns:
            np.ndarray: Cluster labels.
        """
        from kmeans_pytorch import kmeans
        
        cluster_ids_x, cluster_centers = kmeans(
            X=embeddings_tensor,
            num_clusters=self.config['clustering_params']['n_clusters'],
            distance='euclidean',
            device=torch.device(self.device)
        )
        
        return cluster_ids_x.cpu().numpy()
    
    def _calculate_metrics(self, embeddings: np.ndarray) -> Dict:
        """
        Calculate clustering metrics in parallel.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            Dict: Calculated metrics.
        """
        metrics = {}
        
        try:
            with parallel_backend('loky', n_jobs=self.n_jobs):
                if len(np.unique(self.labels_)) > 1:
                    metrics['silhouette'] = silhouette_score(
                        embeddings,
                        self.labels_[self.labels_ != -1]
                    )
                    metrics['davies_bouldin'] = davies_bouldin_score(
                        embeddings[self.labels_ != -1],
                        self.labels_[self.labels_ != -1]
                    )
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
        
        return metrics
    
    def save_results(
        self,
        clusters: Dict[str, List[Dict]],
        metrics: Dict,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save clustering results and metrics.

        Args:
            clusters (Dict[str, List[Dict]]): Cluster assignments.
            metrics (Dict): Clustering metrics.
            output_dir (Union[str, Path]): Directory to save results.
        """
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
