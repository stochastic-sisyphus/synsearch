from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
import logging

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class ClusterEvaluator:
    """Comprehensive evaluation of clustering quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate clustering using multiple metrics.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            labels (np.ndarray): Array of cluster labels.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Dict[str, float]: Dictionary of clustering metrics.
        """
        try:
            dataset = EmbeddingDataset(embeddings)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            all_embeddings = []
            for batch in dataloader:
                all_embeddings.append(batch)
            
            concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
            
            return {
                'silhouette': silhouette_score(concatenated_embeddings, labels),
                'davies_bouldin': davies_bouldin_score(concatenated_embeddings, labels),
                'cluster_sizes': self._get_cluster_sizes(labels),
                'cluster_density': self._calculate_cluster_density(concatenated_embeddings, labels)
            }
        except Exception as e:
            self.logger.error(f"Error evaluating clustering: {e}")
            return {}

    def _get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """
        Calculate the size of each cluster.

        Args:
            labels (np.ndarray): Array of cluster labels.

        Returns:
            Dict[int, int]: Dictionary of cluster sizes.
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def _calculate_cluster_density(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Calculate the density of each cluster.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            labels (np.ndarray): Array of cluster labels.

        Returns:
            Dict[int, float]: Dictionary of cluster densities.
        """
        cluster_density = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_embeddings = embeddings[labels == label]
            if len(cluster_embeddings) > 1:
                distances = np.linalg.norm(
                    cluster_embeddings[:, None] - cluster_embeddings, axis=2
                )
                density = np.mean(np.partition(distances, 5, axis=1)[:, 1:6])
                cluster_density[label] = float(density)
            else:
                cluster_density[label] = 0.0
        
        return cluster_density
