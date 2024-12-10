import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple, List, Any
import logging
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class DynamicClusterManager:
    """Dynamic clustering manager that adapts to data characteristics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DynamicClusterManager with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        self.labels_ = None
        
    def _calculate_density(self, embeddings: np.ndarray) -> float:
        """
        Calculate data density.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            float: Calculated density.
        """
        distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
        density = np.mean(np.partition(distances, 5, axis=1)[:, 1:6])
        return float(density)
    
    def _calculate_spread(self, embeddings: np.ndarray) -> float:
        """
        Calculate data spread.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            float: Calculated spread.
        """
        spread = np.std(embeddings)
        return float(spread)
    
    def _calculate_dimensionality_reduction(self, embeddings: np.ndarray) -> Dict[str, List[float]]:
        """
        Calculate dimensionality reduction for visualization.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            Dict[str, List[float]]: Reduced dimensionality data for visualization.
        """
        if self.config['visualization']['enabled']:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
            return {
                'x': reduced_embeddings[:, 0].tolist(),
                'y': reduced_embeddings[:, 1].tolist()
            }
        return None
    
    def _analyze_data_characteristics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyze embedding space characteristics to inform clustering strategy.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            Dict[str, float]: Data characteristics including density, spread, and dimensionality.
        """
        density = self._calculate_density(embeddings)
        spread = self._calculate_spread(embeddings)
        visualization_data = self._calculate_dimensionality_reduction(embeddings)
        
        return {
            'density': density,
            'spread': spread,
            'dimensionality': embeddings.shape[1],
            'visualization': visualization_data
        }
        
    def _select_algorithm(self, characteristics: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
        """
        Select best clustering algorithm based on data characteristics.

        Args:
            characteristics (Dict[str, float]): Data characteristics.

        Returns:
            Tuple[str, Dict[str, Any]]: Selected algorithm name and parameters.
        """
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
            
    def fit_predict(self, embeddings: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform adaptive clustering and return labels with metrics.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Cluster labels and metrics.
        """
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
            
        # Use DataLoader for batch processing
        dataset = EmbeddingDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_labels = []
        for batch in dataloader:
            labels = self.clusterer.fit_predict(batch)
            all_labels.append(labels)
        
        self.labels_ = np.concatenate(all_labels)
        
        # Calculate clustering metrics
        metrics = self._calculate_metrics(embeddings)
        metrics['algorithm'] = algo_name
        metrics['data_characteristics'] = characteristics
        
        return self.labels_, metrics
        
    def _calculate_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            Dict[str, float]: Calculated metrics.
        """
        metrics = {}
        
        if len(np.unique(self.labels_)) > 1:  # More than one cluster
            metrics['silhouette_score'] = float(silhouette_score(embeddings, self.labels_))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, self.labels_))
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
            
        return metrics
        
    def get_cluster_documents(self, documents: List[Dict], labels: np.ndarray) -> Dict[int, List[Dict]]:
        """
        Group documents by cluster label.

        Args:
            documents (List[Dict]): List of documents.
            labels (np.ndarray): Array of cluster labels.

        Returns:
            Dict[int, List[Dict]]: Grouped documents by cluster label.
        """
        clusters = {}
        for doc, label in zip(documents, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)
        return clusters
