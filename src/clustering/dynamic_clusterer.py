from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from torch.utils.data import DataLoader, Dataset

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class DynamicClusterer:
    """Dynamic clustering manager that adapts to data characteristics."""
    
    def __init__(self, config):
        """
        Initialize the DynamicClusterer with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.metrics = {}
    
    def select_best_algorithm(self, embeddings: np.ndarray, batch_size: int = 32) -> tuple:
        """
        Dynamically select the best clustering algorithm based on data characteristics.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            tuple: Best labels, algorithm name, and silhouette score.
        """
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
        
        dataset = EmbeddingDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for name, (algo, handles_noise) in algorithms.items():
            all_labels = []
            for batch in dataloader:
                labels = algo.fit_predict(batch)
                all_labels.append(labels)
            
            labels = np.concatenate(all_labels)
            if not handles_noise:  # Skip evaluation if algorithm can't handle noise
                labels = labels[labels != -1]
            
            if len(np.unique(labels)) > 1:  # Only evaluate if we have valid clusters
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_algo = name
        
        return best_labels, best_algo, best_score
