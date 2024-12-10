import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
import logging
from datetime import datetime
from .dynamic_cluster_manager import DynamicClusterManager
from torch.utils.data import DataLoader, Dataset

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class StreamingClusterManager:
    """Manages streaming clustering operations with buffer management."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the StreamingClusterManager with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.buffer_size = config.get('buffer_size', 100)
        self.update_interval = config.get('update_interval', 60)  # seconds
        self.buffer = deque(maxlen=self.buffer_size)
        self.cluster_manager = DynamicClusterManager(config)
        self.last_update = datetime.now()
        self.current_labels = None
        
    def update(self, new_embeddings: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Update clusters with new streaming data.

        Args:
            new_embeddings (np.ndarray): New embeddings to be added.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Updated labels and metrics.
        """
        try:
            # Add new embeddings to buffer
            for embedding in new_embeddings:
                self.buffer.append(embedding)
                
            # Check if update is needed
            current_time = datetime.now()
            time_elapsed = (current_time - self.last_update).total_seconds()
            
            if len(self.buffer) >= self.buffer_size or time_elapsed >= self.update_interval:
                # Perform clustering on buffered data
                dataset = EmbeddingDataset(np.array(list(self.buffer)))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                all_labels = []
                for batch in dataloader:
                    labels, _ = self.cluster_manager.fit_predict(batch)
                    all_labels.append(labels)
                
                self.current_labels = np.concatenate(all_labels)
                self.last_update = current_time
                
                metrics = {
                    'buffer_size': len(self.buffer),
                    'time_since_last_update': time_elapsed
                }
                
                return self.current_labels, metrics
                
            # Return current labels if no update needed
            return self.current_labels, {'update_required': False}
            
        except Exception as e:
            self.logger.error(f"Error in streaming update: {e}")
            raise
            
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get current clustering statistics.

        Returns:
            Dict[str, Any]: Clustering statistics.
        """
        return {
            'buffer_size': len(self.buffer),
            'time_since_update': (datetime.now() - self.last_update).total_seconds(),
            'num_clusters': len(np.unique(self.current_labels)) if self.current_labels is not None else 0
        }
