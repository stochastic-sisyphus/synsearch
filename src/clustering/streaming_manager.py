import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
import logging
from datetime import datetime
from .dynamic_cluster_manager import DynamicClusterManager

class StreamingClusterManager:
    """Manages streaming clustering operations with buffer management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.buffer_size = config.get('buffer_size', 100)
        self.update_interval = config.get('update_interval', 60)  # seconds
        self.buffer = deque(maxlen=self.buffer_size)
        self.cluster_manager = DynamicClusterManager(config)
        self.last_update = datetime.now()
        self.current_labels = None
        
    def update(self, new_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Update clusters with new streaming data."""
        try:
            # Add new embeddings to buffer
            for embedding in new_embeddings:
                self.buffer.append(embedding)
                
            # Check if update is needed
            current_time = datetime.now()
            time_elapsed = (current_time - self.last_update).total_seconds()
            
            if len(self.buffer) >= self.buffer_size or time_elapsed >= self.update_interval:
                # Perform clustering on buffered data
                embeddings_array = np.array(list(self.buffer))
                self.current_labels, metrics = self.cluster_manager.fit_predict(embeddings_array)
                self.last_update = current_time
                
                metrics['buffer_size'] = len(self.buffer)
                metrics['time_since_last_update'] = time_elapsed
                
                return self.current_labels, metrics
                
            # Return current labels if no update needed
            return self.current_labels, {'update_required': False}
            
        except Exception as e:
            self.logger.error(f"Error in streaming update: {e}")
            raise
            
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get current clustering statistics."""
        return {
            'buffer_size': len(self.buffer),
            'time_since_update': (datetime.now() - self.last_update).total_seconds(),
            'num_clusters': len(np.unique(self.current_labels)) if self.current_labels is not None else 0
        }
