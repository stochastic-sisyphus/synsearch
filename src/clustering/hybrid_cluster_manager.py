from .attention_clustering import HybridClusteringModule
from .dynamic_cluster_manager import DynamicClusterManager

class HybridClusterManager:
    """Combines attention-based refinement with dynamic algorithm selection"""
    
    def __init__(self, config: Dict):
        self.attention_module = HybridClusteringModule(
            embedding_dim=config['embedding']['dimension']
        )
        self.dynamic_manager = DynamicClusterManager(config)
        
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Refine embeddings using attention
        refined_embeddings = self.attention_module(embeddings)
        
        # Use dynamic selection for clustering
        labels, metrics = self.dynamic_manager.fit_predict(refined_embeddings)
        
        return labels, {
            **metrics,
            'attention_applied': True
        } 