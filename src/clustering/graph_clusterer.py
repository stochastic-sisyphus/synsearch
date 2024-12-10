import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path

# Community detection import with fallbacks
def get_community_detection():
    """Get available community detection implementation."""
    try:
        import community.community_louvain
        return community.community_louvain
    except ImportError:
        try:
            import community.community_louvain
            return community.community_louvain
        except ImportError:
            try:
                import community
                return community
            except ImportError:
                # Fallback to networkx's built-in community detection
                from networkx.algorithms import community
                return community

# Initialize community detection
community_detection = get_community_detection()

class GraphClusterer:
    """Graph-based clustering using community detection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.min_similarity = config.get('min_similarity', 0.5)
        self.graph = None
        self.labels_ = None
        
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform graph-based clustering using community detection."""
        try:
            # Create similarity graph
            similarity_matrix = cosine_similarity(embeddings)
            self.graph = self._create_similarity_graph(similarity_matrix)
            
            # Detect communities using available implementation
            if hasattr(community_detection, 'best_partition'):
                communities = community_detection.best_partition(self.graph)
                self.labels_ = np.array([communities[i] for i in range(len(embeddings))])
            else:
                # Fallback to networkx's implementation
                communities = list(community_detection.louvain_communities(self.graph))
                self.labels_ = np.array([next(i for i, comm in enumerate(communities) if n in comm) 
                                       for n in range(len(embeddings))])
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            return self.labels_, metrics
            
        except Exception as e:
            self.logger.error(f"Error in graph clustering: {e}")
            raise
            
    def _create_similarity_graph(self, similarity_matrix: np.ndarray) -> nx.Graph:
        """Create graph from similarity matrix."""
        graph = nx.Graph()
        
        # Add nodes
        graph.add_nodes_from(range(len(similarity_matrix)))
        
        # Add edges where similarity exceeds threshold
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] >= self.min_similarity:
                    graph.add_edge(i, j, weight=float(similarity_matrix[i, j]))
                    
        return graph
        
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate graph clustering metrics."""
        try:
            if hasattr(community_detection, 'modularity'):
                if hasattr(community_detection, 'best_partition'):
                    partition = community_detection.best_partition(self.graph)
                    modularity = community_detection.modularity(partition, self.graph)
                else:
                    communities = list(community_detection.louvain_communities(self.graph))
                    modularity = community_detection.modularity(self.graph, communities)
            else:
                # Fallback metrics if modularity calculation is not available
                modularity = 0.0

            metrics = {
                'modularity': float(modularity),
                'avg_clustering': float(nx.average_clustering(self.graph)),
                'num_components': int(nx.number_connected_components(self.graph))
            }
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {
                'modularity': 0.0,
                'avg_clustering': 0.0,
                'num_components': 0
            }
