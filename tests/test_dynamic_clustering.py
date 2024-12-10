import pytest
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
import numpy as np
from src.clustering.dynamic_clusterer import DynamicClusterer
from src.clustering.streaming_manager import StreamingClusterManager

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings with clear cluster structure."""
    np.random.seed(42)
    n_samples = 100
    n_features = 768
    
    # Create 3 distinct clusters
    cluster1 = np.random.normal(0, 0.1, (n_samples // 3, n_features))
    cluster2 = np.random.normal(3, 0.1, (n_samples // 3, n_features))
    cluster3 = np.random.normal(-3, 0.1, (n_samples // 3, n_features))
    
    return np.vstack([cluster1, cluster2, cluster3])

@pytest.fixture
def config():
    """Sample configuration for clustering."""
    return {
        'min_cluster_size': 5,
        'min_samples': 3,
        'similarity_threshold': 0.8
    }

def test_adaptive_clustering():
    cluster_manager = DynamicClusterManager({'min_cluster_size': 5})
    embeddings = np.random.rand(100, 768)
    labels = cluster_manager.fit_predict(embeddings)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(embeddings)
    assert len(np.unique(labels)) > 1

def test_online_clustering():
    cluster_manager = DynamicClusterManager({'online_mode': True})
    initial_data = np.random.rand(50, 768)
    new_data = np.random.rand(10, 768)
    
    # Initial clustering
    initial_labels = cluster_manager.fit_predict(initial_data)
    # Update with new data
    updated_labels = cluster_manager.update(new_data)
    
    assert len(updated_labels) == len(new_data)

def test_dynamic_clustering(sample_embeddings, config):
    """Test dynamic clustering algorithm selection."""
    clusterer = DynamicClusterer(config)
    labels, metrics = clusterer.fit_predict(sample_embeddings)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(sample_embeddings)
    assert isinstance(metrics, dict)
    assert metrics['silhouette_score'] > 0.5

def test_streaming_clustering(sample_embeddings, config):
    """Test streaming cluster updates."""
    manager = StreamingClusterManager(config)
    
    # Split embeddings into batches
    batch_size = 20
    n_batches = len(sample_embeddings) // batch_size
    
    all_results = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_embeddings = sample_embeddings[start_idx:end_idx]
        metadata = [{'id': j} for j in range(start_idx, end_idx)]
        
        results = manager.update(batch_embeddings, metadata)
        all_results.append(results)
    
    final_clusters = manager.get_clusters()
    
    assert isinstance(final_clusters, dict)
    assert len(final_clusters) > 0
    assert all(isinstance(cluster, list) for cluster in final_clusters.values())
    
    # Verify cluster statistics
    stats = all_results[-1]['stats']
    assert stats['num_clusters'] > 0
    assert stats['total_docs'] == len(sample_embeddings)
    assert stats['avg_cluster_size'] > 0

def test_empty_clustering(config):
    """Test handling of empty input."""
    clusterer = DynamicClusterer(config)
    empty_embeddings = np.array([])
    
    with pytest.raises(ValueError):
        clusterer.fit_predict(empty_embeddings)