"""
Dynamic Summarization and Adaptive Clustering Framework
====================================================

A framework for real-time research synthesis using dynamic clustering
and adaptive summarization techniques.

Main Components:
---------------
- Enhanced Embedding Generation
- Dynamic Clustering
- Adaptive Summarization
- Interactive Visualization
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__license__ = 'MIT'

from .embedding_generator import EnhancedEmbeddingGenerator
from .clustering.dynamic_cluster_manager import DynamicClusterManager
from .summarization.adaptive_summarizer import AdaptiveSummarizer
from .utils.style_selector import AdaptiveStyleSelector

__all__ = [
    'EnhancedEmbeddingGenerator',
    'DynamicClusterManager',
    'AdaptiveSummarizer',
    'AdaptiveStyleSelector'
] 