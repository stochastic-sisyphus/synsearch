"""
Dynamic Summarization and Adaptive Clustering Framework
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data_loader import DataLoader
from .data_preparation import DataPreparator
from .data_validator import DataValidator
from .embedding_generator import EnhancedEmbeddingGenerator
from .preprocessor import DomainAgnosticPreprocessor
from .clustering.dynamic_cluster_manager import DynamicClusterManager
from .summarization.adaptive_summarizer import AdaptiveSummarizer
from .utils.style_selector import AdaptiveStyleSelector

__all__ = [
    'DataLoader',
    'DataPreparator',
    'DataValidator',
    'EnhancedEmbeddingGenerator',
    'DomainAgnosticPreprocessor',
    'DynamicClusterManager',
    'AdaptiveSummarizer',
    'AdaptiveStyleSelector'
]
