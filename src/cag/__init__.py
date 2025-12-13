"""
Context Augmented Generation (CAG) Framework for SynSearch.

This framework enhances semantic search with multi-dimensional contextual intelligence:
- Temporal Context: Track concept evolution over time
- Authority Context: Assess source credibility and domain expertise
- Sentiment Context: Understand community/domain sentiment
- Event Context: Real-world events affecting search relevance
"""

from .base import (
    ContextProvider,
    ContextResult,
    ContextSignal,
    ContextType,
    ContextAggregator,
)
from .temporal import TemporalContextProvider
from .authority import AuthorityContextProvider
from .sentiment import SentimentContextProvider
from .events import EventContextProvider
from .engine import CAGEnhancedSearchEngine
from .config import CAGConfig

__all__ = [
    # Base infrastructure
    "ContextProvider",
    "ContextResult",
    "ContextSignal",
    "ContextType",
    "ContextAggregator",
    # Context providers
    "TemporalContextProvider",
    "AuthorityContextProvider",
    "SentimentContextProvider",
    "EventContextProvider",
    # Engine
    "CAGEnhancedSearchEngine",
    # Configuration
    "CAGConfig",
]

__version__ = "1.0.0"
