"""
Base infrastructure for the Context Augmented Generation (CAG) framework.

Provides abstract base classes, data models, and core aggregation logic
for contextual intelligence enhancement of semantic search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
import logging
import numpy as np
from collections import defaultdict


class ContextType(Enum):
    """Types of contextual intelligence signals."""
    TEMPORAL = auto()      # Time-based concept evolution
    AUTHORITY = auto()     # Source credibility and expertise
    SENTIMENT = auto()     # Community/domain sentiment
    EVENT = auto()         # Real-world event influence
    COMPOSITE = auto()     # Aggregated multi-context signal


class SignalStrength(Enum):
    """Strength classification for context signals."""
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    VERY_STRONG = 1.0


@dataclass
class ContextSignal:
    """
    Individual contextual signal from a context provider.

    Represents a single piece of contextual intelligence that can
    influence search relevance scoring.
    """
    context_type: ContextType
    name: str
    score: float  # Normalized score between -1.0 and 1.0
    confidence: float  # Confidence in the signal (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    explanation: Optional[str] = None

    def __post_init__(self):
        """Validate signal values."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between -1.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def weighted_score(self) -> float:
        """Score weighted by confidence."""
        return self.score * self.confidence

    @property
    def strength(self) -> SignalStrength:
        """Classify the strength of this signal."""
        abs_score = abs(self.weighted_score)
        if abs_score >= 0.75:
            return SignalStrength.VERY_STRONG
        elif abs_score >= 0.5:
            return SignalStrength.STRONG
        elif abs_score >= 0.25:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize signal to dictionary."""
        return {
            "context_type": self.context_type.name,
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence,
            "weighted_score": self.weighted_score,
            "strength": self.strength.name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "explanation": self.explanation,
        }


@dataclass
class ContextResult:
    """
    Complete context analysis result from a provider.

    Contains all signals and aggregated scoring for a document
    or query in a specific context dimension.
    """
    context_type: ContextType
    signals: List[ContextSignal] = field(default_factory=list)
    aggregate_score: float = 0.0
    aggregate_confidence: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_signal(self, signal: ContextSignal) -> None:
        """Add a signal and update aggregate scores."""
        self.signals.append(signal)
        self._recalculate_aggregates()

    def _recalculate_aggregates(self) -> None:
        """Recalculate aggregate score and confidence from signals."""
        if not self.signals:
            self.aggregate_score = 0.0
            self.aggregate_confidence = 0.0
            return

        total_weight = sum(s.confidence for s in self.signals)
        if total_weight == 0:
            self.aggregate_score = 0.0
            self.aggregate_confidence = 0.0
            return

        self.aggregate_score = sum(
            s.score * s.confidence for s in self.signals
        ) / total_weight
        self.aggregate_confidence = total_weight / len(self.signals)

    @property
    def is_successful(self) -> bool:
        """Check if context analysis was successful."""
        return self.error is None

    @property
    def top_signals(self) -> List[ContextSignal]:
        """Get signals sorted by weighted score magnitude."""
        return sorted(
            self.signals,
            key=lambda s: abs(s.weighted_score),
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "context_type": self.context_type.name,
            "signals": [s.to_dict() for s in self.signals],
            "aggregate_score": self.aggregate_score,
            "aggregate_confidence": self.aggregate_confidence,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """
    Document representation for CAG processing.

    Wraps document content with metadata for contextual analysis.
    """
    id: str
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    domain: Optional[str] = None
    url: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to dictionary."""
        return {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "title": self.title,
            "source": self.source,
            "author": self.author,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "domain": self.domain,
            "url": self.url,
            "metadata": self.metadata,
        }


@dataclass
class Query:
    """
    Query representation for CAG processing.

    Wraps query text with contextual hints and preferences.
    """
    text: str
    temporal_focus: Optional[str] = None  # "recent", "historical", "trending"
    authority_threshold: Optional[float] = None  # Minimum authority score
    sentiment_preference: Optional[str] = None  # "positive", "negative", "neutral"
    event_relevance: bool = True  # Consider event context
    domains: List[str] = field(default_factory=list)  # Domain filters
    date_range: Optional[Tuple[datetime, datetime]] = None
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextProvider(ABC):
    """
    Abstract base class for context providers.

    Each context provider analyzes documents/queries along a specific
    contextual dimension and produces signals that influence relevance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize context provider with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache: Dict[str, ContextResult] = {}
        self._enabled = self.config.get("enabled", True)
        self._weight = self.config.get("weight", 1.0)

    @property
    @abstractmethod
    def context_type(self) -> ContextType:
        """The type of context this provider handles."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name of the provider."""
        return self.__class__.__name__

    @property
    def weight(self) -> float:
        """Weight for this provider in aggregation."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """Set the weight for this provider."""
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"Weight must be between 0.0 and 2.0, got {value}")
        self._weight = value

    @property
    def enabled(self) -> bool:
        """Whether this provider is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable this provider."""
        self._enabled = value

    @abstractmethod
    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> ContextResult:
        """
        Analyze a document and produce context signals.

        Args:
            document: The document to analyze
            query: Optional query for query-aware analysis

        Returns:
            ContextResult with signals and aggregate scores
        """
        pass

    @abstractmethod
    def analyze_query(self, query: Query) -> ContextResult:
        """
        Analyze a query for contextual hints.

        Args:
            query: The query to analyze

        Returns:
            ContextResult with query context signals
        """
        pass

    def batch_analyze(
        self,
        documents: List[Document],
        query: Optional[Query] = None,
        use_cache: bool = True
    ) -> List[ContextResult]:
        """
        Analyze multiple documents in batch.

        Args:
            documents: List of documents to analyze
            query: Optional query for query-aware analysis
            use_cache: Whether to use cached results

        Returns:
            List of ContextResults for each document
        """
        results = []
        for doc in documents:
            cache_key = f"{doc.id}:{query.text if query else 'none'}"
            if use_cache and cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                result = self.analyze_document(doc, query)
                if use_cache:
                    self._cache[cache_key] = result
                results.append(result)
        return results

    def clear_cache(self) -> None:
        """Clear the provider's cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0),
        }


class AggregationStrategy(Enum):
    """Strategies for aggregating context scores."""
    WEIGHTED_AVERAGE = auto()
    WEIGHTED_GEOMETRIC = auto()
    MAX_SCORE = auto()
    MIN_SCORE = auto()
    BAYESIAN = auto()
    CUSTOM = auto()


class ContextAggregator:
    """
    Aggregates context signals from multiple providers.

    Combines signals using configurable strategies to produce
    a final composite context score for search ranking.
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE,
        custom_aggregator: Optional[Callable[[List[ContextResult]], float]] = None
    ):
        """
        Initialize the aggregator.

        Args:
            strategy: Aggregation strategy to use
            custom_aggregator: Custom aggregation function (for CUSTOM strategy)
        """
        self.strategy = strategy
        self.custom_aggregator = custom_aggregator
        self.logger = logging.getLogger(self.__class__.__name__)

        self._strategy_funcs: Dict[AggregationStrategy, Callable] = {
            AggregationStrategy.WEIGHTED_AVERAGE: self._weighted_average,
            AggregationStrategy.WEIGHTED_GEOMETRIC: self._weighted_geometric,
            AggregationStrategy.MAX_SCORE: self._max_score,
            AggregationStrategy.MIN_SCORE: self._min_score,
            AggregationStrategy.BAYESIAN: self._bayesian,
            AggregationStrategy.CUSTOM: self._custom,
        }

    def aggregate(
        self,
        context_results: Dict[ContextType, ContextResult],
        provider_weights: Optional[Dict[ContextType, float]] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Aggregate context results into a final score.

        Args:
            context_results: Results from each context provider
            provider_weights: Optional weights per context type

        Returns:
            Tuple of (aggregate_score, confidence, metadata)
        """
        if not context_results:
            return 0.0, 0.0, {"reason": "no_context_results"}

        # Filter successful results
        valid_results = {
            ct: result for ct, result in context_results.items()
            if result.is_successful
        }

        if not valid_results:
            return 0.0, 0.0, {"reason": "all_contexts_failed"}

        # Apply provider weights
        weights = provider_weights or {}
        weighted_results = []
        for ct, result in valid_results.items():
            weight = weights.get(ct, 1.0)
            weighted_results.append((result, weight))

        # Apply aggregation strategy
        strategy_func = self._strategy_funcs[self.strategy]
        score, confidence, metadata = strategy_func(weighted_results)

        # Add context breakdown to metadata
        metadata["context_breakdown"] = {
            ct.name: {
                "score": result.aggregate_score,
                "confidence": result.aggregate_confidence,
                "signal_count": len(result.signals),
            }
            for ct, result in valid_results.items()
        }

        return score, confidence, metadata

    def _weighted_average(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Weighted average aggregation."""
        total_weight = sum(
            result.aggregate_confidence * weight
            for result, weight in weighted_results
        )

        if total_weight == 0:
            return 0.0, 0.0, {"method": "weighted_average", "note": "zero_weight"}

        score = sum(
            result.aggregate_score * result.aggregate_confidence * weight
            for result, weight in weighted_results
        ) / total_weight

        confidence = total_weight / sum(weight for _, weight in weighted_results)

        return score, confidence, {"method": "weighted_average"}

    def _weighted_geometric(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Weighted geometric mean aggregation."""
        # Shift scores to positive range for geometric mean
        shifted_scores = []
        total_weight = 0.0

        for result, weight in weighted_results:
            if result.aggregate_confidence > 0:
                # Shift from [-1, 1] to [0.1, 2.1]
                shifted = result.aggregate_score + 1.1
                shifted_scores.append((shifted, weight * result.aggregate_confidence))
                total_weight += weight * result.aggregate_confidence

        if not shifted_scores or total_weight == 0:
            return 0.0, 0.0, {"method": "weighted_geometric", "note": "insufficient_data"}

        # Calculate weighted geometric mean
        log_sum = sum(
            weight * np.log(score) for score, weight in shifted_scores
        )
        geometric_mean = np.exp(log_sum / total_weight)

        # Shift back to [-1, 1]
        score = geometric_mean - 1.1
        score = max(-1.0, min(1.0, score))  # Clamp to valid range

        confidence = total_weight / sum(weight for _, weight in weighted_results)

        return score, confidence, {"method": "weighted_geometric"}

    def _max_score(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Maximum score aggregation."""
        max_result = max(
            weighted_results,
            key=lambda x: x[0].aggregate_score * x[1]
        )

        return (
            max_result[0].aggregate_score,
            max_result[0].aggregate_confidence,
            {"method": "max_score", "source": max_result[0].context_type.name}
        )

    def _min_score(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Minimum score aggregation (conservative)."""
        min_result = min(
            weighted_results,
            key=lambda x: x[0].aggregate_score * x[1]
        )

        return (
            min_result[0].aggregate_score,
            min_result[0].aggregate_confidence,
            {"method": "min_score", "source": min_result[0].context_type.name}
        )

    def _bayesian(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Bayesian-inspired aggregation with prior."""
        prior_mean = 0.0
        prior_strength = 0.1  # Weak prior

        total_evidence = prior_strength
        weighted_sum = prior_mean * prior_strength

        for result, weight in weighted_results:
            evidence = result.aggregate_confidence * weight
            total_evidence += evidence
            weighted_sum += result.aggregate_score * evidence

        posterior_score = weighted_sum / total_evidence if total_evidence > 0 else 0.0
        posterior_confidence = 1.0 - (prior_strength / total_evidence) if total_evidence > prior_strength else 0.0

        return posterior_score, posterior_confidence, {
            "method": "bayesian",
            "prior_strength": prior_strength,
            "total_evidence": total_evidence,
        }

    def _custom(
        self,
        weighted_results: List[Tuple[ContextResult, float]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Custom aggregation function."""
        if self.custom_aggregator is None:
            raise ValueError("Custom aggregator function not provided")

        results = [result for result, _ in weighted_results]
        score = self.custom_aggregator(results)
        confidence = np.mean([r.aggregate_confidence for r in results])

        return score, confidence, {"method": "custom"}


@dataclass
class CAGScore:
    """
    Final CAG-enhanced relevance score for a document.

    Combines semantic similarity with contextual intelligence.
    """
    document_id: str
    semantic_score: float  # Original semantic similarity
    context_score: float   # Aggregated context score
    context_confidence: float  # Confidence in context score
    final_score: float     # Combined final score
    context_results: Dict[ContextType, ContextResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        document_id: str,
        semantic_score: float,
        context_results: Dict[ContextType, ContextResult],
        aggregator: ContextAggregator,
        semantic_weight: float = 0.6,
        context_weight: float = 0.4,
        provider_weights: Optional[Dict[ContextType, float]] = None
    ) -> "CAGScore":
        """
        Compute a CAG-enhanced score.

        Args:
            document_id: ID of the document
            semantic_score: Original semantic similarity score
            context_results: Context analysis results
            aggregator: Context aggregator to use
            semantic_weight: Weight for semantic score
            context_weight: Weight for context score
            provider_weights: Optional weights per context type

        Returns:
            CAGScore with combined scoring
        """
        context_score, confidence, agg_metadata = aggregator.aggregate(
            context_results, provider_weights
        )

        # Normalize context score from [-1, 1] to [0, 1]
        normalized_context = (context_score + 1.0) / 2.0

        # Compute final score
        total_weight = semantic_weight + (context_weight * confidence)
        final_score = (
            semantic_score * semantic_weight +
            normalized_context * context_weight * confidence
        ) / total_weight if total_weight > 0 else semantic_score

        return cls(
            document_id=document_id,
            semantic_score=semantic_score,
            context_score=context_score,
            context_confidence=confidence,
            final_score=final_score,
            context_results=context_results,
            metadata={
                "semantic_weight": semantic_weight,
                "context_weight": context_weight,
                "aggregation": agg_metadata,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "semantic_score": self.semantic_score,
            "context_score": self.context_score,
            "context_confidence": self.context_confidence,
            "final_score": self.final_score,
            "context_breakdown": {
                ct.name: result.to_dict()
                for ct, result in self.context_results.items()
            },
            "metadata": self.metadata,
        }
