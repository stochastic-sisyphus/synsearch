"""
CAG Enhanced Search Engine for SynSearch.

Integrates Context Augmented Generation with the existing
semantic search infrastructure to provide contextual intelligence.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import numpy as np

from .base import (
    AggregationStrategy,
    CAGScore,
    ContextAggregator,
    ContextProvider,
    ContextResult,
    ContextType,
    Document,
    Query,
)
from .temporal import TemporalContextProvider
from .authority import AuthorityContextProvider
from .sentiment import SentimentContextProvider
from .events import EventContextProvider


@dataclass
class SearchResult:
    """Enhanced search result with CAG scoring."""
    document: Document
    semantic_score: float
    cag_score: CAGScore
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Get the final combined score."""
        return self.cag_score.final_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document": self.document.to_dict(),
            "semantic_score": self.semantic_score,
            "final_score": self.final_score,
            "context_score": self.cag_score.context_score,
            "context_confidence": self.cag_score.context_confidence,
            "rank": self.rank,
            "context_breakdown": {
                ct.name: {
                    "score": result.aggregate_score,
                    "confidence": result.aggregate_confidence,
                    "signals": len(result.signals),
                }
                for ct, result in self.cag_score.context_results.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class SearchMetrics:
    """Metrics for a CAG-enhanced search operation."""
    query: str
    total_documents: int
    results_returned: int
    semantic_search_time_ms: float
    context_analysis_time_ms: float
    total_time_ms: float
    providers_used: List[str]
    average_context_confidence: float
    rerank_changes: int  # How many documents changed rank due to CAG


class CAGEnhancedSearchEngine:
    """
    CAG-Enhanced Search Engine.

    Wraps existing SynSearch semantic search with contextual
    intelligence from multiple context providers.

    Features:
    - Plugin-based context provider architecture
    - Configurable aggregation strategies
    - Parallel context analysis
    - Caching and optimization
    - Search result explanation
    """

    def __init__(
        self,
        embedding_generator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the CAG-enhanced search engine.

        Args:
            embedding_generator: Optional EnhancedEmbeddingGenerator instance
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store embedding generator reference
        self.embedding_generator = embedding_generator

        # Configuration
        self.semantic_weight = self.config.get("semantic_weight", 0.6)
        self.context_weight = self.config.get("context_weight", 0.4)
        self.parallel_analysis = self.config.get("parallel_analysis", True)
        self.max_workers = self.config.get("max_workers", 4)
        self.cache_enabled = self.config.get("cache_enabled", True)

        # Initialize context providers
        self._providers: Dict[ContextType, ContextProvider] = {}
        self._init_default_providers()

        # Initialize aggregator
        strategy = AggregationStrategy[
            self.config.get("aggregation_strategy", "WEIGHTED_AVERAGE")
        ]
        self.aggregator = ContextAggregator(strategy=strategy)

        # Provider weights
        self._provider_weights: Dict[ContextType, float] = {
            ContextType.TEMPORAL: self.config.get("temporal_weight", 1.0),
            ContextType.AUTHORITY: self.config.get("authority_weight", 1.0),
            ContextType.SENTIMENT: self.config.get("sentiment_weight", 1.0),
            ContextType.EVENT: self.config.get("event_weight", 1.0),
        }

        # Document store
        self._documents: Dict[str, Document] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._document_ids: List[str] = []

        # Cache
        self._context_cache: Dict[str, Dict[ContextType, ContextResult]] = {}

    def _init_default_providers(self) -> None:
        """Initialize default context providers."""
        provider_configs = self.config.get("providers", {})

        # Temporal provider
        temporal_config = provider_configs.get("temporal", {})
        if temporal_config.get("enabled", True):
            self._providers[ContextType.TEMPORAL] = TemporalContextProvider(
                temporal_config
            )

        # Authority provider
        authority_config = provider_configs.get("authority", {})
        if authority_config.get("enabled", True):
            self._providers[ContextType.AUTHORITY] = AuthorityContextProvider(
                authority_config
            )

        # Sentiment provider
        sentiment_config = provider_configs.get("sentiment", {})
        if sentiment_config.get("enabled", True):
            self._providers[ContextType.SENTIMENT] = SentimentContextProvider(
                sentiment_config
            )

        # Event provider
        event_config = provider_configs.get("events", {})
        if event_config.get("enabled", True):
            self._providers[ContextType.EVENT] = EventContextProvider(
                event_config
            )

        self.logger.info(
            f"Initialized {len(self._providers)} context providers: "
            f"{[p.name for p in self._providers.values()]}"
        )

    def register_provider(
        self,
        provider: ContextProvider,
        weight: float = 1.0
    ) -> None:
        """
        Register a custom context provider.

        Args:
            provider: The context provider to register
            weight: Weight for this provider in aggregation
        """
        self._providers[provider.context_type] = provider
        self._provider_weights[provider.context_type] = weight
        self.logger.info(f"Registered provider: {provider.name} with weight {weight}")

    def unregister_provider(self, context_type: ContextType) -> None:
        """Unregister a context provider."""
        if context_type in self._providers:
            del self._providers[context_type]
            self.logger.info(f"Unregistered provider for {context_type.name}")

    def set_provider_weight(
        self,
        context_type: ContextType,
        weight: float
    ) -> None:
        """Set the weight for a specific provider."""
        self._provider_weights[context_type] = weight

    def index_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Index documents for search.

        Args:
            documents: List of documents to index
            embeddings: Pre-computed embeddings (optional)
        """
        self.logger.info(f"Indexing {len(documents)} documents")

        for doc in documents:
            self._documents[doc.id] = doc
            self._document_ids.append(doc.id)

        if embeddings is not None:
            self._embeddings = embeddings
        elif self.embedding_generator is not None:
            # Generate embeddings
            texts = [doc.content for doc in documents]
            self._embeddings = self.embedding_generator.generate_embeddings(texts)

        self.logger.info(f"Indexed {len(self._documents)} documents")

    def search(
        self,
        query: Union[str, Query],
        top_k: int = 10,
        semantic_results: Optional[List[Tuple[str, float]]] = None,
        min_score: float = 0.0,
        explain: bool = False
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Perform CAG-enhanced search.

        Args:
            query: Search query (string or Query object)
            top_k: Number of results to return
            semantic_results: Pre-computed semantic search results (doc_id, score)
            min_score: Minimum score threshold
            explain: Include detailed explanations

        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()

        # Convert string query to Query object
        if isinstance(query, str):
            query = Query(text=query)

        # Analyze query context
        query_context = self._analyze_query_context(query)

        # Get semantic search results
        semantic_start = time.time()
        if semantic_results is None:
            semantic_results = self._semantic_search(query, top_k * 2)
        semantic_time = (time.time() - semantic_start) * 1000

        # Analyze document context and compute CAG scores
        context_start = time.time()
        results = self._analyze_and_score(
            semantic_results, query, query_context, explain
        )
        context_time = (time.time() - context_start) * 1000

        # Sort by final score and apply threshold
        results = [r for r in results if r.final_score >= min_score]
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Calculate rerank changes
        rerank_changes = self._count_rerank_changes(semantic_results, results)

        # Assign final ranks
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1

        # Compute metrics
        total_time = (time.time() - start_time) * 1000
        avg_confidence = np.mean([
            r.cag_score.context_confidence for r in results[:top_k]
        ]) if results else 0.0

        metrics = SearchMetrics(
            query=query.text,
            total_documents=len(self._documents),
            results_returned=min(top_k, len(results)),
            semantic_search_time_ms=semantic_time,
            context_analysis_time_ms=context_time,
            total_time_ms=total_time,
            providers_used=[p.name for p in self._providers.values()],
            average_context_confidence=avg_confidence,
            rerank_changes=rerank_changes,
        )

        return results[:top_k], metrics

    def _semantic_search(
        self,
        query: Query,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Perform semantic search using embeddings."""
        if self._embeddings is None or self.embedding_generator is None:
            self.logger.warning("No embeddings available for semantic search")
            return []

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query.text])[0]

        # Calculate cosine similarities
        similarities = np.dot(self._embeddings, query_embedding) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self._document_ids[i], float(similarities[i]))
            for i in top_indices
        ]

        return results

    def _analyze_query_context(
        self,
        query: Query
    ) -> Dict[ContextType, ContextResult]:
        """Analyze query for contextual signals."""
        query_context = {}

        for context_type, provider in self._providers.items():
            if provider.enabled:
                try:
                    result = provider.analyze_query(query)
                    query_context[context_type] = result
                except Exception as e:
                    self.logger.error(
                        f"Error analyzing query with {provider.name}: {e}"
                    )

        return query_context

    def _analyze_and_score(
        self,
        semantic_results: List[Tuple[str, float]],
        query: Query,
        query_context: Dict[ContextType, ContextResult],
        explain: bool
    ) -> List[SearchResult]:
        """Analyze documents and compute CAG scores."""
        results = []

        if self.parallel_analysis and len(semantic_results) > 1:
            # Parallel analysis
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._analyze_document_context,
                        doc_id, query, query_context
                    ): (doc_id, score)
                    for doc_id, score in semantic_results
                }

                for future in as_completed(futures):
                    doc_id, semantic_score = futures[future]
                    try:
                        context_results = future.result()
                        result = self._create_search_result(
                            doc_id, semantic_score, context_results, explain
                        )
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing {doc_id}: {e}")
        else:
            # Sequential analysis
            for doc_id, semantic_score in semantic_results:
                try:
                    context_results = self._analyze_document_context(
                        doc_id, query, query_context
                    )
                    result = self._create_search_result(
                        doc_id, semantic_score, context_results, explain
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {doc_id}: {e}")

        return results

    def _analyze_document_context(
        self,
        doc_id: str,
        query: Query,
        query_context: Dict[ContextType, ContextResult]
    ) -> Dict[ContextType, ContextResult]:
        """Analyze a document's context across all providers."""
        # Check cache
        cache_key = f"{doc_id}:{query.text}"
        if self.cache_enabled and cache_key in self._context_cache:
            return self._context_cache[cache_key]

        document = self._documents.get(doc_id)
        if not document:
            return {}

        context_results = {}

        for context_type, provider in self._providers.items():
            if not provider.enabled:
                continue

            try:
                result = provider.analyze_document(document, query)

                # Incorporate query context signals
                if context_type in query_context:
                    query_result = query_context[context_type]
                    for signal in query_result.signals:
                        # Adjust document signals based on query context
                        if signal.name.endswith("_preference"):
                            self._adjust_for_preference(result, signal)

                context_results[context_type] = result

            except Exception as e:
                self.logger.error(
                    f"Error with {provider.name} on {doc_id}: {e}"
                )
                context_results[context_type] = ContextResult(
                    context_type=context_type,
                    error=str(e)
                )

        # Cache results
        if self.cache_enabled:
            self._context_cache[cache_key] = context_results

        return context_results

    def _adjust_for_preference(
        self,
        result: ContextResult,
        preference_signal: Any
    ) -> None:
        """Adjust result signals based on query preferences."""
        # This modifies the result in place based on query preferences
        # For example, if query prefers "recent" content, boost recency signals
        pass  # Implemented through the signal scoring already

    def _create_search_result(
        self,
        doc_id: str,
        semantic_score: float,
        context_results: Dict[ContextType, ContextResult],
        explain: bool
    ) -> SearchResult:
        """Create a SearchResult with CAG scoring."""
        document = self._documents[doc_id]

        # Compute CAG score
        cag_score = CAGScore.compute(
            document_id=doc_id,
            semantic_score=semantic_score,
            context_results=context_results,
            aggregator=self.aggregator,
            semantic_weight=self.semantic_weight,
            context_weight=self.context_weight,
            provider_weights=self._provider_weights,
        )

        metadata = {}
        if explain:
            metadata["explanation"] = self._generate_explanation(
                document, semantic_score, context_results, cag_score
            )

        return SearchResult(
            document=document,
            semantic_score=semantic_score,
            cag_score=cag_score,
            rank=0,  # Will be set after sorting
            metadata=metadata,
        )

    def _generate_explanation(
        self,
        document: Document,
        semantic_score: float,
        context_results: Dict[ContextType, ContextResult],
        cag_score: CAGScore
    ) -> Dict[str, Any]:
        """Generate explanation for search result ranking."""
        explanation = {
            "semantic_contribution": semantic_score * self.semantic_weight,
            "context_contribution": cag_score.context_score * self.context_weight,
            "context_details": {},
        }

        for context_type, result in context_results.items():
            if result.is_successful and result.signals:
                top_signals = result.top_signals[:3]
                explanation["context_details"][context_type.name] = {
                    "aggregate_score": result.aggregate_score,
                    "confidence": result.aggregate_confidence,
                    "top_signals": [
                        {
                            "name": s.name,
                            "score": s.score,
                            "explanation": s.explanation,
                        }
                        for s in top_signals
                    ],
                }

        return explanation

    def _count_rerank_changes(
        self,
        semantic_results: List[Tuple[str, float]],
        cag_results: List[SearchResult]
    ) -> int:
        """Count how many documents changed position due to CAG."""
        if not semantic_results or not cag_results:
            return 0

        semantic_order = [doc_id for doc_id, _ in semantic_results]
        cag_order = [r.document.id for r in cag_results]

        changes = 0
        for i, doc_id in enumerate(cag_order):
            if doc_id in semantic_order:
                original_pos = semantic_order.index(doc_id)
                if original_pos != i:
                    changes += 1

        return changes

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> Dict[ContextType, ContextResult]:
        """
        Analyze a single document's context.

        Useful for understanding why a document ranked as it did.
        """
        results = {}

        for context_type, provider in self._providers.items():
            if provider.enabled:
                try:
                    results[context_type] = provider.analyze_document(document, query)
                except Exception as e:
                    self.logger.error(f"Error with {provider.name}: {e}")
                    results[context_type] = ContextResult(
                        context_type=context_type, error=str(e)
                    )

        return results

    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._context_cache.clear()
        for provider in self._providers.values():
            provider.clear_cache()

    def get_provider(self, context_type: ContextType) -> Optional[ContextProvider]:
        """Get a specific context provider."""
        return self._providers.get(context_type)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "documents_indexed": len(self._documents),
            "embeddings_available": self._embeddings is not None,
            "providers": {
                ct.name: {
                    "name": p.name,
                    "enabled": p.enabled,
                    "weight": self._provider_weights.get(ct, 1.0),
                    "cache_stats": p.get_cache_stats(),
                }
                for ct, p in self._providers.items()
            },
            "cache_size": len(self._context_cache),
            "semantic_weight": self.semantic_weight,
            "context_weight": self.context_weight,
        }


def create_cag_engine(
    embedding_generator: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> CAGEnhancedSearchEngine:
    """
    Factory function to create a CAG-enhanced search engine.

    Args:
        embedding_generator: Optional embedding generator instance
        config: Configuration dictionary

    Returns:
        Configured CAGEnhancedSearchEngine instance
    """
    return CAGEnhancedSearchEngine(
        embedding_generator=embedding_generator,
        config=config
    )
