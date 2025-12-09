"""
Temporal Context Provider for the CAG Framework.

Tracks how concepts evolve over time, identifies trending topics,
and adjusts relevance based on temporal patterns.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

from .base import (
    ContextProvider,
    ContextResult,
    ContextSignal,
    ContextType,
    Document,
    Query,
)


@dataclass
class TemporalPattern:
    """Represents a temporal pattern for a concept."""
    concept: str
    frequency_history: List[Tuple[datetime, int]] = field(default_factory=list)
    trend_direction: str = "stable"  # "rising", "falling", "stable", "volatile"
    trend_strength: float = 0.0
    seasonality: Optional[str] = None
    peak_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class ConceptEvolution:
    """Tracks how a concept's meaning or relevance evolves."""
    concept: str
    related_terms_history: Dict[str, List[Tuple[datetime, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    semantic_drift_score: float = 0.0
    current_associations: List[str] = field(default_factory=list)
    deprecated_associations: List[str] = field(default_factory=list)


class TemporalContextProvider(ContextProvider):
    """
    Provides temporal context intelligence for search.

    Features:
    - Recency scoring with configurable decay
    - Trend detection for concepts
    - Concept evolution tracking
    - Temporal query understanding
    - Historical vs current relevance balancing
    """

    # Temporal indicators in queries
    RECENCY_INDICATORS = {
        "recent": 0.9,
        "latest": 0.95,
        "new": 0.85,
        "current": 0.8,
        "today": 1.0,
        "this week": 0.9,
        "this month": 0.8,
        "this year": 0.6,
        "2024": 0.7,
        "2025": 0.9,
    }

    HISTORICAL_INDICATORS = {
        "historical": -0.5,
        "classic": -0.3,
        "traditional": -0.4,
        "original": -0.2,
        "foundational": -0.3,
        "seminal": -0.4,
        "early": -0.5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temporal context provider.

        Config options:
            decay_rate: float - Exponential decay rate for recency (default: 0.1)
            half_life_days: int - Half-life for relevance decay (default: 180)
            trend_window_days: int - Window for trend detection (default: 30)
            min_data_points: int - Minimum data points for trend analysis (default: 3)
            recency_weight: float - Weight for recency in scoring (default: 0.5)
        """
        super().__init__(config)

        self.decay_rate = self.config.get("decay_rate", 0.1)
        self.half_life_days = self.config.get("half_life_days", 180)
        self.trend_window_days = self.config.get("trend_window_days", 30)
        self.min_data_points = self.config.get("min_data_points", 3)
        self.recency_weight = self.config.get("recency_weight", 0.5)

        # Storage for temporal patterns
        self._concept_patterns: Dict[str, TemporalPattern] = {}
        self._concept_evolutions: Dict[str, ConceptEvolution] = {}
        self._document_timestamps: Dict[str, datetime] = {}

    @property
    def context_type(self) -> ContextType:
        return ContextType.TEMPORAL

    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> ContextResult:
        """
        Analyze a document for temporal context.

        Produces signals for:
        - Recency score
        - Trend alignment
        - Concept evolution relevance
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            # Extract document date
            doc_date = self._extract_document_date(document)

            if doc_date:
                # Recency signal
                recency_signal = self._compute_recency_signal(doc_date, query)
                result.add_signal(recency_signal)

                # Store for pattern tracking
                self._document_timestamps[document.id] = doc_date

            # Temporal relevance based on query
            if query:
                query_temporal_signal = self._compute_query_temporal_alignment(
                    document, query, doc_date
                )
                if query_temporal_signal:
                    result.add_signal(query_temporal_signal)

            # Trend alignment signal
            trend_signal = self._compute_trend_signal(document, query)
            if trend_signal:
                result.add_signal(trend_signal)

            # Concept evolution signal
            evolution_signal = self._compute_evolution_signal(document)
            if evolution_signal:
                result.add_signal(evolution_signal)

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            result.error = str(e)

        return result

    def analyze_query(self, query: Query) -> ContextResult:
        """
        Analyze a query for temporal intent.

        Detects:
        - Explicit temporal preferences (recent, historical)
        - Date/time references
        - Trend-related queries
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            query_lower = query.text.lower()

            # Check for recency indicators
            for indicator, strength in self.RECENCY_INDICATORS.items():
                if indicator in query_lower:
                    result.add_signal(ContextSignal(
                        context_type=self.context_type,
                        name="recency_preference",
                        score=strength,
                        confidence=0.8,
                        metadata={"indicator": indicator},
                        explanation=f"Query indicates preference for {indicator} content"
                    ))
                    break

            # Check for historical indicators
            for indicator, strength in self.HISTORICAL_INDICATORS.items():
                if indicator in query_lower:
                    result.add_signal(ContextSignal(
                        context_type=self.context_type,
                        name="historical_preference",
                        score=strength,
                        confidence=0.8,
                        metadata={"indicator": indicator},
                        explanation=f"Query indicates preference for {indicator} content"
                    ))
                    break

            # Check for explicit temporal focus
            if query.temporal_focus:
                focus_scores = {
                    "recent": 0.8,
                    "historical": -0.5,
                    "trending": 0.9,
                    "evergreen": 0.0,
                }
                score = focus_scores.get(query.temporal_focus, 0.0)
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="explicit_temporal_focus",
                    score=score,
                    confidence=0.95,
                    metadata={"focus": query.temporal_focus},
                    explanation=f"Explicit temporal focus: {query.temporal_focus}"
                ))

            # Check date range
            if query.date_range:
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="date_range_filter",
                    score=0.0,  # Neutral, but important filter
                    confidence=1.0,
                    metadata={
                        "start": query.date_range[0].isoformat(),
                        "end": query.date_range[1].isoformat(),
                    },
                    explanation="Query specifies date range filter"
                ))

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in query temporal analysis: {e}")
            result.error = str(e)

        return result

    def _extract_document_date(self, document: Document) -> Optional[datetime]:
        """Extract the publication or creation date from a document."""
        # Direct date from document
        if document.publication_date:
            return document.publication_date

        # Check metadata
        for key in ["date", "published", "created", "timestamp", "pub_date"]:
            if key in document.metadata:
                date_val = document.metadata[key]
                if isinstance(date_val, datetime):
                    return date_val
                elif isinstance(date_val, str):
                    try:
                        return datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                    except ValueError:
                        pass

        # Try to extract from content (arXiv format, etc.)
        date_patterns = [
            r"\b(\d{4})-(\d{2})-(\d{2})\b",  # YYYY-MM-DD
            r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b",
            r"arXiv:(\d{4})\.(\d+)",  # arXiv ID contains year-month
        ]

        for pattern in date_patterns:
            match = re.search(pattern, document.content[:1000])
            if match:
                try:
                    if pattern == date_patterns[0]:
                        return datetime(
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3))
                        )
                    elif pattern == date_patterns[2]:
                        year = 2000 + int(match.group(1)[:2])
                        month = int(match.group(1)[2:]) if len(match.group(1)) > 2 else 1
                        return datetime(year, month, 1)
                except ValueError:
                    continue

        return None

    def _compute_recency_signal(
        self,
        doc_date: datetime,
        query: Optional[Query] = None
    ) -> ContextSignal:
        """Compute recency score with exponential decay."""
        now = datetime.now()
        age_days = (now - doc_date).days

        # Exponential decay
        decay_lambda = np.log(2) / self.half_life_days
        recency_score = np.exp(-decay_lambda * age_days)

        # Normalize to [-1, 1] where 1 is very recent, -1 is very old
        # Score of 0.5 (half-life) maps to 0
        normalized_score = 2 * recency_score - 1

        # Adjust confidence based on data quality
        confidence = 0.9 if doc_date else 0.3

        # Consider query temporal preference
        if query and query.temporal_focus == "historical":
            # Invert for historical queries
            normalized_score = -normalized_score * 0.5

        return ContextSignal(
            context_type=self.context_type,
            name="recency_score",
            score=normalized_score,
            confidence=confidence,
            metadata={
                "age_days": age_days,
                "raw_recency": recency_score,
                "doc_date": doc_date.isoformat(),
            },
            explanation=f"Document is {age_days} days old (recency: {recency_score:.2f})"
        )

    def _compute_query_temporal_alignment(
        self,
        document: Document,
        query: Query,
        doc_date: Optional[datetime]
    ) -> Optional[ContextSignal]:
        """Check if document aligns with query's temporal requirements."""
        if not query.date_range:
            return None

        start_date, end_date = query.date_range

        if doc_date:
            if start_date <= doc_date <= end_date:
                return ContextSignal(
                    context_type=self.context_type,
                    name="date_range_match",
                    score=0.8,
                    confidence=0.95,
                    metadata={
                        "in_range": True,
                        "doc_date": doc_date.isoformat(),
                    },
                    explanation="Document date falls within query date range"
                )
            else:
                return ContextSignal(
                    context_type=self.context_type,
                    name="date_range_mismatch",
                    score=-0.8,
                    confidence=0.95,
                    metadata={
                        "in_range": False,
                        "doc_date": doc_date.isoformat(),
                    },
                    explanation="Document date outside query date range"
                )

        return None

    def _compute_trend_signal(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> Optional[ContextSignal]:
        """Compute trend alignment signal."""
        # Extract key concepts from document
        concepts = self._extract_concepts(document.content)

        if not concepts:
            return None

        # Check for trending concepts
        trending_concepts = []
        declining_concepts = []

        for concept in concepts[:10]:  # Top 10 concepts
            pattern = self._concept_patterns.get(concept)
            if pattern:
                if pattern.trend_direction == "rising":
                    trending_concepts.append((concept, pattern.trend_strength))
                elif pattern.trend_direction == "falling":
                    declining_concepts.append((concept, pattern.trend_strength))

        if not trending_concepts and not declining_concepts:
            return None

        # Calculate trend score
        trend_score = 0.0
        if trending_concepts:
            avg_trend = np.mean([s for _, s in trending_concepts])
            trend_score += avg_trend * 0.5
        if declining_concepts:
            avg_decline = np.mean([s for _, s in declining_concepts])
            trend_score -= avg_decline * 0.3

        trend_score = max(-1.0, min(1.0, trend_score))

        return ContextSignal(
            context_type=self.context_type,
            name="trend_alignment",
            score=trend_score,
            confidence=0.6,
            metadata={
                "trending_concepts": [c for c, _ in trending_concepts],
                "declining_concepts": [c for c, _ in declining_concepts],
            },
            explanation=f"Document contains {len(trending_concepts)} trending and {len(declining_concepts)} declining concepts"
        )

    def _compute_evolution_signal(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Check if document reflects current understanding of evolving concepts."""
        concepts = self._extract_concepts(document.content)

        if not concepts:
            return None

        evolved_concepts = []
        for concept in concepts[:10]:
            evolution = self._concept_evolutions.get(concept)
            if evolution and evolution.semantic_drift_score > 0.3:
                evolved_concepts.append((concept, evolution))

        if not evolved_concepts:
            return None

        # Check if document uses current associations
        doc_content_lower = document.content.lower()
        current_matches = 0
        deprecated_matches = 0

        for concept, evolution in evolved_concepts:
            for term in evolution.current_associations:
                if term.lower() in doc_content_lower:
                    current_matches += 1
            for term in evolution.deprecated_associations:
                if term.lower() in doc_content_lower:
                    deprecated_matches += 1

        total_matches = current_matches + deprecated_matches
        if total_matches == 0:
            return None

        evolution_score = (current_matches - deprecated_matches) / total_matches

        return ContextSignal(
            context_type=self.context_type,
            name="concept_evolution_alignment",
            score=evolution_score,
            confidence=0.5,
            metadata={
                "evolved_concepts": [c for c, _ in evolved_concepts],
                "current_matches": current_matches,
                "deprecated_matches": deprecated_matches,
            },
            explanation=f"Document alignment with evolved concept understanding: {evolution_score:.2f}"
        )

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using simple heuristics."""
        # Simple word frequency-based extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        # Filter common words and sort by frequency
        stopwords = {
            "that", "this", "with", "from", "have", "were", "been",
            "which", "their", "would", "there", "about", "could",
            "other", "these", "than", "into", "more", "some", "such",
            "when", "what", "very", "also", "only", "just", "most",
        }

        concepts = [
            word for word, freq in sorted(
                word_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if word not in stopwords and freq >= 2
        ]

        return concepts[:20]

    def update_trend(
        self,
        concept: str,
        frequency: int,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update trend tracking for a concept."""
        timestamp = timestamp or datetime.now()

        if concept not in self._concept_patterns:
            self._concept_patterns[concept] = TemporalPattern(concept=concept)

        pattern = self._concept_patterns[concept]
        pattern.frequency_history.append((timestamp, frequency))

        # Keep only recent history
        cutoff = timestamp - timedelta(days=self.trend_window_days * 3)
        pattern.frequency_history = [
            (ts, freq) for ts, freq in pattern.frequency_history
            if ts >= cutoff
        ]

        # Recalculate trend
        self._calculate_trend(pattern)

    def _calculate_trend(self, pattern: TemporalPattern) -> None:
        """Calculate trend direction and strength."""
        if len(pattern.frequency_history) < self.min_data_points:
            pattern.trend_direction = "stable"
            pattern.trend_strength = 0.0
            return

        # Sort by timestamp
        sorted_history = sorted(pattern.frequency_history, key=lambda x: x[0])

        # Simple linear regression
        times = np.array([
            (ts - sorted_history[0][0]).total_seconds()
            for ts, _ in sorted_history
        ])
        freqs = np.array([freq for _, freq in sorted_history])

        if np.std(freqs) == 0:
            pattern.trend_direction = "stable"
            pattern.trend_strength = 0.0
            return

        # Normalize
        times_norm = (times - times.mean()) / (times.std() + 1e-8)
        freqs_norm = (freqs - freqs.mean()) / (freqs.std() + 1e-8)

        # Calculate correlation
        correlation = np.corrcoef(times_norm, freqs_norm)[0, 1]

        if np.isnan(correlation):
            pattern.trend_direction = "stable"
            pattern.trend_strength = 0.0
            return

        pattern.trend_strength = abs(correlation)

        if correlation > 0.3:
            pattern.trend_direction = "rising"
        elif correlation < -0.3:
            pattern.trend_direction = "falling"
        else:
            pattern.trend_direction = "stable"

        # Check for volatility
        if np.std(freqs_norm) > 1.5:
            pattern.trend_direction = "volatile"

    def update_concept_evolution(
        self,
        concept: str,
        related_term: str,
        strength: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update concept evolution tracking."""
        timestamp = timestamp or datetime.now()

        if concept not in self._concept_evolutions:
            self._concept_evolutions[concept] = ConceptEvolution(concept=concept)

        evolution = self._concept_evolutions[concept]
        evolution.related_terms_history[related_term].append((timestamp, strength))

        # Update current vs deprecated associations
        self._update_associations(evolution)

    def _update_associations(self, evolution: ConceptEvolution) -> None:
        """Update current and deprecated associations for a concept."""
        current_cutoff = datetime.now() - timedelta(days=90)

        current = []
        deprecated = []

        for term, history in evolution.related_terms_history.items():
            recent = [s for ts, s in history if ts >= current_cutoff]
            old = [s for ts, s in history if ts < current_cutoff]

            if recent:
                avg_recent = np.mean(recent)
                if avg_recent > 0.3:
                    current.append(term)
            elif old:
                avg_old = np.mean(old)
                if avg_old > 0.3:
                    deprecated.append(term)

        evolution.current_associations = current
        evolution.deprecated_associations = deprecated

        # Calculate semantic drift
        if current and deprecated:
            evolution.semantic_drift_score = len(deprecated) / (len(current) + len(deprecated))
        else:
            evolution.semantic_drift_score = 0.0
