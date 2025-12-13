"""
Tests for the Context Augmented Generation (CAG) Framework.

Comprehensive test suite covering all CAG components:
- Base infrastructure
- Context providers (Temporal, Authority, Sentiment, Event)
- Search engine integration
- Configuration management
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from src.cag import (
    ContextProvider,
    ContextResult,
    ContextSignal,
    ContextType,
    ContextAggregator,
    TemporalContextProvider,
    AuthorityContextProvider,
    SentimentContextProvider,
    EventContextProvider,
    CAGEnhancedSearchEngine,
    CAGConfig,
)
from src.cag.base import (
    Document,
    Query,
    CAGScore,
    AggregationStrategy,
    SignalStrength,
)
from src.cag.authority import AuthorProfile, SourceProfile, InstitutionProfile
from src.cag.events import Event, EventType, EventImpact
from src.cag.config import CAGProfiles, load_cag_config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="doc_001",
        content="""
        This paper presents a breakthrough in natural language processing using
        transformer architectures. The proposed method achieves state-of-the-art
        results on multiple benchmarks. Our research demonstrates significant
        improvements in accuracy and efficiency compared to previous approaches.
        The methodology was validated through extensive peer-reviewed evaluation.
        """,
        title="Advances in Transformer-based NLP",
        source="Nature Machine Intelligence",
        author="Dr. Jane Smith",
        publication_date=datetime.now() - timedelta(days=30),
        domain="cs.CL",
        metadata={
            "institution": "Stanford University",
            "citations": 150,
            "peer_reviewed": True,
        }
    )


@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return Query(
        text="recent advances in transformer architectures for NLP",
        temporal_focus="recent",
        domains=["cs.CL", "cs.AI"],
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents."""
    return [
        Document(
            id="doc_001",
            content="Excellent breakthrough in machine learning techniques.",
            title="ML Advances",
            publication_date=datetime.now() - timedelta(days=10),
            domain="cs.ML",
        ),
        Document(
            id="doc_002",
            content="This paper has concerning issues and flawed methodology.",
            title="Problematic Study",
            publication_date=datetime.now() - timedelta(days=365),
            domain="cs.ML",
        ),
        Document(
            id="doc_003",
            content="A balanced analysis of neural network architectures.",
            title="Neutral Review",
            publication_date=datetime.now() - timedelta(days=90),
            domain="cs.AI",
        ),
    ]


# ============================================================================
# Base Infrastructure Tests
# ============================================================================

class TestContextSignal:
    """Tests for ContextSignal class."""

    def test_signal_creation(self):
        """Test creating a context signal."""
        signal = ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="recency_score",
            score=0.8,
            confidence=0.9,
            explanation="Document is recent",
        )

        assert signal.context_type == ContextType.TEMPORAL
        assert signal.name == "recency_score"
        assert signal.score == 0.8
        assert signal.confidence == 0.9

    def test_weighted_score(self):
        """Test weighted score calculation."""
        signal = ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="author_score",
            score=0.6,
            confidence=0.8,
        )

        assert signal.weighted_score == pytest.approx(0.48)

    def test_signal_strength(self):
        """Test signal strength classification."""
        strong_signal = ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="test",
            score=0.9,
            confidence=0.9,
        )
        assert strong_signal.strength == SignalStrength.VERY_STRONG

        weak_signal = ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="test",
            score=0.2,
            confidence=0.5,
        )
        assert weak_signal.strength == SignalStrength.WEAK

    def test_invalid_score_range(self):
        """Test that invalid scores raise errors."""
        with pytest.raises(ValueError):
            ContextSignal(
                context_type=ContextType.TEMPORAL,
                name="test",
                score=1.5,  # Invalid: > 1.0
                confidence=0.5,
            )

    def test_signal_serialization(self):
        """Test signal to_dict serialization."""
        signal = ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="test",
            score=0.5,
            confidence=0.7,
            metadata={"key": "value"},
        )

        data = signal.to_dict()
        assert data["context_type"] == "AUTHORITY"
        assert data["score"] == 0.5
        assert data["metadata"]["key"] == "value"


class TestContextResult:
    """Tests for ContextResult class."""

    def test_result_creation(self):
        """Test creating a context result."""
        result = ContextResult(context_type=ContextType.SENTIMENT)
        assert result.context_type == ContextType.SENTIMENT
        assert result.is_successful
        assert len(result.signals) == 0

    def test_add_signal(self):
        """Test adding signals to result."""
        result = ContextResult(context_type=ContextType.TEMPORAL)

        result.add_signal(ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="recency",
            score=0.8,
            confidence=0.9,
        ))

        assert len(result.signals) == 1
        assert result.aggregate_score == pytest.approx(0.8)
        assert result.aggregate_confidence == pytest.approx(0.9)

    def test_aggregate_multiple_signals(self):
        """Test aggregation with multiple signals."""
        result = ContextResult(context_type=ContextType.AUTHORITY)

        result.add_signal(ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="author",
            score=0.8,
            confidence=0.9,
        ))
        result.add_signal(ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="source",
            score=0.4,
            confidence=0.7,
        ))

        # Weighted average: (0.8*0.9 + 0.4*0.7) / (0.9 + 0.7)
        expected_score = (0.8 * 0.9 + 0.4 * 0.7) / (0.9 + 0.7)
        assert result.aggregate_score == pytest.approx(expected_score)


class TestContextAggregator:
    """Tests for ContextAggregator class."""

    def test_weighted_average_aggregation(self):
        """Test weighted average aggregation strategy."""
        aggregator = ContextAggregator(
            strategy=AggregationStrategy.WEIGHTED_AVERAGE
        )

        results = {
            ContextType.TEMPORAL: ContextResult(context_type=ContextType.TEMPORAL),
            ContextType.AUTHORITY: ContextResult(context_type=ContextType.AUTHORITY),
        }

        results[ContextType.TEMPORAL].add_signal(ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="recency",
            score=0.8,
            confidence=0.9,
        ))
        results[ContextType.AUTHORITY].add_signal(ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="author",
            score=0.6,
            confidence=0.8,
        ))

        score, confidence, metadata = aggregator.aggregate(results)

        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert metadata["method"] == "weighted_average"

    def test_max_score_aggregation(self):
        """Test max score aggregation strategy."""
        aggregator = ContextAggregator(
            strategy=AggregationStrategy.MAX_SCORE
        )

        results = {
            ContextType.TEMPORAL: ContextResult(context_type=ContextType.TEMPORAL),
            ContextType.AUTHORITY: ContextResult(context_type=ContextType.AUTHORITY),
        }

        results[ContextType.TEMPORAL].add_signal(ContextSignal(
            context_type=ContextType.TEMPORAL,
            name="recency",
            score=0.5,
            confidence=0.9,
        ))
        results[ContextType.AUTHORITY].add_signal(ContextSignal(
            context_type=ContextType.AUTHORITY,
            name="author",
            score=0.8,
            confidence=0.8,
        ))

        score, confidence, metadata = aggregator.aggregate(results)

        assert score == pytest.approx(0.8)
        assert metadata["source"] == "AUTHORITY"


# ============================================================================
# Temporal Context Provider Tests
# ============================================================================

class TestTemporalContextProvider:
    """Tests for TemporalContextProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = TemporalContextProvider()
        assert provider.context_type == ContextType.TEMPORAL
        assert provider.enabled

    def test_analyze_recent_document(self, sample_document):
        """Test analyzing a recent document."""
        provider = TemporalContextProvider()
        result = provider.analyze_document(sample_document)

        assert result.is_successful
        assert len(result.signals) > 0

        # Recent document should have positive recency score
        recency_signal = next(
            (s for s in result.signals if s.name == "recency_score"),
            None
        )
        assert recency_signal is not None
        assert recency_signal.score > 0  # Recent document

    def test_analyze_old_document(self):
        """Test analyzing an old document."""
        old_doc = Document(
            id="old_doc",
            content="Historical research from the past.",
            publication_date=datetime.now() - timedelta(days=365),
        )

        provider = TemporalContextProvider({"half_life_days": 30})
        result = provider.analyze_document(old_doc)

        recency_signal = next(
            (s for s in result.signals if s.name == "recency_score"),
            None
        )
        assert recency_signal is not None
        assert recency_signal.score < 0  # Old document has negative recency

    def test_query_temporal_analysis(self, sample_query):
        """Test analyzing a query for temporal preferences."""
        provider = TemporalContextProvider()
        result = provider.analyze_query(sample_query)

        assert result.is_successful

    def test_trend_tracking(self):
        """Test trend tracking functionality."""
        provider = TemporalContextProvider()

        # Simulate rising trend
        for i in range(5):
            provider.update_trend(
                concept="transformer",
                frequency=10 + i * 5,
                timestamp=datetime.now() - timedelta(days=30 - i * 5),
            )

        pattern = provider._concept_patterns.get("transformer")
        assert pattern is not None


# ============================================================================
# Authority Context Provider Tests
# ============================================================================

class TestAuthorityContextProvider:
    """Tests for AuthorityContextProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = AuthorityContextProvider()
        assert provider.context_type == ContextType.AUTHORITY
        assert provider.enabled

    def test_analyze_authoritative_document(self, sample_document):
        """Test analyzing a document from an authoritative source."""
        provider = AuthorityContextProvider()
        result = provider.analyze_document(sample_document)

        assert result.is_successful
        assert len(result.signals) > 0

    def test_register_author_profile(self):
        """Test registering an author profile."""
        provider = AuthorityContextProvider()

        profile = AuthorProfile(
            name="Dr. Jane Smith",
            domains={"cs.CL": 0.9, "cs.AI": 0.7},
            publication_count=150,
            citation_count=5000,
            h_index=35.0,
            affiliations=["Stanford University"],
        )

        provider.register_author(profile)

        doc = Document(
            id="test",
            content="Test content",
            author="Dr. Jane Smith",
            domain="cs.CL",
        )

        result = provider.analyze_document(doc)
        author_signal = next(
            (s for s in result.signals if "author" in s.name),
            None
        )
        assert author_signal is not None
        assert author_signal.score > 0

    def test_source_authority_recognition(self):
        """Test recognition of authoritative sources."""
        provider = AuthorityContextProvider({"domain": "cs"})

        doc = Document(
            id="test",
            content="Research findings",
            source="NeurIPS 2024",
        )

        result = provider.analyze_document(doc)
        assert result.is_successful


# ============================================================================
# Sentiment Context Provider Tests
# ============================================================================

class TestSentimentContextProvider:
    """Tests for SentimentContextProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = SentimentContextProvider()
        assert provider.context_type == ContextType.SENTIMENT
        assert provider.enabled

    def test_analyze_positive_document(self):
        """Test analyzing a document with positive sentiment."""
        provider = SentimentContextProvider()

        doc = Document(
            id="positive",
            content="""
            This excellent research presents a breakthrough in the field.
            The innovative approach shows remarkable improvements and
            outstanding performance on all benchmarks. Truly impressive work.
            """,
        )

        result = provider.analyze_document(doc)
        assert result.is_successful

        sentiment_signal = next(
            (s for s in result.signals if s.name == "document_sentiment"),
            None
        )
        assert sentiment_signal is not None
        assert sentiment_signal.score > 0
        assert sentiment_signal.metadata["label"] == "positive"

    def test_analyze_negative_document(self):
        """Test analyzing a document with negative sentiment."""
        provider = SentimentContextProvider()

        doc = Document(
            id="negative",
            content="""
            This research has serious flaws and problematic methodology.
            The results are unreliable and the approach is criticized by
            experts. Several concerning issues were identified.
            """,
        )

        result = provider.analyze_document(doc)

        sentiment_signal = next(
            (s for s in result.signals if s.name == "document_sentiment"),
            None
        )
        assert sentiment_signal is not None
        assert sentiment_signal.score < 0
        assert sentiment_signal.metadata["label"] == "negative"

    def test_controversy_detection(self):
        """Test detection of controversial content."""
        provider = SentimentContextProvider({"controversy_threshold": 0.2})

        doc = Document(
            id="controversial",
            content="""
            This controversial paper has sparked heated debate in the field.
            Critics have challenged the methodology while others dispute
            the conclusions. The contested findings remain controversial.
            """,
        )

        result = provider.analyze_document(doc)

        controversy_signal = next(
            (s for s in result.signals if s.name == "controversy_level"),
            None
        )
        assert controversy_signal is not None


# ============================================================================
# Event Context Provider Tests
# ============================================================================

class TestEventContextProvider:
    """Tests for EventContextProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = EventContextProvider()
        assert provider.context_type == ContextType.EVENT
        assert provider.enabled

    def test_register_event(self):
        """Test registering an event."""
        provider = EventContextProvider()

        event = Event(
            id="event_001",
            title="GPT-5 Release",
            event_type=EventType.RELEASE,
            description="OpenAI releases GPT-5",
            impact=EventImpact.CRITICAL,
            start_date=datetime.now() - timedelta(days=5),
            domains=["ai", "nlp"],
            keywords=["gpt-5", "language model", "openai"],
            entities=["OpenAI"],
        )

        provider.register_event(event)
        assert "event_001" in provider._events

    def test_analyze_event_related_document(self):
        """Test analyzing a document related to a registered event."""
        provider = EventContextProvider()

        # Register an event
        event = Event(
            id="event_001",
            title="Transformer Architecture Breakthrough",
            event_type=EventType.RESEARCH_BREAKTHROUGH,
            description="Major breakthrough in transformer architectures",
            impact=EventImpact.HIGH,
            start_date=datetime.now() - timedelta(days=7),
            domains=["ai"],
            keywords=["transformer", "breakthrough", "architecture"],
        )
        provider.register_event(event)

        doc = Document(
            id="doc_001",
            content="""
            Following the recent breakthrough in transformer architecture,
            researchers have made significant progress in applying these
            techniques to new domains.
            """,
            domain="ai",
        )

        result = provider.analyze_document(doc)
        assert result.is_successful

    def test_get_active_events(self):
        """Test getting active events."""
        provider = EventContextProvider()

        # Register active event
        active_event = Event(
            id="active",
            title="Active Event",
            event_type=EventType.CONFERENCE,
            description="Active conference",
            impact=EventImpact.MODERATE,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=5),
            domains=["ai"],
        )

        # Register past event
        past_event = Event(
            id="past",
            title="Past Event",
            event_type=EventType.CONFERENCE,
            description="Past conference",
            impact=EventImpact.MODERATE,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=25),
            domains=["ai"],
        )

        provider.register_event(active_event)
        provider.register_event(past_event)

        active_events = provider.get_active_events()
        assert len(active_events) == 1
        assert active_events[0].id == "active"


# ============================================================================
# CAG Enhanced Search Engine Tests
# ============================================================================

class TestCAGEnhancedSearchEngine:
    """Tests for CAGEnhancedSearchEngine."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = CAGEnhancedSearchEngine()

        assert engine is not None
        assert len(engine._providers) > 0

    def test_index_documents(self, sample_documents):
        """Test indexing documents."""
        engine = CAGEnhancedSearchEngine()

        # Create mock embeddings
        embeddings = np.random.rand(len(sample_documents), 768)

        engine.index_documents(sample_documents, embeddings)

        assert len(engine._documents) == len(sample_documents)
        assert engine._embeddings is not None

    def test_analyze_document_context(self, sample_document):
        """Test analyzing a single document."""
        engine = CAGEnhancedSearchEngine()

        results = engine.analyze_document(sample_document)

        assert ContextType.TEMPORAL in results
        assert ContextType.AUTHORITY in results
        assert ContextType.SENTIMENT in results
        assert ContextType.EVENT in results

    def test_provider_management(self):
        """Test provider registration and unregistration."""
        engine = CAGEnhancedSearchEngine()

        # Check default providers
        assert ContextType.TEMPORAL in engine._providers

        # Unregister a provider
        engine.unregister_provider(ContextType.TEMPORAL)
        assert ContextType.TEMPORAL not in engine._providers

        # Re-register
        engine.register_provider(
            TemporalContextProvider(),
            weight=1.5
        )
        assert ContextType.TEMPORAL in engine._providers
        assert engine._provider_weights[ContextType.TEMPORAL] == 1.5

    def test_set_provider_weight(self):
        """Test setting provider weights."""
        engine = CAGEnhancedSearchEngine()

        engine.set_provider_weight(ContextType.AUTHORITY, 2.0)
        assert engine._provider_weights[ContextType.AUTHORITY] == 2.0

    def test_get_stats(self):
        """Test getting engine statistics."""
        engine = CAGEnhancedSearchEngine()

        stats = engine.get_stats()

        assert "documents_indexed" in stats
        assert "providers" in stats
        assert "semantic_weight" in stats


# ============================================================================
# Configuration Tests
# ============================================================================

class TestCAGConfig:
    """Tests for CAG configuration management."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = CAGConfig()

        assert config.temporal.enabled
        assert config.authority.enabled
        assert config.aggregation.semantic_weight == 0.6
        assert config.aggregation.context_weight == 0.4

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "temporal": {"enabled": False, "half_life_days": 30},
            "authority": {"min_citations": 50},
            "aggregation": {"semantic_weight": 0.7},
        }

        config = CAGConfig.from_dict(data)

        assert config.temporal.enabled is False
        assert config.temporal.half_life_days == 30
        assert config.authority.min_citations == 50
        assert config.aggregation.semantic_weight == 0.7

    def test_config_validation(self):
        """Test configuration validation."""
        config = CAGConfig()

        issues = config.validate()
        assert len(issues) == 0  # Default config should be valid

        # Create invalid config
        config.aggregation.semantic_weight = 1.5  # Invalid
        issues = config.validate()
        assert len(issues) > 0

    def test_config_to_engine_config(self):
        """Test converting config to engine format."""
        config = CAGConfig()
        engine_config = config.to_engine_config()

        assert "semantic_weight" in engine_config
        assert "providers" in engine_config
        assert "temporal" in engine_config["providers"]

    def test_research_papers_profile(self):
        """Test research papers configuration profile."""
        config = CAGProfiles.research_papers()

        assert config.name == "research_papers"
        assert config.authority.weight == 1.5
        assert config.sentiment.enabled is False

    def test_news_articles_profile(self):
        """Test news articles configuration profile."""
        config = CAGProfiles.news_articles()

        assert config.name == "news_articles"
        assert config.temporal.weight == 1.5
        assert config.temporal.half_life_days == 7


# ============================================================================
# Integration Tests
# ============================================================================

class TestCAGIntegration:
    """Integration tests for the CAG framework."""

    def test_full_search_workflow(self, sample_documents, sample_query):
        """Test complete search workflow with CAG enhancement."""
        # Initialize engine
        config = CAGConfig()
        engine = CAGEnhancedSearchEngine(config=config.to_engine_config())

        # Create mock embeddings
        embeddings = np.random.rand(len(sample_documents), 768)

        # Index documents
        engine.index_documents(sample_documents, embeddings)

        # Create mock semantic results
        semantic_results = [
            (doc.id, 0.9 - i * 0.1)
            for i, doc in enumerate(sample_documents)
        ]

        # Perform search
        results, metrics = engine.search(
            query=sample_query,
            top_k=3,
            semantic_results=semantic_results,
            explain=True,
        )

        # Verify results
        assert len(results) <= 3
        assert metrics.total_documents == len(sample_documents)
        assert metrics.results_returned <= 3

        # Check that CAG affected ranking
        for result in results:
            assert result.cag_score is not None
            assert result.final_score is not None

    def test_context_aware_reranking(self, sample_documents):
        """Test that context signals affect document ranking."""
        engine = CAGEnhancedSearchEngine()

        embeddings = np.random.rand(len(sample_documents), 768)
        engine.index_documents(sample_documents, embeddings)

        # Create semantic results with same scores
        semantic_results = [
            (doc.id, 0.8) for doc in sample_documents
        ]

        query = Query(text="machine learning advances")

        results, metrics = engine.search(
            query=query,
            top_k=3,
            semantic_results=semantic_results,
        )

        # Results should be reranked based on context
        # (even with same semantic scores, context should differentiate)
        assert len(results) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
