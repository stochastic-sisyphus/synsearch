"""
Sentiment Context Provider for the CAG Framework.

Analyzes community and domain sentiment around topics, concepts,
and sources to enhance search relevance with sentiment intelligence.
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
class SentimentSnapshot:
    """Point-in-time sentiment measurement."""
    score: float  # -1.0 (negative) to 1.0 (positive)
    magnitude: float  # 0.0 (neutral) to 1.0 (strong)
    confidence: float
    timestamp: datetime
    source: str  # Where sentiment was measured


@dataclass
class TopicSentiment:
    """Aggregated sentiment for a topic over time."""
    topic: str
    current_sentiment: float
    sentiment_trend: str  # "improving", "declining", "stable", "volatile"
    controversy_score: float  # 0-1, higher = more controversial
    history: List[SentimentSnapshot] = field(default_factory=list)
    related_emotions: Dict[str, float] = field(default_factory=dict)  # emotion -> strength


@dataclass
class CommunitySentiment:
    """Sentiment from a specific community or domain."""
    community: str
    domain: str
    overall_sentiment: float
    key_topics: Dict[str, float] = field(default_factory=dict)  # topic -> sentiment
    sentiment_drivers: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class SentimentContextProvider(ContextProvider):
    """
    Provides sentiment context intelligence for search.

    Features:
    - Document sentiment analysis
    - Topic-level sentiment tracking
    - Community sentiment aggregation
    - Controversy detection
    - Sentiment-aware query matching
    """

    # Sentiment lexicon (simplified - in production, use a full lexicon)
    POSITIVE_TERMS = {
        "excellent": 0.9, "outstanding": 0.9, "breakthrough": 0.85,
        "innovative": 0.8, "successful": 0.75, "effective": 0.7,
        "promising": 0.65, "improved": 0.6, "better": 0.55,
        "good": 0.5, "positive": 0.5, "beneficial": 0.6,
        "novel": 0.65, "significant": 0.6, "remarkable": 0.75,
        "impressive": 0.7, "efficient": 0.6, "robust": 0.55,
        "accurate": 0.6, "reliable": 0.55, "valuable": 0.6,
    }

    NEGATIVE_TERMS = {
        "poor": -0.6, "failed": -0.75, "failure": -0.8,
        "problematic": -0.6, "flawed": -0.65, "limited": -0.4,
        "weak": -0.5, "incorrect": -0.6, "error": -0.5,
        "bias": -0.55, "biased": -0.55, "unreliable": -0.7,
        "controversial": -0.3, "disputed": -0.4, "criticized": -0.5,
        "concern": -0.4, "issue": -0.3, "problem": -0.4,
        "risk": -0.35, "dangerous": -0.7, "harmful": -0.75,
    }

    # Controversy indicators
    CONTROVERSY_TERMS = {
        "controversial", "debate", "disputed", "contested",
        "criticized", "criticism", "concern", "concerns",
        "questioned", "challenge", "challenges", "disagree",
        "disagreement", "conflict", "contradictory",
    }

    # Emotion categories
    EMOTION_TERMS = {
        "excitement": ["exciting", "enthusiastic", "thrilling", "amazing"],
        "concern": ["worried", "concerning", "troubling", "alarming"],
        "surprise": ["surprising", "unexpected", "shocking", "remarkable"],
        "trust": ["reliable", "trustworthy", "credible", "dependable"],
        "anticipation": ["promising", "potential", "future", "upcoming"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment context provider.

        Config options:
            default_sentiment: float - Default sentiment when unknown (default: 0.0)
            controversy_threshold: float - Threshold for controversy flag (default: 0.3)
            sentiment_decay_days: int - Days for sentiment memory decay (default: 30)
            prefer_positive: bool - Slight preference for positive content (default: False)
            prefer_neutral: bool - Prefer neutral/balanced content (default: False)
        """
        super().__init__(config)

        self.default_sentiment = self.config.get("default_sentiment", 0.0)
        self.controversy_threshold = self.config.get("controversy_threshold", 0.3)
        self.sentiment_decay_days = self.config.get("sentiment_decay_days", 30)
        self.prefer_positive = self.config.get("prefer_positive", False)
        self.prefer_neutral = self.config.get("prefer_neutral", False)

        # Sentiment storage
        self._topic_sentiments: Dict[str, TopicSentiment] = {}
        self._community_sentiments: Dict[str, CommunitySentiment] = {}
        self._document_sentiments: Dict[str, SentimentSnapshot] = {}

    @property
    def context_type(self) -> ContextType:
        return ContextType.SENTIMENT

    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> ContextResult:
        """
        Analyze a document for sentiment signals.

        Produces signals for:
        - Document sentiment
        - Topic sentiment alignment
        - Controversy level
        - Emotional tone
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            # Document sentiment analysis
            doc_sentiment = self._analyze_document_sentiment(document)
            result.add_signal(doc_sentiment)

            # Topic sentiment alignment
            topic_signal = self._compute_topic_sentiment_alignment(document)
            if topic_signal:
                result.add_signal(topic_signal)

            # Controversy detection
            controversy_signal = self._detect_controversy(document)
            if controversy_signal:
                result.add_signal(controversy_signal)

            # Emotional tone analysis
            emotion_signal = self._analyze_emotional_tone(document)
            if emotion_signal:
                result.add_signal(emotion_signal)

            # Community sentiment alignment
            if document.domain:
                community_signal = self._compute_community_alignment(document)
                if community_signal:
                    result.add_signal(community_signal)

            # Query sentiment preference matching
            if query and query.sentiment_preference:
                preference_signal = self._check_sentiment_preference(
                    doc_sentiment.score, query.sentiment_preference
                )
                result.add_signal(preference_signal)

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            result.error = str(e)

        return result

    def analyze_query(self, query: Query) -> ContextResult:
        """
        Analyze a query for sentiment preferences.

        Detects:
        - Sentiment preference (positive/negative/neutral)
        - Controversy tolerance
        - Emotional tone preferences
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            query_lower = query.text.lower()

            # Check for sentiment preference indicators
            positive_indicators = ["best", "top", "excellent", "positive", "successful"]
            negative_indicators = ["worst", "failed", "problems", "issues", "criticism"]
            neutral_indicators = ["balanced", "objective", "unbiased", "neutral", "fair"]

            if any(ind in query_lower for ind in positive_indicators):
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="positive_preference",
                    score=0.6,
                    confidence=0.7,
                    explanation="Query indicates preference for positive content"
                ))

            if any(ind in query_lower for ind in negative_indicators):
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="negative_interest",
                    score=-0.3,
                    confidence=0.7,
                    explanation="Query indicates interest in negative aspects"
                ))

            if any(ind in query_lower for ind in neutral_indicators):
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="neutral_preference",
                    score=0.0,
                    confidence=0.8,
                    metadata={"prefer_neutral": True},
                    explanation="Query indicates preference for neutral/balanced content"
                ))

            # Explicit sentiment preference
            if query.sentiment_preference:
                pref_scores = {
                    "positive": 0.7,
                    "negative": -0.5,
                    "neutral": 0.0,
                }
                score = pref_scores.get(query.sentiment_preference, 0.0)
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="explicit_sentiment_preference",
                    score=score,
                    confidence=0.95,
                    metadata={"preference": query.sentiment_preference},
                    explanation=f"Explicit sentiment preference: {query.sentiment_preference}"
                ))

            # Check for controversy interest
            controversy_keywords = ["controversial", "debate", "disputed", "criticism"]
            if any(kw in query_lower for kw in controversy_keywords):
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="controversy_interest",
                    score=0.0,  # Neutral about controversy
                    confidence=0.75,
                    metadata={"seeks_controversy": True},
                    explanation="Query indicates interest in controversial content"
                ))

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in query sentiment analysis: {e}")
            result.error = str(e)

        return result

    def _analyze_document_sentiment(self, document: Document) -> ContextSignal:
        """Analyze the overall sentiment of a document."""
        content_lower = document.content.lower()
        words = re.findall(r'\b[a-z]+\b', content_lower)

        positive_score = 0.0
        negative_score = 0.0
        total_sentiment_words = 0

        for word in words:
            if word in self.POSITIVE_TERMS:
                positive_score += self.POSITIVE_TERMS[word]
                total_sentiment_words += 1
            elif word in self.NEGATIVE_TERMS:
                negative_score += abs(self.NEGATIVE_TERMS[word])
                total_sentiment_words += 1

        if total_sentiment_words == 0:
            sentiment = self.default_sentiment
            confidence = 0.3
        else:
            # Calculate net sentiment
            sentiment = (positive_score - negative_score) / total_sentiment_words
            sentiment = max(-1.0, min(1.0, sentiment))

            # Confidence based on number of sentiment words
            confidence = min(0.9, 0.3 + (total_sentiment_words / 50))

        # Calculate magnitude (how strong the sentiment is)
        magnitude = (positive_score + negative_score) / max(1, total_sentiment_words)

        # Store for future reference
        self._document_sentiments[document.id] = SentimentSnapshot(
            score=sentiment,
            magnitude=magnitude,
            confidence=confidence,
            timestamp=datetime.now(),
            source="document_analysis"
        )

        # Determine sentiment label
        if sentiment > 0.3:
            label = "positive"
        elif sentiment < -0.3:
            label = "negative"
        else:
            label = "neutral"

        return ContextSignal(
            context_type=self.context_type,
            name="document_sentiment",
            score=sentiment,
            confidence=confidence,
            metadata={
                "magnitude": magnitude,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "sentiment_words": total_sentiment_words,
                "label": label,
            },
            explanation=f"Document has {label} sentiment (score: {sentiment:.2f})"
        )

    def _compute_topic_sentiment_alignment(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Check document alignment with topic-level sentiment."""
        # Extract topics from document
        topics = self._extract_topics(document.content)

        if not topics:
            return None

        topic_scores = []
        for topic in topics[:5]:  # Top 5 topics
            topic_sentiment = self._topic_sentiments.get(topic)
            if topic_sentiment:
                topic_scores.append({
                    "topic": topic,
                    "sentiment": topic_sentiment.current_sentiment,
                    "controversy": topic_sentiment.controversy_score,
                })

        if not topic_scores:
            return None

        avg_topic_sentiment = np.mean([t["sentiment"] for t in topic_scores])
        max_controversy = max(t["controversy"] for t in topic_scores)

        return ContextSignal(
            context_type=self.context_type,
            name="topic_sentiment_alignment",
            score=avg_topic_sentiment,
            confidence=0.6,
            metadata={
                "topics": topic_scores,
                "max_controversy": max_controversy,
            },
            explanation=f"Document topics have average sentiment {avg_topic_sentiment:.2f}"
        )

    def _detect_controversy(self, document: Document) -> Optional[ContextSignal]:
        """Detect level of controversy in the document."""
        content_lower = document.content.lower()

        controversy_count = sum(
            1 for term in self.CONTROVERSY_TERMS
            if term in content_lower
        )

        # Normalize by document length (per 1000 chars)
        doc_length = len(document.content)
        controversy_density = (controversy_count / max(1, doc_length)) * 1000

        # Calculate controversy score (0 to 1)
        controversy_score = min(1.0, controversy_density / 5)

        if controversy_score < self.controversy_threshold:
            return None

        # Controversy affects relevance differently based on config
        if self.prefer_neutral:
            # Penalize controversial content
            score = -controversy_score * 0.5
        else:
            # Neutral about controversy
            score = 0.0

        return ContextSignal(
            context_type=self.context_type,
            name="controversy_level",
            score=score,
            confidence=0.7,
            metadata={
                "controversy_score": controversy_score,
                "controversy_count": controversy_count,
                "is_controversial": controversy_score >= self.controversy_threshold,
            },
            explanation=f"Document has controversy score {controversy_score:.2f}"
        )

    def _analyze_emotional_tone(self, document: Document) -> Optional[ContextSignal]:
        """Analyze the emotional tone of the document."""
        content_lower = document.content.lower()

        emotion_scores = {}
        for emotion, terms in self.EMOTION_TERMS.items():
            count = sum(1 for term in terms if term in content_lower)
            if count > 0:
                emotion_scores[emotion] = min(1.0, count / 3)

        if not emotion_scores:
            return None

        # Determine dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])

        # Map emotions to sentiment-like scores
        emotion_sentiment_map = {
            "excitement": 0.6,
            "concern": -0.4,
            "surprise": 0.2,
            "trust": 0.5,
            "anticipation": 0.4,
        }

        dominant_score = emotion_sentiment_map.get(dominant_emotion[0], 0.0)
        weighted_score = dominant_score * dominant_emotion[1]

        return ContextSignal(
            context_type=self.context_type,
            name="emotional_tone",
            score=weighted_score,
            confidence=0.55,
            metadata={
                "emotion_scores": emotion_scores,
                "dominant_emotion": dominant_emotion[0],
                "dominant_strength": dominant_emotion[1],
            },
            explanation=f"Document has dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})"
        )

    def _compute_community_alignment(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Check alignment with community sentiment."""
        domain = document.domain

        community = self._community_sentiments.get(domain)
        if not community:
            return None

        # Get document sentiment
        doc_sentiment = self._document_sentiments.get(document.id)
        if not doc_sentiment:
            return None

        # Calculate alignment (how close document is to community sentiment)
        sentiment_diff = abs(doc_sentiment.score - community.overall_sentiment)
        alignment = 1.0 - min(1.0, sentiment_diff)

        # Higher alignment with community can be positive or negative
        # depending on whether we want mainstream or contrarian views
        score = alignment * 0.3  # Slight positive for alignment

        return ContextSignal(
            context_type=self.context_type,
            name="community_sentiment_alignment",
            score=score,
            confidence=0.5,
            metadata={
                "community": domain,
                "community_sentiment": community.overall_sentiment,
                "document_sentiment": doc_sentiment.score,
                "alignment": alignment,
            },
            explanation=f"Document sentiment alignment with {domain} community: {alignment:.2f}"
        )

    def _check_sentiment_preference(
        self,
        doc_sentiment: float,
        preference: str
    ) -> ContextSignal:
        """Check if document matches sentiment preference."""
        preference_map = {
            "positive": (0.3, 1.0),
            "negative": (-1.0, -0.3),
            "neutral": (-0.3, 0.3),
        }

        if preference not in preference_map:
            return ContextSignal(
                context_type=self.context_type,
                name="sentiment_preference_check",
                score=0.0,
                confidence=0.5,
                metadata={"unknown_preference": preference},
                explanation=f"Unknown sentiment preference: {preference}"
            )

        low, high = preference_map[preference]
        matches = low <= doc_sentiment <= high

        if matches:
            score = 0.6
            explanation = f"Document matches '{preference}' sentiment preference"
        else:
            score = -0.4
            explanation = f"Document does not match '{preference}' sentiment preference"

        return ContextSignal(
            context_type=self.context_type,
            name="sentiment_preference_match",
            score=score,
            confidence=0.8,
            metadata={
                "preference": preference,
                "doc_sentiment": doc_sentiment,
                "matches": matches,
            },
            explanation=explanation
        )

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        # Simple keyword extraction (in production, use NLP models)
        words = re.findall(r'\b[a-z]{5,}\b', text.lower())

        # Count frequencies
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1

        # Filter common words
        stopwords = {
            "which", "their", "would", "there", "about", "could",
            "these", "other", "being", "through", "should", "where",
            "after", "before", "between", "under", "above", "while",
        }

        topics = [
            word for word, count in sorted(
                freq.items(), key=lambda x: x[1], reverse=True
            )
            if word not in stopwords and count >= 2
        ]

        return topics[:10]

    def update_topic_sentiment(
        self,
        topic: str,
        sentiment: float,
        source: str,
        confidence: float = 0.7
    ) -> None:
        """Update sentiment tracking for a topic."""
        if topic not in self._topic_sentiments:
            self._topic_sentiments[topic] = TopicSentiment(
                topic=topic,
                current_sentiment=sentiment,
                sentiment_trend="stable",
                controversy_score=0.0,
            )

        topic_data = self._topic_sentiments[topic]
        topic_data.history.append(SentimentSnapshot(
            score=sentiment,
            magnitude=abs(sentiment),
            confidence=confidence,
            timestamp=datetime.now(),
            source=source,
        ))

        # Update current sentiment (exponential moving average)
        alpha = 0.3
        topic_data.current_sentiment = (
            alpha * sentiment + (1 - alpha) * topic_data.current_sentiment
        )

        # Update trend
        self._update_sentiment_trend(topic_data)

    def _update_sentiment_trend(self, topic_data: TopicSentiment) -> None:
        """Update the sentiment trend for a topic."""
        recent_cutoff = datetime.now() - timedelta(days=self.sentiment_decay_days)
        recent = [s for s in topic_data.history if s.timestamp >= recent_cutoff]

        if len(recent) < 3:
            topic_data.sentiment_trend = "stable"
            return

        # Calculate trend
        scores = [s.score for s in sorted(recent, key=lambda x: x.timestamp)]
        first_half_avg = np.mean(scores[:len(scores)//2])
        second_half_avg = np.mean(scores[len(scores)//2:])

        diff = second_half_avg - first_half_avg

        if diff > 0.2:
            topic_data.sentiment_trend = "improving"
        elif diff < -0.2:
            topic_data.sentiment_trend = "declining"
        elif np.std(scores) > 0.4:
            topic_data.sentiment_trend = "volatile"
        else:
            topic_data.sentiment_trend = "stable"

    def register_community(self, community: CommunitySentiment) -> None:
        """Register a community sentiment profile."""
        self._community_sentiments[community.domain] = community
