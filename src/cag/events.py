"""
Event Context Provider for the CAG Framework.

Tracks real-world events and their influence on search relevance,
enabling event-driven context enhancement.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
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


class EventType(Enum):
    """Types of events that can influence search."""
    ANNOUNCEMENT = auto()      # Product/service announcements
    RESEARCH_BREAKTHROUGH = auto()  # Scientific discoveries
    REGULATION = auto()        # Policy/regulation changes
    INCIDENT = auto()          # Security incidents, failures
    CONFERENCE = auto()        # Academic/industry conferences
    RELEASE = auto()           # Software/product releases
    ACQUISITION = auto()       # Business acquisitions
    CONTROVERSY = auto()       # Public controversies
    MILESTONE = auto()         # Achievement milestones
    TREND = auto()             # Emerging trends


class EventImpact(Enum):
    """Impact level of events."""
    MINIMAL = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


@dataclass
class Event:
    """Represents a real-world event."""
    id: str
    title: str
    event_type: EventType
    description: str
    impact: EventImpact
    start_date: datetime
    end_date: Optional[datetime] = None
    domains: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Companies, people, etc.
    sentiment_effect: float = 0.0  # -1 to 1
    relevance_boost: float = 0.0  # Extra relevance for related content
    source: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if event is currently active."""
        now = datetime.now()
        if self.end_date:
            return self.start_date <= now <= self.end_date
        return self.start_date <= now

    @property
    def age_days(self) -> float:
        """Days since event started."""
        return (datetime.now() - self.start_date).total_seconds() / 86400

    def relevance_at(self, date: datetime) -> float:
        """Calculate event relevance at a given date."""
        if date < self.start_date:
            return 0.0

        days_since = (date - self.start_date).total_seconds() / 86400

        # Events have peak relevance at start, then decay
        decay_rate = 0.1  # Decay constant
        relevance = np.exp(-decay_rate * days_since) * self.impact.value

        return relevance


@dataclass
class EventStream:
    """Stream of events for a domain or topic."""
    name: str
    events: List[Event] = field(default_factory=list)
    active_events: List[Event] = field(default_factory=list)
    event_frequency: float = 0.0  # Events per day
    volatility: float = 0.0  # How volatile the space is


class EventContextProvider(ContextProvider):
    """
    Provides event-driven context intelligence for search.

    Features:
    - Real-world event tracking
    - Event-document relevance mapping
    - Event impact scoring
    - Temporal event decay
    - Event-query alignment
    """

    # Event type keywords for detection
    EVENT_TYPE_KEYWORDS = {
        EventType.ANNOUNCEMENT: [
            "announce", "announced", "announces", "announcement",
            "unveil", "unveiled", "launch", "launched", "introduce",
        ],
        EventType.RESEARCH_BREAKTHROUGH: [
            "breakthrough", "discovery", "discovered", "novel",
            "first-ever", "unprecedented", "revolutionary",
            "state-of-the-art", "sota", "achieves", "surpasses",
        ],
        EventType.REGULATION: [
            "regulation", "regulatory", "compliance", "law",
            "legislation", "policy", "ban", "banned", "mandate",
            "requirement", "gdpr", "legal",
        ],
        EventType.INCIDENT: [
            "breach", "hack", "hacked", "vulnerability", "exploit",
            "failure", "outage", "incident", "attack", "leak",
        ],
        EventType.CONFERENCE: [
            "conference", "summit", "symposium", "workshop",
            "neurips", "icml", "iclr", "cvpr", "acl", "emnlp",
            "keynote", "presentation",
        ],
        EventType.RELEASE: [
            "release", "released", "version", "update", "upgrade",
            "v2", "v3", "beta", "alpha", "ga", "generally available",
        ],
        EventType.ACQUISITION: [
            "acquire", "acquired", "acquisition", "merger",
            "buyout", "purchase", "deal", "billion",
        ],
        EventType.CONTROVERSY: [
            "controversy", "controversial", "backlash", "criticism",
            "scandal", "accused", "allegations", "concerns",
        ],
        EventType.MILESTONE: [
            "milestone", "achievement", "record", "billion users",
            "million users", "anniversary", "celebrates",
        ],
        EventType.TREND: [
            "trending", "trend", "emerging", "growing", "rise of",
            "adoption", "surge", "boom", "explosion",
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event context provider.

        Config options:
            event_decay_days: int - Days for event relevance decay (default: 30)
            min_event_relevance: float - Minimum relevance threshold (default: 0.1)
            max_events_per_query: int - Max events to consider (default: 10)
            event_boost_multiplier: float - Boost for event-related content (default: 1.5)
            track_domains: List[str] - Domains to track events for
        """
        super().__init__(config)

        self.event_decay_days = self.config.get("event_decay_days", 30)
        self.min_event_relevance = self.config.get("min_event_relevance", 0.1)
        self.max_events_per_query = self.config.get("max_events_per_query", 10)
        self.event_boost_multiplier = self.config.get("event_boost_multiplier", 1.5)
        self.track_domains = self.config.get("track_domains", [])

        # Event storage
        self._events: Dict[str, Event] = {}
        self._event_streams: Dict[str, EventStream] = {}
        self._keyword_to_events: Dict[str, List[str]] = defaultdict(list)
        self._entity_to_events: Dict[str, List[str]] = defaultdict(list)

    @property
    def context_type(self) -> ContextType:
        return ContextType.EVENT

    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> ContextResult:
        """
        Analyze a document for event-related context.

        Produces signals for:
        - Event relevance
        - Event impact on document
        - Event type detection
        - Timeliness relative to events
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            # Find related events
            related_events = self._find_related_events(document)

            if related_events:
                # Event relevance signal
                event_signal = self._compute_event_relevance(
                    document, related_events
                )
                result.add_signal(event_signal)

                # Event impact signal
                impact_signal = self._compute_event_impact(
                    document, related_events
                )
                if impact_signal:
                    result.add_signal(impact_signal)

                # Timeliness signal
                timeliness_signal = self._compute_event_timeliness(
                    document, related_events
                )
                if timeliness_signal:
                    result.add_signal(timeliness_signal)

            # Detect event type in document
            event_type_signal = self._detect_document_event_type(document)
            if event_type_signal:
                result.add_signal(event_type_signal)

            # Query event alignment
            if query and query.event_relevance:
                alignment_signal = self._compute_query_event_alignment(
                    document, query
                )
                if alignment_signal:
                    result.add_signal(alignment_signal)

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in event analysis: {e}")
            result.error = str(e)

        return result

    def analyze_query(self, query: Query) -> ContextResult:
        """
        Analyze a query for event-related intent.

        Detects:
        - Event references
        - Temporal event context
        - Event type preferences
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            query_lower = query.text.lower()

            # Find events mentioned in query
            query_events = self._find_events_in_text(query_lower)

            if query_events:
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="query_event_reference",
                    score=0.7,
                    confidence=0.8,
                    metadata={
                        "events": [e.id for e in query_events],
                        "event_types": [e.event_type.name for e in query_events],
                    },
                    explanation=f"Query references {len(query_events)} known event(s)"
                ))

            # Check for event type keywords in query
            for event_type, keywords in self.EVENT_TYPE_KEYWORDS.items():
                if any(kw in query_lower for kw in keywords):
                    result.add_signal(ContextSignal(
                        context_type=self.context_type,
                        name="event_type_interest",
                        score=0.5,
                        confidence=0.65,
                        metadata={"event_type": event_type.name},
                        explanation=f"Query shows interest in {event_type.name} events"
                    ))
                    break

            # Check for recency indicators (event-related)
            recency_keywords = ["latest", "recent", "new", "just", "today", "breaking"]
            if any(kw in query_lower for kw in recency_keywords):
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="event_recency_preference",
                    score=0.6,
                    confidence=0.7,
                    explanation="Query prefers recent events/developments"
                ))

            # Event relevance flag
            if query.event_relevance:
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="event_relevance_enabled",
                    score=0.3,
                    confidence=1.0,
                    metadata={"enabled": True},
                    explanation="Event context consideration is enabled"
                ))

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in query event analysis: {e}")
            result.error = str(e)

        return result

    def _find_related_events(self, document: Document) -> List[Tuple[Event, float]]:
        """Find events related to a document with relevance scores."""
        related = []
        content_lower = document.content.lower()
        doc_date = document.publication_date or datetime.now()

        # Check keyword matches
        for keyword, event_ids in self._keyword_to_events.items():
            if keyword.lower() in content_lower:
                for event_id in event_ids:
                    event = self._events.get(event_id)
                    if event:
                        relevance = event.relevance_at(doc_date)
                        if relevance >= self.min_event_relevance:
                            related.append((event, relevance))

        # Check entity matches
        for entity, event_ids in self._entity_to_events.items():
            if entity.lower() in content_lower:
                for event_id in event_ids:
                    event = self._events.get(event_id)
                    if event and (event, event.relevance_at(doc_date)) not in related:
                        relevance = event.relevance_at(doc_date) * 1.2  # Entity bonus
                        if relevance >= self.min_event_relevance:
                            related.append((event, relevance))

        # Check domain matches
        if document.domain:
            for event in self._events.values():
                if document.domain in event.domains:
                    relevance = event.relevance_at(doc_date)
                    if relevance >= self.min_event_relevance:
                        if (event, relevance) not in related:
                            related.append((event, relevance * 0.8))

        # Sort by relevance and limit
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:self.max_events_per_query]

    def _find_events_in_text(self, text: str) -> List[Event]:
        """Find specific events mentioned in text."""
        found = []

        for event in self._events.values():
            # Check title match
            if event.title.lower() in text:
                found.append(event)
                continue

            # Check keyword match (require multiple)
            keyword_matches = sum(
                1 for kw in event.keywords
                if kw.lower() in text
            )
            if keyword_matches >= 2:
                found.append(event)
                continue

            # Check entity match
            for entity in event.entities:
                if entity.lower() in text:
                    found.append(event)
                    break

        return found

    def _compute_event_relevance(
        self,
        document: Document,
        related_events: List[Tuple[Event, float]]
    ) -> ContextSignal:
        """Compute overall event relevance signal."""
        if not related_events:
            return ContextSignal(
                context_type=self.context_type,
                name="event_relevance",
                score=0.0,
                confidence=0.5,
                explanation="No related events found"
            )

        # Aggregate relevance scores
        total_relevance = sum(rel for _, rel in related_events)
        avg_relevance = total_relevance / len(related_events)

        # Get event types
        event_types = list(set(e.event_type.name for e, _ in related_events))

        # Calculate score (higher relevance = higher score)
        score = min(1.0, avg_relevance * self.event_boost_multiplier)

        return ContextSignal(
            context_type=self.context_type,
            name="event_relevance",
            score=score,
            confidence=min(0.9, 0.5 + len(related_events) * 0.1),
            metadata={
                "related_event_count": len(related_events),
                "event_types": event_types,
                "total_relevance": total_relevance,
                "top_events": [
                    {"id": e.id, "title": e.title, "relevance": rel}
                    for e, rel in related_events[:3]
                ],
            },
            explanation=f"Document relates to {len(related_events)} event(s) with avg relevance {avg_relevance:.2f}"
        )

    def _compute_event_impact(
        self,
        document: Document,
        related_events: List[Tuple[Event, float]]
    ) -> Optional[ContextSignal]:
        """Compute the impact of related events on document relevance."""
        if not related_events:
            return None

        # Find highest impact event
        max_impact_event = max(related_events, key=lambda x: x[0].impact.value * x[1])
        event, relevance = max_impact_event

        impact_score = event.impact.value * relevance

        # Determine if this is a high-impact event
        is_high_impact = event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]

        return ContextSignal(
            context_type=self.context_type,
            name="event_impact",
            score=impact_score,
            confidence=0.7,
            metadata={
                "highest_impact_event": event.id,
                "event_title": event.title,
                "impact_level": event.impact.name,
                "is_high_impact": is_high_impact,
            },
            explanation=f"Highest impact event: '{event.title}' ({event.impact.name})"
        )

    def _compute_event_timeliness(
        self,
        document: Document,
        related_events: List[Tuple[Event, float]]
    ) -> Optional[ContextSignal]:
        """Compute how timely the document is relative to events."""
        if not related_events:
            return None

        doc_date = document.publication_date or datetime.now()

        timeliness_scores = []
        for event, _ in related_events:
            # How close is the document to the event start?
            days_diff = abs((doc_date - event.start_date).total_seconds() / 86400)

            # Documents close to event start are more timely
            if days_diff <= 7:
                timeliness = 1.0
            elif days_diff <= 30:
                timeliness = 0.7
            elif days_diff <= 90:
                timeliness = 0.4
            else:
                timeliness = 0.1

            timeliness_scores.append(timeliness)

        avg_timeliness = np.mean(timeliness_scores)

        return ContextSignal(
            context_type=self.context_type,
            name="event_timeliness",
            score=avg_timeliness * 2 - 1,  # Convert to [-1, 1]
            confidence=0.65,
            metadata={
                "avg_timeliness": avg_timeliness,
                "event_count": len(related_events),
            },
            explanation=f"Document timeliness relative to events: {avg_timeliness:.2f}"
        )

    def _detect_document_event_type(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Detect if document describes an event and its type."""
        content_lower = document.content[:3000].lower()  # Check first 3000 chars

        detected_types = []
        for event_type, keywords in self.EVENT_TYPE_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count >= 2:  # Require at least 2 keyword matches
                detected_types.append((event_type, match_count))

        if not detected_types:
            return None

        # Get the most likely event type
        primary_type = max(detected_types, key=lambda x: x[1])

        # Determine sentiment impact of event type
        positive_types = [
            EventType.ANNOUNCEMENT, EventType.RESEARCH_BREAKTHROUGH,
            EventType.RELEASE, EventType.MILESTONE,
        ]
        negative_types = [EventType.INCIDENT, EventType.CONTROVERSY]

        if primary_type[0] in positive_types:
            score = 0.3
        elif primary_type[0] in negative_types:
            score = -0.2
        else:
            score = 0.0

        return ContextSignal(
            context_type=self.context_type,
            name="document_event_type",
            score=score,
            confidence=0.6,
            metadata={
                "detected_type": primary_type[0].name,
                "keyword_matches": primary_type[1],
                "all_detected": [t.name for t, _ in detected_types],
            },
            explanation=f"Document appears to describe a {primary_type[0].name} event"
        )

    def _compute_query_event_alignment(
        self,
        document: Document,
        query: Query
    ) -> Optional[ContextSignal]:
        """Check alignment between document and query event context."""
        # Find events in query
        query_events = self._find_events_in_text(query.text.lower())

        if not query_events:
            return None

        # Find events in document
        doc_related = self._find_related_events(document)
        doc_event_ids = set(e.id for e, _ in doc_related)

        # Check overlap
        query_event_ids = set(e.id for e in query_events)
        overlap = query_event_ids.intersection(doc_event_ids)

        if overlap:
            score = 0.8
            explanation = f"Document shares {len(overlap)} event(s) with query"
        else:
            # Check for related domains
            query_domains = set()
            for e in query_events:
                query_domains.update(e.domains)

            doc_domains = set()
            for e, _ in doc_related:
                doc_domains.update(e.domains)

            domain_overlap = query_domains.intersection(doc_domains)
            if domain_overlap:
                score = 0.3
                explanation = f"Document events share domains with query events"
            else:
                score = 0.0
                explanation = "No event overlap between query and document"

        return ContextSignal(
            context_type=self.context_type,
            name="query_event_alignment",
            score=score,
            confidence=0.75,
            metadata={
                "query_events": [e.id for e in query_events],
                "doc_events": list(doc_event_ids),
                "overlap": list(overlap) if overlap else [],
            },
            explanation=explanation
        )

    def register_event(self, event: Event) -> None:
        """Register a new event."""
        self._events[event.id] = event

        # Index by keywords
        for keyword in event.keywords:
            self._keyword_to_events[keyword.lower()].append(event.id)

        # Index by entities
        for entity in event.entities:
            self._entity_to_events[entity.lower()].append(event.id)

        # Add to domain streams
        for domain in event.domains:
            if domain not in self._event_streams:
                self._event_streams[domain] = EventStream(name=domain)
            self._event_streams[domain].events.append(event)

        self.logger.info(f"Registered event: {event.id} - {event.title}")

    def get_active_events(
        self,
        domain: Optional[str] = None
    ) -> List[Event]:
        """Get currently active events."""
        active = [e for e in self._events.values() if e.is_active]

        if domain:
            active = [e for e in active if domain in e.domains]

        return sorted(active, key=lambda e: e.impact.value, reverse=True)

    def get_recent_events(
        self,
        days: int = 7,
        domain: Optional[str] = None
    ) -> List[Event]:
        """Get events from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)

        recent = [
            e for e in self._events.values()
            if e.start_date >= cutoff
        ]

        if domain:
            recent = [e for e in recent if domain in e.domains]

        return sorted(recent, key=lambda e: e.start_date, reverse=True)

    def create_event_from_document(
        self,
        document: Document,
        event_type: Optional[EventType] = None
    ) -> Optional[Event]:
        """Create an event from a document (e.g., news article)."""
        # Auto-detect event type if not provided
        if event_type is None:
            signal = self._detect_document_event_type(document)
            if signal and signal.metadata.get("detected_type"):
                event_type = EventType[signal.metadata["detected_type"]]
            else:
                event_type = EventType.ANNOUNCEMENT

        # Extract keywords from document
        content_lower = document.content.lower()
        words = re.findall(r'\b[a-z]{4,}\b', content_lower)
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        keywords = [
            word for word, count in sorted(
                word_freq.items(), key=lambda x: x[1], reverse=True
            )
            if count >= 3
        ][:10]

        event = Event(
            id=f"doc_{document.id}",
            title=document.title or f"Event from {document.id}",
            event_type=event_type,
            description=document.content[:500],
            impact=EventImpact.MODERATE,
            start_date=document.publication_date or datetime.now(),
            domains=[document.domain] if document.domain else [],
            keywords=keywords,
            entities=[document.author] if document.author else [],
            source=document.source,
            url=document.url,
        )

        return event
