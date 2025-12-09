"""
Authority Context Provider for the CAG Framework.

Assesses source credibility, author expertise, and domain authority
to enhance search relevance with trust signals.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
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
class AuthorProfile:
    """Profile tracking an author's authority metrics."""
    name: str
    domains: Dict[str, float] = field(default_factory=dict)  # domain -> expertise score
    publication_count: int = 0
    citation_count: int = 0
    h_index: float = 0.0
    affiliations: List[str] = field(default_factory=list)
    known_works: List[str] = field(default_factory=list)
    collaboration_network: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SourceProfile:
    """Profile tracking a publication source's authority."""
    name: str
    source_type: str  # "journal", "conference", "preprint", "blog", "news"
    domain: str
    impact_factor: float = 0.0
    acceptance_rate: Optional[float] = None
    peer_reviewed: bool = True
    reputation_score: float = 0.5
    established_year: Optional[int] = None
    known_quality_issues: List[str] = field(default_factory=list)


@dataclass
class InstitutionProfile:
    """Profile for institutional authority."""
    name: str
    institution_type: str  # "university", "research_lab", "company", "government"
    global_ranking: Optional[int] = None
    domain_rankings: Dict[str, int] = field(default_factory=dict)
    reputation_score: float = 0.5
    notable_researchers: List[str] = field(default_factory=list)


class AuthorityContextProvider(ContextProvider):
    """
    Provides authority and credibility context for search.

    Features:
    - Author expertise assessment
    - Source/venue credibility scoring
    - Institutional authority
    - Citation impact analysis
    - Domain-specific authority calibration
    """

    # Well-known high-authority sources by domain
    HIGH_AUTHORITY_SOURCES = {
        "cs": [
            "nature", "science", "neurips", "icml", "iclr", "cvpr", "acl",
            "emnlp", "naacl", "aaai", "ijcai", "jmlr", "tacl", "ieee",
        ],
        "medical": [
            "lancet", "nejm", "jama", "bmj", "nature medicine", "cell",
        ],
        "physics": [
            "physical review", "nature physics", "science", "arxiv",
        ],
        "general": [
            "nature", "science", "pnas", "cell", "ieee", "acm",
        ],
    }

    # Source type authority multipliers
    SOURCE_TYPE_AUTHORITY = {
        "journal": 1.0,
        "conference": 0.9,
        "preprint": 0.6,
        "thesis": 0.7,
        "book": 0.85,
        "blog": 0.3,
        "news": 0.4,
        "social_media": 0.2,
        "unknown": 0.5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize authority context provider.

        Config options:
            domain: str - Primary domain for authority assessment
            min_citations: int - Minimum citations for high authority (default: 10)
            author_weight: float - Weight for author authority (default: 0.4)
            source_weight: float - Weight for source authority (default: 0.4)
            institution_weight: float - Weight for institutional authority (default: 0.2)
            require_peer_review: bool - Prefer peer-reviewed sources (default: True)
        """
        super().__init__(config)

        self.domain = self.config.get("domain", "general")
        self.min_citations = self.config.get("min_citations", 10)
        self.author_weight = self.config.get("author_weight", 0.4)
        self.source_weight = self.config.get("source_weight", 0.4)
        self.institution_weight = self.config.get("institution_weight", 0.2)
        self.require_peer_review = self.config.get("require_peer_review", True)

        # Authority profiles storage
        self._author_profiles: Dict[str, AuthorProfile] = {}
        self._source_profiles: Dict[str, SourceProfile] = {}
        self._institution_profiles: Dict[str, InstitutionProfile] = {}

        # Citation graph
        self._citation_counts: Dict[str, int] = {}

    @property
    def context_type(self) -> ContextType:
        return ContextType.AUTHORITY

    def analyze_document(
        self,
        document: Document,
        query: Optional[Query] = None
    ) -> ContextResult:
        """
        Analyze a document for authority signals.

        Produces signals for:
        - Author expertise
        - Source credibility
        - Institutional authority
        - Citation impact
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            # Determine target domain
            target_domain = document.domain or self.domain

            # Author authority
            author_signal = self._compute_author_authority(document, target_domain)
            if author_signal:
                result.add_signal(author_signal)

            # Source authority
            source_signal = self._compute_source_authority(document, target_domain)
            if source_signal:
                result.add_signal(source_signal)

            # Institutional authority
            institution_signal = self._compute_institution_authority(document)
            if institution_signal:
                result.add_signal(institution_signal)

            # Citation impact
            citation_signal = self._compute_citation_impact(document)
            if citation_signal:
                result.add_signal(citation_signal)

            # Peer review status
            peer_review_signal = self._compute_peer_review_signal(document)
            if peer_review_signal:
                result.add_signal(peer_review_signal)

            # Query-specific authority requirements
            if query and query.authority_threshold:
                threshold_signal = self._check_authority_threshold(
                    result, query.authority_threshold
                )
                if threshold_signal:
                    result.add_signal(threshold_signal)

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in authority analysis: {e}")
            result.error = str(e)

        return result

    def analyze_query(self, query: Query) -> ContextResult:
        """
        Analyze a query for authority preferences.

        Detects:
        - Authority threshold requirements
        - Domain expertise preferences
        - Source type preferences
        """
        import time
        start_time = time.time()

        result = ContextResult(context_type=self.context_type)

        try:
            query_lower = query.text.lower()

            # Check for authority indicators
            authority_keywords = {
                "expert": 0.8,
                "authoritative": 0.9,
                "peer-reviewed": 0.85,
                "published": 0.7,
                "cited": 0.75,
                "reputable": 0.8,
                "academic": 0.7,
                "scientific": 0.75,
                "verified": 0.8,
                "trusted": 0.85,
            }

            for keyword, strength in authority_keywords.items():
                if keyword in query_lower:
                    result.add_signal(ContextSignal(
                        context_type=self.context_type,
                        name="authority_preference",
                        score=strength,
                        confidence=0.7,
                        metadata={"keyword": keyword},
                        explanation=f"Query indicates preference for {keyword} sources"
                    ))
                    break

            # Explicit authority threshold
            if query.authority_threshold:
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="explicit_authority_threshold",
                    score=query.authority_threshold,
                    confidence=0.95,
                    metadata={"threshold": query.authority_threshold},
                    explanation=f"Query requires minimum authority of {query.authority_threshold}"
                ))

            # Domain-specific expertise
            if query.domains:
                result.add_signal(ContextSignal(
                    context_type=self.context_type,
                    name="domain_expertise_required",
                    score=0.5,
                    confidence=0.8,
                    metadata={"domains": query.domains},
                    explanation=f"Query requires expertise in: {', '.join(query.domains)}"
                ))

            result.processing_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error in query authority analysis: {e}")
            result.error = str(e)

        return result

    def _compute_author_authority(
        self,
        document: Document,
        target_domain: str
    ) -> Optional[ContextSignal]:
        """Compute authority signal based on author expertise."""
        author = document.author
        if not author:
            return None

        # Check for known author profile
        profile = self._author_profiles.get(author)

        if profile:
            # Domain-specific expertise
            domain_expertise = profile.domains.get(target_domain, 0.0)

            # Overall authority metrics
            h_index_score = min(1.0, profile.h_index / 50)  # Normalize h-index
            citation_score = min(1.0, profile.citation_count / 1000)
            pub_score = min(1.0, profile.publication_count / 100)

            authority_score = (
                domain_expertise * 0.4 +
                h_index_score * 0.3 +
                citation_score * 0.2 +
                pub_score * 0.1
            )

            return ContextSignal(
                context_type=self.context_type,
                name="author_authority",
                score=authority_score * 2 - 1,  # Convert to [-1, 1]
                confidence=0.8,
                metadata={
                    "author": author,
                    "h_index": profile.h_index,
                    "citations": profile.citation_count,
                    "domain_expertise": domain_expertise,
                    "affiliations": profile.affiliations,
                },
                explanation=f"Author {author} has h-index {profile.h_index:.1f} with {profile.citation_count} citations"
            )
        else:
            # Infer authority from document metadata
            inferred_authority = self._infer_author_authority(document)
            return ContextSignal(
                context_type=self.context_type,
                name="author_authority_inferred",
                score=inferred_authority * 2 - 1,
                confidence=0.4,
                metadata={"author": author, "inferred": True},
                explanation=f"Inferred author authority for {author}"
            )

    def _infer_author_authority(self, document: Document) -> float:
        """Infer author authority from document signals."""
        authority = 0.5  # Default neutral

        # Check for academic titles
        if document.author:
            author_lower = document.author.lower()
            if any(title in author_lower for title in ["dr.", "prof.", "phd"]):
                authority += 0.15
            if any(title in author_lower for title in ["professor", "researcher"]):
                authority += 0.1

        # Check metadata for affiliation clues
        if document.metadata:
            affiliation = str(document.metadata.get("affiliation", "")).lower()
            if any(inst in affiliation for inst in ["university", "institute", "lab"]):
                authority += 0.1
            if any(inst in affiliation for inst in ["mit", "stanford", "berkeley", "oxford", "cambridge"]):
                authority += 0.15

        return min(1.0, authority)

    def _compute_source_authority(
        self,
        document: Document,
        target_domain: str
    ) -> Optional[ContextSignal]:
        """Compute authority signal based on publication source."""
        source = document.source
        if not source:
            # Try to extract from URL or metadata
            source = self._extract_source(document)

        if not source:
            return None

        source_lower = source.lower()

        # Check for known source profile
        profile = self._source_profiles.get(source_lower)

        if profile:
            # Use stored reputation
            authority_score = profile.reputation_score

            # Apply source type multiplier
            type_multiplier = self.SOURCE_TYPE_AUTHORITY.get(
                profile.source_type, 0.5
            )
            authority_score *= type_multiplier

            # Consider impact factor for journals
            if profile.impact_factor > 0:
                if_score = min(1.0, profile.impact_factor / 50)
                authority_score = authority_score * 0.7 + if_score * 0.3

            return ContextSignal(
                context_type=self.context_type,
                name="source_authority",
                score=authority_score * 2 - 1,
                confidence=0.85,
                metadata={
                    "source": source,
                    "source_type": profile.source_type,
                    "impact_factor": profile.impact_factor,
                    "peer_reviewed": profile.peer_reviewed,
                },
                explanation=f"Source '{source}' ({profile.source_type}) has reputation {profile.reputation_score:.2f}"
            )
        else:
            # Check against known high-authority sources
            high_auth_sources = self.HIGH_AUTHORITY_SOURCES.get(
                target_domain,
                self.HIGH_AUTHORITY_SOURCES["general"]
            )

            if any(auth_source in source_lower for auth_source in high_auth_sources):
                return ContextSignal(
                    context_type=self.context_type,
                    name="source_authority",
                    score=0.7,
                    confidence=0.7,
                    metadata={"source": source, "recognized": True},
                    explanation=f"Source '{source}' is a recognized authority in {target_domain}"
                )

            # Infer source type and authority
            source_type, inferred_auth = self._infer_source_authority(document)
            return ContextSignal(
                context_type=self.context_type,
                name="source_authority_inferred",
                score=inferred_auth * 2 - 1,
                confidence=0.5,
                metadata={
                    "source": source,
                    "source_type": source_type,
                    "inferred": True,
                },
                explanation=f"Inferred source authority for '{source}' ({source_type})"
            )

    def _extract_source(self, document: Document) -> Optional[str]:
        """Extract source from document URL or metadata."""
        if document.url:
            # Extract domain from URL
            match = re.search(r'https?://(?:www\.)?([^/]+)', document.url)
            if match:
                return match.group(1)

        # Check metadata
        for key in ["source", "venue", "journal", "conference", "publisher"]:
            if key in document.metadata:
                return str(document.metadata[key])

        return None

    def _infer_source_authority(
        self,
        document: Document
    ) -> Tuple[str, float]:
        """Infer source type and authority from signals."""
        source_type = "unknown"
        authority = 0.5

        if document.url:
            url_lower = document.url.lower()

            # ArXiv (preprint)
            if "arxiv" in url_lower:
                source_type = "preprint"
                authority = 0.6

            # Academic domains
            elif any(domain in url_lower for domain in [".edu", ".ac.uk", ".ac."]):
                source_type = "academic"
                authority = 0.7

            # Known platforms
            elif any(platform in url_lower for platform in ["github", "gitlab"]):
                source_type = "repository"
                authority = 0.5

            elif any(platform in url_lower for platform in ["medium.com", "blog"]):
                source_type = "blog"
                authority = 0.35

            elif any(platform in url_lower for platform in ["twitter", "x.com"]):
                source_type = "social_media"
                authority = 0.25

        # Check content for source type clues
        content_lower = document.content[:2000].lower()

        if any(marker in content_lower for marker in ["abstract", "methodology", "references"]):
            source_type = source_type if source_type != "unknown" else "academic"
            authority = max(authority, 0.65)

        if "peer-reviewed" in content_lower or "peer reviewed" in content_lower:
            authority = max(authority, 0.75)

        return source_type, authority

    def _compute_institution_authority(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Compute authority signal based on institutional affiliation."""
        # Extract institution from metadata or content
        institution = document.metadata.get("institution") or document.metadata.get("affiliation")

        if not institution:
            # Try to extract from content
            institution = self._extract_institution(document)

        if not institution:
            return None

        institution_lower = institution.lower()

        # Check for known institution profile
        profile = self._institution_profiles.get(institution_lower)

        if profile:
            authority = profile.reputation_score

            if profile.global_ranking:
                rank_score = 1.0 - min(1.0, profile.global_ranking / 500)
                authority = authority * 0.6 + rank_score * 0.4

            return ContextSignal(
                context_type=self.context_type,
                name="institution_authority",
                score=authority * 2 - 1,
                confidence=0.75,
                metadata={
                    "institution": institution,
                    "type": profile.institution_type,
                    "ranking": profile.global_ranking,
                },
                explanation=f"Institution '{institution}' has reputation {profile.reputation_score:.2f}"
            )
        else:
            # Infer institution type and authority
            inst_type, authority = self._infer_institution_authority(institution)
            return ContextSignal(
                context_type=self.context_type,
                name="institution_authority_inferred",
                score=authority * 2 - 1,
                confidence=0.4,
                metadata={
                    "institution": institution,
                    "type": inst_type,
                    "inferred": True,
                },
                explanation=f"Inferred institution authority for '{institution}'"
            )

    def _extract_institution(self, document: Document) -> Optional[str]:
        """Extract institution from document content."""
        # Look for common patterns
        patterns = [
            r"(?:University|Institute|Laboratory|Lab)\s+of\s+[\w\s]+",
            r"[\w\s]+\s+(?:University|Institute|College)",
            r"(?:MIT|Stanford|Berkeley|Oxford|Cambridge|Harvard|Google|Microsoft|Meta|OpenAI|DeepMind)",
        ]

        for pattern in patterns:
            match = re.search(pattern, document.content[:2000])
            if match:
                return match.group(0).strip()

        return None

    def _infer_institution_authority(
        self,
        institution: str
    ) -> Tuple[str, float]:
        """Infer institution type and authority."""
        institution_lower = institution.lower()

        # Top-tier research institutions
        top_tier = [
            "mit", "stanford", "berkeley", "harvard", "oxford", "cambridge",
            "caltech", "princeton", "yale", "columbia", "cornell", "cmu",
            "eth zurich", "tsinghua", "peking", "tokyo",
        ]

        tech_companies = [
            "google", "deepmind", "openai", "microsoft", "meta", "apple",
            "amazon", "nvidia", "anthropic", "hugging face",
        ]

        research_labs = [
            "bell labs", "ibm research", "fair", "allen institute",
            "max planck", "fraunhofer",
        ]

        if any(inst in institution_lower for inst in top_tier):
            return "university", 0.9

        if any(inst in institution_lower for inst in tech_companies):
            return "company", 0.85

        if any(inst in institution_lower for inst in research_labs):
            return "research_lab", 0.85

        if "university" in institution_lower or "college" in institution_lower:
            return "university", 0.6

        if "institute" in institution_lower or "laboratory" in institution_lower:
            return "research_lab", 0.65

        return "unknown", 0.5

    def _compute_citation_impact(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Compute authority signal based on citation count."""
        citations = document.metadata.get("citations") or document.metadata.get("citation_count")

        if citations is None:
            # Check document store
            citations = self._citation_counts.get(document.id, 0)

        if citations == 0:
            return None

        # Normalize citation count
        citation_score = min(1.0, np.log1p(citations) / np.log1p(1000))

        high_impact = citations >= self.min_citations

        return ContextSignal(
            context_type=self.context_type,
            name="citation_impact",
            score=citation_score * 2 - 1,
            confidence=0.85 if citations > 0 else 0.3,
            metadata={
                "citations": citations,
                "high_impact": high_impact,
            },
            explanation=f"Document has {citations} citations {'(high impact)' if high_impact else ''}"
        )

    def _compute_peer_review_signal(
        self,
        document: Document
    ) -> Optional[ContextSignal]:
        """Check if document is peer-reviewed."""
        is_peer_reviewed = document.metadata.get("peer_reviewed", False)

        # Infer from source type
        source = document.source or ""
        source_lower = source.lower()

        # Known peer-reviewed venues
        if any(venue in source_lower for venue in ["journal", "conference", "proceedings"]):
            is_peer_reviewed = True

        # Known non-peer-reviewed
        if any(venue in source_lower for venue in ["arxiv", "preprint", "blog", "medium"]):
            is_peer_reviewed = False

        if self.require_peer_review:
            score = 0.5 if is_peer_reviewed else -0.3
        else:
            score = 0.3 if is_peer_reviewed else 0.0

        return ContextSignal(
            context_type=self.context_type,
            name="peer_review_status",
            score=score,
            confidence=0.7,
            metadata={"peer_reviewed": is_peer_reviewed},
            explanation=f"Document is {'peer-reviewed' if is_peer_reviewed else 'not peer-reviewed'}"
        )

    def _check_authority_threshold(
        self,
        result: ContextResult,
        threshold: float
    ) -> Optional[ContextSignal]:
        """Check if document meets authority threshold."""
        current_score = result.aggregate_score

        # Convert from [-1, 1] to [0, 1]
        normalized_score = (current_score + 1) / 2

        meets_threshold = normalized_score >= threshold

        return ContextSignal(
            context_type=self.context_type,
            name="authority_threshold_check",
            score=0.5 if meets_threshold else -0.8,
            confidence=0.9,
            metadata={
                "threshold": threshold,
                "current_score": normalized_score,
                "meets_threshold": meets_threshold,
            },
            explanation=f"Authority {'meets' if meets_threshold else 'below'} threshold ({normalized_score:.2f} vs {threshold})"
        )

    def register_author(self, profile: AuthorProfile) -> None:
        """Register an author profile."""
        self._author_profiles[profile.name] = profile

    def register_source(self, profile: SourceProfile) -> None:
        """Register a source profile."""
        self._source_profiles[profile.name.lower()] = profile

    def register_institution(self, profile: InstitutionProfile) -> None:
        """Register an institution profile."""
        self._institution_profiles[profile.name.lower()] = profile

    def update_citations(self, document_id: str, citations: int) -> None:
        """Update citation count for a document."""
        self._citation_counts[document_id] = citations
