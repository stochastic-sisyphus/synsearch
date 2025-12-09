"""
Configuration module for the CAG Framework.

Provides configuration management, validation, and default settings
for the Context Augmented Generation system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import yaml
import json
from pathlib import Path
import logging


class CAGConfigError(Exception):
    """Configuration error for CAG framework."""
    pass


@dataclass
class ProviderConfig:
    """Configuration for a single context provider."""
    enabled: bool = True
    weight: float = 1.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalConfig(ProviderConfig):
    """Configuration for temporal context provider."""
    decay_rate: float = 0.1
    half_life_days: int = 180
    trend_window_days: int = 30
    min_data_points: int = 3
    recency_weight: float = 0.5


@dataclass
class AuthorityConfig(ProviderConfig):
    """Configuration for authority context provider."""
    domain: str = "general"
    min_citations: int = 10
    author_weight: float = 0.4
    source_weight: float = 0.4
    institution_weight: float = 0.2
    require_peer_review: bool = True


@dataclass
class SentimentConfig(ProviderConfig):
    """Configuration for sentiment context provider."""
    default_sentiment: float = 0.0
    controversy_threshold: float = 0.3
    sentiment_decay_days: int = 30
    prefer_positive: bool = False
    prefer_neutral: bool = False


@dataclass
class EventConfig(ProviderConfig):
    """Configuration for event context provider."""
    event_decay_days: int = 30
    min_event_relevance: float = 0.1
    max_events_per_query: int = 10
    event_boost_multiplier: float = 1.5
    track_domains: List[str] = field(default_factory=list)


@dataclass
class AggregationConfig:
    """Configuration for context aggregation."""
    strategy: str = "WEIGHTED_AVERAGE"  # WEIGHTED_AVERAGE, WEIGHTED_GEOMETRIC, MAX_SCORE, MIN_SCORE, BAYESIAN
    semantic_weight: float = 0.6
    context_weight: float = 0.4
    temporal_weight: float = 1.0
    authority_weight: float = 1.0
    sentiment_weight: float = 1.0
    event_weight: float = 1.0


@dataclass
class EngineConfig:
    """Configuration for CAG search engine."""
    parallel_analysis: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    cache_max_size: int = 10000
    min_score_threshold: float = 0.0
    explain_results: bool = False


@dataclass
class CAGConfig:
    """
    Complete CAG framework configuration.

    Manages all settings for the Context Augmented Generation system.
    """
    # Provider configurations
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    authority: AuthorityConfig = field(default_factory=AuthorityConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    events: EventConfig = field(default_factory=EventConfig)

    # Aggregation configuration
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    # Engine configuration
    engine: EngineConfig = field(default_factory=EngineConfig)

    # Metadata
    version: str = "1.0.0"
    name: str = "default"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAGConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Update temporal config
        if "temporal" in data:
            for key, value in data["temporal"].items():
                if hasattr(config.temporal, key):
                    setattr(config.temporal, key, value)

        # Update authority config
        if "authority" in data:
            for key, value in data["authority"].items():
                if hasattr(config.authority, key):
                    setattr(config.authority, key, value)

        # Update sentiment config
        if "sentiment" in data:
            for key, value in data["sentiment"].items():
                if hasattr(config.sentiment, key):
                    setattr(config.sentiment, key, value)

        # Update events config
        if "events" in data:
            for key, value in data["events"].items():
                if hasattr(config.events, key):
                    setattr(config.events, key, value)

        # Update aggregation config
        if "aggregation" in data:
            for key, value in data["aggregation"].items():
                if hasattr(config.aggregation, key):
                    setattr(config.aggregation, key, value)

        # Update engine config
        if "engine" in data:
            for key, value in data["engine"].items():
                if hasattr(config.engine, key):
                    setattr(config.engine, key, value)

        # Update metadata
        if "version" in data:
            config.version = data["version"]
        if "name" in data:
            config.name = data["name"]

        return config

    @classmethod
    def from_yaml(cls, path: str) -> "CAGConfig":
        """Load configuration from YAML file."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Extract CAG section if it exists
            if "cag" in data:
                data = data["cag"]

            return cls.from_dict(data)
        except Exception as e:
            raise CAGConfigError(f"Failed to load config from {path}: {e}")

    @classmethod
    def from_json(cls, path: str) -> "CAGConfig":
        """Load configuration from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "cag" in data:
                data = data["cag"]

            return cls.from_dict(data)
        except Exception as e:
            raise CAGConfigError(f"Failed to load config from {path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "temporal": {
                "enabled": self.temporal.enabled,
                "weight": self.temporal.weight,
                "cache_enabled": self.temporal.cache_enabled,
                "decay_rate": self.temporal.decay_rate,
                "half_life_days": self.temporal.half_life_days,
                "trend_window_days": self.temporal.trend_window_days,
                "min_data_points": self.temporal.min_data_points,
                "recency_weight": self.temporal.recency_weight,
            },
            "authority": {
                "enabled": self.authority.enabled,
                "weight": self.authority.weight,
                "cache_enabled": self.authority.cache_enabled,
                "domain": self.authority.domain,
                "min_citations": self.authority.min_citations,
                "author_weight": self.authority.author_weight,
                "source_weight": self.authority.source_weight,
                "institution_weight": self.authority.institution_weight,
                "require_peer_review": self.authority.require_peer_review,
            },
            "sentiment": {
                "enabled": self.sentiment.enabled,
                "weight": self.sentiment.weight,
                "cache_enabled": self.sentiment.cache_enabled,
                "default_sentiment": self.sentiment.default_sentiment,
                "controversy_threshold": self.sentiment.controversy_threshold,
                "sentiment_decay_days": self.sentiment.sentiment_decay_days,
                "prefer_positive": self.sentiment.prefer_positive,
                "prefer_neutral": self.sentiment.prefer_neutral,
            },
            "events": {
                "enabled": self.events.enabled,
                "weight": self.events.weight,
                "cache_enabled": self.events.cache_enabled,
                "event_decay_days": self.events.event_decay_days,
                "min_event_relevance": self.events.min_event_relevance,
                "max_events_per_query": self.events.max_events_per_query,
                "event_boost_multiplier": self.events.event_boost_multiplier,
                "track_domains": self.events.track_domains,
            },
            "aggregation": {
                "strategy": self.aggregation.strategy,
                "semantic_weight": self.aggregation.semantic_weight,
                "context_weight": self.aggregation.context_weight,
                "temporal_weight": self.aggregation.temporal_weight,
                "authority_weight": self.aggregation.authority_weight,
                "sentiment_weight": self.aggregation.sentiment_weight,
                "event_weight": self.aggregation.event_weight,
            },
            "engine": {
                "parallel_analysis": self.engine.parallel_analysis,
                "max_workers": self.engine.max_workers,
                "cache_enabled": self.engine.cache_enabled,
                "cache_max_size": self.engine.cache_max_size,
                "min_score_threshold": self.engine.min_score_threshold,
                "explain_results": self.engine.explain_results,
            },
            "version": self.version,
            "name": self.name,
        }

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump({"cag": self.to_dict()}, f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump({"cag": self.to_dict()}, f, indent=2)

    def to_engine_config(self) -> Dict[str, Any]:
        """Convert to format expected by CAGEnhancedSearchEngine."""
        return {
            "semantic_weight": self.aggregation.semantic_weight,
            "context_weight": self.aggregation.context_weight,
            "parallel_analysis": self.engine.parallel_analysis,
            "max_workers": self.engine.max_workers,
            "cache_enabled": self.engine.cache_enabled,
            "aggregation_strategy": self.aggregation.strategy,
            "temporal_weight": self.aggregation.temporal_weight,
            "authority_weight": self.aggregation.authority_weight,
            "sentiment_weight": self.aggregation.sentiment_weight,
            "event_weight": self.aggregation.event_weight,
            "providers": {
                "temporal": {
                    "enabled": self.temporal.enabled,
                    "decay_rate": self.temporal.decay_rate,
                    "half_life_days": self.temporal.half_life_days,
                    "trend_window_days": self.temporal.trend_window_days,
                    "min_data_points": self.temporal.min_data_points,
                    "recency_weight": self.temporal.recency_weight,
                },
                "authority": {
                    "enabled": self.authority.enabled,
                    "domain": self.authority.domain,
                    "min_citations": self.authority.min_citations,
                    "author_weight": self.authority.author_weight,
                    "source_weight": self.authority.source_weight,
                    "institution_weight": self.authority.institution_weight,
                    "require_peer_review": self.authority.require_peer_review,
                },
                "sentiment": {
                    "enabled": self.sentiment.enabled,
                    "default_sentiment": self.sentiment.default_sentiment,
                    "controversy_threshold": self.sentiment.controversy_threshold,
                    "sentiment_decay_days": self.sentiment.sentiment_decay_days,
                    "prefer_positive": self.sentiment.prefer_positive,
                    "prefer_neutral": self.sentiment.prefer_neutral,
                },
                "events": {
                    "enabled": self.events.enabled,
                    "event_decay_days": self.events.event_decay_days,
                    "min_event_relevance": self.events.min_event_relevance,
                    "max_events_per_query": self.events.max_events_per_query,
                    "event_boost_multiplier": self.events.event_boost_multiplier,
                    "track_domains": self.events.track_domains,
                },
            },
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate weights sum to reasonable value
        total_weight = (
            self.aggregation.semantic_weight +
            self.aggregation.context_weight
        )
        if not 0.9 <= total_weight <= 1.1:
            issues.append(
                f"Semantic + context weights should sum to ~1.0, got {total_weight}"
            )

        # Validate individual weights
        if not 0.0 <= self.aggregation.semantic_weight <= 1.0:
            issues.append("Semantic weight must be between 0.0 and 1.0")

        if not 0.0 <= self.aggregation.context_weight <= 1.0:
            issues.append("Context weight must be between 0.0 and 1.0")

        # Validate provider weights
        for name, weight in [
            ("temporal", self.aggregation.temporal_weight),
            ("authority", self.aggregation.authority_weight),
            ("sentiment", self.aggregation.sentiment_weight),
            ("event", self.aggregation.event_weight),
        ]:
            if not 0.0 <= weight <= 2.0:
                issues.append(f"{name} weight must be between 0.0 and 2.0")

        # Validate decay rates
        if not 0.0 < self.temporal.half_life_days <= 365:
            issues.append("Temporal half_life_days should be between 1 and 365")

        # Validate thresholds
        if not 0.0 <= self.sentiment.controversy_threshold <= 1.0:
            issues.append("Controversy threshold must be between 0.0 and 1.0")

        # Validate engine settings
        if self.engine.max_workers < 1:
            issues.append("max_workers must be at least 1")

        return issues

    def __str__(self) -> str:
        """String representation of config."""
        return (
            f"CAGConfig(name='{self.name}', version='{self.version}', "
            f"providers=[temporal={self.temporal.enabled}, "
            f"authority={self.authority.enabled}, "
            f"sentiment={self.sentiment.enabled}, "
            f"events={self.events.enabled}])"
        )


# Pre-defined configuration profiles
class CAGProfiles:
    """Pre-defined CAG configuration profiles for common use cases."""

    @staticmethod
    def research_papers() -> CAGConfig:
        """Configuration optimized for academic research papers."""
        config = CAGConfig(name="research_papers")
        config.authority.weight = 1.5  # Emphasize authority
        config.authority.require_peer_review = True
        config.temporal.half_life_days = 365  # Slower decay for research
        config.sentiment.enabled = False  # Less relevant for research
        config.aggregation.authority_weight = 1.5
        return config

    @staticmethod
    def news_articles() -> CAGConfig:
        """Configuration optimized for news and current events."""
        config = CAGConfig(name="news_articles")
        config.temporal.weight = 1.5  # Emphasize recency
        config.temporal.half_life_days = 7  # Fast decay
        config.events.weight = 1.5  # Event-driven
        config.events.event_decay_days = 14
        config.aggregation.temporal_weight = 1.5
        config.aggregation.event_weight = 1.5
        return config

    @staticmethod
    def technical_docs() -> CAGConfig:
        """Configuration optimized for technical documentation."""
        config = CAGConfig(name="technical_docs")
        config.authority.weight = 1.2
        config.temporal.weight = 0.8  # Some decay but slower
        config.temporal.half_life_days = 180
        config.sentiment.enabled = False
        config.events.enabled = False
        return config

    @staticmethod
    def social_media() -> CAGConfig:
        """Configuration optimized for social media content."""
        config = CAGConfig(name="social_media")
        config.temporal.weight = 2.0  # Very time-sensitive
        config.temporal.half_life_days = 1
        config.sentiment.weight = 1.5  # Sentiment matters
        config.sentiment.prefer_neutral = False
        config.authority.weight = 0.5  # Less emphasis on authority
        config.events.weight = 1.5
        config.aggregation.temporal_weight = 2.0
        config.aggregation.sentiment_weight = 1.5
        config.aggregation.authority_weight = 0.5
        return config

    @staticmethod
    def balanced() -> CAGConfig:
        """Balanced configuration for general use."""
        return CAGConfig(name="balanced")


def load_cag_config(
    path: Optional[str] = None,
    profile: Optional[str] = None
) -> CAGConfig:
    """
    Load CAG configuration from file or use a profile.

    Args:
        path: Path to configuration file (YAML or JSON)
        profile: Name of pre-defined profile to use

    Returns:
        CAGConfig instance
    """
    logger = logging.getLogger(__name__)

    if profile:
        profiles = {
            "research_papers": CAGProfiles.research_papers,
            "news_articles": CAGProfiles.news_articles,
            "technical_docs": CAGProfiles.technical_docs,
            "social_media": CAGProfiles.social_media,
            "balanced": CAGProfiles.balanced,
        }

        if profile not in profiles:
            raise CAGConfigError(
                f"Unknown profile: {profile}. "
                f"Available: {list(profiles.keys())}"
            )

        logger.info(f"Using CAG profile: {profile}")
        return profiles[profile]()

    if path:
        path_obj = Path(path)
        if not path_obj.exists():
            raise CAGConfigError(f"Config file not found: {path}")

        if path_obj.suffix in [".yaml", ".yml"]:
            return CAGConfig.from_yaml(path)
        elif path_obj.suffix == ".json":
            return CAGConfig.from_json(path)
        else:
            raise CAGConfigError(
                f"Unsupported config format: {path_obj.suffix}"
            )

    logger.info("Using default CAG configuration")
    return CAGConfig()
