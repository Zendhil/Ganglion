"""
Semantic router for query dispatch to specialist agents.

Description: Loads route definitions from routes.yaml, builds RouteLayer
    with FastEmbed encoder, and provides routing function for queries.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third party imports
import numpy as np
import yaml
from semantic_router import Route
from semantic_router import SemanticRouter as SRRouter
from semantic_router.encoders import FastEmbedEncoder

# App imports
from ganglion.router.cache import AbstractSemanticCache, InMemorySemanticCache, CacheStats

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ROUTES_CONFIG_FILENAME = "routes.yaml"
DEFAULT_ROUTES_PATH = Path(__file__).parent / ROUTES_CONFIG_FILENAME

DEFAULT_HIGH_THRESHOLD = 0.75
DEFAULT_MEDIUM_THRESHOLD = 0.45
DEFAULT_FALLBACK_ROUTE = "head_agent"


# ── Route Result Schema ───────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """
    Task packet returned by the router, passed to head agent.

    :param query: Original user query string.
    :param route: Name of the specialist agent to handle the query.
    :param confidence: Cosine similarity score (0.0-1.0).
    :param cache_hit: Whether result came from semantic cache.
    :param timestamp: ISO format timestamp of routing decision.
    """

    query: str
    route: str
    confidence: float
    cache_hit: bool
    timestamp: str


# ── Configuration Loader ──────────────────────────────────────────────────────

def load_routes_config(
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load route definitions from YAML config file.

    Description: Reads routes.yaml and returns parsed configuration
        containing route definitions and threshold settings.

    :param config_path: Path to config file. Defaults to package's routes.yaml.
    :return: Parsed configuration dictionary.
    :raises FileNotFoundError: If config file does not exist.
    :raises yaml.YAMLError: If config file is invalid YAML.
    """
    logger.info("In function load_routes_config: Entered")

    if config_path is None:
        config_path = DEFAULT_ROUTES_PATH

    if not config_path.exists():
        logger.error(f"In function load_routes_config: Config file not found at {config_path}")
        raise FileNotFoundError(f"Routes config not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"In function load_routes_config: Loaded {len(config.get('routes', []))} routes")
        return config
    except yaml.YAMLError:
        logger.exception("In function load_routes_config: Failed to parse YAML config")
        raise


def _build_routes(config: Dict[str, Any]) -> List[Route]:
    """
    Build Route objects from config dictionary.

    Description: Converts route definitions from YAML into semantic_router
        Route objects for use in RouteLayer.

    :param config: Parsed routes.yaml configuration.
    :return: List of Route objects.
    """
    logger.info("In function _build_routes: Entered")

    routes = []
    for route_def in config.get("routes", []):
        route = Route(
            name=route_def["name"],
            utterances=route_def["utterances"],
        )
        routes.append(route)
        logger.info(
            f"In function _build_routes: Added route '{route_def['name']}' "
            f"with {len(route_def['utterances'])} utterances"
        )

    return routes


# ── Router Class ──────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Query router for dispatching to specialist agents.

    Description: Wraps semantic_router with configuration loading,
        threshold-based routing logic, optional caching, and result formatting.

    :param config_path: Path to routes.yaml. Uses default if not specified.
    :param cache: Optional semantic cache instance. If None, caching is disabled.
    :param enable_cache: If True and cache is None, creates InMemorySemanticCache.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        cache: Optional[AbstractSemanticCache] = None,
        enable_cache: bool = False,
    ):
        logger.info("In class QueryRouter, function __init__: Entered")

        self._config = load_routes_config(config_path)
        self._routes = _build_routes(self._config)
        self._encoder = self._init_encoder()
        self._sr_router = SRRouter(
            encoder=self._encoder,
            routes=self._routes,
            auto_sync="local",
        )

        # Load thresholds from config
        thresholds = self._config.get("thresholds", {})
        self._high_threshold = thresholds.get("high", DEFAULT_HIGH_THRESHOLD)
        self._medium_threshold = thresholds.get("medium", DEFAULT_MEDIUM_THRESHOLD)

        # Load fallback route
        fallback = self._config.get("fallback", {})
        self._fallback_route = fallback.get("route", DEFAULT_FALLBACK_ROUTE)

        # Initialize cache
        if cache is not None:
            self._cache = cache
        elif enable_cache:
            self._cache = InMemorySemanticCache()
        else:
            self._cache = None

        cache_status = "enabled" if self._cache else "disabled"
        logger.info(
            f"In class QueryRouter, function __init__: Initialized with "
            f"{len(self._routes)} routes, thresholds=({self._medium_threshold}, {self._high_threshold}), "
            f"cache={cache_status}"
        )

    def _init_encoder(self) -> FastEmbedEncoder:
        """
        Initialize the FastEmbed encoder.

        Description: Creates encoder instance from config settings.
            Runs locally (~90MB model, no GPU needed).

        :return: Configured FastEmbedEncoder instance.
        """
        logger.info("In class QueryRouter, function _init_encoder: Entered")

        encoder_config = self._config.get("encoder", {})
        model_name = encoder_config.get("model")

        try:
            if model_name:
                encoder = FastEmbedEncoder(model_name=model_name)
            else:
                encoder = FastEmbedEncoder()
            logger.info("In class QueryRouter, function _init_encoder: Encoder initialized")
            return encoder
        except Exception:
            logger.exception("In class QueryRouter, function _init_encoder: Failed to initialize encoder")
            raise

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query string to embedding vector.

        :param query: Query string to encode.
        :return: Embedding as numpy array.
        """
        embeddings = self._encoder([query])
        return np.array(embeddings[0])

    def route(
        self,
        query: str,
    ) -> RouteResult:
        """
        Route a query to the appropriate specialist agent.

        Description: Checks cache first if enabled, then embeds query,
            computes cosine similarity against route utterances, and
            returns routing decision with confidence score.

        Decision logic:
            - confidence >= high_threshold: route directly to specialist
            - confidence >= medium_threshold: route to head agent to decide
            - confidence < medium_threshold: head agent handles with LLM planning

        :param query: User query string to route.
        :return: RouteResult containing routing decision and metadata.
        """
        logger.info(f"In class QueryRouter, function route: Entered with query length={len(query)}")

        cache_hit = False
        embedding = None

        # Check cache first
        if self._cache is not None:
            embedding = self._encode_query(query)
            cache_result = self._cache.get(query, embedding)

            if cache_result is not None:
                route_name, confidence, similarity = cache_result
                cache_hit = True
                timestamp = datetime.now(timezone.utc).isoformat()

                logger.info(
                    f"In class QueryRouter, function route: Cache hit "
                    f"(route={route_name}, similarity={similarity:.3f})"
                )

                return RouteResult(
                    query=query,
                    route=route_name,
                    confidence=confidence,
                    cache_hit=cache_hit,
                    timestamp=timestamp,
                )

        # Cache miss - perform routing
        result = self._sr_router(query)

        # Extract route name and confidence
        route_name = result.name if result.name else self._fallback_route
        confidence = getattr(result, "similarity_score", 0.0) or 0.0

        # Apply confidence thresholds
        if confidence < self._medium_threshold:
            route_name = self._fallback_route
            logger.info(
                f"In class QueryRouter, function route: Low confidence ({confidence:.3f}), "
                f"routing to {self._fallback_route}"
            )
        elif confidence < self._high_threshold:
            route_name = self._fallback_route
            logger.info(
                f"In class QueryRouter, function route: Medium confidence ({confidence:.3f}), "
                f"routing to {self._fallback_route} for decision"
            )
        else:
            logger.info(
                f"In class QueryRouter, function route: High confidence ({confidence:.3f}), "
                f"routing directly to {route_name}"
            )

        timestamp = datetime.now(timezone.utc).isoformat()

        route_result = RouteResult(
            query=query,
            route=route_name,
            confidence=confidence,
            cache_hit=cache_hit,
            timestamp=timestamp,
        )

        # Store in cache
        if self._cache is not None:
            if embedding is None:
                embedding = self._encode_query(query)
            self._cache.put(query, embedding, route_name, confidence)

        return route_result

    def get_cache_stats(self) -> Optional[CacheStats]:
        """
        Get cache statistics if caching is enabled.

        :return: CacheStats object or None if cache disabled.
        """
        if self._cache is None:
            return None
        return self._cache.get_stats()

    @property
    def route_names(self) -> List[str]:
        """
        Get list of configured route names.

        :return: List of route name strings.
        """
        return [r.name for r in self._routes]

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.

        :return: Configuration dictionary.
        """
        return self._config


# ── Module-level convenience ──────────────────────────────────────────────────

_router: Optional[QueryRouter] = None


def get_router(config_path: Optional[Path] = None) -> QueryRouter:
    """
    Get or create the global router instance.

    Description: Lazy initialization of router. Pass config_path on first
        call to override default configuration.

    :param config_path: Path to routes.yaml config file.
    :return: QueryRouter instance.
    """
    global _router
    if _router is None:
        _router = QueryRouter(config_path=config_path)
    return _router


def route_query(query: str) -> RouteResult:
    """
    Route a query using the global router instance.

    Description: Convenience function that uses the module-level router.

    :param query: User query string to route.
    :return: RouteResult containing routing decision and metadata.
    """
    router = get_router()
    return router.route(query)
