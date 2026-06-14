"""
Router package for Ganglion multi-agent system.

Handles semantic routing of user queries to specialist agents
via cosine similarity matching and LiteLLM model gateway.
"""

from ganglion.router.routes import QueryRouter, RouteResult, get_router, route_query
from ganglion.router.config import load_litellm_config, apply_litellm_config, get_model_config
from ganglion.router.cache import (
    AbstractSemanticCache,
    InMemorySemanticCache,
    CacheEntry,
    CacheStats,
    create_cache,
)

__all__ = [
    # Router
    "QueryRouter",
    "RouteResult",
    "get_router",
    "route_query",
    # Config
    "load_litellm_config",
    "apply_litellm_config",
    "get_model_config",
    # Cache
    "AbstractSemanticCache",
    "InMemorySemanticCache",
    "CacheEntry",
    "CacheStats",
    "create_cache",
]
