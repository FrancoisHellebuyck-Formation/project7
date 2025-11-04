"""
Package API pour l'interrogation du vector store FAISS.
"""

from .main import app
from .models import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    AskQuery,
    AskResponse,
    StatsResponse,
    HealthResponse,
)

__all__ = [
    "app",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "AskQuery",
    "AskResponse",
    "StatsResponse",
    "HealthResponse",
]
