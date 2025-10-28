"""
Package pour la gestion des bases vectorielles (vector stores).

Ce package fournit les outils pour créer, sauvegarder, charger et rechercher
dans des bases vectorielles FAISS.
"""

from .vectors import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    search_similar_documents,
    add_documents_to_vector_store,
    delete_vector_store,
    get_vector_store_stats,
)

__all__ = [
    "create_vector_store",
    "save_vector_store",
    "load_vector_store",
    "search_similar_documents",
    "add_documents_to_vector_store",
    "delete_vector_store",
    "get_vector_store_stats",
]
