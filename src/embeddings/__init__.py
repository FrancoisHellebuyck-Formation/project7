"""
Package pour la génération d'embeddings.

Ce package fournit les outils pour générer des embeddings vectoriels
multilingues en utilisant le modèle E5.
"""

from .embeddings import E5Embeddings, get_embeddings_model

__all__ = ["E5Embeddings", "get_embeddings_model"]
