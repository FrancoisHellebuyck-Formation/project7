"""
Modèles Pydantic pour l'API FastAPI.

Ce module contient tous les modèles de données utilisés pour les requêtes
et réponses de l'API de recherche d'événements culturels.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Modèles pour la recherche sémantique
# ============================================================================

class SearchQuery(BaseModel):
    """Modèle pour une requête de recherche."""
    query: str = Field(..., description="Texte de la requête de recherche", min_length=1)
    k: int = Field(5, description="Nombre de résultats à retourner", ge=1, le=100)


class SearchResult(BaseModel):
    """Modèle pour un résultat de recherche."""
    score: float = Field(..., description="Score de similarité (distance L2)")
    title: str = Field(..., description="Titre de l'événement")
    content: str = Field(..., description="Contenu du document")
    location: Optional[str] = Field(None, description="Lieu de l'événement")
    metadata: dict = Field(default_factory=dict, description="Métadonnées supplémentaires")


class SearchResponse(BaseModel):
    """Modèle pour la réponse de recherche."""
    query: str = Field(..., description="Requête effectuée")
    results: List[SearchResult] = Field(..., description="Liste des résultats")
    total_results: int = Field(..., description="Nombre de résultats retournés")


# ============================================================================
# Modèles pour le chatbot avec RAG
# ============================================================================

class AskQuery(BaseModel):
    """Modèle pour une question avec RAG."""
    question: str = Field(..., description="Question de l'utilisateur", min_length=1)
    k: int = Field(5, description="Nombre de documents de contexte à récupérer", ge=1, le=20)
    system_prompt: Optional[str] = Field(None, description="Prompt système personnalisé (optionnel)")


class AskResponse(BaseModel):
    """Modèle pour la réponse à une question."""
    question: str = Field(..., description="Question posée")
    answer: str = Field(..., description="Réponse générée par Mistral AI")
    context_used: List[SearchResult] = Field(..., description="Documents utilisés comme contexte")
    tokens_used: dict = Field(..., description="Statistiques d'utilisation des tokens")


# ============================================================================
# Modèles pour les métadonnées et statistiques
# ============================================================================

class StatsResponse(BaseModel):
    """Modèle pour les statistiques du vector store."""
    num_vectors: int = Field(..., description="Nombre de vecteurs dans l'index")
    dimension: int = Field(..., description="Dimension des vecteurs")
    index_path: str = Field(..., description="Chemin du vector store")


class HealthResponse(BaseModel):
    """Modèle pour le health check."""
    status: str = Field(..., description="Statut de l'API")
    vector_store_loaded: bool = Field(..., description="Indique si le vector store est chargé")
    embeddings_model_loaded: bool = Field(..., description="Indique si le modèle d'embeddings est chargé")
    mistral_client_loaded: bool = Field(..., description="Indique si le client Mistral AI est chargé")
